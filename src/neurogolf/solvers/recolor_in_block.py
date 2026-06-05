"""Solver: recolour one colour inside another colour's bounding box (task 70).

A "region" colour R occupies a rectangular area; every source-colour (S) cell
inside R's bounding box becomes the destination colour (D).  R, S and D are
fixed per task and detected, then baked.

Build: slice channel R, take the bbox span as (any R in this row) x (any R in
this column) - flooded with a log-doubling shift-`Max` so interior rows/cols
with no R cell are still inside the box - intersect with the channel-S cells,
and add `e_D - e_S` there.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from ..grids import CHANNELS, HEIGHT, WIDTH, all_examples

OPSET = 11
IR_VERSION = 8
FULL = [1, CHANNELS, HEIGHT, WIDTH]


def _ref(g: np.ndarray, R: int, S: int, D: int) -> Optional[np.ndarray]:
    ys, xs = np.where(g == R)
    if len(ys) == 0:
        return None
    r0, r1, c0, c1 = ys.min(), ys.max(), xs.min(), xs.max()
    out = g.copy()
    sub = out[r0:r1 + 1, c0:c1 + 1]
    sub[sub == S] = D
    out[r0:r1 + 1, c0:c1 + 1] = sub
    return out


def _params(task: dict) -> Optional[tuple[int, int, int]]:
    exs = []
    src = dst = None
    for ex in all_examples(task):
        i, o = ex["input"], ex["output"]
        if not i or not i[0] or len(i) > HEIGHT or len(i[0]) > WIDTH:
            continue
        g = np.array(i); out = np.array(o)
        if g.shape != out.shape:
            return None
        exs.append((g, out))
        d = g != out
        if d.any():
            ss, dd = set(g[d].tolist()), set(out[d].tolist())
            if len(ss) != 1 or len(dd) != 1:
                return None
            if src is None:
                src, dst = ss.pop(), dd.pop()
            elif src != next(iter(ss)) or dst != next(iter(dd)):
                return None
    if not exs or src is None:
        return None
    cands = set()
    for g, _ in exs:
        cands |= set(int(c) for c in np.unique(g))
    cands -= {src, dst}
    for R in sorted(cands):
        if all(_ref(g, R, src, dst) is not None and np.array_equal(_ref(g, R, src, dst), o)
               for g, o in exs):
            return R, int(src), int(dst)
    return None


def _build(R: int, S: int, D: int) -> onnx.ModelProto:
    F = TensorProto.FLOAT
    n = helper.make_node

    def chan_slice(ch, lo, hi):
        return (np.array([0, ch, 0, 0], np.int64), np.array([1, ch + 1, HEIGHT, WIDTH], np.int64))

    e_delta = np.zeros((1, CHANNELS, 1, 1), np.float32)
    e_delta[0, D] = 1.0
    e_delta[0, S] = -1.0
    init = [
        numpy_helper.from_array(np.array([0, R, 0, 0], np.int64), "r_s"),
        numpy_helper.from_array(np.array([1, R + 1, HEIGHT, WIDTH], np.int64), "r_e"),
        numpy_helper.from_array(np.array([0, S, 0, 0], np.int64), "s_s"),
        numpy_helper.from_array(np.array([1, S + 1, HEIGHT, WIDTH], np.int64), "s_e"),
        numpy_helper.from_array(np.array([0, 1, 2, 3], np.int64), "ax4"),
        numpy_helper.from_array(np.array([1], np.int64), "st1"),
        numpy_helper.from_array(np.array(0.0, np.float32), "zero"),
        numpy_helper.from_array(e_delta, "e_delta"),
    ]
    nodes = [
        n("Slice", ["input", "r_s", "r_e", "ax4"], ["chR"]),            # (1,1,H,W)
        n("Slice", ["input", "s_s", "s_e", "ax4"], ["chS"]),
        n("ReduceMax", ["chR"], ["row_has"], axes=[3], keepdims=1),     # (1,1,H,1)
        n("ReduceMax", ["chR"], ["col_has"], axes=[2], keepdims=1),     # (1,1,1,W)
    ]

    # flood helper: prefix/suffix max along an axis (log-doubling)
    def flood(src, axis, hi, tag):
        D_ = HEIGHT if axis == 2 else WIDTH
        cur = src; d = 1; s = 0
        while d < D_:
            pad = [0, 0, 0, 0, 0, 0, 0, 0]
            if hi:
                pad[4 + axis] = d; slo, shi = d, D_ + d
            else:
                pad[axis] = d; slo, shi = 0, D_
            init.append(numpy_helper.from_array(np.array(pad, np.int64), f"{tag}pad{s}"))
            init.append(numpy_helper.from_array(np.array([slo], np.int64), f"{tag}lo{s}"))
            init.append(numpy_helper.from_array(np.array([shi], np.int64), f"{tag}hi{s}"))
            init.append(numpy_helper.from_array(np.array([axis], np.int64), f"{tag}ax{s}"))
            nodes.append(n("Pad", [cur, f"{tag}pad{s}", "zero"], [f"{tag}p{s}"]))
            nodes.append(n("Slice", [f"{tag}p{s}", f"{tag}lo{s}", f"{tag}hi{s}",
                                     f"{tag}ax{s}", "st1"], [f"{tag}sl{s}"]))
            nodes.append(n("Max", [cur, f"{tag}sl{s}"], [f"{tag}m{s}"]))
            cur = f"{tag}m{s}"; d *= 2; s += 1
        return cur

    rpre = flood("row_has", 2, False, "rp")   # any R at or above
    rsuf = flood("row_has", 2, True, "rs")     # any R at or below
    cpre = flood("col_has", 3, False, "cp")
    csuf = flood("col_has", 3, True, "cs")
    nodes += [
        n("Mul", [rpre, rsuf], ["row_span"]),     # (1,1,H,1)
        n("Mul", [cpre, csuf], ["col_span"]),     # (1,1,1,W)
        n("Mul", ["row_span", "col_span"], ["bbox"]),  # (1,1,H,W)
        n("Mul", ["chS", "bbox"], ["inside_S"]),
        n("Mul", ["e_delta", "inside_S"], ["delta"]),  # (1,10,H,W)
        n("Add", ["input", "delta"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "recolor_in_block",
                              [helper.make_tensor_value_info("input", F, FULL)],
                              [helper.make_tensor_value_info("output", F, FULL)],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_recolor_in_block(task: dict) -> Optional[onnx.ModelProto]:
    p = _params(task)
    if p is None:
        return None
    return _build(*p)
