"""Solver: replace every marker with a fixed 3x3 colour motif (a stamp).

Each non-background cell is the centre of a fixed 3x3 stamp that is painted into
the output; the stamp (detected from the task) may recolour the centre too
(task 282). Implemented as one 3x3 convolution per stamp colour over the marker
mask -- the kernel is the stamp reflected 180 so cross-correlation lands each
offset in the right place.

One-hot channel arithmetic on the (1, 10, 30, 30) tensor:
  nonbg   = marker cells
  conv_k  = marker mask convolved with the (flipped) "stamp == k" kernel
  output  = colour k wherever conv_k fires (clipped to the grid) + background
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from ..grids import CHANNELS, HEIGHT, WIDTH, all_examples


OPSET = 11
IR_VERSION = 8


def _markers(g):
    return [(r, c) for r in range(len(g)) for c in range(len(g[0])) if g[r][c] != 0]


def _extract_stamp(task: dict):
    stamp, colors = {}, set()
    for ex in all_examples(task):
        i, o = ex["input"], ex["output"]
        if len(i) > 30 or len(i[0]) > 30:
            continue
        if len(i) != len(o) or len(i[0]) != len(o[0]):
            return None
        H, W = len(i), len(i[0])
        for (r, c) in _markers(i):
            colors.add(i[r][c])
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr, cc = r + dr, c + dc
                    val = o[rr][cc] if (0 <= rr < H and 0 <= cc < W) else 0
                    k = (dr, dc)
                    if stamp.get(k, val) != val:
                        return None
                    stamp[k] = val
    if len(colors) != 1 or not stamp:
        return None
    return stamp


def _apply(g, stamp):
    H, W = len(g), len(g[0])
    out = [[0] * W for _ in range(H)]
    for (r, c) in _markers(g):
        for (dr, dc), val in stamp.items():
            rr, cc = r + dr, c + dc
            if val and 0 <= rr < H and 0 <= cc < W:
                out[rr][cc] = val
    return out


def _params(task: dict):
    stamp = _extract_stamp(task)
    if stamp is None:
        return None
    changed = False
    for ex in all_examples(task):
        i, o = ex["input"], ex["output"]
        if len(i) > 30 or len(i[0]) > 30:
            continue
        if _apply(i, stamp) != o:
            return None
        if i != o:
            changed = True
    return stamp if changed else None


def _build(stamp: dict) -> onnx.ModelProto:
    F = TensorProto.FLOAT
    B = TensorProto.BOOL
    colors = sorted({v for v in stamp.values() if v != 0})

    def f32(name, arr):
        return numpy_helper.from_array(arr.astype(np.float32), name)

    def onehot(ch):
        a = np.zeros((1, CHANNELS, 1, 1), dtype=np.float32)
        a[0, ch, 0, 0] = 1.0
        return a

    init = [f32("e0", onehot(0)), f32("one", np.array([[[[1.0]]]])),
            f32("half", np.array([[[[0.5]]]])),
            numpy_helper.from_array(np.array([0], dtype=np.int64), "idx0")]
    for k in colors:
        ker = np.zeros((1, 1, 3, 3), dtype=np.float32)
        for (dr, dc), v in stamp.items():
            if v == k:
                ker[0, 0, 1 - dr, 1 - dc] = 1.0      # 180-flip for cross-corr
        init.append(f32(f"ker_{k}", ker))
        init.append(f32(f"e_{k}", onehot(k)))

    n = helper.make_node
    nodes = [
        n("ReduceSum", ["input"], ["content"], axes=[1], keepdims=1),
        n("Gather", ["input", "idx0"], ["ch0"], axis=1),
        n("Sub", ["content", "ch0"], ["nonbg"]),
    ]
    contribs, covers = [], []
    for k in colors:
        nodes += [
            n("Conv", ["nonbg", f"ker_{k}"], [f"conv_{k}"],
              kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
            n("Greater", [f"conv_{k}", "half"], [f"onb_{k}"]),
            n("Cast", [f"onb_{k}"], [f"on_{k}"], to=F),
            n("Mul", [f"on_{k}", "content"], [f"onc_{k}"]),
            n("Mul", [f"e_{k}", f"onc_{k}"], [f"contrib_{k}"]),
        ]
        contribs.append(f"contrib_{k}")
        covers.append(f"onc_{k}")

    # covered = sum of onc_k
    cov = covers[0]
    for j, name in enumerate(covers[1:], 1):
        out = f"cov{j}" if j < len(covers) - 1 else "covered"
        nodes.append(n("Add", [cov, name], [out]))
        cov = out
    if len(covers) == 1:
        nodes.append(n("Identity", [cov], ["covered"]))
    nodes += [
        n("Sub", ["one", "covered"], ["notcov"]),
        n("Mul", ["content", "notcov"], ["bg_cells"]),
        n("Mul", ["e0", "bg_cells"], ["out_bg"]),
    ]
    # output = out_bg + sum(contribs)
    acc = "out_bg"
    for j, name in enumerate(contribs):
        out = f"acc{j}" if j < len(contribs) - 1 else "output"
        nodes.append(n("Add", [acc, name], [out]))
        acc = out

    def vi(name, shape, dt=F):
        return helper.make_tensor_value_info(name, dt, shape)

    g4 = [1, CHANNELS, HEIGHT, WIDTH]
    s1 = [1, 1, HEIGHT, WIDTH]
    value_info = [vi("content", s1), vi("ch0", s1), vi("nonbg", s1),
                  vi("covered", s1), vi("notcov", s1), vi("bg_cells", s1),
                  vi("out_bg", g4)]
    for k in colors:
        value_info += [vi(f"conv_{k}", s1), vi(f"onb_{k}", s1, B),
                       vi(f"on_{k}", s1), vi(f"onc_{k}", s1),
                       vi(f"contrib_{k}", g4)]
    for j in range(1, len(covers) - 1):
        value_info.append(vi(f"cov{j}", s1))
    for j in range(len(contribs) - 1):
        value_info.append(vi(f"acc{j}", g4))

    graph = helper.make_graph(nodes, "stamp", [vi("input", g4)],
                              [vi("output", g4)], initializer=init,
                              value_info=value_info)
    return helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
        ir_version=IR_VERSION)


def solve_stamp(task: dict) -> Optional[onnx.ModelProto]:
    params = _params(task)
    if params is None:
        return None
    return _build(params)
