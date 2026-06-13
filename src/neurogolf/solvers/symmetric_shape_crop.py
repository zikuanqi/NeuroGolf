"""Solver: crop the one mirror-symmetric shape (task 174).

Several coloured shapes are scattered on the grid; exactly one is symmetric
under horizontal mirroring (within its bounding box).  The output is that
shape's bounding-box crop::

    2 2 .      7 7      6 6 6 6          6 6 6 6
    . 2 2      7 . 7    . 6 6 .    ->    . 6 6 .   (the symmetric one)

All nine colours are processed at once on the channel axis: per-channel bbox
bounds come from ``ArgMax`` / ``ReduceMax`` over (1,10,30,1) projections, and
the mirror test plus the final crop-to-top-left use **batched per-channel
``MatMul`` shift matrices** (``S = |D - delta_v| < 0.5`` with ``delta_v`` a
per-channel scalar broadcast against the static offset matrix ``D``).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from ..grids import CHANNELS, HEIGHT, WIDTH, all_examples

OPSET = 11
IR_VERSION = 8
F = TensorProto.FLOAT


def _ref(g: np.ndarray) -> Optional[np.ndarray]:
    best = None
    nq = 0
    for v in np.unique(g):
        if v == 0:
            continue
        ys, xs = np.where(g == v)
        crop = (g[ys.min():ys.max() + 1, xs.min():xs.max() + 1] == v).astype(int) * int(v)
        if np.array_equal(crop, crop[:, ::-1]):
            nq += 1
            best = crop
    if nq != 1:
        return None
    return best


def _detect(task: dict) -> bool:
    saw = False
    for ex in all_examples(task):
        i, o = ex["input"], ex["output"]
        if not i or not i[0] or len(i) > HEIGHT or len(i[0]) > WIDTH:
            continue
        r = _ref(np.array(i))
        if r is None or r.shape != np.array(o).shape or not np.array_equal(r, np.array(o)):
            return False
        saw = True
    return saw


def _build() -> onnx.ModelProto:
    n = helper.make_node
    e0 = np.zeros((1, CHANNELS, 1, 1), np.float32); e0[0, 0] = 1.0
    notbg = np.ones((1, CHANNELS, 1, 1), np.float32); notbg[0, 0] = 0.0
    idx = np.arange(WIDTH)
    D2 = (idx[None, :] - idx[:, None]).astype(np.float32).reshape(1, 1, WIDTH, WIDTH)  # D2[j,c]=c-j
    Pflip = np.zeros((1, 1, WIDTH, WIDTH), np.float32)
    for c in range(WIDTH):
        Pflip[0, 0, WIDTH - 1 - c, c] = 1.0          # F = mask @ Pflip -> mask[:, 29-c]
    init = [
        numpy_helper.from_array(e0, "e0"),
        numpy_helper.from_array(notbg, "notbg"),
        numpy_helper.from_array(D2, "D2"),
        numpy_helper.from_array(Pflip, "Pflip"),
        numpy_helper.from_array(np.arange(HEIGHT, dtype=np.float32).reshape(1, 1, HEIGHT, 1), "ah"),
        numpy_helper.from_array(np.arange(WIDTH, dtype=np.float32).reshape(1, 1, 1, WIDTH), "aw"),
        numpy_helper.from_array(np.array(0.5, np.float32), "half"),
        numpy_helper.from_array(np.array(1.0, np.float32), "one"),
        numpy_helper.from_array(np.array(float(WIDTH - 1), np.float32), "w29"),
    ]
    nodes = [
        # per-channel bbox bounds (1,10,1,1)
        n("ReduceMax", ["input"], ["rowHas"], axes=[3], keepdims=1),
        n("ReduceMax", ["input"], ["colHas"], axes=[2], keepdims=1),
        n("ArgMax", ["rowHas"], ["minr_i"], axis=2, keepdims=1), n("Cast", ["minr_i"], ["minr"], to=F),
        n("ArgMax", ["colHas"], ["minc_i"], axis=3, keepdims=1), n("Cast", ["minc_i"], ["minc"], to=F),
        n("Mul", ["rowHas", "ah"], ["rp"]), n("ReduceMax", ["rp"], ["maxr"], axes=[2], keepdims=1),
        n("Mul", ["colHas", "aw"], ["cp"]), n("ReduceMax", ["cp"], ["maxc"], axes=[3], keepdims=1),
        n("ReduceSum", ["input"], ["cnt"], axes=[2, 3], keepdims=1),
        # mirror test: aligned flip == mask ?
        n("MatMul", ["input", "Pflip"], ["Fm"]),
        n("Add", ["minc", "maxc"], ["mm"]),
        n("Sub", ["mm", "w29"], ["delta"]),
        n("Sub", ["D2", "delta"], ["dd"]), n("Abs", ["dd"], ["dda"]),
        n("Less", ["dda", "half"], ["S2_b"]), n("Cast", ["S2_b"], ["S2"], to=F),
        n("MatMul", ["Fm", "S2"], ["aligned"]),
        n("Sub", ["input", "aligned"], ["df"]), n("Abs", ["df"], ["dfa"]),
        n("ReduceSum", ["dfa"], ["mism"], axes=[2, 3], keepdims=1),
        n("Less", ["mism", "half"], ["sy_b"]), n("Cast", ["sy_b"], ["sy0"], to=F),
        n("Greater", ["cnt", "half"], ["hs_b"]), n("Cast", ["hs_b"], ["hs"], to=F),
        n("Mul", ["sy0", "hs"], ["sy1"]),
        n("Mul", ["sy1", "notbg"], ["sym"]),                       # (1,10,1,1)
        # crop to top-left: rows up by minr, cols left by minc (batched MatMul)
        n("Neg", ["D2"], ["D2n"]),                                  # D2n[j,c] = j-c
        n("Sub", ["D2", "minr"], ["dr_"]), n("Abs", ["dr_"], ["dra"]),
        n("Less", ["dra", "half"], ["Sr_b"]), n("Cast", ["Sr_b"], ["Sr"], to=F),
        n("MatMul", ["Sr", "input"], ["rsh"]),
        n("Sub", ["D2n", "minc"], ["dc_"]), n("Abs", ["dc_"], ["dca"]),
        n("Less", ["dca", "half"], ["Sc_b"]), n("Cast", ["Sc_b"], ["Sc"], to=F),
        n("MatMul", ["rsh", "Sc"], ["shifted"]),
        n("Mul", ["shifted", "sym"], ["sel"]),                      # only the symmetric shape
        # window of the selected bbox
        n("Sub", ["maxr", "minr"], ["hm1"]), n("Add", ["hm1", "one"], ["hv"]),
        n("Sub", ["maxc", "minc"], ["wm1"]), n("Add", ["wm1", "one"], ["wv"]),
        n("Mul", ["hv", "sym"], ["hsel0"]), n("ReduceSum", ["hsel0"], ["hsel"], axes=[1], keepdims=1),
        n("Mul", ["wv", "sym"], ["wsel0"]), n("ReduceSum", ["wsel0"], ["wsel"], axes=[1], keepdims=1),
        n("Less", ["ah", "hsel"], ["ir_b"]), n("Cast", ["ir_b"], ["ir"], to=F),
        n("Less", ["aw", "wsel"], ["ic_b"]), n("Cast", ["ic_b"], ["ic"], to=F),
        n("Mul", ["ir", "ic"], ["win"]),
        n("ReduceSum", ["sel"], ["cm"], axes=[1], keepdims=1),
        n("Sub", ["win", "cm"], ["bgm"]),
        n("Mul", ["bgm", "e0"], ["bgp"]),
        n("Add", ["sel", "bgp"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "symmetric_shape_crop",
                              [helper.make_tensor_value_info("input", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              [helper.make_tensor_value_info("output", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_symmetric_shape_crop(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
