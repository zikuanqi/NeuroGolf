"""Solver: float each colour bar up by its own height (task 128).

Every colour occupies a single solid bar; each bar is translated upward by its
own height (a bar in rows [r0, r1] of height h = r1-r0+1 moves to [r0-h, r1-h]).

The per-colour shift is done with a batched `MatMul`: build a (10, 30, 30)
stack of shift matrices M where M[c, i, j] = 1 iff j == i + h_c, so
`M @ channels` sends output row i to input row i + h_c (a clean upward shift of
each colour channel).  Colour 0 is excluded from the shift and rebuilt as the
in-grid background afterwards.
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


def _ref(g: np.ndarray) -> Optional[np.ndarray]:
    H, W = g.shape
    out = np.zeros_like(g)
    moved = False
    for c in [int(x) for x in np.unique(g) if x != 0]:
        ys, xs = np.where(g == c)
        r0, r1 = ys.min(), ys.max()
        h = r1 - r0 + 1
        for (y, x) in zip(ys, xs):
            ny = y - h
            if 0 <= ny < H:
                out[ny, x] = c
        if h > 0:
            moved = True
    return out if moved else None


def _detect(task: dict) -> bool:
    saw = False
    for ex in all_examples(task):
        i, o = ex["input"], ex["output"]
        if not i or not i[0] or len(i) > HEIGHT or len(i[0]) > WIDTH:
            continue
        r = _ref(np.array(i))
        if r is None or not np.array_equal(r, np.array(o)):
            return False
        saw = True
    return saw


def _build() -> onnx.ModelProto:
    F = TensorProto.FLOAT
    n = helper.make_node
    row_idx = np.arange(HEIGHT, dtype=np.float32).reshape(1, 1, HEIGHT, 1)
    jmi = (np.arange(WIDTH, dtype=np.float32).reshape(1, WIDTH)
           - np.arange(HEIGHT, dtype=np.float32).reshape(HEIGHT, 1))  # (H,W)[i,j]=j-i
    note0 = np.ones((1, CHANNELS, 1, 1), np.float32); note0[0, 0] = 0.0
    e0 = np.zeros((1, CHANNELS, 1, 1), np.float32); e0[0, 0] = 1.0
    init = [
        numpy_helper.from_array(row_idx, "row_idx"),
        numpy_helper.from_array(jmi, "jmi"),
        numpy_helper.from_array(note0, "note0"),
        numpy_helper.from_array(e0, "e0"),
        numpy_helper.from_array(np.array(0.5, np.float32), "half"),
        numpy_helper.from_array(np.array(1.0, np.float32), "one"),
        numpy_helper.from_array(np.array(1000.0, np.float32), "big"),
        numpy_helper.from_array(np.array([CHANNELS, 1, 1], np.int64), "shp_h"),
        numpy_helper.from_array(np.array([CHANNELS, HEIGHT, WIDTH], np.int64), "shp3"),
        numpy_helper.from_array(np.array([1, CHANNELS, HEIGHT, WIDTH], np.int64), "shp4"),
    ]
    nodes = [
        n("Mul", ["input", "note0"], ["ic"]),                         # colours only
        # per-colour bar height h_c
        n("ReduceMax", ["ic"], ["row_pres"], axes=[3], keepdims=1),    # (1,10,H,1)
        n("Mul", ["row_pres", "row_idx"], ["rp_idx"]),
        n("ReduceMax", ["rp_idx"], ["rmax"], axes=[2], keepdims=1),
        n("Sub", ["one", "row_pres"], ["notpres"]),
        n("Mul", ["notpres", "big"], ["nb"]),
        n("Add", ["rp_idx", "nb"], ["rmin_src"]),
        n("ReduceMin", ["rmin_src"], ["rmin"], axes=[2], keepdims=1),
        n("Sub", ["rmax", "rmin"], ["hdiff"]),
        n("Add", ["hdiff", "one"], ["h"]),                            # (1,10,1,1)
        n("Reshape", ["h", "shp_h"], ["h3"]),                         # (10,1,1)
        # shift matrices M[c,i,j] = (j == i + h_c)
        n("Sub", ["jmi", "h3"], ["mdiff"]),                           # (10,H,W) broadcast
        n("Abs", ["mdiff"], ["mabs"]),
        n("Less", ["mabs", "half"], ["m_b"]), n("Cast", ["m_b"], ["M"], to=F),  # (10,H,H)
        # batched shift
        n("Reshape", ["ic", "shp3"], ["ic_r"]),                       # (10,H,W)
        n("MatMul", ["M", "ic_r"], ["shifted_r"]),                    # (10,H,W)
        n("Reshape", ["shifted_r", "shp4"], ["shifted"]),             # (1,10,H,W)
        # rebuild background channel 0
        n("ReduceSum", ["input"], ["content"], axes=[1], keepdims=1),
        n("ReduceSum", ["shifted"], ["hasColor"], axes=[1], keepdims=1),
        n("Sub", ["one", "hasColor"], ["nocolor"]),
        n("Mul", ["content", "nocolor"], ["ch0v"]),
        n("Mul", ["e0", "ch0v"], ["bg"]),
        n("Add", ["shifted", "bg"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "float_up",
                              [helper.make_tensor_value_info("input", F, FULL)],
                              [helper.make_tensor_value_info("output", F, FULL)],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_float_up(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
