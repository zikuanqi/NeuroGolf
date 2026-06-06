"""Solver: grow a single-row colour run into a staircase (task 295).

The input is one row of width W with a left-aligned run of K cells of colour C.
The output is a (W/2) x W grid whose row r is the run grown to K + r cells
(left-aligned), the rest of the grid being background.

Build: C is the present non-background channel, K its cell count, W the content
width; the staircase mask is `col < K + row` intersected with the (W/2) x W
region, painted C, and the remaining in-region cells rebuilt as background 0.
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
    if H != 1 or W % 2 != 0:
        return None
    row = g[0]
    nz = row[row != 0]
    if len(nz) == 0:
        return None
    C = int(nz[0])
    K = int((row == C).sum())
    if not (row[:K] == C).all() or (row[K:] != 0).any():
        return None
    OH = W // 2
    out = np.zeros((OH, W), int)
    for r in range(OH):
        out[r, :min(K + r, W)] = C
    return out


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
    F = TensorProto.FLOAT
    n = helper.make_node
    row_idx = np.arange(HEIGHT, dtype=np.float32).reshape(1, 1, HEIGHT, 1)
    col_idx = np.arange(WIDTH, dtype=np.float32).reshape(1, 1, 1, WIDTH)
    note0 = np.ones((1, CHANNELS, 1, 1), np.float32); note0[0, 0] = 0.0
    e0 = np.zeros((1, CHANNELS, 1, 1), np.float32); e0[0, 0] = 1.0
    init = [
        numpy_helper.from_array(row_idx, "row_idx"),
        numpy_helper.from_array(col_idx, "col_idx"),
        numpy_helper.from_array(note0, "note0"),
        numpy_helper.from_array(e0, "e0"),
        numpy_helper.from_array(np.array(0.5, np.float32), "half"),
        numpy_helper.from_array(np.array(1.0, np.float32), "one"),
    ]
    nodes = [
        n("ReduceSum", ["input"], ["tot"], axes=[2, 3], keepdims=1),       # (1,10,1,1)
        n("Greater", ["tot", "half"], ["pres_b"]), n("Cast", ["pres_b"], ["present"], to=F),
        n("Mul", ["present", "note0"], ["eC"]),                            # (1,10,1,1)
        n("Mul", ["tot", "eC"], ["Kc"]),
        n("ReduceSum", ["Kc"], ["K"]),                                     # (1,1,1,1)
        n("ReduceSum", ["input"], ["content"], axes=[1], keepdims=1),
        n("ReduceMax", ["content"], ["col_has"], axes=[2], keepdims=1),    # (1,1,1,W)
        n("ReduceSum", ["col_has"], ["W"]),                                # (1,1,1,1)
        n("Mul", ["W", "half"], ["Hh"]),
        n("Add", ["K", "row_idx"], ["thr"]),                               # (1,1,H,1)
        n("Less", ["col_idx", "thr"], ["lt_b"]), n("Cast", ["lt_b"], ["lt"], to=F),  # (1,1,H,W)
        n("Less", ["row_idx", "Hh"], ["rm_b"]), n("Cast", ["rm_b"], ["rmask"], to=F),
        n("Less", ["col_idx", "W"], ["cm_b"]), n("Cast", ["cm_b"], ["cmask"], to=F),
        n("Mul", ["lt", "rmask"], ["m1"]),
        n("Mul", ["m1", "cmask"], ["mask"]),                               # (1,1,H,W)
        n("Mul", ["rmask", "cmask"], ["gridmask"]),
        n("Sub", ["one", "mask"], ["notmask"]),
        n("Mul", ["gridmask", "notmask"], ["ch0v"]),
        n("Mul", ["eC", "mask"], ["paintC"]),
        n("Mul", ["e0", "ch0v"], ["paintBg"]),
        n("Add", ["paintC", "paintBg"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "staircase",
                              [helper.make_tensor_value_info("input", F, FULL)],
                              [helper.make_tensor_value_info("output", F, FULL)],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_staircase(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
