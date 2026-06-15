"""Solver: recolour every all-background row or column to 2 (task 303).

A row or column of the grid that contains no foreground cell is filled solid
``2``; everything else is left untouched.

Row/column emptiness = the grid mask minus the per-row / per-column ``ReduceMax``
of the foreground mask.
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
    H, W = g.shape
    out = g.copy()
    changed = False
    for r in range(H):
        if np.all(g[r] == 0):
            out[r, :] = 2; changed = True
    for c in range(W):
        if np.all(g[:, c] == 0):
            out[:, c] = 2; changed = True
    return out if changed else None


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
    n = helper.make_node
    e2me0 = np.zeros((1, CHANNELS, 1, 1), np.float32); e2me0[0, 2] = 1.0; e2me0[0, 0] = -1.0
    init = [
        numpy_helper.from_array(e2me0, "e2me0"),
        numpy_helper.from_array(np.array([0], np.int64), "c0"),
        numpy_helper.from_array(np.array([1], np.int64), "c1"),
        numpy_helper.from_array(np.array([1], np.int64), "ax1"),
    ]
    nodes = [
        n("ReduceSum", ["input"], ["occ"], axes=[1], keepdims=1),
        n("Slice", ["input", "c0", "c1", "ax1"], ["is0"]),
        n("Sub", ["occ", "is0"], ["nonbg"]),
        # rows
        n("ReduceMax", ["nonbg"], ["rowNb"], axes=[3], keepdims=1),
        n("ReduceMax", ["occ"], ["gridRow"], axes=[3], keepdims=1),
        n("Sub", ["gridRow", "rowNb"], ["emptyRow"]),
        # cols
        n("ReduceMax", ["nonbg"], ["colNb"], axes=[2], keepdims=1),
        n("ReduceMax", ["occ"], ["gridCol"], axes=[2], keepdims=1),
        n("Sub", ["gridCol", "colNb"], ["emptyCol"]),
        n("Max", ["emptyRow", "emptyCol"], ["emptyAny"]),
        n("Mul", ["emptyAny", "occ"], ["fill"]),
        n("Mul", ["fill", "e2me0"], ["delta"]),
        n("Add", ["input", "delta"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "empty_line_fill",
                              [helper.make_tensor_value_info("input", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              [helper.make_tensor_value_info("output", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_empty_line_fill(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
