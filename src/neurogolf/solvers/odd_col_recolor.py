"""Solver: recolour the non-background cells in odd columns to 4 (task 252).

A single-colour diagonal pattern keeps its colour in even columns but turns 4 in
odd columns.

Build: a baked odd-column mask times the non-background mask selects the cells to
repaint; `e_4` is painted there and the rest of the input is kept.
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
    out = g.copy()
    changed = False
    for r in range(H):
        for c in range(W):
            if g[r, c] != 0 and c % 2 == 1:
                out[r, c] = 4
                changed = True
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
    F = TensorProto.FLOAT
    n = helper.make_node
    note0 = np.ones((1, CHANNELS, 1, 1), np.float32); note0[0, 0] = 0.0
    e4 = np.zeros((1, CHANNELS, 1, 1), np.float32); e4[0, 4] = 1.0
    oddcol = (np.arange(WIDTH) % 2 == 1).astype(np.float32).reshape(1, 1, 1, WIDTH)
    init = [
        numpy_helper.from_array(note0, "note0"),
        numpy_helper.from_array(e4, "e4"),
        numpy_helper.from_array(oddcol, "oddcol"),
        numpy_helper.from_array(np.array(1.0, np.float32), "one"),
    ]
    nodes = [
        n("Mul", ["input", "note0"], ["nb"]),
        n("ReduceSum", ["nb"], ["cmask"], axes=[1], keepdims=1),     # (1,1,H,W) non-bg
        n("Mul", ["cmask", "oddcol"], ["recolor"]),                  # non-bg in odd cols
        n("Sub", ["one", "recolor"], ["keep"]),
        n("Mul", ["input", "keep"], ["kept"]),
        n("Mul", ["e4", "recolor"], ["paint"]),
        n("Add", ["kept", "paint"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "odd_col_recolor",
                              [helper.make_tensor_value_info("input", F, FULL)],
                              [helper.make_tensor_value_info("output", F, FULL)],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_odd_col_recolor(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
