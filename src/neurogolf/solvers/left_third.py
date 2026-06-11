"""Solver: keep the left third of a horizontally 3x-tiled grid (task 67).

The input is a square block repeated three times across the width; the output
is one copy::

    A A A | A A A | A A A   ->   A A A     (input W -> output W/3)

The grid width ``W`` is recovered as the number of non-padding columns; every
column at index ``>= W/3`` is masked to all-zero (padding), leaving the leftmost
block at the top-left.
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
    if W < 3 or W % 3:
        return None
    return g[:, :W // 3]


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
    init = [
        numpy_helper.from_array(np.arange(WIDTH, dtype=np.float32).reshape(1, 1, 1, WIDTH), "aw"),
        numpy_helper.from_array(np.array(3.0, np.float32), "three"),
    ]
    nodes = [
        n("ReduceSum", ["input"], ["occ"], axes=[1], keepdims=1),       # 1 in grid, 0 in padding
        n("ReduceMax", ["occ"], ["colData"], axes=[2], keepdims=1),      # (1,1,1,30) grid columns
        n("ReduceSum", ["colData"], ["W"], axes=[3], keepdims=1),        # grid width
        n("Div", ["W", "three"], ["third"]),
        n("Less", ["aw", "third"], ["keepb"]),
        n("Cast", ["keepb"], ["keep"], to=F),
        n("Mul", ["input", "keep"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "left_third",
                              [helper.make_tensor_value_info("input", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              [helper.make_tensor_value_info("output", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_left_third(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
