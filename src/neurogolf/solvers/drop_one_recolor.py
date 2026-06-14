"""Solver: shift every coloured cell down one row and recolour to 2 (task 261).

The single block of colour falls one row and is repainted ``2``; cells that would
leave the grid are dropped.  Implemented as a one-row downward ``Pad``/``Slice``
shift of the non-background mask, re-confined to the grid by the occupancy mask.
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
    out = np.zeros_like(g)
    ys, xs = np.where(g != 0)
    if len(ys) == 0:
        return None
    for r, c in zip(ys, xs):
        if r + 1 < H:
            out[r + 1, c] = 2
    return out


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
    e0 = np.zeros((1, CHANNELS, 1, 1), np.float32); e0[0, 0] = 1.0
    e2 = np.zeros((1, CHANNELS, 1, 1), np.float32); e2[0, 2] = 1.0
    init = [
        numpy_helper.from_array(e0, "e0"),
        numpy_helper.from_array(e2, "e2"),
        numpy_helper.from_array(np.array([0], np.int64), "c0"),
        numpy_helper.from_array(np.array([1], np.int64), "c1"),
        numpy_helper.from_array(np.array([1], np.int64), "ax1"),
        numpy_helper.from_array(np.array([2], np.int64), "ax2"),
        numpy_helper.from_array(np.array([0], np.int64), "s0"),
        numpy_helper.from_array(np.array([HEIGHT], np.int64), "sH"),
        numpy_helper.from_array(np.array([0, 0, 1, 0, 0, 0, 0, 0], np.int64), "padTop"),
    ]
    nodes = [
        n("ReduceSum", ["input"], ["occ"], axes=[1], keepdims=1),
        n("Slice", ["input", "c0", "c1", "ax1"], ["is0"]),
        n("Sub", ["occ", "is0"], ["mark"]),
        n("Pad", ["mark", "padTop"], ["padded"]),
        n("Slice", ["padded", "s0", "sH", "ax2"], ["shifted"]),
        n("Mul", ["shifted", "occ"], ["shiftedG"]),
        n("Mul", ["shiftedG", "e2"], ["L2"]),
        n("Sub", ["occ", "shiftedG"], ["bgv"]),
        n("Mul", ["bgv", "e0"], ["L0"]),
        n("Add", ["L2", "L0"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "drop_one_recolor",
                              [helper.make_tensor_value_info("input", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              [helper.make_tensor_value_info("output", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_drop_one_recolor(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
