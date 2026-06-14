"""Solver: fill the single-cell gap between two horizontal 1s with 2 (task 258).

Any background cell that has a ``1`` immediately to its left and a ``1``
immediately to its right becomes ``2``::

    1 0 1   ->   1 2 1

A pure neighbour rule: shift the ``1`` mask left and right by one column and
intersect with the background mask.
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
        for c in range(1, W - 1):
            if g[r, c] == 0 and g[r, c - 1] == 1 and g[r, c + 1] == 1:
                out[r, c] = 2
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
    n = helper.make_node
    e2me0 = np.zeros((1, CHANNELS, 1, 1), np.float32); e2me0[0, 2] = 1.0; e2me0[0, 0] = -1.0
    init = [
        numpy_helper.from_array(e2me0, "e2me0"),
        numpy_helper.from_array(np.array([0], np.int64), "c0"),
        numpy_helper.from_array(np.array([1], np.int64), "c1"),
        numpy_helper.from_array(np.array([2], np.int64), "c2"),
        numpy_helper.from_array(np.array([1], np.int64), "ax1"),
        numpy_helper.from_array(np.array([3], np.int64), "ax3"),
        numpy_helper.from_array(np.array([0], np.int64), "s0"),
        numpy_helper.from_array(np.array([1], np.int64), "s1"),
        numpy_helper.from_array(np.array([WIDTH], np.int64), "sW"),
        numpy_helper.from_array(np.array([WIDTH + 1], np.int64), "sW1"),
        numpy_helper.from_array(np.array([0, 0, 0, 1, 0, 0, 0, 0], np.int64), "padL"),
        numpy_helper.from_array(np.array([0, 0, 0, 0, 0, 0, 0, 1], np.int64), "padR"),
    ]
    nodes = [
        n("Slice", ["input", "c0", "c1", "ax1"], ["is0"]),
        n("Slice", ["input", "c1", "c2", "ax1"], ["is1"]),
        # is1 shifted right (value of the left neighbour at this cell)
        n("Pad", ["is1", "padL"], ["padLeft"]),
        n("Slice", ["padLeft", "s0", "sW", "ax3"], ["is1L"]),
        # is1 shifted left (value of the right neighbour at this cell)
        n("Pad", ["is1", "padR"], ["padRight"]),
        n("Slice", ["padRight", "s1", "sW1", "ax3"], ["is1R"]),
        n("Mul", ["is1L", "is1R"], ["both"]),
        n("Mul", ["both", "is0"], ["fill"]),
        n("Mul", ["fill", "e2me0"], ["delta"]),
        n("Add", ["input", "delta"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "midpoint_fill_h",
                              [helper.make_tensor_value_info("input", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              [helper.make_tensor_value_info("output", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_midpoint_fill_h(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
