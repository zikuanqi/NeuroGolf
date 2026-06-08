"""Solver: recolour every third column's 4-cells to 6 (task 292).

The grid holds a regular 4/0 texture; the output repaints the colour-4 cells
whose column index is a multiple of 3 to colour 6::

    4 0 4 0 4 0 4      6 0 4 0 4 0 6
    4 4 4 4 4 4 4  ->  6 4 4 6 4 4 6
    0 4 0 4 0 4 0      0 4 0 6 0 4 0

Build: a baked column mask ``col % 3 == 0`` intersected with the channel-4 mask
selects the cells; ``e_6`` is painted there.
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
    C = np.arange(W).reshape(1, -1)
    mask = ((C % 3) == 0) & (g == 4)
    if not mask.any():
        return None
    out = g.copy()
    out[mask] = 6
    return out if not np.array_equal(out, g) else None


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

    colmask = np.zeros((1, 1, 1, WIDTH), np.float32)
    colmask[0, 0, 0, ::3] = 1.0
    e6 = np.zeros((1, CHANNELS, 1, 1), np.float32); e6[0, 6] = 1.0
    init = [
        numpy_helper.from_array(colmask, "colmask"),
        numpy_helper.from_array(e6, "e6"),
        numpy_helper.from_array(np.array(1.0, np.float32), "one"),
        numpy_helper.from_array(np.array([4], np.int64), "c4s"),
        numpy_helper.from_array(np.array([5], np.int64), "c5e"),
        numpy_helper.from_array(np.array([1], np.int64), "ax1"),
    ]
    nodes = [
        n("Slice", ["input", "c4s", "c5e", "ax1"], ["is4"]),       # (1,1,H,W)
        n("Mul", ["is4", "colmask"], ["paint"]),                   # 4-cells in col%3==0
        n("Sub", ["one", "paint"], ["inv"]),
        n("Mul", ["input", "inv"], ["kept"]),
        n("Mul", ["e6", "paint"], ["addc"]),
        n("Add", ["kept", "addc"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "col3_recolor",
                              [helper.make_tensor_value_info("input", F, FULL)],
                              [helper.make_tensor_value_info("output", F, FULL)],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_col3_recolor(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
