"""Solver: an 8-segment and a 2-segment expand into a full cross (task 299).

A short vertical ``8`` segment marks a column and a short horizontal ``2``
segment marks a row.  The output fills that whole column with ``8`` and that
whole row with ``2``, placing a ``4`` at their intersection::

    . . . . 8 .          . . . . 8 .
    . . . . 8 .          . . . . 8 .
    2 2 . . . .    ->    2 2 2 2 4 2
    . . . . . .          . . . . 8 .

Build: ``ReduceMax`` of the 8-mask over rows gives the column flag, of the
2-mask over columns the row flag; broadcast both over the real grid, paint
``e_8`` on the column, ``e_2`` on the row, ``e_4`` on the overlap.
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
    y8, x8 = np.where(g == 8); y2, x2 = np.where(g == 2)
    if len(x8) == 0 or len(y2) == 0:
        return None
    out = g.copy()
    cols8 = sorted(set(x8.tolist())); rows2 = sorted(set(y2.tolist()))
    for c in cols8:
        out[:, c] = 8
    for r in rows2:
        out[r, :] = 2
    for c in cols8:
        for r in rows2:
            out[r, c] = 4
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

    def onehot(k):
        v = np.zeros((1, CHANNELS, 1, 1), np.float32); v[0, k] = 1.0
        return v

    init = [
        numpy_helper.from_array(onehot(2), "e2"),
        numpy_helper.from_array(onehot(4), "e4"),
        numpy_helper.from_array(onehot(8), "e8"),
        numpy_helper.from_array(np.array(1.0, np.float32), "one"),
        numpy_helper.from_array(np.array([2], np.int64), "c2s"),
        numpy_helper.from_array(np.array([3], np.int64), "c3e"),
        numpy_helper.from_array(np.array([8], np.int64), "c8s"),
        numpy_helper.from_array(np.array([9], np.int64), "c9e"),
        numpy_helper.from_array(np.array([1], np.int64), "ax1"),
    ]
    nodes = [
        n("ReduceSum", ["input"], ["content"], axes=[1], keepdims=1),
        n("Slice", ["input", "c8s", "c9e", "ax1"], ["is8"]),
        n("Slice", ["input", "c2s", "c3e", "ax1"], ["is2"]),
        n("ReduceMax", ["is8"], ["col8"], axes=[2], keepdims=1),   # (1,1,1,W)
        n("ReduceMax", ["is2"], ["row2"], axes=[3], keepdims=1),   # (1,1,H,1)
        n("Mul", ["col8", "content"], ["colf"]),                   # (1,1,H,W)
        n("Mul", ["row2", "content"], ["rowf"]),
        n("Mul", ["colf", "rowf"], ["inter"]),
        n("Sub", ["colf", "inter"], ["p8"]),
        n("Sub", ["rowf", "inter"], ["p2"]),
        n("Add", ["colf", "rowf"], ["u0"]),
        n("Sub", ["u0", "inter"], ["union"]),
        n("Sub", ["one", "union"], ["keep"]),
        n("Mul", ["input", "keep"], ["kept"]),
        n("Mul", ["e8", "p8"], ["a8"]),
        n("Mul", ["e2", "p2"], ["a2"]),
        n("Mul", ["e4", "inter"], ["a4"]),
        n("Add", ["a8", "a2"], ["s1"]), n("Add", ["s1", "a4"], ["s2"]),
        n("Add", ["kept", "s2"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "laser_cross",
                              [helper.make_tensor_value_info("input", F, FULL)],
                              [helper.make_tensor_value_info("output", F, FULL)],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_laser_cross(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
