"""Solver: each marker shoots an alternating ray to the right (task 232).

Every isolated marker fills its row from that column to the right edge with an
alternating pattern of the marker colour and ``5``::

    . . 2 . . . . .   ->   . . 2 5 2 5 2 5

The active region (cells at or right of the marker) is ``CumSum(markers) > 0``;
a second ``CumSum`` over that region gives the offset, whose parity selects the
marker colour (odd cumulative count) or ``5`` (even).  The marker colour per row
is the row's only non-background channel.
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
    ys, xs = np.where(g != 0)
    if len(ys) == 0:
        return None
    for r, c in zip(ys, xs):
        col = int(g[r, c])
        for k, cc in enumerate(range(c, W)):
            out[r, cc] = col if k % 2 == 0 else 5
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
    n = helper.make_node
    e0 = np.zeros((1, CHANNELS, 1, 1), np.float32); e0[0, 0] = 1.0
    e5 = np.zeros((1, CHANNELS, 1, 1), np.float32); e5[0, 5] = 1.0
    notbg = np.ones((1, CHANNELS, 1, 1), np.float32); notbg[0, 0] = 0.0
    init = [
        numpy_helper.from_array(e0, "e0"),
        numpy_helper.from_array(e5, "e5"),
        numpy_helper.from_array(notbg, "notbg"),
        numpy_helper.from_array(np.array(0.5, np.float32), "half"),
        numpy_helper.from_array(np.array(2.0, np.float32), "two"),
        numpy_helper.from_array(np.array(3, np.int64), "axis3"),
        numpy_helper.from_array(np.array([0], np.int64), "c0s"),
        numpy_helper.from_array(np.array([1], np.int64), "c1e"),
        numpy_helper.from_array(np.array([1], np.int64), "ax1"),
    ]
    nodes = [
        n("ReduceSum", ["input"], ["occ"], axes=[1], keepdims=1),
        n("Slice", ["input", "c0s", "c1e", "ax1"], ["is0"]),
        n("Sub", ["occ", "is0"], ["mark"]),
        n("CumSum", ["mark", "axis3"], ["cumM"]),
        n("Greater", ["cumM", "half"], ["act_b"]), n("Cast", ["act_b"], ["active"], to=F),
        n("CumSum", ["active", "axis3"], ["cumA"]),
        n("Mul", ["cumA", "half"], ["ha"]), n("Floor", ["ha"], ["fl"]),
        n("Mul", ["fl", "two"], ["dbl"]), n("Sub", ["cumA", "dbl"], ["rem"]),
        n("Mul", ["active", "occ"], ["actG"]),
        n("Mul", ["actG", "rem"], ["mcell"]),
        n("Sub", ["actG", "mcell"], ["fcell"]),
        n("ReduceMax", ["input"], ["rowHas"], axes=[3], keepdims=1),
        n("Mul", ["rowHas", "notbg"], ["rowColor"]),
        n("Mul", ["mcell", "rowColor"], ["mlayer"]),
        n("Mul", ["fcell", "e5"], ["flayer"]),
        n("Sub", ["occ", "actG"], ["ch0v"]),
        n("Mul", ["ch0v", "e0"], ["ch0layer"]),
        n("Add", ["mlayer", "flayer"], ["t1"]),
        n("Add", ["t1", "ch0layer"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "alt_ray_right",
                              [helper.make_tensor_value_info("input", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              [helper.make_tensor_value_info("output", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_alt_ray_right(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
