"""Solver: fill the interior of rectangles given by 4-corner markers (task 273).

Colour-4 markers sit at the corners of one or more axis-aligned rectangles; the
strict interior of each rectangle is filled with ``2``::

    4 . 4            4 . 4
    . . .     ->     . 2 .       (and with two rectangles, each interior
    4 . 4            4 . 4        is filled independently)

A background cell is interior to some rectangle iff there is a 4-marker in each
of its four diagonal quadrants (up-left, up-right, down-left, down-right).  This
holds for any number of rectangles and is computed with four exclusive
``CumSum`` passes over the 4-mask (no per-rectangle grouping needed).
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
F = TensorProto.FLOAT


def _ref(g: np.ndarray) -> Optional[np.ndarray]:
    H, W = g.shape
    y4, x4 = np.where(g == 4)
    if len(y4) < 4:
        return None
    is4 = (g == 4).astype(int)
    cu_up = np.cumsum(is4, axis=0) - is4          # strictly above
    cu_dn = (np.cumsum(is4[::-1], axis=0)[::-1]) - is4  # strictly below
    def excl_l(a):
        return np.cumsum(a, axis=1) - a
    def excl_r(a):
        return np.cumsum(a[:, ::-1], axis=1)[:, ::-1] - a
    hasUL = excl_l(cu_up) > 0; hasUR = excl_r(cu_up) > 0
    hasDL = excl_l(cu_dn) > 0; hasDR = excl_r(cu_dn) > 0
    inside = hasUL & hasUR & hasDL & hasDR
    out = g.copy()
    out[(g == 0) & inside] = 2
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
    e2 = np.zeros((1, CHANNELS, 1, 1), np.float32); e2[0, 2] = 1.0
    init = [
        numpy_helper.from_array(e2, "e2"),
        numpy_helper.from_array(np.array(0.5, np.float32), "half"),
        numpy_helper.from_array(np.array(1.0, np.float32), "one"),
        numpy_helper.from_array(np.array([0], np.int64), "c0s"),
        numpy_helper.from_array(np.array([1], np.int64), "c1e"),
        numpy_helper.from_array(np.array([4], np.int64), "c4s"),
        numpy_helper.from_array(np.array([5], np.int64), "c5e"),
        numpy_helper.from_array(np.array([1], np.int64), "ax1"),
        numpy_helper.from_array(np.array(2, np.int64), "axis2"),
        numpy_helper.from_array(np.array(3, np.int64), "axis3"),
    ]
    nodes = [
        n("Slice", ["input", "c4s", "c5e", "ax1"], ["is4"]),
        n("Slice", ["input", "c0s", "c1e", "ax1"], ["is0"]),
        n("CumSum", ["is4", "axis2"], ["cu_up"], exclusive=1, reverse=0),
        n("CumSum", ["is4", "axis2"], ["cu_dn"], exclusive=1, reverse=1),
        n("CumSum", ["cu_up", "axis3"], ["ul"], exclusive=1, reverse=0),
        n("CumSum", ["cu_up", "axis3"], ["ur"], exclusive=1, reverse=1),
        n("CumSum", ["cu_dn", "axis3"], ["dl"], exclusive=1, reverse=0),
        n("CumSum", ["cu_dn", "axis3"], ["dr"], exclusive=1, reverse=1),
        n("Greater", ["ul", "half"], ["hul_b"]), n("Cast", ["hul_b"], ["hul"], to=F),
        n("Greater", ["ur", "half"], ["hur_b"]), n("Cast", ["hur_b"], ["hur"], to=F),
        n("Greater", ["dl", "half"], ["hdl_b"]), n("Cast", ["hdl_b"], ["hdl"], to=F),
        n("Greater", ["dr", "half"], ["hdr_b"]), n("Cast", ["hdr_b"], ["hdr"], to=F),
        n("Mul", ["hul", "hur"], ["i1"]), n("Mul", ["hdl", "hdr"], ["i2"]),
        n("Mul", ["i1", "i2"], ["inside"]),
        n("Mul", ["inside", "is0"], ["paint"]),
        n("Sub", ["one", "paint"], ["keep"]),
        n("Mul", ["input", "keep"], ["kept"]),
        n("Mul", ["e2", "paint"], ["addc"]),
        n("Add", ["kept", "addc"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "corner_rect_fill",
                              [helper.make_tensor_value_info("input", F, FULL)],
                              [helper.make_tensor_value_info("output", F, FULL)],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_corner_rect_fill(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
