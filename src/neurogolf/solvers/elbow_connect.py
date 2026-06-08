"""Solver: L-shaped path of 8 connecting a 2-marker to a 3-marker (task 246).

A single ``2`` and a single ``3`` sit on the grid.  An elbow path of ``8`` is
drawn -- horizontally along the ``2``'s row to the ``3``'s column, then
vertically down that column to the ``3``.  Both endpoints keep their colour::

    . 2 . . . .        . 2 8 8 8 .
    . . . . . .   ->   . . . . . 8
    . . . . . 3        . . . . . 3

Build: centroids of the ``2`` / ``3`` masks give ``(r2,c2)`` and ``(r3,c3)``;
index masks select row ``r2`` ∩ cols ``[c2,c3]`` and col ``c3`` ∩ rows
``[r2,r3]``; the union (minus the endpoints) is painted ``e_8``.
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
    y2, x2 = np.where(g == 2); y3, x3 = np.where(g == 3)
    if len(y2) != 1 or len(y3) != 1:
        return None
    r2, c2, r3, c3 = y2[0], x2[0], y3[0], x3[0]
    out = g.copy()
    lo, hi = sorted((c2, c3))
    for c in range(lo, hi + 1):
        out[r2, c] = 8
    lo, hi = sorted((r2, r3))
    for r in range(lo, hi + 1):
        out[r, c3] = 8
    out[r2, c2] = 2; out[r3, c3] = 3
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

    row_idx = np.arange(HEIGHT, dtype=np.float32).reshape(1, 1, HEIGHT, 1)
    col_idx = np.arange(WIDTH, dtype=np.float32).reshape(1, 1, 1, WIDTH)
    e8 = np.zeros((1, CHANNELS, 1, 1), np.float32); e8[0, 8] = 1.0
    init = [
        numpy_helper.from_array(row_idx, "row_idx"),
        numpy_helper.from_array(col_idx, "col_idx"),
        numpy_helper.from_array(e8, "e8"),
        numpy_helper.from_array(np.array(0.5, np.float32), "half"),
        numpy_helper.from_array(np.array(1.0, np.float32), "one"),
        numpy_helper.from_array(np.array([2], np.int64), "two_s"),
        numpy_helper.from_array(np.array([3], np.int64), "three_s"),
        numpy_helper.from_array(np.array([4], np.int64), "four_s"),
        numpy_helper.from_array(np.array([1], np.int64), "ax1"),
    ]

    def centroid(mask, pre):
        return [
            n("ReduceSum", [mask], [pre + "_cnt"], keepdims=0),
            n("Mul", [mask, "row_idx"], [pre + "_mr"]),
            n("ReduceSum", [pre + "_mr"], [pre + "_sr"], keepdims=0),
            n("Div", [pre + "_sr", pre + "_cnt"], [pre + "_r"]),
            n("Mul", [mask, "col_idx"], [pre + "_mc"]),
            n("ReduceSum", [pre + "_mc"], [pre + "_sc"], keepdims=0),
            n("Div", [pre + "_sc", pre + "_cnt"], [pre + "_c"]),
        ]

    nodes = [
        n("Slice", ["input", "two_s", "three_s", "ax1"], ["is2"]),
        n("Slice", ["input", "three_s", "four_s", "ax1"], ["is3"]),
    ]
    nodes += centroid("is2", "a")   # a_r, a_c
    nodes += centroid("is3", "b")   # b_r, b_c
    nodes += [
        # row a_r ; col b_c
        n("Sub", ["row_idx", "a_r"], ["dr"]), n("Abs", ["dr"], ["adr"]),
        n("Less", ["adr", "half"], ["onr_b"]), n("Cast", ["onr_b"], ["on_r2"], to=F),
        n("Sub", ["col_idx", "b_c"], ["dc"]), n("Abs", ["dc"], ["adc"]),
        n("Less", ["adc", "half"], ["onc_b"]), n("Cast", ["onc_b"], ["on_c3"], to=F),
        # between cols a_c .. b_c
        n("Min", ["a_c", "b_c"], ["cmin"]), n("Max", ["a_c", "b_c"], ["cmax"]),
        n("Sub", ["cmin", "half"], ["cminh"]), n("Add", ["cmax", "half"], ["cmaxh"]),
        n("Greater", ["col_idx", "cminh"], ["gc_b"]), n("Cast", ["gc_b"], ["gc"], to=F),
        n("Less", ["col_idx", "cmaxh"], ["lc_b"]), n("Cast", ["lc_b"], ["lc"], to=F),
        n("Mul", ["gc", "lc"], ["betw_c"]),
        # between rows a_r .. b_r
        n("Min", ["a_r", "b_r"], ["rmin"]), n("Max", ["a_r", "b_r"], ["rmax"]),
        n("Sub", ["rmin", "half"], ["rminh"]), n("Add", ["rmax", "half"], ["rmaxh"]),
        n("Greater", ["row_idx", "rminh"], ["gr_b"]), n("Cast", ["gr_b"], ["gr"], to=F),
        n("Less", ["row_idx", "rmaxh"], ["lr_b"]), n("Cast", ["lr_b"], ["lr"], to=F),
        n("Mul", ["gr", "lr"], ["betw_r"]),
        # segments and union
        n("Mul", ["on_r2", "betw_c"], ["hseg"]),
        n("Mul", ["on_c3", "betw_r"], ["vseg"]),
        n("Max", ["hseg", "vseg"], ["path"]),
        # exclude endpoints, clip to real grid, paint e8
        n("Sub", ["one", "is2"], ["nin2"]), n("Sub", ["one", "is3"], ["nin3"]),
        n("Mul", ["path", "nin2"], ["pa1"]), n("Mul", ["pa1", "nin3"], ["pa2"]),
        n("ReduceSum", ["input"], ["content"], axes=[1], keepdims=1),
        n("Mul", ["pa2", "content"], ["paint"]),
        n("Sub", ["one", "paint"], ["inv"]), n("Mul", ["input", "inv"], ["kept"]),
        n("Mul", ["e8", "paint"], ["addc"]), n("Add", ["kept", "addc"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "elbow_connect",
                              [helper.make_tensor_value_info("input", F, FULL)],
                              [helper.make_tensor_value_info("output", F, FULL)],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_elbow_connect(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
