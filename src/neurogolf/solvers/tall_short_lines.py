"""Solver: keep only the tallest and shortest vertical lines (task 254).

The grid holds several vertical line segments of one colour.  The tallest column
(most cells) is recoloured ``1`` and the shortest non-empty column ``2``; every
other line is erased to background.

Per-column counts come from a row ``ReduceSum`` of the marker mask; the max and
the non-zero min select the two surviving columns.
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
    col = (g != 0).sum(0)
    nz = np.where(col > 0)[0]
    if len(nz) == 0:
        return None
    mx, mn = col.max(), col[nz].min()
    if mx == mn:
        return None
    for c in range(W):
        if col[c] == mx:
            out[(g[:, c] != 0), c] = 1
        elif col[c] == mn and col[c] > 0:
            out[(g[:, c] != 0), c] = 2
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
    e1 = np.zeros((1, CHANNELS, 1, 1), np.float32); e1[0, 1] = 1.0
    e2 = np.zeros((1, CHANNELS, 1, 1), np.float32); e2[0, 2] = 1.0
    init = [
        numpy_helper.from_array(e0, "e0"),
        numpy_helper.from_array(e1, "e1"),
        numpy_helper.from_array(e2, "e2"),
        numpy_helper.from_array(np.array(0.5, np.float32), "half"),
        numpy_helper.from_array(np.array(-0.5, np.float32), "neghalf"),
        numpy_helper.from_array(np.array(1e4, np.float32), "big"),
        numpy_helper.from_array(np.array([0], np.int64), "c0s"),
        numpy_helper.from_array(np.array([1], np.int64), "c1e"),
        numpy_helper.from_array(np.array([1], np.int64), "ax1"),
    ]
    nodes = [
        n("ReduceSum", ["input"], ["occ"], axes=[1], keepdims=1),
        n("Slice", ["input", "c0s", "c1e", "ax1"], ["is0"]),
        n("Sub", ["occ", "is0"], ["mark"]),
        # per-column counts (1,1,1,W)
        n("ReduceSum", ["mark"], ["colCnt"], axes=[2], keepdims=1),
        n("ReduceMax", ["colCnt"], ["maxC"], axes=[3], keepdims=1),
        # min over non-empty columns
        n("Less", ["colCnt", "half"], ["empty_b"]), n("Cast", ["empty_b"], ["empty"], to=F),
        n("Mul", ["empty", "big"], ["emptyBig"]),
        n("Add", ["colCnt", "emptyBig"], ["colAdj"]),
        n("ReduceMin", ["colAdj"], ["minC"], axes=[3], keepdims=1),
        # tall columns: colCnt == maxC
        n("Sub", ["colCnt", "maxC"], ["dMax"]),
        n("Greater", ["dMax", "neghalf"], ["tall_b"]), n("Cast", ["tall_b"], ["tallCol"], to=F),
        # short columns: colCnt == minC (and non-empty, not tall)
        n("Sub", ["colCnt", "minC"], ["dMin"]),
        n("Greater", ["dMin", "neghalf"], ["sLo_b"]), n("Cast", ["sLo_b"], ["sLo"], to=F),
        n("Less", ["dMin", "half"], ["sHi_b"]), n("Cast", ["sHi_b"], ["sHi"], to=F),
        n("Mul", ["sLo", "sHi"], ["shortRaw"]),
        n("Sub", ["shortRaw", "tallCol"], ["shortTmp"]),  # exclude any column that is also the tallest
        n("Relu", ["shortTmp"], ["shortCol"]),
        # apply to cells
        n("Mul", ["mark", "tallCol"], ["tallCell"]),
        n("Mul", ["mark", "shortCol"], ["shortCell"]),
        n("Mul", ["tallCell", "e1"], ["L1"]),
        n("Mul", ["shortCell", "e2"], ["L2"]),
        n("Add", ["tallCell", "shortCell"], ["keep"]),
        n("Sub", ["occ", "keep"], ["bgv"]),
        n("Mul", ["bgv", "e0"], ["L0"]),
        n("Add", ["L1", "L2"], ["t1"]),
        n("Add", ["t1", "L0"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "tall_short_lines",
                              [helper.make_tensor_value_info("input", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              [helper.make_tensor_value_info("output", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_tall_short_lines(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
