"""Solver: L-shaped path of 4 connecting an 8 and a 2 marker (task 335).

The grid holds a single colour-8 cell and a single colour-2 cell.  The output
draws an L of colour 4: down (or up) the 8's column to the 2's row, then across
the 2's row to the 2.  Both markers are kept; only background cells become 4.

Build: the marker positions come from `ReduceSum` of the channel mask times row/
col index ramps.  The vertical leg is `col == c8` within `[min,max](r8,r2)`; the
horizontal leg is `row == r2` within `[min,max](c8,c2)`.  Their union, minus the
two marker cells, is painted colour 4 over the (background) input.
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
    p8 = np.argwhere(g == 8)
    p2 = np.argwhere(g == 2)
    if len(p8) != 1 or len(p2) != 1:
        return None
    r8, c8 = p8[0]; r2, c2 = p2[0]
    out = g.copy()
    for r in range(min(r8, r2), max(r8, r2) + 1):
        if out[r, c8] == 0:
            out[r, c8] = 4
    for c in range(min(c8, c2), max(c8, c2) + 1):
        if out[r2, c] == 0:
            out[r2, c] = 4
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
    F = TensorProto.FLOAT
    n = helper.make_node

    def cf(b, name):
        return n("Cast", [b], [name], to=F)

    row_idx = np.arange(HEIGHT, dtype=np.float32).reshape(1, 1, HEIGHT, 1)
    col_idx = np.arange(WIDTH, dtype=np.float32).reshape(1, 1, 1, WIDTH)
    e4 = np.zeros((1, CHANNELS, 1, 1), np.float32); e4[0, 4] = 1.0
    init = [
        numpy_helper.from_array(row_idx, "row_idx"),
        numpy_helper.from_array(col_idx, "col_idx"),
        numpy_helper.from_array(e4, "e4"),
        numpy_helper.from_array(np.array(0.5, np.float32), "half"),
        numpy_helper.from_array(np.array(1.0, np.float32), "one"),
        numpy_helper.from_array(np.array([0, 8, 0, 0], np.int64), "s8"),
        numpy_helper.from_array(np.array([1, 9, HEIGHT, WIDTH], np.int64), "e8"),
        numpy_helper.from_array(np.array([0, 2, 0, 0], np.int64), "s2"),
        numpy_helper.from_array(np.array([1, 3, HEIGHT, WIDTH], np.int64), "e2"),
        numpy_helper.from_array(np.array([0, 1, 2, 3], np.int64), "ax4"),
    ]
    nodes = [
        n("Slice", ["input", "s8", "e8", "ax4"], ["ch8"]),       # (1,1,H,W)
        n("Slice", ["input", "s2", "e2", "ax4"], ["ch2"]),
        n("Mul", ["ch8", "row_idx"], ["m8r"]), n("ReduceSum", ["m8r"], ["r8"], keepdims=0),
        n("Mul", ["ch8", "col_idx"], ["m8c"]), n("ReduceSum", ["m8c"], ["c8"], keepdims=0),
        n("Mul", ["ch2", "row_idx"], ["m2r"]), n("ReduceSum", ["m2r"], ["r2"], keepdims=0),
        n("Mul", ["ch2", "col_idx"], ["m2c"]), n("ReduceSum", ["m2c"], ["c2"], keepdims=0),
        n("Min", ["r8", "r2"], ["rmin"]), n("Max", ["r8", "r2"], ["rmax"]),
        n("Min", ["c8", "c2"], ["cmin"]), n("Max", ["c8", "c2"], ["cmax"]),
        # vertical leg: col == c8 and rmin <= row <= rmax
        n("Sub", ["col_idx", "c8"], ["dc8"]), n("Abs", ["dc8"], ["adc8"]),
        n("Less", ["adc8", "half"], ["colsel_b"]), cf("colsel_b", "colsel"),   # (1,1,1,W)
        n("Sub", ["rmin", "half"], ["rmin_l"]), n("Greater", ["row_idx", "rmin_l"], ["rge_b"]), cf("rge_b", "rge"),
        n("Add", ["rmax", "half"], ["rmax_u"]), n("Less", ["row_idx", "rmax_u"], ["rle_b"]), cf("rle_b", "rle"),
        n("Mul", ["rge", "rle"], ["rowin"]),                                   # (1,1,H,1)
        n("Mul", ["colsel", "rowin"], ["vmask"]),                             # (1,1,H,W)
        # horizontal leg: row == r2 and cmin <= col <= cmax
        n("Sub", ["row_idx", "r2"], ["dr2"]), n("Abs", ["dr2"], ["adr2"]),
        n("Less", ["adr2", "half"], ["rowsel_b"]), cf("rowsel_b", "rowsel"),   # (1,1,H,1)
        n("Sub", ["cmin", "half"], ["cmin_l"]), n("Greater", ["col_idx", "cmin_l"], ["cge_b"]), cf("cge_b", "cge"),
        n("Add", ["cmax", "half"], ["cmax_u"]), n("Less", ["col_idx", "cmax_u"], ["cle_b"]), cf("cle_b", "cle"),
        n("Mul", ["cge", "cle"], ["colin"]),
        n("Mul", ["rowsel", "colin"], ["hmask"]),
        n("Max", ["vmask", "hmask"], ["path"]),
        n("Sub", ["path", "ch8"], ["p1"]), n("Sub", ["p1", "ch2"], ["path_ne"]),
        n("Sub", ["one", "path_ne"], ["keep"]),
        n("Mul", ["input", "keep"], ["kept"]),
        n("Mul", ["e4", "path_ne"], ["paint"]),
        n("Add", ["kept", "paint"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "l_connect",
                              [helper.make_tensor_value_info("input", F, FULL)],
                              [helper.make_tensor_value_info("output", F, FULL)],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_l_connect(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
