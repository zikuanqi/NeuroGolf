"""Solver: move the 3-marker one step toward the 4-marker (task 353).

A single colour-3 cell and a single colour-4 cell sit on a blank grid.  The
output keeps the 4 where it is and shifts the 3 one cell toward the 4 (one step
in each axis by the sign of the offset); everything else is background.

Build: marker positions come from index-ramp `ReduceSum`; `Sign` of the row/col
offset gives the step, and an index mask places the 3 at its new cell.  If the
step lands on the 4 (adjacent markers) the 3 takes precedence.
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
    p3 = np.argwhere(g == 3)
    p4 = np.argwhere(g == 4)
    if len(p3) != 1 or len(p4) != 1:
        return None
    r3, c3 = p3[0]; r4, c4 = p4[0]
    nr, nc = r3 + int(np.sign(r4 - r3)), c3 + int(np.sign(c4 - c3))
    out = np.zeros_like(g)
    out[r4, c4] = 4
    out[nr, nc] = 3
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
    e3 = np.zeros((1, CHANNELS, 1, 1), np.float32); e3[0, 3] = 1.0
    e4 = np.zeros((1, CHANNELS, 1, 1), np.float32); e4[0, 4] = 1.0
    e0 = np.zeros((1, CHANNELS, 1, 1), np.float32); e0[0, 0] = 1.0
    init = [
        numpy_helper.from_array(row_idx, "row_idx"),
        numpy_helper.from_array(col_idx, "col_idx"),
        numpy_helper.from_array(e3, "e3"),
        numpy_helper.from_array(e4, "e4"),
        numpy_helper.from_array(e0, "e0"),
        numpy_helper.from_array(np.array(0.5, np.float32), "half"),
        numpy_helper.from_array(np.array(1.0, np.float32), "one"),
        numpy_helper.from_array(np.array([0, 3, 0, 0], np.int64), "s3"),
        numpy_helper.from_array(np.array([1, 4, HEIGHT, WIDTH], np.int64), "e3s"),
        numpy_helper.from_array(np.array([0, 4, 0, 0], np.int64), "s4"),
        numpy_helper.from_array(np.array([1, 5, HEIGHT, WIDTH], np.int64), "e4s"),
        numpy_helper.from_array(np.array([0, 1, 2, 3], np.int64), "ax4"),
    ]
    nodes = [
        n("Slice", ["input", "s3", "e3s", "ax4"], ["ch3"]),
        n("Slice", ["input", "s4", "e4s", "ax4"], ["ch4"]),
        n("Mul", ["ch3", "row_idx"], ["c3r"]), n("ReduceSum", ["c3r"], ["r3"], keepdims=0),
        n("Mul", ["ch3", "col_idx"], ["c3c"]), n("ReduceSum", ["c3c"], ["c3"], keepdims=0),
        n("Mul", ["ch4", "row_idx"], ["c4r"]), n("ReduceSum", ["c4r"], ["r4"], keepdims=0),
        n("Mul", ["ch4", "col_idx"], ["c4c"]), n("ReduceSum", ["c4c"], ["c4"], keepdims=0),
        n("Sub", ["r4", "r3"], ["ddr"]), n("Sign", ["ddr"], ["dr"]),
        n("Sub", ["c4", "c3"], ["ddc"]), n("Sign", ["ddc"], ["dc"]),
        n("Add", ["r3", "dr"], ["nr"]), n("Add", ["c3", "dc"], ["nc"]),
        n("Sub", ["row_idx", "nr"], ["drow"]), n("Abs", ["drow"], ["adrow"]),
        n("Less", ["adrow", "half"], ["rsel_b"]), cf("rsel_b", "rsel"),       # (1,1,H,1)
        n("Sub", ["col_idx", "nc"], ["dcol"]), n("Abs", ["dcol"], ["adcol"]),
        n("Less", ["adcol", "half"], ["csel_b"]), cf("csel_b", "csel"),       # (1,1,1,W)
        n("Mul", ["rsel", "csel"], ["new3"]),                                 # (1,1,H,W)
        n("Sub", ["one", "new3"], ["nn3"]),
        n("Mul", ["ch4", "nn3"], ["n4"]),                                     # 4 except where 3 lands
        n("ReduceSum", ["input"], ["grid"], axes=[1], keepdims=1),
        n("Add", ["new3", "n4"], ["occ"]), n("Sub", ["grid", "occ"], ["bg"]),
        n("Mul", ["e3", "new3"], ["p3"]),
        n("Mul", ["e4", "n4"], ["p4"]),
        n("Mul", ["e0", "bg"], ["pbg"]),
        n("Add", ["p3", "p4"], ["pa"]), n("Add", ["pa", "pbg"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "move_toward",
                              [helper.make_tensor_value_info("input", F, FULL)],
                              [helper.make_tensor_value_info("output", F, FULL)],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_move_toward(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
