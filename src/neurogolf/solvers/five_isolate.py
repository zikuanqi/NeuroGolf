"""Solver: keep the colour-5 cells, repaint them the other colour (task 389).

Every grid contains background 0, a scaffold of colour 5, and exactly one other
colour C.  The output blanks everything, then paints the cells that held a 5 in
colour C (the 5 footprint becomes C, the old C cells become background).

Build: C is recovered as the one-hot channel that survives after zeroing the
background (ch0) and marker (ch5) entries of the channel histogram.  The 5-mask
(input channel 5) is painted C; the remaining in-grid cells get background ch0.
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
MARK = 5


def _ref(g: np.ndarray) -> Optional[np.ndarray]:
    others = [c for c in np.unique(g) if c not in (0, MARK)]
    if len(others) != 1:
        return None
    return np.where(g == MARK, others[0], 0)


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
    note05 = np.ones((1, CHANNELS, 1, 1), np.float32)
    note05[0, 0] = 0.0
    note05[0, MARK] = 0.0
    e0 = np.zeros((1, CHANNELS, 1, 1), np.float32)
    e0[0, 0] = 1.0
    init = [
        numpy_helper.from_array(note05, "note05"),
        numpy_helper.from_array(e0, "e0"),
        numpy_helper.from_array(np.array(0.5, np.float32), "half"),
        numpy_helper.from_array(np.array([0, MARK, 0, 0], np.int64), "m_s"),
        numpy_helper.from_array(np.array([1, MARK + 1, HEIGHT, WIDTH], np.int64), "m_e"),
        numpy_helper.from_array(np.array([0, 1, 2, 3], np.int64), "ax4"),
    ]
    nodes = [
        n("ReduceSum", ["input"], ["hist"], axes=[2, 3], keepdims=1),     # (1,10,1,1)
        n("Mul", ["hist", "note05"], ["hm"]),
        n("Greater", ["hm", "half"], ["eC_b"]),
        n("Cast", ["eC_b"], ["eC"], to=F),                                # (1,10,1,1) colour C
        n("Slice", ["input", "m_s", "m_e", "ax4"], ["ch5"]),              # (1,1,H,W) 5-mask
        n("ReduceSum", ["input"], ["content"], axes=[1], keepdims=1),     # (1,1,H,W)
        n("Sub", ["content", "ch5"], ["bg"]),                             # non-5 grid cells
        n("Mul", ["eC", "ch5"], ["paint5"]),                              # C at 5 footprint
        n("Mul", ["e0", "bg"], ["paintBg"]),                              # bg at the rest
        n("Add", ["paint5", "paintBg"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "five_isolate",
                              [helper.make_tensor_value_info("input", F, FULL)],
                              [helper.make_tensor_value_info("output", F, FULL)],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_five_isolate(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
