"""Solver: stamp the top-row pattern at marked rows (task 43).

The top row holds a pattern of colour-5 cells; the right edge holds colour-5
markers on certain rows. Each marked row receives a copy of the top-row pattern
painted in colour 2 (only on background cells; the original 5s stay).

Build: the colour-5 channel's row 0 gives the column pattern; a per-row
`ReduceMax` (excluding row 0) gives the marked rows; their outer product,
restricted to background, is painted colour 2.
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
SRC, NEW = 5, 2


def _ref(g: np.ndarray) -> Optional[np.ndarray]:
    H, W = g.shape
    toppat = [c for c in range(W) if g[0, c] == SRC]
    marked = [r for r in range(1, H) if (g[r] == SRC).any()]
    if not toppat or not marked:
        return None
    out = g.copy()
    for r in marked:
        for c in toppat:
            if out[r, c] == 0:
                out[r, c] = NEW
    return out


def _detect(task: dict) -> bool:
    saw = False
    for ex in all_examples(task):
        i, o = ex["input"], ex["output"]
        if not i or not i[0] or len(i) > HEIGHT or len(i[0]) > WIDTH:
            return False
        ref = _ref(np.array(i))
        if ref is None or not np.array_equal(ref, np.array(o)):
            return False
        saw = True
    return saw


def _build() -> onnx.ModelProto:
    F = TensorProto.FLOAT
    n = helper.make_node
    row_ar = np.arange(HEIGHT, dtype=np.float32).reshape(1, 1, HEIGHT, 1)
    e_delta = np.zeros((1, CHANNELS, 1, 1), np.float32)
    e_delta[0, NEW] = 1.0
    e_delta[0, 0] = -1.0  # turn background (ch0) cell into colour NEW
    init = [
        numpy_helper.from_array(row_ar, "row_ar"),
        numpy_helper.from_array(np.array(0.5, np.float32), "half"),
        numpy_helper.from_array(np.array(1.0, np.float32), "one"),
        numpy_helper.from_array(e_delta, "e_delta"),
        numpy_helper.from_array(np.array([0, SRC, 0, 0], np.int64), "s_s"),
        numpy_helper.from_array(np.array([1, SRC + 1, HEIGHT, WIDTH], np.int64), "s_e"),
        numpy_helper.from_array(np.array([0, 0, 0, 0], np.int64), "r0_s"),
        numpy_helper.from_array(np.array([1, 1, 1, WIDTH], np.int64), "r0_e"),
        numpy_helper.from_array(np.array([0, 0, 0, 0], np.int64), "c0_s"),
        numpy_helper.from_array(np.array([1, 1, HEIGHT, WIDTH], np.int64), "c0_e"),
        numpy_helper.from_array(np.array([0, 1, 2, 3], np.int64), "ax4"),
    ]
    s1 = [1, 1, HEIGHT, WIDTH]; rc = [1, 1, HEIGHT, 1]; cc = [1, 1, 1, WIDTH]
    B = TensorProto.BOOL
    nodes = [
        n("Slice", ["input", "s_s", "s_e", "ax4"], ["ch5"]),
        n("Slice", ["ch5", "r0_s", "r0_e", "ax4"], ["toppat"]),       # (1,1,1,W)
        n("ReduceMax", ["ch5"], ["rowhas"], axes=[3], keepdims=1),     # (1,1,H,1)
        n("Greater", ["row_ar", "half"], ["notr0_b"]),
        n("Cast", ["notr0_b"], ["notr0"], to=F),
        n("Mul", ["rowhas", "notr0"], ["marked"]),
        n("Mul", ["marked", "toppat"], ["stamp"]),                     # (1,1,H,W)
        n("Slice", ["input", "c0_s", "c0_e", "ax4"], ["ch0"]),         # background mask
        n("Mul", ["stamp", "ch0"], ["stamp2"]),
        n("Mul", ["e_delta", "stamp2"], ["delta"]),                    # (1,10,H,W)
        n("Add", ["input", "delta"], ["output"]),
    ]
    vi = [
        helper.make_tensor_value_info("ch5", F, s1),
        helper.make_tensor_value_info("toppat", F, cc),
        helper.make_tensor_value_info("rowhas", F, rc),
        helper.make_tensor_value_info("notr0_b", F, rc, ) if False else helper.make_tensor_value_info("notr0_b", B, rc),
        helper.make_tensor_value_info("notr0", F, rc),
        helper.make_tensor_value_info("marked", F, rc),
        helper.make_tensor_value_info("stamp", F, s1),
        helper.make_tensor_value_info("ch0", F, s1),
        helper.make_tensor_value_info("stamp2", F, s1),
        helper.make_tensor_value_info("delta", F, FULL),
    ]
    graph = helper.make_graph(nodes, "stamp_top_row",
                              [helper.make_tensor_value_info("input", F, FULL)],
                              [helper.make_tensor_value_info("output", F, FULL)],
                              initializer=init, value_info=vi)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_stamp_top_row(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
