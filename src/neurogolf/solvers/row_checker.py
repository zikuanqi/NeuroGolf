"""Solver: two uniform rows -> diagonal checkerboard (task 373).

The input is exactly two rows, each a solid colour A (top) and B (bottom).
The output keeps the same 2xW shape but interleaves the two colours on a
diagonal checkerboard: output[r, c] = A when (r + c) is even, else B.

Build: A and B are read straight off the one-hot input as the channel vectors
at (row0, col0) and (row1, col0).  A precomputed parity mask P[r,c]=((r+c)%2==0)
times the content mask selects where A goes; the complement (content - A-cells)
selects B.  Padding cells stay all-zero because they are excluded by content.
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
    if H != 2:
        return None
    a, b = g[0, 0], g[1, 0]
    if not (g[0] == a).all() or not (g[1] == b).all():
        return None
    out = np.empty_like(g)
    for r in range(H):
        for c in range(W):
            out[r, c] = a if (r + c) % 2 == 0 else b
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
    rr = np.arange(HEIGHT).reshape(HEIGHT, 1)
    cc = np.arange(WIDTH).reshape(1, WIDTH)
    parity = (((rr + cc) % 2) == 0).astype(np.float32).reshape(1, 1, HEIGHT, WIDTH)
    init = [
        numpy_helper.from_array(parity, "parity"),
        numpy_helper.from_array(np.array([0, 0, 0, 0], np.int64), "a_s"),
        numpy_helper.from_array(np.array([1, CHANNELS, 1, 1], np.int64), "a_e"),
        numpy_helper.from_array(np.array([0, 0, 1, 0], np.int64), "b_s"),
        numpy_helper.from_array(np.array([1, CHANNELS, 2, 1], np.int64), "b_e"),
        numpy_helper.from_array(np.array([0, 1, 2, 3], np.int64), "ax4"),
    ]
    nodes = [
        n("ReduceSum", ["input"], ["content"], axes=[1], keepdims=1),  # (1,1,H,W)
        n("Mul", ["content", "parity"], ["mA"]),                       # A cells
        n("Sub", ["content", "mA"], ["mB"]),                           # B cells
        n("Slice", ["input", "a_s", "a_e", "ax4"], ["eA"]),            # (1,10,1,1)
        n("Slice", ["input", "b_s", "b_e", "ax4"], ["eB"]),            # (1,10,1,1)
        n("Mul", ["eA", "mA"], ["outA"]),
        n("Mul", ["eB", "mB"], ["outB"]),
        n("Add", ["outA", "outB"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "row_checker",
                              [helper.make_tensor_value_info("input", F, FULL)],
                              [helper.make_tensor_value_info("output", F, FULL)],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_row_checker(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
