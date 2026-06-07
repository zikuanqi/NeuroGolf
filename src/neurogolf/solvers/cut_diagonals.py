"""Solver: erase both diagonals of a solid square (task 375).

The input is a solid NxN square of one colour (often with the centre already
blank).  The output blanks every cell on the main diagonal (r==c) or the
anti-diagonal (r+c==N-1), leaving an X-shaped hole; the rest keeps its colour.

Build: N-1 is the largest content row index.  Index masks `|row-col|<0.5` and
`|row+col-(N-1)|<0.5` give the two diagonals; their union is cleared to
background 0 over the grid, the complement keeps the input.
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
    if H != W:
        return None
    N = H
    out = g.copy()
    for r in range(N):
        for c in range(N):
            if r == c or r + c == N - 1:
                out[r, c] = 0
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
    e0 = np.zeros((1, CHANNELS, 1, 1), np.float32); e0[0, 0] = 1.0
    init = [
        numpy_helper.from_array(row_idx, "row_idx"),
        numpy_helper.from_array(col_idx, "col_idx"),
        numpy_helper.from_array(e0, "e0"),
        numpy_helper.from_array(np.array(0.5, np.float32), "half"),
        numpy_helper.from_array(np.array(1.0, np.float32), "one"),
    ]
    nodes = [
        n("ReduceSum", ["input"], ["content"], axes=[1], keepdims=1),     # (1,1,H,W)
        n("ReduceMax", ["content"], ["rowhas"], axes=[3], keepdims=1),    # (1,1,H,1)
        n("Mul", ["rowhas", "row_idx"], ["rr"]), n("ReduceMax", ["rr"], ["nm1"], keepdims=0),
        n("Sub", ["row_idx", "col_idx"], ["dmain"]), n("Abs", ["dmain"], ["admain"]),
        n("Less", ["admain", "half"], ["main_b"]), cf("main_b", "main"),
        n("Add", ["row_idx", "col_idx"], ["sumrc"]), n("Sub", ["sumrc", "nm1"], ["dalt"]),
        n("Abs", ["dalt"], ["adalt"]), n("Less", ["adalt", "half"], ["anti_b"]), cf("anti_b", "anti"),
        n("Max", ["main", "anti"], ["remove"]),                          # (1,1,H,W)
        n("Sub", ["one", "remove"], ["keep"]),
        n("Mul", ["input", "keep"], ["kept"]),
        n("Mul", ["remove", "content"], ["rg"]),                         # in-grid diagonal cells
        n("Mul", ["e0", "rg"], ["pbg"]),
        n("Add", ["kept", "pbg"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "cut_diagonals",
                              [helper.make_tensor_value_info("input", F, FULL)],
                              [helper.make_tensor_value_info("output", F, FULL)],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_cut_diagonals(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
