"""Solver: colour-2 markers fire vertical lines, other colours fire horizontals.

Every column that holds a colour-2 cell is filled with colour 2 (a vertical
line); every row that holds a non-2 marker is filled with that marker's colour
(a horizontal line). Horizontal lines are drawn on top, so in a row that has a
non-2 marker the whole row takes that colour even where a vertical line crosses
it. Cells on no line stay background (task 24).

One-hot channel arithmetic on the (1, 10, 30, 30) tensor:
  col_has2  = column contains colour 2            -> vertical fill of colour 2
  row_chan  = per-row non-2 marker colour         -> horizontal fill
  row_hline = row carries a non-2 marker
  output    = horizontal colour on row-line rows,
              else colour 2 on colour-2 columns,
              else background
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from ..grids import CHANNELS, HEIGHT, WIDTH, all_examples

OPSET = 11
IR_VERSION = 8


def _transform(grid):
    g = np.array(grid)
    H, W = g.shape
    for r in range(H):                       # each row: <=1 distinct non-2 colour
        cols = {int(v) for v in g[r] if v not in (0, 2)}
        if len(cols) > 1:
            return None
    out = [[0] * W for _ in range(H)]
    for c in range(W):                       # vertical colour-2 lines
        if any(g[r, c] == 2 for r in range(H)):
            for r in range(H):
                out[r][c] = 2
    for r in range(H):                       # horizontal lines on top
        cols = [int(v) for v in g[r] if v not in (0, 2)]
        if cols:
            for c in range(W):
                out[r][c] = cols[0]
    return out


def _detect(task: dict) -> bool:
    examples = list(all_examples(task))
    if not examples:
        return False
    changed = False
    for ex in examples:
        inp, out = ex["input"], ex["output"]
        if len(inp) > 30 or len(inp[0]) > 30:
            continue
        if len(inp) != len(out) or len(inp[0]) != len(out[0]):
            return False
        if _transform(inp) != out:
            return False
        if inp != out:
            changed = True
    return changed


def _build() -> onnx.ModelProto:
    F = TensorProto.FLOAT

    def f32(name, arr):
        return numpy_helper.from_array(arr.astype(np.float32), name)

    e_not02 = np.ones((1, CHANNELS, 1, 1), dtype=np.float32)
    e_not02[0, 0, 0, 0] = 0.0
    e_not02[0, 2, 0, 0] = 0.0
    e2 = np.zeros((1, CHANNELS, 1, 1), dtype=np.float32)
    e2[0, 2, 0, 0] = 1.0
    e0 = np.zeros((1, CHANNELS, 1, 1), dtype=np.float32)
    e0[0, 0, 0, 0] = 1.0

    init = [
        f32("e_not02", e_not02), f32("e2", e2), f32("e0", e0),
        f32("one", np.array([[[[1.0]]]])),
        numpy_helper.from_array(np.array([2], dtype=np.int64), "idx2"),
    ]

    n = helper.make_node
    nodes = [
        n("ReduceSum", ["input"], ["content"], axes=[1], keepdims=1),
        n("Gather", ["input", "idx2"], ["ch2"], axis=1),
        n("ReduceMax", ["ch2"], ["col_has2"], axes=[2], keepdims=1),
        n("Mul", ["input", "e_not02"], ["masked"]),
        n("ReduceMax", ["masked"], ["row_chan"], axes=[3], keepdims=1),
        n("ReduceMax", ["row_chan"], ["row_hline"], axes=[1], keepdims=1),
        n("Mul", ["content", "row_hline"], ["use_h"]),
        n("Sub", ["one", "row_hline"], ["not_hline"]),
        n("Mul", ["content", "not_hline"], ["real_nonh"]),
        n("Mul", ["real_nonh", "col_has2"], ["use_v"]),
        n("Sub", ["one", "col_has2"], ["not_col2"]),
        n("Mul", ["real_nonh", "not_col2"], ["use_bg"]),
        n("Mul", ["row_chan", "use_h"], ["out_h"]),
        n("Mul", ["e2", "use_v"], ["out_v"]),
        n("Mul", ["e0", "use_bg"], ["out_bg"]),
        n("Add", ["out_h", "out_v"], ["tmp"]),
        n("Add", ["tmp", "out_bg"], ["output"]),
    ]

    def vi(name, shape):
        return helper.make_tensor_value_info(name, F, shape)

    g4 = [1, CHANNELS, HEIGHT, WIDTH]
    s1 = [1, 1, HEIGHT, WIDTH]
    value_info = [
        vi("content", s1), vi("ch2", s1), vi("col_has2", [1, 1, 1, WIDTH]),
        vi("masked", g4), vi("row_chan", [1, CHANNELS, HEIGHT, 1]),
        vi("row_hline", [1, 1, HEIGHT, 1]),
        vi("use_h", s1), vi("not_hline", [1, 1, HEIGHT, 1]),
        vi("real_nonh", s1), vi("use_v", s1),
        vi("not_col2", [1, 1, 1, WIDTH]), vi("use_bg", s1),
        vi("out_h", g4), vi("out_v", g4), vi("out_bg", g4), vi("tmp", g4),
    ]

    inputs = [vi("input", g4)]
    outputs = [vi("output", g4)]
    graph = helper.make_graph(nodes, "color_lines", inputs, outputs,
                              initializer=init, value_info=value_info)
    return helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
        ir_version=IR_VERSION)


def solve_color_lines(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
