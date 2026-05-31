"""Solver: repaint every colour-5 cell with the marker colour of its row.

Each row carries a single marker colour (anything other than background or 5)
plus one or more colour-5 cells; every 5 in the row is recoloured to that
marker, while the marker and background stay put (task 312).

One-hot channel arithmetic on the (1, 10, 30, 30) tensor:
  ch5        = the colour-5 cells
  row_marker = per-row marker colour (max over columns, channels 0 and 5 zeroed)
  output     = input with the channel-5 cells cleared, then row_marker painted
               back onto those cells
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
    H = len(grid); W = len(grid[0])
    out = [row[:] for row in grid]
    for r in range(H):
        marks = {x for x in grid[r] if x not in (0, 5)}
        has5 = any(x == 5 for x in grid[r])
        if len(marks) > 1:
            return None
        m = next(iter(marks)) if marks else None
        if has5 and m is None:
            return None
        if m is None:
            continue
        for c in range(W):
            if grid[r][c] == 5:
                out[r][c] = m
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

    e_mark = np.ones((1, CHANNELS, 1, 1), dtype=np.float32)
    e_mark[0, 0, 0, 0] = 0.0
    e_mark[0, 5, 0, 0] = 0.0
    e5 = np.zeros((1, CHANNELS, 1, 1), dtype=np.float32)
    e5[0, 5, 0, 0] = 1.0

    init = [
        f32("e_mark", e_mark), f32("e5", e5),
        numpy_helper.from_array(np.array([5], dtype=np.int64), "idx5"),
    ]

    n = helper.make_node
    nodes = [
        n("Gather", ["input", "idx5"], ["ch5"], axis=1),
        n("Mul", ["input", "e_mark"], ["masked"]),
        n("ReduceMax", ["masked"], ["row_marker"], axes=[3], keepdims=1),
        n("Mul", ["e5", "ch5"], ["e5ch5"]),
        n("Sub", ["input", "e5ch5"], ["base"]),
        n("Mul", ["row_marker", "ch5"], ["fill"]),
        n("Add", ["base", "fill"], ["output"]),
    ]

    def vi(name, shape):
        return helper.make_tensor_value_info(name, F, shape)

    g4 = [1, CHANNELS, HEIGHT, WIDTH]
    value_info = [
        vi("ch5", [1, 1, HEIGHT, WIDTH]),
        vi("masked", g4),
        vi("row_marker", [1, CHANNELS, HEIGHT, 1]),
        vi("e5ch5", g4), vi("base", g4), vi("fill", g4),
    ]

    inputs = [vi("input", g4)]
    outputs = [vi("output", g4)]
    graph = helper.make_graph(nodes, "recolor_fives", inputs, outputs,
                              initializer=init, value_info=value_info)
    return helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
        ir_version=IR_VERSION)


def solve_recolor_fives(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
