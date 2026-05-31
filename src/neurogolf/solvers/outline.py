"""Solver: keep each shape's perimeter, erase its interior.

A coloured cell survives only if at least one of its four orthogonal neighbours
is background (or off the real grid); cells whose four neighbours are all
coloured (interior cells) are turned to background. Colours are preserved on the
surviving border (task 98).

One-hot channel arithmetic on the (1, 10, 30, 30) tensor:
  nonbg     = coloured cells
  nbr       = count of coloured 4-neighbours (3x3 cross convolution)
  interior  = coloured cell with all four neighbours coloured (nbr == 4)
  perimeter = nonbg - interior
  output    = input on the perimeter + background everywhere else
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
    H, W = len(grid), len(grid[0])
    out = [[0] * W for _ in range(H)]
    for r in range(H):
        for c in range(W):
            if grid[r][c] == 0:
                continue
            border = False
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if not (0 <= rr < H and 0 <= cc < W) or grid[rr][cc] == 0:
                    border = True
            if border:
                out[r][c] = grid[r][c]
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
    B = TensorProto.BOOL

    def f32(name, arr):
        return numpy_helper.from_array(arr.astype(np.float32), name)

    e0 = np.zeros((1, CHANNELS, 1, 1), dtype=np.float32)
    e0[0, 0, 0, 0] = 1.0
    cross = np.zeros((1, 1, 3, 3), dtype=np.float32)
    for (r, c) in ((0, 1), (1, 0), (1, 2), (2, 1)):
        cross[0, 0, r, c] = 1.0

    init = [
        f32("e0", e0), f32("cross", cross),
        f32("thr", np.array([[[[3.5]]]])),
        numpy_helper.from_array(np.array([0], dtype=np.int64), "idx0"),
    ]

    n = helper.make_node
    nodes = [
        n("ReduceSum", ["input"], ["content"], axes=[1], keepdims=1),
        n("Gather", ["input", "idx0"], ["ch0"], axis=1),
        n("Sub", ["content", "ch0"], ["nonbg"]),
        n("Conv", ["nonbg", "cross"], ["nbr"],
          kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        n("Greater", ["nbr", "thr"], ["ge4_b"]),
        n("Cast", ["ge4_b"], ["ge4"], to=F),
        n("Mul", ["nonbg", "ge4"], ["interior"]),
        n("Sub", ["nonbg", "interior"], ["perimeter"]),
        n("Mul", ["input", "perimeter"], ["keep"]),
        n("Sub", ["content", "perimeter"], ["notper"]),
        n("Mul", ["e0", "notper"], ["out_bg"]),
        n("Add", ["keep", "out_bg"], ["output"]),
    ]

    def vi(name, shape, dt=F):
        return helper.make_tensor_value_info(name, dt, shape)

    g4 = [1, CHANNELS, HEIGHT, WIDTH]
    s1 = [1, 1, HEIGHT, WIDTH]
    value_info = [
        vi("content", s1), vi("ch0", s1), vi("nonbg", s1), vi("nbr", s1),
        vi("ge4_b", s1, B), vi("ge4", s1), vi("interior", s1),
        vi("perimeter", s1), vi("keep", g4), vi("notper", s1), vi("out_bg", g4),
    ]

    inputs = [vi("input", g4)]
    outputs = [vi("output", g4)]
    graph = helper.make_graph(nodes, "outline", inputs, outputs,
                              initializer=init, value_info=value_info)
    return helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
        ir_version=IR_VERSION)


def solve_outline(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
