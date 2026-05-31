"""Solver: paint each marker's colour straight down its column.

Going down each column, every cell takes the colour of the nearest marker at or
above it (a downward forward-fill); cells above the first marker stay background
(task 322).

One-hot channel arithmetic on the (1, 10, 30, 30) tensor:
  markpos = (row+1) at each coloured cell (channel 0 zeroed)
  cumpos  = cumulative max of markpos down each column (log-doubling Pad/Max),
            i.e. the row of the nearest marker at or above, per colour
  owner   = the colour whose cumpos equals the column-wide max (nearest marker)
  output  = owner colour where a marker lies above + background elsewhere
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
    out = [row[:] for row in grid]
    for c in range(W):
        cur = 0
        for r in range(H):
            if grid[r][c] != 0:
                cur = grid[r][c]
            out[r][c] = cur
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

    def i64(name, vals):
        return numpy_helper.from_array(np.array(vals, dtype=np.int64), name)

    rowp1 = (np.arange(HEIGHT) + 1).astype(np.float32).reshape(1, 1, HEIGHT, 1)
    e_nonbg = np.ones((1, CHANNELS, 1, 1), dtype=np.float32)
    e_nonbg[0, 0, 0, 0] = 0.0
    e0 = np.zeros((1, CHANNELS, 1, 1), dtype=np.float32)
    e0[0, 0, 0, 0] = 1.0

    init = [
        f32("rowp1", rowp1), f32("e_nonbg", e_nonbg), f32("e0", e0),
        f32("one", np.array([[[[1.0]]]])), f32("half", np.array([[[[0.5]]]])),
        i64("s_start", [0]), i64("s_end", [HEIGHT]), i64("s_axis", [2]),
    ]
    shifts = [1, 2, 4, 8, 16]
    for s in shifts:
        init.append(i64(f"pads_{s}", [0, 0, s, 0, 0, 0, 0, 0]))

    n = helper.make_node
    nodes = [
        n("ReduceSum", ["input"], ["content"], axes=[1], keepdims=1),
        n("Mul", ["input", "rowp1"], ["markpos0"]),
        n("Mul", ["markpos0", "e_nonbg"], ["cp0"]),
    ]
    prev = "cp0"
    for j, s in enumerate(shifts):
        nodes += [
            n("Pad", [prev, f"pads_{s}"], [f"pad{j}"]),
            n("Slice", [f"pad{j}", "s_start", "s_end", "s_axis"], [f"shift{j}"]),
            n("Max", [prev, f"shift{j}"], [f"cp{j+1}"]),
        ]
        prev = f"cp{j+1}"
    cumpos = prev
    nodes += [
        n("ReduceMax", [cumpos], ["maxcum"], axes=[1], keepdims=1),
        n("Equal", [cumpos, "maxcum"], ["eq_b"]),
        n("Cast", ["eq_b"], ["eqf"], to=F),
        n("Greater", ["maxcum", "half"], ["hof_b"]),
        n("Cast", ["hof_b"], ["hof"], to=F),
        n("Mul", ["eqf", "hof"], ["colored"]),
        n("Mul", ["colored", "content"], ["colored_real"]),
        n("Sub", ["one", "hof"], ["noowner"]),
        n("Mul", ["content", "noowner"], ["bg_mask"]),
        n("Mul", ["e0", "bg_mask"], ["out_bg"]),
        n("Add", ["colored_real", "out_bg"], ["output"]),
    ]

    def vi(name, shape, dt=F):
        return helper.make_tensor_value_info(name, dt, shape)

    g4 = [1, CHANNELS, HEIGHT, WIDTH]
    s1 = [1, 1, HEIGHT, WIDTH]
    value_info = [
        vi("content", s1), vi("markpos0", g4), vi("cp0", g4),
        vi("maxcum", s1), vi("eq_b", g4, B), vi("eqf", g4),
        vi("hof_b", s1, B), vi("hof", s1), vi("colored", g4),
        vi("colored_real", g4), vi("noowner", s1), vi("bg_mask", s1),
        vi("out_bg", g4),
    ]
    for j, s in enumerate(shifts):
        value_info += [
            vi(f"pad{j}", [1, CHANNELS, HEIGHT + s, WIDTH]),
            vi(f"shift{j}", g4), vi(f"cp{j+1}", g4),
        ]

    inputs = [vi("input", g4)]
    outputs = [vi("output", g4)]
    graph = helper.make_graph(nodes, "ray_down", inputs, outputs,
                              initializer=init, value_info=value_info)
    return helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
        ir_version=IR_VERSION)


def solve_ray_down(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
