"""Solver: fill the row-span between two wall cells with a fixed colour.

Within each row, any background cell that has a wall-colour cell somewhere to
its left *and* somewhere to its right is repainted with a fixed fill colour;
walls and everything else stay put (task 258). The wall and fill colours are
detected from the task and baked into the graph.

One-hot channel arithmetic on the (1, 10, 30, 30) tensor:
  ch_src   = the wall-colour cells
  left/right = a wall lies to the left / right (exclusive prefix/suffix CumSum)
  fill_mask  = background cells flanked on both sides
  output     = input with those cells recoloured to the fill colour
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from ..grids import CHANNELS, HEIGHT, WIDTH, all_examples


OPSET = 11
IR_VERSION = 8


def _transform(grid, source, fill):
    H = len(grid); W = len(grid[0])
    out = [row[:] for row in grid]
    for r in range(H):
        for c in range(W):
            if grid[r][c] != 0:
                continue
            left = any(grid[r][cc] == source for cc in range(0, c))
            right = any(grid[r][cc] == source for cc in range(c + 1, W))
            if left and right:
                out[r][c] = fill
    return out


def _params(task: dict):
    """Infer (source, fill) from the examples, or None if the rule doesn't fit."""
    examples = list(all_examples(task))
    if not examples:
        return None
    fills, sources = set(), set()
    for ex in examples:
        i, o = ex["input"], ex["output"]
        if len(i) > 30 or len(i[0]) > 30:
            continue
        if len(i) != len(o) or len(i[0]) != len(o[0]):
            return None
        for r in range(len(i)):
            for c in range(len(i[0])):
                if i[r][c] != 0:
                    sources.add(i[r][c])
                elif o[r][c] != 0:
                    fills.add(o[r][c])
    if len(fills) != 1 or len(sources) != 1:
        return None
    source = next(iter(sources)); fill = next(iter(fills))
    changed = False
    for ex in examples:
        i, o = ex["input"], ex["output"]
        if len(i) > 30 or len(i[0]) > 30:
            continue
        if _transform(i, source, fill) != o:
            return None
        if i != o:
            changed = True
    return (source, fill) if changed else None


def _build(source: int, fill: int) -> onnx.ModelProto:
    F = TensorProto.FLOAT
    B = TensorProto.BOOL

    def f32(name, arr):
        return numpy_helper.from_array(arr.astype(np.float32), name)

    def i64(name, vals):
        return numpy_helper.from_array(np.array(vals, dtype=np.int64), name)

    e0 = np.zeros((1, CHANNELS, 1, 1), dtype=np.float32)
    e0[0, 0, 0, 0] = 1.0
    e_fill = np.zeros((1, CHANNELS, 1, 1), dtype=np.float32)
    e_fill[0, fill, 0, 0] = 1.0

    init = [
        f32("e0", e0), f32("e_fill", e_fill),
        f32("half", np.array([[[[0.5]]]])),
        i64("idx_src", [source]), i64("idx0", [0]),
        numpy_helper.from_array(np.array(3, dtype=np.int64), "axis_w"),
    ]

    n = helper.make_node
    nodes = [
        n("Gather", ["input", "idx_src"], ["ch_src"], axis=1),
        n("Gather", ["input", "idx0"], ["ch0"], axis=1),
        n("CumSum", ["ch_src", "axis_w"], ["lcum"], exclusive=1, reverse=0),
        n("CumSum", ["ch_src", "axis_w"], ["rcum"], exclusive=1, reverse=1),
        n("Greater", ["lcum", "half"], ["lh_b"]), n("Cast", ["lh_b"], ["lh"], to=F),
        n("Greater", ["rcum", "half"], ["rh_b"]), n("Cast", ["rh_b"], ["rh"], to=F),
        n("Mul", ["lh", "rh"], ["between"]),
        n("Mul", ["between", "ch0"], ["fill_mask"]),
        n("Mul", ["e0", "fill_mask"], ["e0fm"]),
        n("Mul", ["e_fill", "fill_mask"], ["efill_fm"]),
        n("Sub", ["input", "e0fm"], ["sub1"]),
        n("Add", ["sub1", "efill_fm"], ["output"]),
    ]

    def vi(name, shape, dt=F):
        return helper.make_tensor_value_info(name, dt, shape)

    g4 = [1, CHANNELS, HEIGHT, WIDTH]
    s1 = [1, 1, HEIGHT, WIDTH]
    value_info = [
        vi("ch_src", s1), vi("ch0", s1), vi("lcum", s1), vi("rcum", s1),
        vi("lh_b", s1, B), vi("lh", s1), vi("rh_b", s1, B), vi("rh", s1),
        vi("between", s1), vi("fill_mask", s1),
        vi("e0fm", g4), vi("efill_fm", g4), vi("sub1", g4),
    ]

    inputs = [vi("input", g4)]
    outputs = [vi("output", g4)]
    graph = helper.make_graph(nodes, "hspan_fill", inputs, outputs,
                              initializer=init, value_info=value_info)
    return helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
        ir_version=IR_VERSION)


def solve_hspan_fill(task: dict) -> Optional[onnx.ModelProto]:
    params = _params(task)
    if params is None:
        return None
    return _build(*params)
