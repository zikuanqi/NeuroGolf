"""Solver: every marker fires a full row + column "plus", crossings become 2.

Each non-background cell projects its colour across its entire row and its
entire column. Where one marker's row line meets a *different* marker's column
line, the cell is painted colour 2 instead. A marker's own row/column share its
colour, so the marker and its arms keep that colour (task 47).

Channel arithmetic on the one-hot (1, 10, 30, 30) tensor (channel 0 = bg):
  row_chan = per-row colour   = max over columns, with channel 0 zeroed
  col_chan = per-column colour = max over rows,    with channel 0 zeroed
  h_line / v_line  = 1 where a row / column carries a marker
  same     = row colour equals column colour (one-hot dot product)
  output   = row_chan on (only-row OR matching cross)
           + col_chan on (only-column)
           + colour-2 on (row and column present but differ)
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
    rowcol: dict[int, int] = {}
    colcol: dict[int, int] = {}
    for r in range(H):
        for c in range(W):
            v = int(g[r, c])
            if v == 0:
                continue
            if (r in rowcol and rowcol[r] != v) or (c in colcol and colcol[c] != v):
                return None              # mixed colours on a line: not this rule
            rowcol[r] = v
            colcol[c] = v
    out = [[0] * W for _ in range(H)]
    for r in range(H):
        for c in range(W):
            hr = rowcol.get(r)
            vc = colcol.get(c)
            if hr is not None and vc is not None:
                out[r][c] = hr if hr == vc else 2
            elif hr is not None:
                out[r][c] = hr
            elif vc is not None:
                out[r][c] = vc
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
        res = _transform(inp)
        if res is None or res != out:
            return False
        if inp != out:
            changed = True
    return changed


def _build() -> onnx.ModelProto:
    F = TensorProto.FLOAT

    def f32(name, arr):
        return numpy_helper.from_array(arr.astype(np.float32), name)

    e_notbg = np.ones((1, CHANNELS, 1, 1), dtype=np.float32)
    e_notbg[0, 0, 0, 0] = 0.0          # zero out the background channel
    e2 = np.zeros((1, CHANNELS, 1, 1), dtype=np.float32)
    e2[0, 2, 0, 0] = 1.0               # one-hot of colour 2 (crossings)
    e0 = np.zeros((1, CHANNELS, 1, 1), dtype=np.float32)
    e0[0, 0, 0, 0] = 1.0               # one-hot of background colour 0

    init = [
        f32("e_notbg", e_notbg),
        f32("e2", e2),
        f32("e0", e0),
        f32("one", np.array([[[[1.0]]]])),
    ]

    n = helper.make_node
    nodes = [
        n("ReduceMax", ["input"], ["rc_raw"], axes=[3], keepdims=1),
        n("Mul", ["rc_raw", "e_notbg"], ["row_chan"]),
        n("ReduceMax", ["input"], ["cc_raw"], axes=[2], keepdims=1),
        n("Mul", ["cc_raw", "e_notbg"], ["col_chan"]),
        n("ReduceMax", ["row_chan"], ["h_line"], axes=[1], keepdims=1),
        n("ReduceMax", ["col_chan"], ["v_line"], axes=[1], keepdims=1),
        n("Mul", ["h_line", "v_line"], ["both"]),
        n("Mul", ["row_chan", "col_chan"], ["prod_cc"]),
        n("ReduceSum", ["prod_cc"], ["same_cc"], axes=[1], keepdims=1),
        n("Sub", ["one", "v_line"], ["notv"]),
        n("Sub", ["one", "h_line"], ["noth"]),
        n("Mul", ["h_line", "notv"], ["only_h"]),
        n("Mul", ["noth", "v_line"], ["only_v"]),
        n("Mul", ["both", "same_cc"], ["both_same"]),
        n("Sub", ["one", "same_cc"], ["notsame"]),
        n("Mul", ["both", "notsame"], ["both_diff"]),
        n("Add", ["only_h", "both_same"], ["use_h"]),
        n("Mul", ["row_chan", "use_h"], ["out_h"]),
        n("Mul", ["col_chan", "only_v"], ["out_v"]),
        n("Mul", ["e2", "both_diff"], ["out_2"]),
        # real cells crossed by neither a row nor a column line stay background
        n("ReduceSum", ["input"], ["content"], axes=[1], keepdims=1),
        n("Mul", ["noth", "notv"], ["neither"]),
        n("Mul", ["content", "neither"], ["bg_mask"]),
        n("Mul", ["e0", "bg_mask"], ["out_bg"]),
        n("Add", ["out_h", "out_v"], ["tmp"]),
        n("Add", ["tmp", "out_2"], ["lines"]),
        n("Mul", ["lines", "content"], ["lines_masked"]),  # clip to real grid
        n("Add", ["lines_masked", "out_bg"], ["output"]),
    ]

    def vi(name, shape):
        return helper.make_tensor_value_info(name, F, shape)

    g4 = [1, CHANNELS, HEIGHT, WIDTH]
    s1 = [1, 1, HEIGHT, WIDTH]
    value_info = [
        vi("rc_raw", [1, CHANNELS, HEIGHT, 1]),
        vi("row_chan", [1, CHANNELS, HEIGHT, 1]),
        vi("cc_raw", [1, CHANNELS, 1, WIDTH]),
        vi("col_chan", [1, CHANNELS, 1, WIDTH]),
        vi("h_line", [1, 1, HEIGHT, 1]), vi("v_line", [1, 1, 1, WIDTH]),
        vi("both", s1), vi("prod_cc", g4), vi("same_cc", s1),
        vi("notv", [1, 1, 1, WIDTH]), vi("noth", [1, 1, HEIGHT, 1]),
        vi("only_h", s1), vi("only_v", s1),
        vi("both_same", s1), vi("notsame", s1), vi("both_diff", s1),
        vi("use_h", s1),
        vi("content", s1), vi("neither", s1), vi("bg_mask", s1),
        vi("out_h", g4), vi("out_v", g4), vi("out_2", g4),
        vi("out_bg", g4), vi("tmp", g4), vi("lines", g4),
        vi("lines_masked", g4),
    ]

    inputs = [vi("input", g4)]
    outputs = [vi("output", g4)]
    graph = helper.make_graph(nodes, "cross_laser", inputs, outputs,
                              initializer=init, value_info=value_info)
    return helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
        ir_version=IR_VERSION)


def solve_cross_laser(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
