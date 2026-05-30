"""Solver: connect aligned same-colour dots with a fixed fill colour.

Like `connect_dots`, but the gap between two same-colour dots (in a row or a
column) is filled with one fixed colour instead of the dots' own colour, and
the original cells are kept unchanged (tasks 50, 350, 356):

    a background cell is painted `fill` iff, for some colour k, it lies between
    two k-cells along its row OR along its column.

Per colour channel k and axis, `between = (forward cumsum > 0) AND
(reverse cumsum > 0)`. The union over both axes and all colours, intersected
with the background, gives the cells to paint; the input is otherwise copied.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from ..grids import CHANNELS, HEIGHT, WIDTH, all_examples

OPSET = 11
IR_VERSION = 8
NC = CHANNELS - 1


def _connect_fill(grid, fill):
    g = np.array(grid)
    H, W = g.shape
    between = np.zeros((H, W), bool)
    for k in range(1, 10):
        m = (g == k).astype(int)
        lh = np.cumsum(m, axis=1)
        rh = np.cumsum(m[:, ::-1], axis=1)[:, ::-1]
        lv = np.cumsum(m, axis=0)
        rv = np.cumsum(m[::-1], axis=0)[::-1]
        between |= ((lh > 0) & (rh > 0)) | ((lv > 0) & (rv > 0))
    out = g.copy()
    out[between & (g == 0)] = fill
    return out.tolist()


def _detect(task: dict):
    """Return the fixed fill colour, or None."""
    examples = list(all_examples(task))
    if not examples:
        return None
    for ex in examples:
        if (len(ex["input"]) != len(ex["output"])
                or len(ex["input"][0]) != len(ex["output"][0])):
            return None
    for fill in range(1, 10):
        if all(_connect_fill(ex["input"], fill) == ex["output"]
               for ex in examples) and \
           any(ex["input"] != ex["output"] for ex in examples):
            return fill
    return None


def _build(fill: int) -> onnx.ModelProto:
    def i64(name, vals):
        return numpy_helper.from_array(np.array(vals, dtype=np.int64), name)

    def f32(name, arr):
        return numpy_helper.from_array(arr.astype(np.float32), name)

    rev_w = list(range(WIDTH - 1, -1, -1))
    rev_h = list(range(HEIGHT - 1, -1, -1))
    init = [
        i64("c1", [1]), i64("c10", [CHANNELS]), i64("ax1", [1]), i64("st1", [1]),
        i64("axis_w", [3]), i64("axis_h", [2]),
        i64("rev_w", rev_w), i64("rev_h", rev_h),
        f32("zero", np.array([0.0])), f32("one_f", np.array([1.0])),
        f32("fill_f", np.array([float(fill)])),
    ]

    nodes = [
        helper.make_node("Slice", ["input", "c1", "c10", "ax1", "st1"],
                         ["colors"]),                       # [1,9,H,W]
        # horizontal between
        helper.make_node("CumSum", ["colors", "axis_w"], ["preh"]),
        helper.make_node("Gather", ["colors", "rev_w"], ["revh"], axis=3),
        helper.make_node("CumSum", ["revh", "axis_w"], ["revh_cs"]),
        helper.make_node("Gather", ["revh_cs", "rev_w"], ["sufh"], axis=3),
        helper.make_node("Greater", ["preh", "zero"], ["preh_b"]),
        helper.make_node("Greater", ["sufh", "zero"], ["sufh_b"]),
        helper.make_node("And", ["preh_b", "sufh_b"], ["bh"]),
        # vertical between
        helper.make_node("CumSum", ["colors", "axis_h"], ["prev"]),
        helper.make_node("Gather", ["colors", "rev_h"], ["revv"], axis=2),
        helper.make_node("CumSum", ["revv", "axis_h"], ["revv_cs"]),
        helper.make_node("Gather", ["revv_cs", "rev_h"], ["sufv"], axis=2),
        helper.make_node("Greater", ["prev", "zero"], ["prev_b"]),
        helper.make_node("Greater", ["sufv", "zero"], ["sufv_b"]),
        helper.make_node("And", ["prev_b", "sufv_b"], ["bv"]),
        # between (per colour) = bh OR bv ; reduce over colours
        helper.make_node("Or", ["bh", "bv"], ["between_c"]),
        helper.make_node("Cast", ["between_c"], ["between_f"],
                         to=TensorProto.FLOAT),              # [1,9,H,W]
        helper.make_node("ReduceMax", ["between_f"], ["between_any"], axes=[1],
                         keepdims=1),                        # [1,1,H,W]
        # background channel (channel 0): paint only here
        helper.make_node("Slice", ["input", "c0_s", "c1", "ax1", "st1"],
                         ["bg_ch"]),                         # [1,1,H,W]
        helper.make_node("Mul", ["between_any", "bg_ch"], ["paint"]),
        # new background = bg minus painted; fill channel gets the painted cells
        helper.make_node("Sub", ["bg_ch", "paint"], ["bg_new"]),
        helper.make_node("Slice", ["input", "fc_s", "fc_e", "ax1", "st1"],
                         ["fill_ch_in"]),
        helper.make_node("Add", ["fill_ch_in", "paint"], ["fill_new"]),
    ]
    init += [i64("c0_s", [0]),
             i64("fc_s", [fill]), i64("fc_e", [fill + 1])]

    # reassemble channels: [bg_new][1..fill-1][fill_new][fill+1..9]
    parts = ["bg_new"]
    if fill > 1:
        nodes.append(helper.make_node(
            "Slice", ["input", "mid_s", "mid_e", "ax1", "st1"], ["mid"]))
        init += [i64("mid_s", [1]), i64("mid_e", [fill])]
        parts.append("mid")
    parts.append("fill_new")
    if fill < CHANNELS - 1:
        nodes.append(helper.make_node(
            "Slice", ["input", "aft_s", "aft_e", "ax1", "st1"], ["aft"]))
        init += [i64("aft_s", [fill + 1]), i64("aft_e", [CHANNELS])]
        parts.append("aft")
    nodes.append(helper.make_node("Concat", parts, ["output"], axis=1))

    inputs = [helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, CHANNELS, HEIGHT, WIDTH])]
    outputs = [helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, CHANNELS, HEIGHT, WIDTH])]
    value_info = [
        helper.make_tensor_value_info("colors", TensorProto.FLOAT,
                                      [1, NC, HEIGHT, WIDTH]),
        helper.make_tensor_value_info("between_any", TensorProto.FLOAT,
                                      [1, 1, HEIGHT, WIDTH]),
        helper.make_tensor_value_info("bg_ch", TensorProto.FLOAT,
                                      [1, 1, HEIGHT, WIDTH]),
        helper.make_tensor_value_info("paint", TensorProto.FLOAT,
                                      [1, 1, HEIGHT, WIDTH]),
        helper.make_tensor_value_info("bg_new", TensorProto.FLOAT,
                                      [1, 1, HEIGHT, WIDTH]),
        helper.make_tensor_value_info("fill_ch_in", TensorProto.FLOAT,
                                      [1, 1, HEIGHT, WIDTH]),
        helper.make_tensor_value_info("fill_new", TensorProto.FLOAT,
                                      [1, 1, HEIGHT, WIDTH]),
    ]
    graph = helper.make_graph(nodes, "connect_fill", inputs, outputs,
                              initializer=init, value_info=value_info)
    return helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
        ir_version=IR_VERSION)


def solve_connect_fill(task: dict) -> Optional[onnx.ModelProto]:
    fill = _detect(task)
    if fill is None:
        return None
    return _build(fill)
