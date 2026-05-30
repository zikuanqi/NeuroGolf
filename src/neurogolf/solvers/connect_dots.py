"""Solver: connect collinear same-colour dots along rows or columns.

For each colour, a cell becomes that colour iff there is a same-colour cell on
one side of it AND another on the other side along the chosen axis - i.e. the
gap between the first and last dot of a colour in a row (or column) is filled
(tasks 41 and 45).

Per colour channel k:
    left  = cumsum(mask_k)            # how many k at-or-before this cell
    right = cumsum(mask_k reversed)   # how many k at-or-after this cell
    fill  = (left > 0) AND (right > 0)
Channels are combined and the background channel rebuilt so the output stays a
valid one-hot. Colours never collide because a filled span only covers cells
between two dots of the *same* colour.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from ..grids import CHANNELS, HEIGHT, WIDTH, all_examples

OPSET = 11
IR_VERSION = 8


def _connect(grid, axis: int):
    g = np.array(grid)
    h, w = g.shape
    out = np.zeros((h, w), dtype=int)
    for k in range(1, 10):
        m = (g == k).astype(int)
        if axis == 1:                       # along rows (horizontal)
            left = np.cumsum(m, axis=1)
            right = np.cumsum(m[:, ::-1], axis=1)[:, ::-1]
        else:                               # along columns (vertical)
            left = np.cumsum(m, axis=0)
            right = np.cumsum(m[::-1], axis=0)[::-1]
        between = (left > 0) & (right > 0)
        out = np.where(between, k, out)
    return out.tolist()


def _detect(task: dict) -> Optional[int]:
    """Return spatial axis (3 = rows/horizontal, 2 = cols/vertical) or None."""
    examples = list(all_examples(task))
    if not examples:
        return None
    for np_axis, onnx_axis in ((1, 3), (0, 2)):
        ok = True
        changed = False
        for ex in examples:
            inp, out = ex["input"], ex["output"]
            if len(inp) != len(out) or len(inp[0]) != len(out[0]):
                ok = False
                break
            if _connect(inp, np_axis) != out:
                ok = False
                break
            if inp != out:
                changed = True
        if ok and changed:
            return onnx_axis
    return None


def _build(onnx_axis: int) -> onnx.ModelProto:
    def i64(name, vals):
        return numpy_helper.from_array(np.array(vals, dtype=np.int64), name)

    def f32(name, arr):
        return numpy_helper.from_array(arr.astype(np.float32), name)

    # reverse index along the chosen spatial axis (length 30)
    dim = WIDTH if onnx_axis == 3 else HEIGHT
    rev = list(range(dim - 1, -1, -1))

    init = [
        i64("c1", [1]), i64("c10", [CHANNELS]), i64("ax1", [1]), i64("st1", [1]),
        i64("axis_v", [onnx_axis]),
        i64("rev_idx", rev),
        f32("zero", np.array([0.0])), f32("one_f", np.array([1.0])),
    ]

    nodes = [
        # colour channels only (drop background channel 0)
        helper.make_node("Slice", ["input", "c1", "c10", "ax1", "st1"],
                         ["colors"]),                       # [1,9,H,W]
        # left/before prefix count along the axis
        helper.make_node("CumSum", ["colors", "axis_v"], ["pre"]),
        # right/after prefix: reverse, cumsum, reverse back
        helper.make_node("Gather", ["colors", "rev_idx"], ["rev"],
                         axis=onnx_axis),
        helper.make_node("CumSum", ["rev", "axis_v"], ["rev_cs"]),
        helper.make_node("Gather", ["rev_cs", "rev_idx"], ["suf"],
                         axis=onnx_axis),
        # both sides present -> fill
        helper.make_node("Greater", ["pre", "zero"], ["pre_b"]),
        helper.make_node("Greater", ["suf", "zero"], ["suf_b"]),
        helper.make_node("And", ["pre_b", "suf_b"], ["fill_b"]),
        helper.make_node("Cast", ["fill_b"], ["fill_colors"],
                         to=TensorProto.FLOAT),             # [1,9,H,W]
        # rebuild background = 1 - sum(colour channels), but only inside grid
        helper.make_node("ReduceSum", ["input"], ["content"], axes=[1],
                         keepdims=1),
        helper.make_node("Greater", ["content", "zero"], ["in_grid_b"]),
        helper.make_node("Cast", ["in_grid_b"], ["in_grid"],
                         to=TensorProto.FLOAT),
        helper.make_node("ReduceSum", ["fill_colors"], ["csum"], axes=[1],
                         keepdims=1),
        helper.make_node("Sub", ["one_f", "csum"], ["not_color"]),
        helper.make_node("Mul", ["in_grid", "not_color"], ["bg"]),
        helper.make_node("Concat", ["bg", "fill_colors"], ["output"], axis=1),
    ]

    inputs = [helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, CHANNELS, HEIGHT, WIDTH])]
    outputs = [helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, CHANNELS, HEIGHT, WIDTH])]
    value_info = [
        helper.make_tensor_value_info("colors", TensorProto.FLOAT,
                                      [1, CHANNELS - 1, HEIGHT, WIDTH]),
        helper.make_tensor_value_info("pre", TensorProto.FLOAT,
                                      [1, CHANNELS - 1, HEIGHT, WIDTH]),
        helper.make_tensor_value_info("suf", TensorProto.FLOAT,
                                      [1, CHANNELS - 1, HEIGHT, WIDTH]),
        helper.make_tensor_value_info("fill_colors", TensorProto.FLOAT,
                                      [1, CHANNELS - 1, HEIGHT, WIDTH]),
        helper.make_tensor_value_info("bg", TensorProto.FLOAT,
                                      [1, 1, HEIGHT, WIDTH]),
    ]
    graph = helper.make_graph(nodes, "connect_dots", inputs, outputs,
                              initializer=init, value_info=value_info)
    return helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
        ir_version=IR_VERSION)


def solve_connect_dots(task: dict) -> Optional[onnx.ModelProto]:
    axis = _detect(task)
    if axis is None:
        return None
    return _build(axis)
