"""Solver: gravity down - each column's cells fall to the bottom edge.

Every non-background cell slides straight down until it rests on the bottom
edge or on the stack of cells already below it. This targets the
single-colour-per-column case (task 032), where per-channel sinking is exact:
for colour channel k and column c the output fills the bottom `count(k, c)`
rows of the real grid, where `count(k, c)` is how many cells of colour k sit
in that column of the input.

The scorer always feeds a 30x30 one-hot tensor with the real grid in the
top-left corner and *all channels zero* outside it, so the true grid height
can't be read from the tensor shape. We recover it from a content mask
(`H_real = number of rows that contain any colour`) and sink each colour into
`[H_real - count, H_real)`. The implementation is pure arithmetic on fixed
30-length index ramps - no Range, no sort - and rebuilds channel 0 inside the
real grid so the output stays a valid one-hot tensor.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from ..grids import CHANNELS, HEIGHT, WIDTH, all_examples

OPSET = 11
IR_VERSION = 8


def _detect(task: dict) -> bool:
    examples = list(all_examples(task))
    if not examples:
        return False
    for ex in examples:
        inp, out = ex["input"], ex["output"]
        h, w = len(inp), len(inp[0])
        if len(out) != h or len(out[0]) != w:
            return False
        for c in range(w):
            vals = [inp[r][c] for r in range(h) if inp[r][c] != 0]
            col = [0] * (h - len(vals)) + vals
            for r in range(h):
                if out[r][c] != col[r]:
                    return False
    return True


def _build() -> onnx.ModelProto:
    def i64(name, vals):
        return numpy_helper.from_array(np.array(vals, dtype=np.int64), name)

    def f32(name, arr):
        return numpy_helper.from_array(arr.astype(np.float32), name)

    rows = np.arange(HEIGHT, dtype=np.float32).reshape(1, 1, HEIGHT, 1)
    cols = np.arange(WIDTH, dtype=np.float32).reshape(1, 1, 1, WIDTH)

    init = [
        f32("zero_f", np.array([0.0])),
        f32("one_f", np.array([1.0])),
        f32("rows", rows),
        f32("cols", cols),
        i64("c1", [1]),
        i64("c10", [CHANNELS]),
        i64("axis_ch", [1]),
        i64("step1", [1]),
    ]

    nodes = [
        # content mask: 1 inside the real grid, 0 in the all-zero padding region
        helper.make_node("ReduceSum", ["input"], ["content"], axes=[1],
                         keepdims=1),                       # [1,1,H,W]
        # real grid height: count rows that hold any content
        helper.make_node("ReduceMax", ["content"], ["row_has"], axes=[3],
                         keepdims=1),                       # [1,1,H,1]
        helper.make_node("ReduceSum", ["row_has"], ["H_real"], axes=[2],
                         keepdims=1),                       # [1,1,1,1]
        # real grid width: count columns that hold any content
        helper.make_node("ReduceMax", ["content"], ["col_has"], axes=[2],
                         keepdims=1),                       # [1,1,1,W]
        helper.make_node("ReduceSum", ["col_has"], ["W_real"], axes=[3],
                         keepdims=1),                       # [1,1,1,1]

        # per-channel, per-column colour counts
        helper.make_node("Greater", ["input", "zero_f"], ["mask_b"]),
        helper.make_node("Cast", ["mask_b"], ["mask"], to=TensorProto.FLOAT),
        helper.make_node("ReduceSum", ["mask"], ["count"], axes=[2],
                         keepdims=1),                       # [1,10,1,W]

        # fill rows [H_real - count, H_real): low <= row AND row < H_real
        helper.make_node("Sub", ["H_real", "count"], ["low"]),  # [1,10,1,W]
        helper.make_node("Less", ["rows", "low"], ["below_low_b"]),
        helper.make_node("Cast", ["below_low_b"], ["below_low"],
                         to=TensorProto.FLOAT),
        helper.make_node("Sub", ["one_f", "below_low"], ["ge_low"]),  # row>=low
        helper.make_node("Less", ["rows", "H_real"], ["lt_h_b"]),
        helper.make_node("Cast", ["lt_h_b"], ["lt_h"], to=TensorProto.FLOAT),
        helper.make_node("Mul", ["ge_low", "lt_h"], ["sunk_all"]),  # [1,10,H,W]

        # colour channels 1..9; channel 0 rebuilt as the in-grid complement
        helper.make_node("Slice", ["sunk_all", "c1", "c10", "axis_ch", "step1"],
                         ["colors"]),                       # [1,9,H,W]
        helper.make_node("ReduceSum", ["colors"], ["color_sum"], axes=[1],
                         keepdims=1),                       # [1,1,H,W]
        # in-grid mask = (row < H_real) AND (col < W_real)
        helper.make_node("Less", ["rows", "H_real"], ["row_in_b"]),
        helper.make_node("Cast", ["row_in_b"], ["row_in"], to=TensorProto.FLOAT),
        helper.make_node("Less", ["cols", "W_real"], ["col_in_b"]),
        helper.make_node("Cast", ["col_in_b"], ["col_in"], to=TensorProto.FLOAT),
        helper.make_node("Mul", ["row_in", "col_in"], ["in_grid"]),  # [1,1,H,W]
        helper.make_node("Sub", ["one_f", "color_sum"], ["not_color"]),
        helper.make_node("Mul", ["in_grid", "not_color"], ["bg"]),   # [1,1,H,W]
        helper.make_node("Concat", ["bg", "colors"], ["output"], axis=1),
    ]

    inputs = [helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, CHANNELS, HEIGHT, WIDTH])]
    outputs = [helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, CHANNELS, HEIGHT, WIDTH])]
    graph = helper.make_graph(nodes, "gravity_down", inputs, outputs,
                              initializer=init)
    return helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
        ir_version=IR_VERSION)


def solve_gravity_down(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
