"""Color-count pattern solver for task 167.

Task 167: 3x3 input with colors {2,3,4}, 3x3 output with color {0,5}.
Rule based on number of distinct non-zero colors in input:
- 1 color  -> top row all 5s
- 2 colors -> main diagonal 5s
- 3 colors -> anti-diagonal 5s
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from ..grids import CHANNELS, HEIGHT, WIDTH, all_examples, to_onehot

OPSET = 10
IR_VERSION_V = 8
DATA_TYPE = TensorProto.FLOAT
GRID_SHAPE = [1, CHANNELS, HEIGHT, WIDTH]


def _check_task(task: dict) -> bool:
    for ex in all_examples(task):
        inp, out = ex["input"], ex["output"]
        if len(inp) != 3 or any(len(row) != 3 for row in inp):
            return False
        if len(out) != 3 or any(len(row) != 3 for row in out):
            return False
        in_colors = {v for row in inp for v in row if v != 0}
        out_colors = {v for row in out for v in row if v != 0}
        if not in_colors.issubset({2, 3, 4}):
            return False
        if not out_colors.issubset({5}):
            return False
    return True


def _n_colors(inp: list) -> int:
    return len({v for row in inp for v in row if v != 0})


def _expected_output(n: int) -> list:
    if n == 1:
        return [[5, 5, 5], [0, 0, 0], [0, 0, 0]]
    elif n == 2:
        return [[5, 0, 0], [0, 5, 0], [0, 0, 5]]
    elif n == 3:
        return [[0, 0, 5], [0, 5, 0], [5, 0, 0]]
    return [[0, 0, 0], [0, 0, 0], [0, 0, 0]]


def _verify_rule(task: dict) -> bool:
    for ex in all_examples(task):
        inp, out = ex["input"], ex["output"]
        n = _n_colors(inp)
        if out != _expected_output(n):
            return False
    return True


def solve_colorcount_pattern(task: dict) -> Optional[onnx.ModelProto]:
    if not _check_task(task):
        return None
    if not _verify_rule(task):
        return None

    def _i64(name, vals):
        return numpy_helper.from_array(np.array(vals, dtype=np.int64), name)

    def _f32(name, vals):
        return numpy_helper.from_array(np.array(vals, dtype=np.float32), name)

    nodes = []
    initializers = []

    # Step 1: Slice to 3x3 content (skip channel 0)
    initializers.extend([
        _i64("slice_starts", [0, 1, 0, 0]),
        _i64("slice_ends", [1, CHANNELS, 3, 3]),
        _i64("slice_axes", [0, 1, 2, 3]),
        _i64("slice_steps", [1, 1, 1, 1]),
    ])
    nodes.append(helper.make_node(
        "Slice", ["input", "slice_starts", "slice_ends",
                  "slice_axes", "slice_steps"], ["content"]))

    # Step 2: Count distinct non-zero colors by checking which channels
    # have any 1s in the 3x3 region.
    nodes.append(helper.make_node(
        "ReduceSum", ["content"], ["per_ch"], axes=[2, 3], keepdims=0))
    initializers.append(_f32("zero_f", [0.0]))
    nodes.append(helper.make_node(
        "Greater", ["per_ch", "zero_f"], ["has_ink"]))
    nodes.append(helper.make_node(
        "Cast", ["has_ink"], ["has_ink_f"], to=int(TensorProto.FLOAT)))
    nodes.append(helper.make_node(
        "ReduceSum", ["has_ink_f"], ["n_colors_f"], axes=[1], keepdims=0))
    initializers.append(_i64("scalar_shape", [1]))
    nodes.append(helper.make_node(
        "Reshape", ["n_colors_f", "scalar_shape"], ["n_colors_flat"]))
    nodes.append(helper.make_node(
        "Cast", ["n_colors_flat"], ["n_colors"], to=int(TensorProto.INT64)))

    # Step 3: For each n_colors value, add matching canvas
    n_to_pattern = {1: _expected_output(1),
                     2: _expected_output(2),
                     3: _expected_output(3)}

    idx = 0
    for n in sorted(n_to_pattern.keys()):
        pattern = n_to_pattern[n]
        canvas = to_onehot(pattern)
        initializers.append(_f32(f"canvas_{idx}", canvas))
        initializers.append(_i64(f"target_{idx}", [n]))

        nodes.append(helper.make_node(
            "Equal", ["n_colors", f"target_{idx}"], [f"eq_{idx}"]))
        nodes.append(helper.make_node(
            "Cast", [f"eq_{idx}"], [f"match_{idx}"],
            to=int(TensorProto.FLOAT)))
        nodes.append(helper.make_node(
            "Mul", [f"match_{idx}", f"canvas_{idx}"], [f"scaled_{idx}"]))

        if idx == 0:
            nodes.append(helper.make_node(
                "Identity", [f"scaled_{idx}"], [f"acc_{idx}"]))
        else:
            nodes.append(helper.make_node(
                "Add", [f"acc_{idx-1}", f"scaled_{idx}"], [f"acc_{idx}"]))
        idx += 1

    nodes.append(helper.make_node("Identity", [f"acc_{idx - 1}"], ["output"]))

    x = helper.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = helper.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    graph = helper.make_graph(nodes, "colorcount_pattern", [x], [y],
                              initializers)
    return helper.make_model(graph, ir_version=IR_VERSION_V,
                             opset_imports=[helper.make_opsetid("", OPSET)])