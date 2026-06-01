"""Count-based output solver for task 186.

Task 186: 3x3 input with color 1, 3x3 output with color 2.
Rule: count number of 1s in input mapped to fixed pattern of 2s.
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
        in_colors = set()
        for row in inp:
            for v in row:
                in_colors.add(v)
        out_colors = set()
        for row in out:
            for v in row:
                out_colors.add(v)
        if in_colors != {0, 1} and in_colors != {0}:
            return False
        if out_colors != {0, 2} and out_colors != {0}:
            return False
    return True


def _count_ones(inp: list) -> int:
    return sum(v == 1 for row in inp for v in row)


def _expected_output(count: int) -> list:
    if count == 1:
        return [[2, 0, 0], [0, 0, 0], [0, 0, 0]]
    elif count == 2:
        return [[2, 2, 0], [0, 0, 0], [0, 0, 0]]
    elif count == 3:
        return [[2, 2, 2], [0, 0, 0], [0, 0, 0]]
    elif count == 4:
        return [[2, 2, 2], [0, 2, 0], [0, 0, 0]]
    else:
        return [[0, 0, 0], [0, 0, 0], [0, 0, 0]]


def _verify_rule(task: dict) -> bool:
    for ex in all_examples(task):
        inp, out = ex["input"], ex["output"]
        n = _count_ones(inp)
        expected = _expected_output(n)
        if out != expected:
            return False
    return True


def solve_count_pattern(task: dict) -> Optional[onnx.ModelProto]:
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

    # Step 1: Slice to 3x3 content, skip channel 0 (background).
    initializers.extend([
        _i64("slice_starts", [0, 1, 0, 0]),
        _i64("slice_ends", [1, CHANNELS, 3, 3]),
        _i64("slice_axes", [0, 1, 2, 3]),
        _i64("slice_steps", [1, 1, 1, 1]),
    ])
    nodes.append(helper.make_node(
        "Slice", ["input", "slice_starts", "slice_ends", "slice_axes", "slice_steps"],
        ["content"]))

    # Step 2: Sum over channels, then sum spatial -> scalar count
    nodes.append(helper.make_node(
        "ReduceSum", ["content"], ["ink"], axes=[1], keepdims=0))

    initializers.append(_i64("shape_1x9", [1, 9]))
    nodes.append(helper.make_node(
        "Reshape", ["ink", "shape_1x9"], ["ink_flat"]))
    nodes.append(helper.make_node(
        "ReduceSum", ["ink_flat"], ["count_2d"], axes=[1], keepdims=0))
    initializers.append(_i64("scalar_shape", [1]))
    nodes.append(helper.make_node(
        "Reshape", ["count_2d", "scalar_shape"], ["count_f"]))

    # Step 3: Cast to int64 for Equal
    nodes.append(helper.make_node(
        "Cast", ["count_f"], ["count_i64"], to=int(TensorProto.INT64)))

    # Step 4: For each (count, pattern) pair, create full one-hot canvas.
    count_to_pattern = {}
    for ex in task["train"]:
        n = _count_ones(ex["input"])
        count_to_pattern[n] = _expected_output(n)

    idx = 0
    for count in sorted(count_to_pattern.keys()):
        pattern = count_to_pattern[count]
        canvas = to_onehot(pattern)
        initializers.append(_f32(f"canvas_{idx}", canvas))
        initializers.append(_i64(f"target_{idx}", [count]))

        nodes.append(helper.make_node(
            "Equal", ["count_i64", f"target_{idx}"], [f"eq_{idx}"]))
        nodes.append(helper.make_node(
            "Cast", [f"eq_{idx}"], [f"match_{idx}"], to=int(TensorProto.FLOAT)))
        nodes.append(helper.make_node(
            "Mul", [f"match_{idx}", f"canvas_{idx}"], [f"scaled_{idx}"]))

        if idx == 0:
            nodes.append(helper.make_node(
                "Identity", ["scaled_0"], ["acc_0"]))
        else:
            nodes.append(helper.make_node(
                "Add", [f"acc_{idx-1}", f"scaled_{idx}"], [f"acc_{idx}"]))
        idx += 1

    nodes.append(helper.make_node("Identity", [f"acc_{idx - 1}"], ["output"]))

    x = helper.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = helper.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    graph = helper.make_graph(nodes, "count_pattern", [x], [y], initializers)
    return helper.make_model(graph, ir_version=IR_VERSION_V,
                             opset_imports=[helper.make_opsetid("", OPSET)])