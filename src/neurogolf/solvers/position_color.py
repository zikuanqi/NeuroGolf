"""Position-to-color solver for task 262. Zero-channel-corrected version."""
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

COLOR_MAP = {0: 2, 1: 4, 2: 3}


def _check_task(task: dict) -> bool:
    for ex in all_examples(task):
        inp, out = ex["input"], ex["output"]
        if len(inp) != 3 or any(len(row) != 3 for row in inp):
            return False
        if len(out) != 3 or any(len(row) != 3 for row in out):
            return False
        in_colors = {v for row in inp for v in row}
        out_colors = {v for row in out for v in row}
        if not in_colors.issubset({0, 5}):
            return False
        if not out_colors.issubset({0, 2, 3, 4}):
            return False
    return True


def _expected_output(inp: list) -> list:
    out = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for r in range(3):
        for c in range(3):
            if inp[r][c] == 5:
                clr = COLOR_MAP[c]
                out[r] = [clr, clr, clr]
    return out


def _verify_rule(task: dict) -> bool:
    for ex in all_examples(task):
        inp, out = ex["input"], ex["output"]
        if out != _expected_output(inp):
            return False
    return True


def solve_position_color(task: dict) -> Optional[onnx.ModelProto]:
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

    # Step 1: Slice channel 5, 3x3 region
    initializers.extend([
        _i64("starts", [0, 5, 0, 0]),
        _i64("ends", [1, 6, 3, 3]),
        _i64("axes", [0, 1, 2, 3]),
        _i64("steps", [1, 1, 1, 1]),
    ])
    nodes.append(helper.make_node(
        "Slice", ["input", "starts", "ends", "axes", "steps"], ["ch5"]))

    idx = 0
    for r in range(3):
        for c in range(3):
            clr = COLOR_MAP[c]
            initializers.extend([
                _i64(f"s_{idx}", [0, 0, r, c]),
                _i64(f"e_{idx}", [1, 1, r + 1, c + 1]),
                _i64(f"a_{idx}", [0, 1, 2, 3]),
                _i64(f"t_{idx}", [1, 1, 1, 1]),
            ])
            nodes.append(helper.make_node(
                "Slice", ["ch5", f"s_{idx}", f"e_{idx}",
                          f"a_{idx}", f"t_{idx}"], [f"pos_{idx}"]))
            initializers.append(_i64(f"sc_{idx}", [1]))
            nodes.append(helper.make_node(
                "Reshape", [f"pos_{idx}", f"sc_{idx}"], [f"val_{idx}"]))

            out_grid = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
            out_grid[r] = [clr, clr, clr]
            canvas = to_onehot(out_grid)
            initializers.append(_f32(f"can_{idx}", canvas))

            nodes.append(helper.make_node(
                "Mul", [f"val_{idx}", f"can_{idx}"], [f"add_{idx}"]))
            if idx == 0:
                nodes.append(helper.make_node(
                    "Identity", [f"add_{idx}"], [f"acc_{idx}"]))
            else:
                nodes.append(helper.make_node(
                    "Add", [f"acc_{idx - 1}", f"add_{idx}"],
                    [f"acc_{idx}"]))
            idx += 1

    accumulated = f"acc_{idx - 1}"

    # Zero out channel 0 (accumulated 2.0 from empty rows in each canvas)
    initializers.append(_f32("zero_f", [0.0]))
    initializers.extend([
        _i64("ch0_s", [0, 0, 0, 0]),
        _i64("ch0_e", [1, 1, HEIGHT, WIDTH]),
        _i64("ch0_a", [0, 1, 2, 3]),
        _i64("ch0_t", [1, 1, 1, 1]),
    ])
    nodes.append(helper.make_node(
        "Slice", [accumulated, "ch0_s", "ch0_e", "ch0_a", "ch0_t"],
        ["ch0_raw"]))
    nodes.append(helper.make_node(
        "Mul", ["ch0_raw", "zero_f"], ["ch0_zero"]))

    initializers.extend([
        _i64("rest_s", [0, 1, 0, 0]),
        _i64("rest_e", [1, CHANNELS, HEIGHT, WIDTH]),
        _i64("rest_a", [0, 1, 2, 3]),
        _i64("rest_t", [1, 1, 1, 1]),
    ])
    nodes.append(helper.make_node(
        "Slice", [accumulated, "rest_s", "rest_e", "rest_a", "rest_t"],
        ["rest"]))

    nodes.append(helper.make_node(
        "Concat", ["ch0_zero", "rest"], ["output"], axis=1))

    x = helper.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = helper.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    graph = helper.make_graph(nodes, "position_color", [x], [y],
                              initializers)
    return helper.make_model(graph, ir_version=IR_VERSION_V,
                             opset_imports=[helper.make_opsetid("", OPSET)])