"""Solver: AND of left/right grids separated by a color-5 column.

Pattern: input has a column of color 5 dividing it into left and right
sub-grids. Output is the AND of corresponding cells (both color 1 = on),
output as color 2, same shape as each half.

Example: task 6: 3x7 input -> two 3x3 halves ANDed -> 3x3 output.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from ..grids import CHANNELS, HEIGHT, WIDTH, all_examples

OPSET = 11
IR_VERSION = 8


def _detect(task: dict) -> tuple[int, int, int, int, int] | None:
    """Return (sep_col, in_color, out_color, oh, ow) or None."""
    examples = list(all_examples(task))
    if not examples:
        return None

    # Find separator column from first example
    inp0 = examples[0]["input"]
    ih0, iw0 = len(inp0), len(inp0[0])
    sep_col = None
    for c in range(iw0):
        if all(inp0[r][c] == 5 for r in range(ih0)):
            sep_col = c
            break
    if sep_col is None:
        return None

    # Check all examples have same separator
    for ex in examples:
        inp = ex["input"]
        if not all(inp[r][sep_col] == 5 for r in range(len(inp))):
            return None

    # Left and right halves must have same width
    left_w = sep_col
    right_w = iw0 - sep_col - 1
    if left_w != right_w or left_w <= 0:
        return None

    # Detect pattern from first example
    ex0 = examples[0]
    inp, out = ex0["input"], ex0["output"]
    ih, iw = len(inp), len(inp[0])
    oh, ow = len(out), len(out[0])

    # Check half colors
    in_half_colors = set()
    out_half_colors = set()
    for r in range(ih):
        for c in range(left_w):
            if inp[r][c] != 0:
                in_half_colors.add(inp[r][c])
        for c in range(sep_col + 1, iw):
            if inp[r][c] != 0:
                in_half_colors.add(inp[r][c])

    for r in range(oh):
        for c in range(ow):
            if out[r][c] != 0:
                out_half_colors.add(out[r][c])

    if len(in_half_colors) != 1 or len(out_half_colors) != 1:
        return None

    in_color = list(in_half_colors)[0]
    out_color = list(out_half_colors)[0]

    # Verify AND pattern across all examples
    for ex in examples:
        inp, out = ex["input"], ex["output"]
        ih, iw = len(inp), len(inp[0])
        eoh, eow = len(out), len(out[0])

        if eoh != oh or eow != ow:
            return None

        for r in range(min(ih, oh)):
            for c in range(min(left_w, ow)):
                left_val = inp[r][c]
                right_val = inp[r][sep_col + 1 + c]
                expected = out_color if (left_val == in_color and right_val == in_color) else 0
                if out[r][c] != expected:
                    return None

    return (sep_col, in_color, out_color, oh, ow)


def _build(sep_col: int, in_color: int, out_color: int,
           out_h: int, out_w: int) -> onnx.ModelProto:
    """Build ONNX graph for two-half AND operation."""

    def i64(name: str, vals) -> onnx.TensorProto:
        return numpy_helper.from_array(np.array(vals, dtype=np.int64), name)

    def f32(name: str, vals) -> onnx.TensorProto:
        return numpy_helper.from_array(np.array(vals, dtype=np.float32), name)

    # Region mask: 1.0 in first out_h x out_w cells, 0.0 elsewhere
    region_mask = np.zeros((1, 1, HEIGHT, WIDTH), dtype=np.float32)
    region_mask[:, :, :out_h, :out_w] = 1.0

    # Slice left half: channel in_color, cols 0:left_w
    starts_left = i64("starts_left", [0, in_color, 0, 0])
    ends_left = i64("ends_left", [1, in_color + 1, HEIGHT, sep_col])
    axes_full = i64("axes_full", [0, 1, 2, 3])

    # Slice right half: channel in_color, cols (sep_col+1):(sep_col+1+out_w)
    starts_right = i64("starts_right", [0, in_color, 0, sep_col + 1])
    ends_right = i64("ends_right", [1, in_color + 1, HEIGHT, sep_col + 1 + out_w])

    # Pad right side to WIDTH
    pad_right = WIDTH - out_w
    pad_const = f32("pad_const", np.array(0.0, dtype=np.float32))
    pads_init = i64("pads", [0, 0, 0, 0, 0, 0, 0, pad_right])

    # Build output: [bg_ch(1), zeros_before, and_padded, zeros_after]
    region_mask_t = f32("region_mask", region_mask)

    if out_color > 1:
        zeros_before = f32("zeros_before",
                           np.zeros((1, out_color - 1, HEIGHT, WIDTH),
                                    dtype=np.float32))
    if out_color < CHANNELS - 1:
        zeros_after = f32("zeros_after",
                          np.zeros((1, CHANNELS - out_color - 1, HEIGHT, WIDTH),
                                   dtype=np.float32))

    nodes = [
        helper.make_node(
            "Slice", ["input", "starts_left", "ends_left", "axes_full"],
            ["left_ch"], name="slice_left"),
        helper.make_node(
            "Slice", ["input", "starts_right", "ends_right", "axes_full"],
            ["right_ch"], name="slice_right"),
        helper.make_node(
            "Mul", ["left_ch", "right_ch"], ["and_result"], name="and_op"),
        helper.make_node(
            "Pad", ["and_result", "pads", "pad_const"], ["and_padded"],
            mode="constant", name="pad_op"),
        # Background channel: region_mask - and_padded
        helper.make_node(
            "Sub", ["region_mask", "and_padded"], ["bg_ch"], name="bg_op"),
    ]

    concat_parts = ["bg_ch"]
    init_parts = [starts_left, ends_left, axes_full,
                  starts_right, ends_right, pads_init, pad_const,
                  region_mask_t]
    if out_color > 1:
        concat_parts.append("zeros_before")
        init_parts.append(zeros_before)
    concat_parts.append("and_padded")
    if out_color < CHANNELS - 1:
        concat_parts.append("zeros_after")
        init_parts.append(zeros_after)

    nodes.append(helper.make_node(
        "Concat", concat_parts, ["output"],
        axis=1, name="concat_output"))

    inputs = [helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, CHANNELS, HEIGHT, WIDTH])]
    outputs = [helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, CHANNELS, HEIGHT, WIDTH])]

    value_info = [
        helper.make_tensor_value_info(
            "left_ch", TensorProto.FLOAT, [1, 1, HEIGHT, sep_col]),
        helper.make_tensor_value_info(
            "right_ch", TensorProto.FLOAT, [1, 1, HEIGHT, out_w]),
        helper.make_tensor_value_info(
            "and_result", TensorProto.FLOAT, [1, 1, HEIGHT, out_w]),
        helper.make_tensor_value_info(
            "and_padded", TensorProto.FLOAT, [1, 1, HEIGHT, WIDTH]),
        helper.make_tensor_value_info(
            "bg_ch", TensorProto.FLOAT, [1, 1, HEIGHT, WIDTH]),
    ]

    graph = helper.make_graph(
        nodes, "split_and", inputs, outputs,
        initializer=init_parts, value_info=value_info)
    return helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
        ir_version=IR_VERSION)


def solve_split_and(task: dict) -> Optional[onnx.ModelProto]:
    params = _detect(task)
    if params is None:
        return None
    sep_col, in_color, out_color, out_h, out_w = params
    return _build(sep_col, in_color, out_color, out_h, out_w)