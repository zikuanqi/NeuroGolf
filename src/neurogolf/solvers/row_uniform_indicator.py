"""Solver for "constant-shape input where output is row-uniform indicator".

Detects tasks where every example has the same input shape (H, W), the same
output shape (H, W), and the rule is: for each row r, output row r is filled
with color X if input row r is uniform (all cells the same color), else
filled with color Y. The (X, Y) pair is consistent across all examples.

Currently picks up task 052 with (X=5, Y=0).

ONNX pipeline (opset 11):
  row_chan_sum  = ReduceSum(input, axis=3)               (1, 10, 30, 1)
  is_uniform_b  = row_chan_sum > (W - 0.5)               bool
  row_uniform_f = ReduceMax(cast to float, axis=1)       (1, 1, 30, 1)
  row_5_or_0    = row_uniform_f * (X - Y) + Y            (still constant per row)
  in_grid       = (arange_h < H) ⊗ (arange_w < W)        (1, 1, 30, 30)
  color_indices = int(row_5_or_0 * in_grid)
                  → X or Y inside the grid, 0 outside (but we then mask)
  onehot_raw    = OneHot(color_indices, depth=10, [0, 1])    (1, 10, 30, 30)
  output        = onehot_raw * in_grid                       padding cleared

We discovered the hard way that OneHot in opset 11 wraps negative indices
modulo depth, so an "indices = -1 → all-zero one-hot" trick doesn't work.
Multiplying the OneHot output by the in_grid mask is the safest fix.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from ..grids import CHANNELS, HEIGHT, WIDTH, all_examples

OPSET = 11
IR_VERSION = 8


def _detect(task: dict) -> tuple[int, int, int, int] | None:
    examples = list(all_examples(task))
    if not examples:
        return None
    shapes_in = {(len(e["input"]), len(e["input"][0])) for e in examples}
    shapes_out = {(len(e["output"]), len(e["output"][0])) for e in examples}
    if len(shapes_in) != 1 or len(shapes_out) != 1:
        return None
    H, W = shapes_in.pop()
    OH, OW = shapes_out.pop()
    if (H, W) != (OH, OW) or H == 0 or W == 0:
        return None
    if H > HEIGHT or W > WIDTH:
        return None
    for X in range(CHANNELS):
        for Y in range(CHANNELS):
            if X == Y:
                continue
            ok = True
            for ex in examples:
                inp, out = ex["input"], ex["output"]
                for r in range(H):
                    is_unif = len(set(inp[r])) == 1
                    expected = X if is_unif else Y
                    for c in range(W):
                        if out[r][c] != expected:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if ok:
                return H, W, X, Y
    return None


def _build(H: int, W: int, X: int, Y: int) -> onnx.ModelProto:
    def i64(name: str, vals) -> onnx.TensorProto:
        return numpy_helper.from_array(np.array(vals, dtype=np.int64), name)

    def f32(name: str, vals) -> onnx.TensorProto:
        return numpy_helper.from_array(np.array(vals, dtype=np.float32), name)

    init = [
        f32("arange_h", np.arange(HEIGHT)),
        f32("arange_w", np.arange(WIDTH)),
        i64("shape4_h", [1, 1, HEIGHT, 1]),
        i64("shape4_w", [1, 1, 1, WIDTH]),
        i64("flat_3d", [1, HEIGHT, WIDTH]),
        f32("thresh_unif", float(W - 0.5)),  # row_chan_sum > W-0.5 iff == W
        f32("h_max", float(H)),
        f32("w_max", float(W)),
        f32("color_diff", float(X - Y)),
        f32("color_y", float(Y)),
        i64("depth", [10]),
        f32("values", [0.0, 1.0]),
    ]

    value_info = [
        helper.make_tensor_value_info(
            "row_chan_sum", TensorProto.FLOAT, [1, CHANNELS, HEIGHT, 1]),
        helper.make_tensor_value_info(
            "is_unif_b", TensorProto.BOOL, [1, CHANNELS, HEIGHT, 1]),
        helper.make_tensor_value_info(
            "is_unif_f", TensorProto.FLOAT, [1, CHANNELS, HEIGHT, 1]),
        helper.make_tensor_value_info(
            "row_unif", TensorProto.FLOAT, [1, 1, HEIGHT, 1]),
        helper.make_tensor_value_info(
            "row_scaled", TensorProto.FLOAT, [1, 1, HEIGHT, 1]),
        helper.make_tensor_value_info(
            "row_with_y", TensorProto.FLOAT, [1, 1, HEIGHT, 1]),
        helper.make_tensor_value_info(
            "arange_h_4d", TensorProto.FLOAT, [1, 1, HEIGHT, 1]),
        helper.make_tensor_value_info(
            "arange_w_4d", TensorProto.FLOAT, [1, 1, 1, WIDTH]),
        helper.make_tensor_value_info(
            "h_mask_b", TensorProto.BOOL, [1, 1, HEIGHT, 1]),
        helper.make_tensor_value_info(
            "w_mask_b", TensorProto.BOOL, [1, 1, 1, WIDTH]),
        helper.make_tensor_value_info(
            "h_mask_f", TensorProto.FLOAT, [1, 1, HEIGHT, 1]),
        helper.make_tensor_value_info(
            "w_mask_f", TensorProto.FLOAT, [1, 1, 1, WIDTH]),
        helper.make_tensor_value_info(
            "in_grid", TensorProto.FLOAT, [1, 1, HEIGHT, WIDTH]),
        helper.make_tensor_value_info(
            "color_inside", TensorProto.FLOAT, [1, 1, HEIGHT, WIDTH]),
        helper.make_tensor_value_info(
            "color_i", TensorProto.INT64, [1, 1, HEIGHT, WIDTH]),
        helper.make_tensor_value_info(
            "color_3d", TensorProto.INT64, [1, HEIGHT, WIDTH]),
        helper.make_tensor_value_info(
            "onehot_raw", TensorProto.FLOAT,
            [1, CHANNELS, HEIGHT, WIDTH]),
    ]

    nodes = [
        # Detect uniform rows: row_chan_sum[ch, r] == W iff all W cells in row r have color ch.
        helper.make_node(
            "ReduceSum", ["input"], ["row_chan_sum"],
            axes=[3], keepdims=1, name="row_sum"),
        helper.make_node(
            "Greater", ["row_chan_sum", "thresh_unif"], ["is_unif_b"],
            name="is_unif_gt"),
        helper.make_node(
            "Cast", ["is_unif_b"], ["is_unif_f"], to=TensorProto.FLOAT,
            name="is_unif_cast"),
        helper.make_node(
            "ReduceMax", ["is_unif_f"], ["row_unif"],
            axes=[1], keepdims=1, name="row_unif_any"),
        # row_5_or_0 = row_unif * (X-Y) + Y
        helper.make_node(
            "Mul", ["row_unif", "color_diff"], ["row_scaled"],
            name="scale_by_diff"),
        helper.make_node(
            "Add", ["row_scaled", "color_y"], ["row_with_y"],
            name="add_baseline"),
        # in-grid mask = (arange_h < H) * (arange_w < W)
        helper.make_node(
            "Reshape", ["arange_h", "shape4_h"], ["arange_h_4d"],
            name="resh_ah"),
        helper.make_node(
            "Reshape", ["arange_w", "shape4_w"], ["arange_w_4d"],
            name="resh_aw"),
        helper.make_node(
            "Less", ["arange_h_4d", "h_max"], ["h_mask_b"], name="h_mask_op"),
        helper.make_node(
            "Less", ["arange_w_4d", "w_max"], ["w_mask_b"], name="w_mask_op"),
        helper.make_node(
            "Cast", ["h_mask_b"], ["h_mask_f"], to=TensorProto.FLOAT,
            name="h_mask_cast"),
        helper.make_node(
            "Cast", ["w_mask_b"], ["w_mask_f"], to=TensorProto.FLOAT,
            name="w_mask_cast"),
        helper.make_node(
            "Mul", ["h_mask_f", "w_mask_f"], ["in_grid"], name="in_grid_op"),
        # Inside grid: color is row_with_y (X or Y). Outside grid: 0 (will be
        # masked). We cast to int and feed into OneHot at axis=1.
        helper.make_node(
            "Mul", ["row_with_y", "in_grid"], ["color_inside"],
            name="color_inside_op"),
        helper.make_node(
            "Cast", ["color_inside"], ["color_i"], to=TensorProto.INT64,
            name="color_to_i"),
        helper.make_node(
            "Reshape", ["color_i", "flat_3d"], ["color_3d"],
            name="squeeze_to_3d"),
        helper.make_node(
            "OneHot", ["color_3d", "depth", "values"], ["onehot_raw"],
            axis=1, name="onehot_op"),
        # Zero out padding cells (outside the grid). Multiply by in_grid;
        # broadcasts (1, 1, 30, 30) to (1, 10, 30, 30).
        helper.make_node(
            "Mul", ["onehot_raw", "in_grid"], ["output"],
            name="clear_padding"),
    ]

    inputs = [helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, CHANNELS, HEIGHT, WIDTH])]
    outputs = [helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, CHANNELS, HEIGHT, WIDTH])]
    graph = helper.make_graph(
        nodes, "row_uniform_indicator", inputs, outputs,
        initializer=init, value_info=value_info)
    return helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
        ir_version=IR_VERSION)


def solve_row_uniform_indicator(task: dict) -> Optional[onnx.ModelProto]:
    spec = _detect(task)
    if spec is None:
        return None
    return _build(*spec)
