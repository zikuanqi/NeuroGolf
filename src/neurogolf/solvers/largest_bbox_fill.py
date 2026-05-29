"""Solver: constant-shape output filled with the color whose input bbox is
largest in area.

Detected when (a) every example's output is the same H×W, single-color, and
(b) that color is exactly the color whose non-bg cells span the largest
bounding-box area in the input. Ties break the detector (returns None).

Pipeline (opset 11):

  per-channel row presence  =  ReduceMax(input, axis=3)          (1, 10, 30, 1)
  per-channel col presence  =  ReduceMax(input, axis=2)          (1, 10, 1, 30)
  drop bg channel (Slice)                                        (1, 9, *, *)
  r_min/c_min = ArgMax(presence)
  r_max/c_max = ReduceMax(presence * arange)
  area        = (r_max - r_min + 1) * (c_max - c_min + 1)
  has_any     = ReduceMax(presence) over both spatial dims        (1, 9, 1, 1)
  area_masked = area * has_any
  winning_color = ArgMax(area_masked over channel) + 1            (1,)
  OneHot → broadcast against ones(H, W) → Pad to canvas
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from ..grids import CHANNELS, HEIGHT, WIDTH, all_examples

OPSET = 11
IR_VERSION = 8


def _detect(task: dict) -> tuple[int, int] | None:
    examples = list(all_examples(task))
    if not examples:
        return None
    shapes = {(len(e["output"]), len(e["output"][0])) for e in examples}
    if len(shapes) != 1:
        return None
    H, W = shapes.pop()
    if H == 0 or W == 0:
        return None
    for ex in examples:
        out_colors = {c for row in ex["output"] for c in row}
        if len(out_colors) != 1:
            return None
        out_color = next(iter(out_colors))
        inp = ex["input"]
        ih, iw = len(inp), len(inp[0])
        max_area = -1
        max_color = -1
        for C in range(1, 10):
            rs = [r for r in range(ih) if any(inp[r][c] == C for c in range(iw))]
            cs = [c for c in range(iw) if any(inp[r][c] == C for r in range(ih))]
            if not rs or not cs:
                continue
            area = (rs[-1] - rs[0] + 1) * (cs[-1] - cs[0] + 1)
            if area > max_area:
                max_area = area
                max_color = C
            elif area == max_area:
                max_color = -2  # tied → reject
        if max_color < 0 or max_color != out_color:
            return None
    return H, W


def _build(H: int, W: int) -> onnx.ModelProto:
    def i64(name: str, vals) -> onnx.TensorProto:
        return numpy_helper.from_array(np.array(vals, dtype=np.int64), name)

    def f32(name: str, vals) -> onnx.TensorProto:
        return numpy_helper.from_array(np.array(vals, dtype=np.float32), name)

    init = [
        i64("ch_start", [1]),
        i64("ch_end", [10]),
        i64("ch_axes_chan", [1]),
        f32("arange_h", np.arange(HEIGHT)),
        f32("arange_w", np.arange(WIDTH)),
        i64("shape4_h", [1, 1, HEIGHT, 1]),
        i64("shape4_w", [1, 1, 1, WIDTH]),
        i64("flat9", [1, 9]),
        i64("chan_shape", [1, 10, 1, 1]),
        i64("depth", [10]),
        f32("values", [0.0, 1.0]),
        i64("one_idx", [1]),
        f32("one_f", 1.0),
        f32("ones_spatial", np.ones((1, 1, H, W), dtype=np.float32)),
        i64("pads", [0, 0, 0, 0, 0, 0, HEIGHT - H, WIDTH - W]),
    ]

    value_info = [
        helper.make_tensor_value_info(
            "row_max_chan", TensorProto.FLOAT, [1, CHANNELS, HEIGHT, 1]),
        helper.make_tensor_value_info(
            "col_max_chan", TensorProto.FLOAT, [1, CHANNELS, 1, WIDTH]),
        helper.make_tensor_value_info(
            "nb_row", TensorProto.FLOAT, [1, 9, HEIGHT, 1]),
        helper.make_tensor_value_info(
            "nb_col", TensorProto.FLOAT, [1, 9, 1, WIDTH]),
        helper.make_tensor_value_info(
            "arange_h_4d", TensorProto.FLOAT, [1, 1, HEIGHT, 1]),
        helper.make_tensor_value_info(
            "arange_w_4d", TensorProto.FLOAT, [1, 1, 1, WIDTH]),
        helper.make_tensor_value_info(
            "row_pos", TensorProto.FLOAT, [1, 9, HEIGHT, 1]),
        helper.make_tensor_value_info(
            "col_pos", TensorProto.FLOAT, [1, 9, 1, WIDTH]),
        helper.make_tensor_value_info(
            "r_max_f", TensorProto.FLOAT, [1, 9, 1, 1]),
        helper.make_tensor_value_info(
            "c_max_f", TensorProto.FLOAT, [1, 9, 1, 1]),
        helper.make_tensor_value_info(
            "r_min_i", TensorProto.INT64, [1, 9, 1, 1]),
        helper.make_tensor_value_info(
            "c_min_i", TensorProto.INT64, [1, 9, 1, 1]),
        helper.make_tensor_value_info(
            "r_min_f", TensorProto.FLOAT, [1, 9, 1, 1]),
        helper.make_tensor_value_info(
            "c_min_f", TensorProto.FLOAT, [1, 9, 1, 1]),
        helper.make_tensor_value_info(
            "h_minus_1", TensorProto.FLOAT, [1, 9, 1, 1]),
        helper.make_tensor_value_info(
            "w_minus_1", TensorProto.FLOAT, [1, 9, 1, 1]),
        helper.make_tensor_value_info(
            "h_f", TensorProto.FLOAT, [1, 9, 1, 1]),
        helper.make_tensor_value_info(
            "w_f", TensorProto.FLOAT, [1, 9, 1, 1]),
        helper.make_tensor_value_info(
            "area_f", TensorProto.FLOAT, [1, 9, 1, 1]),
        helper.make_tensor_value_info(
            "has_any", TensorProto.FLOAT, [1, 9, 1, 1]),
        helper.make_tensor_value_info(
            "area_masked", TensorProto.FLOAT, [1, 9, 1, 1]),
        helper.make_tensor_value_info(
            "area_flat", TensorProto.FLOAT, [1, 9]),
        helper.make_tensor_value_info(
            "argmax_idx", TensorProto.INT64, [1]),
        helper.make_tensor_value_info(
            "color_1d", TensorProto.INT64, [1]),
        helper.make_tensor_value_info(
            "onehot10", TensorProto.FLOAT, [1, 10]),
        helper.make_tensor_value_info(
            "onehot_4d", TensorProto.FLOAT, [1, CHANNELS, 1, 1]),
        helper.make_tensor_value_info(
            "fill", TensorProto.FLOAT, [1, CHANNELS, H, W]),
    ]

    nodes = [
        helper.make_node(
            "ReduceMax", ["input"], ["row_max_chan"],
            axes=[3], keepdims=1, name="row_max"),
        helper.make_node(
            "ReduceMax", ["input"], ["col_max_chan"],
            axes=[2], keepdims=1, name="col_max"),
        helper.make_node(
            "Slice",
            ["row_max_chan", "ch_start", "ch_end", "ch_axes_chan"],
            ["nb_row"], name="drop_bg_row"),
        helper.make_node(
            "Slice",
            ["col_max_chan", "ch_start", "ch_end", "ch_axes_chan"],
            ["nb_col"], name="drop_bg_col"),
        # Position-weighted maxes for r_max / c_max.
        helper.make_node(
            "Reshape", ["arange_h", "shape4_h"], ["arange_h_4d"],
            name="resh_arange_h"),
        helper.make_node(
            "Reshape", ["arange_w", "shape4_w"], ["arange_w_4d"],
            name="resh_arange_w"),
        helper.make_node(
            "Mul", ["nb_row", "arange_h_4d"], ["row_pos"], name="row_pos_op"),
        helper.make_node(
            "Mul", ["nb_col", "arange_w_4d"], ["col_pos"], name="col_pos_op"),
        helper.make_node(
            "ReduceMax", ["row_pos"], ["r_max_f"],
            axes=[2], keepdims=1, name="r_max_op"),
        helper.make_node(
            "ReduceMax", ["col_pos"], ["c_max_f"],
            axes=[3], keepdims=1, name="c_max_op"),
        # ArgMax-based r_min / c_min.
        helper.make_node(
            "ArgMax", ["nb_row"], ["r_min_i"],
            axis=2, keepdims=1, name="r_min_op"),
        helper.make_node(
            "ArgMax", ["nb_col"], ["c_min_i"],
            axis=3, keepdims=1, name="c_min_op"),
        helper.make_node(
            "Cast", ["r_min_i"], ["r_min_f"], to=TensorProto.FLOAT,
            name="cast_r_min"),
        helper.make_node(
            "Cast", ["c_min_i"], ["c_min_f"], to=TensorProto.FLOAT,
            name="cast_c_min"),
        # h = r_max - r_min + 1; w = c_max - c_min + 1.
        helper.make_node(
            "Sub", ["r_max_f", "r_min_f"], ["h_minus_1"], name="h_sub"),
        helper.make_node(
            "Sub", ["c_max_f", "c_min_f"], ["w_minus_1"], name="w_sub"),
        helper.make_node(
            "Add", ["h_minus_1", "one_f"], ["h_f"], name="h_add"),
        helper.make_node(
            "Add", ["w_minus_1", "one_f"], ["w_f"], name="w_add"),
        helper.make_node(
            "Mul", ["h_f", "w_f"], ["area_f"], name="area_op"),
        # has_any: did this channel have any cell? Use either nb_row or nb_col.
        helper.make_node(
            "ReduceMax", ["nb_row"], ["has_any"],
            axes=[2, 3], keepdims=1, name="has_any_op"),
        helper.make_node(
            "Mul", ["area_f", "has_any"], ["area_masked"],
            name="area_mask"),
        # Flatten + ArgMax over channels.
        helper.make_node(
            "Reshape", ["area_masked", "flat9"], ["area_flat"],
            name="flatten_area"),
        helper.make_node(
            "ArgMax", ["area_flat"], ["argmax_idx"],
            axis=1, keepdims=0, name="argmax_chan"),
        helper.make_node(
            "Add", ["argmax_idx", "one_idx"], ["color_1d"],
            name="color_shift"),
        # OneHot → broadcast → Pad.
        helper.make_node(
            "OneHot", ["color_1d", "depth", "values"], ["onehot10"],
            axis=1, name="onehot_op"),
        helper.make_node(
            "Reshape", ["onehot10", "chan_shape"], ["onehot_4d"],
            name="reshape_chan"),
        helper.make_node(
            "Mul", ["onehot_4d", "ones_spatial"], ["fill"],
            name="broadcast_fill"),
        helper.make_node(
            "Pad", ["fill", "pads"], ["output"], mode="constant",
            name="pad_to_canvas"),
    ]

    inputs = [helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, CHANNELS, HEIGHT, WIDTH])]
    outputs = [helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, CHANNELS, HEIGHT, WIDTH])]
    graph = helper.make_graph(
        nodes, "largest_bbox_fill", inputs, outputs,
        initializer=init, value_info=value_info)
    return helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
        ir_version=IR_VERSION)


def solve_largest_bbox_fill(task: dict) -> Optional[onnx.ModelProto]:
    spec = _detect(task)
    if spec is None:
        return None
    return _build(*spec)
