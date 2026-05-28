"""Solver: horizontal-tile by an integer factor with variable input width.

Detects tasks where `output[r, c] = input[r, c mod W]` for c < N*W and zero
beyond. We support N in {2, 3}. The runtime steps:

  1. Find W via row-summed content mask + ReduceMax(mask * arange) + 1.
  2. Compute mod indices `arange mod W` (int64).
  3. Gather along the column axis to lay down the repeating pattern.
  4. Build a boolean mask `arange < N*W`, broadcast over (1, 10, 30, 30),
     multiply to zero out cells past the tiled region.

The output ends up Hx(N*W) content sitting at the top-left of a 30x30 canvas;
rows past H are already zero because the input one-hot is zero there.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from ..grids import CHANNELS, HEIGHT, WIDTH, all_examples

OPSET = 11
IR_VERSION = 8


def _detect(task: dict) -> int | None:
    """Return N in {2, 3} if the task is horizontal tile_N; else None."""
    examples = list(all_examples(task))
    if not examples:
        return None
    for N in (2, 3):
        ok = True
        for ex in examples:
            i, o = ex["input"], ex["output"]
            if not i or not o:
                ok = False
                break
            ih, iw = len(i), len(i[0])
            if len(o) != ih or len(o[0]) != N * iw:
                ok = False
                break
            if N * iw > WIDTH:
                ok = False
                break
            for r in range(ih):
                for c in range(N * iw):
                    if o[r][c] != i[r][c % iw]:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            return N
    return None


def _build(N: int) -> onnx.ModelProto:
    def i64(name: str, vals) -> onnx.TensorProto:
        return numpy_helper.from_array(np.array(vals, dtype=np.int64), name)

    def f32(name: str, vals) -> onnx.TensorProto:
        return numpy_helper.from_array(np.array(vals, dtype=np.float32), name)

    # Initializers --------------------------------------------------------
    init = [
        i64("arange_w_i64", np.arange(WIDTH)),
        f32("arange_w_f32", np.arange(WIDTH)),
        i64("arange_w_shape4", [1, 1, 1, WIDTH]),
        i64("flat_shape", [WIDTH]),
        f32("one_f", 1.0),
        f32("N_f", float(N)),
    ]

    value_info = [
        helper.make_tensor_value_info(
            "chan_sum", TensorProto.FLOAT, [1, 1, HEIGHT, WIDTH]),
        helper.make_tensor_value_info(
            "col_has", TensorProto.FLOAT, [1, 1, 1, WIDTH]),
        helper.make_tensor_value_info(
            "arange_w_4d_f32", TensorProto.FLOAT, [1, 1, 1, WIDTH]),
        helper.make_tensor_value_info(
            "arange_w_4d_i64", TensorProto.INT64, [1, 1, 1, WIDTH]),
        helper.make_tensor_value_info(
            "col_pos", TensorProto.FLOAT, [1, 1, 1, WIDTH]),
        helper.make_tensor_value_info(
            "w_minus_1_f", TensorProto.FLOAT, [1, 1, 1, 1]),
        helper.make_tensor_value_info(
            "w_f", TensorProto.FLOAT, [1, 1, 1, 1]),
        helper.make_tensor_value_info(
            "w_i64", TensorProto.INT64, [1, 1, 1, 1]),
        helper.make_tensor_value_info(
            "mod_idx_4d", TensorProto.INT64, [1, 1, 1, WIDTH]),
        helper.make_tensor_value_info(
            "mod_idx_1d", TensorProto.INT64, [WIDTH]),
        helper.make_tensor_value_info(
            "gathered", TensorProto.FLOAT,
            [1, CHANNELS, HEIGHT, WIDTH]),
        helper.make_tensor_value_info(
            "nw_f", TensorProto.FLOAT, [1, 1, 1, 1]),
        helper.make_tensor_value_info(
            "in_tile_bool", TensorProto.BOOL, [1, 1, 1, WIDTH]),
        helper.make_tensor_value_info(
            "in_tile_f", TensorProto.FLOAT, [1, 1, 1, WIDTH]),
    ]

    nodes = [
        # 1. Channel sum → "any color in this cell?" mask.
        helper.make_node(
            "ReduceSum", ["input"], ["chan_sum"],
            axes=[1], keepdims=1, name="chan_sum_op"),
        # 2. Project to columns: 1 if any row of that column has content.
        helper.make_node(
            "ReduceMax", ["chan_sum"], ["col_has"],
            axes=[2], keepdims=1, name="col_has_op"),
        # 3. Reshape arange to broadcast against col_has.
        helper.make_node(
            "Reshape", ["arange_w_f32", "arange_w_shape4"],
            ["arange_w_4d_f32"], name="arange_resh_f32"),
        helper.make_node(
            "Reshape", ["arange_w_i64", "arange_w_shape4"],
            ["arange_w_4d_i64"], name="arange_resh_i64"),
        # 4. Position-weighted mask → ReduceMax = last occupied index = W - 1.
        helper.make_node(
            "Mul", ["col_has", "arange_w_4d_f32"], ["col_pos"],
            name="col_pos_op"),
        helper.make_node(
            "ReduceMax", ["col_pos"], ["w_minus_1_f"],
            axes=[3], keepdims=1, name="w_minus_1_op"),
        helper.make_node(
            "Add", ["w_minus_1_f", "one_f"], ["w_f"], name="w_add_one"),
        helper.make_node(
            "Cast", ["w_f"], ["w_i64"], to=TensorProto.INT64,
            name="cast_w_i64"),
        # 5. c_mod = arange % W (int64, broadcasts (1,30) and (1,1,1,1)).
        helper.make_node(
            "Mod", ["arange_w_4d_i64", "w_i64"], ["mod_idx_4d"],
            name="mod_op"),
        helper.make_node(
            "Reshape", ["mod_idx_4d", "flat_shape"], ["mod_idx_1d"],
            name="mod_flatten"),
        # 6. Tile by gathering input columns.
        helper.make_node(
            "Gather", ["input", "mod_idx_1d"], ["gathered"],
            axis=3, name="gather_cols"),
        # 7. Mask cells past column index N*W.
        helper.make_node(
            "Mul", ["w_f", "N_f"], ["nw_f"], name="nw_op"),
        helper.make_node(
            "Less", ["arange_w_4d_f32", "nw_f"], ["in_tile_bool"],
            name="in_tile_op"),
        helper.make_node(
            "Cast", ["in_tile_bool"], ["in_tile_f"], to=TensorProto.FLOAT,
            name="in_tile_cast"),
        helper.make_node(
            "Mul", ["gathered", "in_tile_f"], ["output"],
            name="apply_tile_mask"),
    ]

    inputs = [helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, CHANNELS, HEIGHT, WIDTH])]
    outputs = [helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, CHANNELS, HEIGHT, WIDTH])]
    graph = helper.make_graph(
        nodes, "tile_h", inputs, outputs,
        initializer=init, value_info=value_info)
    return helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
        ir_version=IR_VERSION)


def solve_tile_h(task: dict) -> Optional[onnx.ModelProto]:
    N = _detect(task)
    if N is None:
        return None
    return _build(N)
