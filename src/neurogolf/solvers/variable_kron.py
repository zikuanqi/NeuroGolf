"""Solver: kron-scale where the scale factor varies per example.

Some ARC tasks scale a fixed-shape (H_in × W_in) input by an integer factor N
where N is a function of the input — for instance the count of non-zero
cells, or the count of distinct non-zero colors. The output is
`(N·H_in × N·W_in)` with `output[r, c] = input[r // N, c // N]`.

To keep the canvas static at 30×30 we compute the gather indices for the
worst case (30 positions) and zero out cells past row/col N·H_in with a Less
mask. Indices that overshoot `H_in - 1` are clamped via Min before Gather so
we don't hit out-of-bounds.

We try two rules for deriving N:

  * `cells`     — sum over channels 1..9 of ReduceSum(input).
  * `distinct`  — sum over channels 1..9 of (channel-sum > 0).

The detector tries them in order and keeps the first that fits every example.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from ..grids import CHANNELS, HEIGHT, WIDTH, all_examples

OPSET = 11
IR_VERSION = 8


def _detect(task: dict) -> tuple[int, int, str] | None:
    """Return (H_in, W_in, rule) if every example fits one of the two rules."""
    examples = list(all_examples(task))
    if not examples:
        return None
    shapes_in = {(len(e["input"]), len(e["input"][0])) for e in examples}
    if len(shapes_in) != 1:
        return None
    H_in, W_in = shapes_in.pop()
    if H_in == 0 or W_in == 0:
        return None

    def n_cells(inp):
        return sum(1 for row in inp for c in row if c != 0)

    def n_distinct(inp):
        return len({c for row in inp for c in row if c != 0})

    for rule_name, rule_fn in (("cells", n_cells), ("distinct", n_distinct)):
        ok = True
        for ex in examples:
            inp, out = ex["input"], ex["output"]
            N = rule_fn(inp)
            if N <= 0:
                ok = False
                break
            if N * H_in > HEIGHT or N * W_in > WIDTH:
                ok = False
                break
            if len(out) != N * H_in or len(out[0]) != N * W_in:
                ok = False
                break
            for r in range(N * H_in):
                row_src = inp[r // N]
                row_out = out[r]
                for c in range(N * W_in):
                    if row_out[c] != row_src[c // N]:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            return H_in, W_in, rule_name
    return None


def _build(H_in: int, W_in: int, rule: str) -> onnx.ModelProto:
    def i64(name: str, vals) -> onnx.TensorProto:
        return numpy_helper.from_array(np.array(vals, dtype=np.int64), name)

    def f32(name: str, vals) -> onnx.TensorProto:
        return numpy_helper.from_array(np.array(vals, dtype=np.float32), name)

    init = [
        # Compute N --------------------------------------------------------
        i64("ch_start", [1]),
        i64("ch_end", [10]),
        i64("ch_axes", [1]),
        # arange constants for row + col index computation.
        i64("arange_h", np.arange(HEIGHT)),
        i64("arange_w", np.arange(WIDTH)),
        f32("arange_h_f", np.arange(HEIGHT)),
        f32("arange_w_f", np.arange(WIDTH)),
        i64("shape4_h", [1, 1, HEIGHT, 1]),
        i64("shape4_w", [1, 1, 1, WIDTH]),
        i64("flat_h", [HEIGHT]),
        i64("flat_w", [WIDTH]),
        # Clamping the gather index to [0, H_in-1] / [0, W_in-1]. We need a
        # 4-D shape that broadcasts against (1, 1, HEIGHT, 1) and similar.
        f32("h_max_f", [[[[float(H_in - 1)]]]]),
        f32("w_max_f", [[[[float(W_in - 1)]]]]),
        # Multiplier H_in, W_in for the boundary mask N*H_in, N*W_in.
        f32("h_in_f", float(H_in)),
        f32("w_in_f", float(W_in)),
        # Zero constant for the "is non-zero" comparison.
        f32("zero_f", 0.0),
    ]

    value_info = [
        helper.make_tensor_value_info(
            "counts_all", TensorProto.FLOAT, [1, CHANNELS, 1, 1]),
        helper.make_tensor_value_info(
            "counts_nobg", TensorProto.FLOAT, [1, 9, 1, 1]),
        helper.make_tensor_value_info(
            "n_f", TensorProto.FLOAT, [1, 1, 1, 1]),
        helper.make_tensor_value_info(
            "n_i", TensorProto.INT64, [1, 1, 1, 1]),
        helper.make_tensor_value_info(
            "arange_h_4d", TensorProto.INT64, [1, 1, HEIGHT, 1]),
        helper.make_tensor_value_info(
            "arange_w_4d", TensorProto.INT64, [1, 1, 1, WIDTH]),
        helper.make_tensor_value_info(
            "div_h_4d", TensorProto.INT64, [1, 1, HEIGHT, 1]),
        helper.make_tensor_value_info(
            "div_w_4d", TensorProto.INT64, [1, 1, 1, WIDTH]),
        helper.make_tensor_value_info(
            "div_h_4d_f", TensorProto.FLOAT, [1, 1, HEIGHT, 1]),
        helper.make_tensor_value_info(
            "div_w_4d_f", TensorProto.FLOAT, [1, 1, 1, WIDTH]),
        helper.make_tensor_value_info(
            "clip_h_4d_f", TensorProto.FLOAT, [1, 1, HEIGHT, 1]),
        helper.make_tensor_value_info(
            "clip_w_4d_f", TensorProto.FLOAT, [1, 1, 1, WIDTH]),
        helper.make_tensor_value_info(
            "clip_h_4d", TensorProto.INT64, [1, 1, HEIGHT, 1]),
        helper.make_tensor_value_info(
            "clip_w_4d", TensorProto.INT64, [1, 1, 1, WIDTH]),
        helper.make_tensor_value_info(
            "idx_h_1d", TensorProto.INT64, [HEIGHT]),
        helper.make_tensor_value_info(
            "idx_w_1d", TensorProto.INT64, [WIDTH]),
        helper.make_tensor_value_info(
            "gather_rows", TensorProto.FLOAT,
            [1, CHANNELS, HEIGHT, WIDTH]),
        helper.make_tensor_value_info(
            "gather_full", TensorProto.FLOAT,
            [1, CHANNELS, HEIGHT, WIDTH]),
        helper.make_tensor_value_info(
            "arange_h_4d_f", TensorProto.FLOAT, [1, 1, HEIGHT, 1]),
        helper.make_tensor_value_info(
            "arange_w_4d_f", TensorProto.FLOAT, [1, 1, 1, WIDTH]),
        helper.make_tensor_value_info(
            "nh_f", TensorProto.FLOAT, [1, 1, 1, 1]),
        helper.make_tensor_value_info(
            "nw_f", TensorProto.FLOAT, [1, 1, 1, 1]),
        helper.make_tensor_value_info(
            "in_h_b", TensorProto.BOOL, [1, 1, HEIGHT, 1]),
        helper.make_tensor_value_info(
            "in_w_b", TensorProto.BOOL, [1, 1, 1, WIDTH]),
        helper.make_tensor_value_info(
            "in_h_f", TensorProto.FLOAT, [1, 1, HEIGHT, 1]),
        helper.make_tensor_value_info(
            "in_w_f", TensorProto.FLOAT, [1, 1, 1, WIDTH]),
        helper.make_tensor_value_info(
            "mask_f", TensorProto.FLOAT, [1, 1, HEIGHT, WIDTH]),
    ]

    nodes = [
        # 1. Compute per-channel cell counts, drop bg.
        helper.make_node(
            "ReduceSum", ["input"], ["counts_all"],
            axes=[2, 3], keepdims=1, name="reduce_spatial"),
        helper.make_node(
            "Slice", ["counts_all", "ch_start", "ch_end", "ch_axes"],
            ["counts_nobg"], name="drop_bg"),
    ]

    if rule == "cells":
        # N = sum over channels 1..9 of channel counts.
        nodes.append(helper.make_node(
            "ReduceSum", ["counts_nobg"], ["n_f"],
            axes=[1, 2, 3], keepdims=1, name="sum_nobg"))
    else:  # 'distinct'
        # N = sum over channels 1..9 of (count > 0).
        value_info.append(helper.make_tensor_value_info(
            "any_b", TensorProto.BOOL, [1, 9, 1, 1]))
        value_info.append(helper.make_tensor_value_info(
            "any_f", TensorProto.FLOAT, [1, 9, 1, 1]))
        nodes += [
            helper.make_node(
                "Greater", ["counts_nobg", "zero_f"], ["any_b"],
                name="any_op"),
            helper.make_node(
                "Cast", ["any_b"], ["any_f"], to=TensorProto.FLOAT,
                name="any_cast"),
            helper.make_node(
                "ReduceSum", ["any_f"], ["n_f"],
                axes=[1, 2, 3], keepdims=1, name="sum_distinct"),
        ]

    nodes += [
        helper.make_node(
            "Cast", ["n_f"], ["n_i"], to=TensorProto.INT64, name="n_to_i"),
        # 2. Build row / col gather indices: clip(arange // N, 0, H_in - 1).
        helper.make_node(
            "Reshape", ["arange_h", "shape4_h"], ["arange_h_4d"],
            name="reshape_arange_h"),
        helper.make_node(
            "Reshape", ["arange_w", "shape4_w"], ["arange_w_4d"],
            name="reshape_arange_w"),
        helper.make_node(
            "Div", ["arange_h_4d", "n_i"], ["div_h_4d"], name="div_h"),
        helper.make_node(
            "Div", ["arange_w_4d", "n_i"], ["div_w_4d"], name="div_w"),
        # Opset 11 Min doesn't accept int64. Cast through float for the clip.
        helper.make_node(
            "Cast", ["div_h_4d"], ["div_h_4d_f"], to=TensorProto.FLOAT,
            name="div_h_to_f"),
        helper.make_node(
            "Cast", ["div_w_4d"], ["div_w_4d_f"], to=TensorProto.FLOAT,
            name="div_w_to_f"),
        helper.make_node(
            "Min", ["div_h_4d_f", "h_max_f"], ["clip_h_4d_f"],
            name="clip_h"),
        helper.make_node(
            "Min", ["div_w_4d_f", "w_max_f"], ["clip_w_4d_f"],
            name="clip_w"),
        helper.make_node(
            "Cast", ["clip_h_4d_f"], ["clip_h_4d"], to=TensorProto.INT64,
            name="clip_h_to_i"),
        helper.make_node(
            "Cast", ["clip_w_4d_f"], ["clip_w_4d"], to=TensorProto.INT64,
            name="clip_w_to_i"),
        helper.make_node(
            "Reshape", ["clip_h_4d", "flat_h"], ["idx_h_1d"],
            name="flat_h_idx"),
        helper.make_node(
            "Reshape", ["clip_w_4d", "flat_w"], ["idx_w_1d"],
            name="flat_w_idx"),
        # 3. Two-axis Gather across rows then columns.
        helper.make_node(
            "Gather", ["input", "idx_h_1d"], ["gather_rows"],
            axis=2, name="gather_h"),
        helper.make_node(
            "Gather", ["gather_rows", "idx_w_1d"], ["gather_full"],
            axis=3, name="gather_w"),
        # 4. Mask cells with r >= N*H_in or c >= N*W_in.
        helper.make_node(
            "Reshape", ["arange_h_f", "shape4_h"], ["arange_h_4d_f"],
            name="resh_arange_h_f"),
        helper.make_node(
            "Reshape", ["arange_w_f", "shape4_w"], ["arange_w_4d_f"],
            name="resh_arange_w_f"),
        helper.make_node(
            "Mul", ["n_f", "h_in_f"], ["nh_f"], name="nh"),
        helper.make_node(
            "Mul", ["n_f", "w_in_f"], ["nw_f"], name="nw"),
        helper.make_node(
            "Less", ["arange_h_4d_f", "nh_f"], ["in_h_b"], name="in_h_op"),
        helper.make_node(
            "Less", ["arange_w_4d_f", "nw_f"], ["in_w_b"], name="in_w_op"),
        helper.make_node(
            "Cast", ["in_h_b"], ["in_h_f"], to=TensorProto.FLOAT,
            name="in_h_cast"),
        helper.make_node(
            "Cast", ["in_w_b"], ["in_w_f"], to=TensorProto.FLOAT,
            name="in_w_cast"),
        helper.make_node(
            "Mul", ["in_h_f", "in_w_f"], ["mask_f"], name="mask_op"),
        helper.make_node(
            "Mul", ["gather_full", "mask_f"], ["output"],
            name="apply_mask"),
    ]

    inputs = [helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, CHANNELS, HEIGHT, WIDTH])]
    outputs = [helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, CHANNELS, HEIGHT, WIDTH])]
    graph = helper.make_graph(
        nodes, "variable_kron", inputs, outputs,
        initializer=init, value_info=value_info)
    return helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
        ir_version=IR_VERSION)


def solve_variable_kron(task: dict) -> Optional[onnx.ModelProto]:
    spec = _detect(task)
    if spec is None:
        return None
    return _build(*spec)
