"""Solver: output is the bounding box of "non-background" cells.

For a fixed background color `bg`, the output is `input` cropped to the
tightest rectangle that contains every cell whose color != bg, shifted to the
top-left of the 30x30 canvas. The bbox dimensions vary per example.

Pipeline (entirely static-shape, opset 11):

  1.  no_bg     = Slice(input, channels != bg) → ReduceSum across channel.
  2.  row_proj  = ReduceMax(no_bg, axis=col)      — per-row presence.
  3.  col_proj  = ReduceMax(no_bg, axis=row)      — per-col presence.
  4.  r_min     = ArgMax(row_proj)                — first non-bg row.
  5.  c_min     = ArgMax(col_proj)                — first non-bg col.
  6.  r_max     = ReduceMax(row_proj * arange)    — last non-bg row.
  7.  c_max     = ReduceMax(col_proj * arange).
  8.  h, w      = r_max - r_min + 1, c_max - c_min + 1.
  9.  shift idx = Clip(arange + r_min, 0, 29)     — translation indices.
 10.  shifted   = Gather(input, shift_idx_row) then Gather along col axis.
 11.  bbox_mask = (arange < h) ⊗ (arange < w)     — zero outside (h, w).
 12.  output    = shifted * bbox_mask.

Step 10 produces a (1, 10, 30, 30) tensor with the bbox content at the
top-left. Cells past the bbox boundary carry whatever input had there (often
the background color itself, which would otherwise leak as channel-`bg`
ones); the mask in step 12 zeros all 10 channels for those cells so the
output's `from_onehot` decode treats them as empty padding.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from ..grids import CHANNELS, HEIGHT, WIDTH, all_examples

OPSET = 11
IR_VERSION = 8


def _detect_bg(task: dict) -> int | None:
    """Return the bg color in {0..9} that makes output = bbox-of-non-bg, else None."""
    examples = list(all_examples(task))
    if not examples:
        return None
    for bg in range(CHANNELS):
        ok = True
        for ex in examples:
            i, o = ex["input"], ex["output"]
            if not i or not o:
                ok = False
                break
            H, W = len(i), len(i[0])
            rs = [r for r in range(H) if any(i[r][c] != bg for c in range(W))]
            cs = [c for c in range(W) if any(i[r][c] != bg for r in range(H))]
            if not rs or not cs:
                ok = False
                break
            r0, r1 = rs[0], rs[-1]
            c0, c1 = cs[0], cs[-1]
            h, w = r1 - r0 + 1, c1 - c0 + 1
            if len(o) != h or len(o[0]) != w:
                ok = False
                break
            for r in range(h):
                for c in range(w):
                    if o[r][c] != i[r0 + r][c0 + c]:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            return bg
    return None


def _build(bg: int) -> onnx.ModelProto:
    def i64(name: str, vals) -> onnx.TensorProto:
        return numpy_helper.from_array(np.array(vals, dtype=np.int64), name)

    def f32(name: str, vals) -> onnx.TensorProto:
        return numpy_helper.from_array(np.array(vals, dtype=np.float32), name)

    # Channel masking. To minimize intermediate memory, we sum across ALL
    # channels (cheap: (1, 1, 30, 30)) and then subtract just the bg
    # channel — also (1, 1, 30, 30). This avoids any (1, 9, 30, 30)
    # intermediate from a slice-and-concat.
    init: list[onnx.TensorProto] = []
    nodes: list = []
    value_info: list = []

    bg_start = i64("bg_start", [bg])
    bg_end = i64("bg_end", [bg + 1])
    bg_axes = i64("bg_axes", [1])
    init += [bg_start, bg_end, bg_axes]

    value_info += [
        helper.make_tensor_value_info(
            "all_chan_sum", TensorProto.FLOAT, [1, 1, HEIGHT, WIDTH]),
        helper.make_tensor_value_info(
            "bg_only", TensorProto.FLOAT, [1, 1, HEIGHT, WIDTH]),
    ]
    nodes += [
        helper.make_node(
            "ReduceSum", ["input"], ["all_chan_sum"],
            axes=[1], keepdims=1, name="sum_all_channels"),
        helper.make_node(
            "Slice", ["input", "bg_start", "bg_end", "bg_axes"],
            ["bg_only"], name="slice_bg"),
    ]
    non_bg_source = "non_bg_mask"

    # Build the rest of the graph -----------------------------------------
    init += [
        f32("arange_h", np.arange(HEIGHT)),
        f32("arange_w", np.arange(WIDTH)),
        i64("arange_h_shape", [1, 1, HEIGHT, 1]),
        i64("arange_w_shape", [1, 1, 1, WIDTH]),
        i64("h_idx_shape", [HEIGHT]),
        i64("w_idx_shape", [WIDTH]),
        f32("clip_min", 0.0),
        f32("clip_max", float(HEIGHT - 1)),
        f32("one_f", 1.0),
    ]

    value_info += [
        helper.make_tensor_value_info(
            "non_bg_mask", TensorProto.FLOAT, [1, 1, HEIGHT, WIDTH]),
        helper.make_tensor_value_info(
            "row_proj", TensorProto.FLOAT, [1, 1, HEIGHT, 1]),
        helper.make_tensor_value_info(
            "col_proj", TensorProto.FLOAT, [1, 1, 1, WIDTH]),
        helper.make_tensor_value_info(
            "arange_h_4d", TensorProto.FLOAT, [1, 1, HEIGHT, 1]),
        helper.make_tensor_value_info(
            "arange_w_4d", TensorProto.FLOAT, [1, 1, 1, WIDTH]),
        helper.make_tensor_value_info(
            "row_pos", TensorProto.FLOAT, [1, 1, HEIGHT, 1]),
        helper.make_tensor_value_info(
            "col_pos", TensorProto.FLOAT, [1, 1, 1, WIDTH]),
        helper.make_tensor_value_info(
            "r_max_f", TensorProto.FLOAT, [1, 1, 1, 1]),
        helper.make_tensor_value_info(
            "c_max_f", TensorProto.FLOAT, [1, 1, 1, 1]),
        helper.make_tensor_value_info(
            "r_min_i", TensorProto.INT64, [1, 1, 1, 1]),
        helper.make_tensor_value_info(
            "c_min_i", TensorProto.INT64, [1, 1, 1, 1]),
        helper.make_tensor_value_info(
            "r_min_f", TensorProto.FLOAT, [1, 1, 1, 1]),
        helper.make_tensor_value_info(
            "c_min_f", TensorProto.FLOAT, [1, 1, 1, 1]),
        helper.make_tensor_value_info(
            "h_minus_1", TensorProto.FLOAT, [1, 1, 1, 1]),
        helper.make_tensor_value_info(
            "w_minus_1", TensorProto.FLOAT, [1, 1, 1, 1]),
        helper.make_tensor_value_info(
            "h_f", TensorProto.FLOAT, [1, 1, 1, 1]),
        helper.make_tensor_value_info(
            "w_f", TensorProto.FLOAT, [1, 1, 1, 1]),
        helper.make_tensor_value_info(
            "shift_row_f", TensorProto.FLOAT, [1, 1, HEIGHT, 1]),
        helper.make_tensor_value_info(
            "shift_row_clip", TensorProto.FLOAT, [1, 1, HEIGHT, 1]),
        helper.make_tensor_value_info(
            "shift_row_i", TensorProto.INT64, [1, 1, HEIGHT, 1]),
        helper.make_tensor_value_info(
            "shift_row_1d", TensorProto.INT64, [HEIGHT]),
        helper.make_tensor_value_info(
            "shift_col_f", TensorProto.FLOAT, [1, 1, 1, WIDTH]),
        helper.make_tensor_value_info(
            "shift_col_clip", TensorProto.FLOAT, [1, 1, 1, WIDTH]),
        helper.make_tensor_value_info(
            "shift_col_i", TensorProto.INT64, [1, 1, 1, WIDTH]),
        helper.make_tensor_value_info(
            "shift_col_1d", TensorProto.INT64, [WIDTH]),
        helper.make_tensor_value_info(
            "shifted_rows", TensorProto.FLOAT,
            [1, CHANNELS, HEIGHT, WIDTH]),
        helper.make_tensor_value_info(
            "shifted", TensorProto.FLOAT,
            [1, CHANNELS, HEIGHT, WIDTH]),
        helper.make_tensor_value_info(
            "h_le", TensorProto.BOOL, [1, 1, HEIGHT, 1]),
        helper.make_tensor_value_info(
            "w_le", TensorProto.BOOL, [1, 1, 1, WIDTH]),
        helper.make_tensor_value_info(
            "h_le_f", TensorProto.FLOAT, [1, 1, HEIGHT, 1]),
        helper.make_tensor_value_info(
            "w_le_f", TensorProto.FLOAT, [1, 1, 1, WIDTH]),
        helper.make_tensor_value_info(
            "bbox_mask", TensorProto.FLOAT, [1, 1, HEIGHT, WIDTH]),
    ]

    nodes += [
        # 1. non-bg presence = sum-of-all-channels - bg channel.
        helper.make_node(
            "Sub", ["all_chan_sum", "bg_only"], ["non_bg_mask"],
            name="non_bg_mask_op"),
        # 2. Row / col presence projections (still float).
        helper.make_node(
            "ReduceMax", ["non_bg_mask"], ["row_proj"],
            axes=[3], keepdims=1, name="row_proj_op"),
        helper.make_node(
            "ReduceMax", ["non_bg_mask"], ["col_proj"],
            axes=[2], keepdims=1, name="col_proj_op"),
        # 3. Reshape arange to broadcast.
        helper.make_node(
            "Reshape", ["arange_h", "arange_h_shape"], ["arange_h_4d"],
            name="resh_arange_h"),
        helper.make_node(
            "Reshape", ["arange_w", "arange_w_shape"], ["arange_w_4d"],
            name="resh_arange_w"),
        # 4. Position = projection * arange.
        helper.make_node(
            "Mul", ["row_proj", "arange_h_4d"], ["row_pos"], name="mul_row"),
        helper.make_node(
            "Mul", ["col_proj", "arange_w_4d"], ["col_pos"], name="mul_col"),
        # 5. r_max / c_max as floats.
        helper.make_node(
            "ReduceMax", ["row_pos"], ["r_max_f"], axes=[2], keepdims=1,
            name="r_max"),
        helper.make_node(
            "ReduceMax", ["col_pos"], ["c_max_f"], axes=[3], keepdims=1,
            name="c_max"),
        # 6. r_min / c_min via ArgMax + Cast to float.
        helper.make_node(
            "ArgMax", ["row_proj"], ["r_min_i"], axis=2, keepdims=1,
            name="r_min"),
        helper.make_node(
            "ArgMax", ["col_proj"], ["c_min_i"], axis=3, keepdims=1,
            name="c_min"),
        helper.make_node(
            "Cast", ["r_min_i"], ["r_min_f"], to=TensorProto.FLOAT,
            name="cast_r_min"),
        helper.make_node(
            "Cast", ["c_min_i"], ["c_min_f"], to=TensorProto.FLOAT,
            name="cast_c_min"),
        # 7. h-1 = r_max - r_min (so cells with arange <= h-1 are inside).
        helper.make_node(
            "Sub", ["r_max_f", "r_min_f"], ["h_minus_1"], name="h_sub"),
        helper.make_node(
            "Sub", ["c_max_f", "c_min_f"], ["w_minus_1"], name="w_sub"),
        # 8. Shift indices: arange + r_min (resp. c_min), clipped to [0, 29].
        helper.make_node(
            "Add", ["arange_h_4d", "r_min_f"], ["shift_row_f"],
            name="shift_row_add"),
        helper.make_node(
            "Clip", ["shift_row_f", "clip_min", "clip_max"],
            ["shift_row_clip"], name="shift_row_clip_op"),
        helper.make_node(
            "Cast", ["shift_row_clip"], ["shift_row_i"],
            to=TensorProto.INT64, name="shift_row_cast"),
        helper.make_node(
            "Reshape", ["shift_row_i", "h_idx_shape"], ["shift_row_1d"],
            name="shift_row_resh"),
        helper.make_node(
            "Add", ["arange_w_4d", "c_min_f"], ["shift_col_f"],
            name="shift_col_add"),
        helper.make_node(
            "Clip", ["shift_col_f", "clip_min", "clip_max"],
            ["shift_col_clip"], name="shift_col_clip_op"),
        helper.make_node(
            "Cast", ["shift_col_clip"], ["shift_col_i"],
            to=TensorProto.INT64, name="shift_col_cast"),
        helper.make_node(
            "Reshape", ["shift_col_i", "w_idx_shape"], ["shift_col_1d"],
            name="shift_col_resh"),
        # 9. Two-axis Gather.
        helper.make_node(
            "Gather", ["input", "shift_row_1d"], ["shifted_rows"],
            axis=2, name="gather_rows"),
        helper.make_node(
            "Gather", ["shifted_rows", "shift_col_1d"], ["shifted"],
            axis=3, name="gather_cols"),
        # 10. Build the bbox mask: arange < h and arange < w. Opset 11 has no
        #     LessOrEqual, so use Less(arange, h) with h = (h-1) + 1.
        helper.make_node(
            "Add", ["h_minus_1", "one_f"], ["h_f"], name="h_add_one"),
        helper.make_node(
            "Add", ["w_minus_1", "one_f"], ["w_f"], name="w_add_one"),
        helper.make_node(
            "Less", ["arange_h_4d", "h_f"], ["h_le"], name="h_le_op"),
        helper.make_node(
            "Less", ["arange_w_4d", "w_f"], ["w_le"], name="w_le_op"),
        helper.make_node(
            "Cast", ["h_le"], ["h_le_f"], to=TensorProto.FLOAT,
            name="h_le_cast"),
        helper.make_node(
            "Cast", ["w_le"], ["w_le_f"], to=TensorProto.FLOAT,
            name="w_le_cast"),
        helper.make_node(
            "Mul", ["h_le_f", "w_le_f"], ["bbox_mask"], name="bbox_mask_op"),
        # 11. Mask the shifted content.
        helper.make_node(
            "Mul", ["shifted", "bbox_mask"], ["output"], name="apply_mask"),
    ]

    inputs = [helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, CHANNELS, HEIGHT, WIDTH])]
    outputs = [helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, CHANNELS, HEIGHT, WIDTH])]
    graph = helper.make_graph(
        nodes, "bbox_strip", inputs, outputs,
        initializer=init, value_info=value_info)
    return helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
        ir_version=IR_VERSION)


def solve_bbox_strip(task: dict) -> Optional[onnx.ModelProto]:
    bg = _detect_bg(task)
    if bg is None:
        return None
    return _build(bg)
