"""Solver: output is the input cropped to the bounding box of cells of a
selected color C.

The color C is chosen at runtime by one of two rules:

  * `majority` — C is the non-bg color with the most cells in input. The
    detector requires a strict (non-tied) winner on every example.
  * `rarest`   — C is the non-bg color with the fewest (but non-zero) cells.
    Same strict-winner requirement.

The output content is the *actual* bbox content of the input (preserving the
other colors that happen to lie within the bbox), shifted to the top-left
of the 30x30 canvas with zero padding past the bbox extent.

Pipeline (opset 11):

  counts_all  = ReduceSum(input, axes=[2, 3])               (1, 10, 1, 1)
  counts_nobg = Slice channels 1..10                         (1, 9, 1, 1)
  flat        = Reshape → (1, 9)
  target_idx  = ArgMax(flat) or ArgMin(Where(has_any, flat, BIG))
  color       = target_idx + 1                               (1,)
  chan_mask   = OneHot(color, depth=10, axis=1) reshaped     (1, 10, 1, 1)
  target_mask = ReduceSum(input * chan_mask, axes=[1])       (1, 1, 30, 30)
  r_min/c_min = ArgMax(row/col projections)
  r_max/c_max = ReduceMax(projection * arange)
  shifted     = Gather(input, arange + r_min) ×2             (1, 10, 30, 30)
  in_bbox     = (arange < h) * (arange < w)
  output      = shifted * in_bbox
"""
from __future__ import annotations

from collections import Counter
from typing import Optional

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from ..grids import CHANNELS, HEIGHT, WIDTH, all_examples

OPSET = 11
IR_VERSION = 8


def _detect(task: dict) -> str | None:
    """Return 'majority' or 'rarest' if every example fits the rule."""
    examples = list(all_examples(task))
    if not examples:
        return None
    for rule in ("majority", "rarest"):
        ok = True
        for ex in examples:
            inp, out = ex["input"], ex["output"]
            cnt = Counter()
            for row in inp:
                for c in row:
                    if c != 0:
                        cnt[c] += 1
            if not cnt:
                ok = False
                break
            if rule == "majority":
                ranked = sorted(cnt.items(), key=lambda x: (-x[1], x[0]))
            else:
                ranked = sorted(cnt.items(), key=lambda x: (x[1], x[0]))
            if len(ranked) >= 2 and ranked[0][1] == ranked[1][1]:
                ok = False
                break
            target = ranked[0][0]
            H, W = len(inp), len(inp[0])
            rs = [r for r in range(H) if any(inp[r][c] == target for c in range(W))]
            cs = [c for c in range(W) if any(inp[r][c] == target for r in range(H))]
            if not rs or not cs:
                ok = False
                break
            r0, r1, c0, c1 = rs[0], rs[-1], cs[0], cs[-1]
            h, w = r1 - r0 + 1, c1 - c0 + 1
            if len(out) != h or len(out[0]) != w:
                ok = False
                break
            for r in range(h):
                for c in range(w):
                    if out[r][c] != inp[r0 + r][c0 + c]:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            return rule
    return None


def _build(rule: str) -> onnx.ModelProto:
    def i64(name: str, vals) -> onnx.TensorProto:
        return numpy_helper.from_array(np.array(vals, dtype=np.int64), name)

    def f32(name: str, vals) -> onnx.TensorProto:
        return numpy_helper.from_array(np.array(vals, dtype=np.float32), name)

    init = [
        i64("ch_start", [1]),
        i64("ch_end", [10]),
        i64("ch_axes", [1]),
        i64("flat9_shape", [1, 9]),
        i64("chan_shape", [1, CHANNELS, 1, 1]),
        i64("depth", [10]),
        f32("values", [0.0, 1.0]),
        i64("one_idx", [1]),
        f32("one_f", 1.0),
        f32("zero_f", 0.0),
        f32("big_f", 1e6),
        # arange for shift index computation and bbox masking.
        i64("arange_h", np.arange(HEIGHT)),
        i64("arange_w", np.arange(WIDTH)),
        f32("arange_h_f", np.arange(HEIGHT)),
        f32("arange_w_f", np.arange(WIDTH)),
        i64("shape4_h", [1, 1, HEIGHT, 1]),
        i64("shape4_w", [1, 1, 1, WIDTH]),
        i64("flat_h", [HEIGHT]),
        i64("flat_w", [WIDTH]),
        i64("height_i", [HEIGHT]),
        i64("width_i", [WIDTH]),
    ]

    value_info = [
        helper.make_tensor_value_info(
            "counts_all", TensorProto.FLOAT, [1, CHANNELS, 1, 1]),
        helper.make_tensor_value_info(
            "counts_nobg", TensorProto.FLOAT, [1, 9, 1, 1]),
        helper.make_tensor_value_info(
            "flat", TensorProto.FLOAT, [1, 9]),
        helper.make_tensor_value_info(
            "target_idx", TensorProto.INT64, [1]),
        helper.make_tensor_value_info(
            "color_1d", TensorProto.INT64, [1]),
        helper.make_tensor_value_info(
            "onehot10", TensorProto.FLOAT, [1, 10]),
        helper.make_tensor_value_info(
            "chan_mask", TensorProto.FLOAT, [1, CHANNELS, 1, 1]),
        helper.make_tensor_value_info(
            "target_cells", TensorProto.FLOAT,
            [1, CHANNELS, HEIGHT, WIDTH]),
        helper.make_tensor_value_info(
            "target_mask", TensorProto.FLOAT, [1, 1, HEIGHT, WIDTH]),
        helper.make_tensor_value_info(
            "row_proj", TensorProto.FLOAT, [1, 1, HEIGHT, 1]),
        helper.make_tensor_value_info(
            "col_proj", TensorProto.FLOAT, [1, 1, 1, WIDTH]),
        helper.make_tensor_value_info(
            "arange_h_4d_f", TensorProto.FLOAT, [1, 1, HEIGHT, 1]),
        helper.make_tensor_value_info(
            "arange_w_4d_f", TensorProto.FLOAT, [1, 1, 1, WIDTH]),
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
            "h_m1", TensorProto.FLOAT, [1, 1, 1, 1]),
        helper.make_tensor_value_info(
            "w_m1", TensorProto.FLOAT, [1, 1, 1, 1]),
        helper.make_tensor_value_info(
            "h_f", TensorProto.FLOAT, [1, 1, 1, 1]),
        helper.make_tensor_value_info(
            "w_f", TensorProto.FLOAT, [1, 1, 1, 1]),
        helper.make_tensor_value_info(
            "arange_h_4d_i", TensorProto.INT64, [1, 1, HEIGHT, 1]),
        helper.make_tensor_value_info(
            "arange_w_4d_i", TensorProto.INT64, [1, 1, 1, WIDTH]),
        helper.make_tensor_value_info(
            "shift_h_4d_f", TensorProto.FLOAT, [1, 1, HEIGHT, 1]),
        helper.make_tensor_value_info(
            "shift_w_4d_f", TensorProto.FLOAT, [1, 1, 1, WIDTH]),
        helper.make_tensor_value_info(
            "shift_h_4d_i", TensorProto.INT64, [1, 1, HEIGHT, 1]),
        helper.make_tensor_value_info(
            "shift_w_4d_i", TensorProto.INT64, [1, 1, 1, WIDTH]),
        helper.make_tensor_value_info(
            "shift_h_1d", TensorProto.INT64, [HEIGHT]),
        helper.make_tensor_value_info(
            "shift_w_1d", TensorProto.INT64, [WIDTH]),
        helper.make_tensor_value_info(
            "mod_h_1d", TensorProto.INT64, [HEIGHT]),
        helper.make_tensor_value_info(
            "mod_w_1d", TensorProto.INT64, [WIDTH]),
        helper.make_tensor_value_info(
            "shifted_rows", TensorProto.FLOAT,
            [1, CHANNELS, HEIGHT, WIDTH]),
        helper.make_tensor_value_info(
            "shifted", TensorProto.FLOAT,
            [1, CHANNELS, HEIGHT, WIDTH]),
        helper.make_tensor_value_info(
            "h_le_b", TensorProto.BOOL, [1, 1, HEIGHT, 1]),
        helper.make_tensor_value_info(
            "w_le_b", TensorProto.BOOL, [1, 1, 1, WIDTH]),
        helper.make_tensor_value_info(
            "h_le_f", TensorProto.FLOAT, [1, 1, HEIGHT, 1]),
        helper.make_tensor_value_info(
            "w_le_f", TensorProto.FLOAT, [1, 1, 1, WIDTH]),
        helper.make_tensor_value_info(
            "bbox_mask", TensorProto.FLOAT, [1, 1, HEIGHT, WIDTH]),
    ]

    nodes = [
        helper.make_node(
            "ReduceSum", ["input"], ["counts_all"],
            axes=[2, 3], keepdims=1, name="counts"),
        helper.make_node(
            "Slice", ["counts_all", "ch_start", "ch_end", "ch_axes"],
            ["counts_nobg"], name="drop_bg"),
        helper.make_node(
            "Reshape", ["counts_nobg", "flat9_shape"], ["flat"],
            name="flat_counts"),
    ]

    if rule == "majority":
        nodes.append(helper.make_node(
            "ArgMax", ["flat"], ["target_idx"],
            axis=1, keepdims=0, name="argmax_target"))
    else:  # rarest
        value_info += [
            helper.make_tensor_value_info(
                "has_any_b", TensorProto.BOOL, [1, 9]),
            helper.make_tensor_value_info(
                "candidate", TensorProto.FLOAT, [1, 9]),
        ]
        nodes += [
            # Mask "no cells" channels to a large value so ArgMin skips them.
            helper.make_node(
                "Greater", ["flat", "zero_f"], ["has_any_b"],
                name="has_any_op"),
            helper.make_node(
                "Where", ["has_any_b", "flat", "big_f"], ["candidate"],
                name="candidate_op"),
            helper.make_node(
                "ArgMin", ["candidate"], ["target_idx"],
                axis=1, keepdims=0, name="argmin_target"),
        ]

    nodes += [
        helper.make_node(
            "Add", ["target_idx", "one_idx"], ["color_1d"],
            name="color_shift"),
        helper.make_node(
            "OneHot", ["color_1d", "depth", "values"], ["onehot10"],
            axis=1, name="onehot_op"),
        helper.make_node(
            "Reshape", ["onehot10", "chan_shape"], ["chan_mask"],
            name="reshape_chan_mask"),
        helper.make_node(
            "Mul", ["input", "chan_mask"], ["target_cells"],
            name="target_cells_op"),
        helper.make_node(
            "ReduceSum", ["target_cells"], ["target_mask"],
            axes=[1], keepdims=1, name="target_mask_op"),
        # Row / col projections → r_min, r_max, c_min, c_max.
        helper.make_node(
            "ReduceMax", ["target_mask"], ["row_proj"],
            axes=[3], keepdims=1, name="row_proj_op"),
        helper.make_node(
            "ReduceMax", ["target_mask"], ["col_proj"],
            axes=[2], keepdims=1, name="col_proj_op"),
        helper.make_node(
            "Reshape", ["arange_h_f", "shape4_h"], ["arange_h_4d_f"],
            name="resh_ah_f"),
        helper.make_node(
            "Reshape", ["arange_w_f", "shape4_w"], ["arange_w_4d_f"],
            name="resh_aw_f"),
        helper.make_node(
            "Mul", ["row_proj", "arange_h_4d_f"], ["row_pos"],
            name="row_pos_op"),
        helper.make_node(
            "Mul", ["col_proj", "arange_w_4d_f"], ["col_pos"],
            name="col_pos_op"),
        helper.make_node(
            "ReduceMax", ["row_pos"], ["r_max_f"],
            axes=[2], keepdims=1, name="r_max"),
        helper.make_node(
            "ReduceMax", ["col_pos"], ["c_max_f"],
            axes=[3], keepdims=1, name="c_max"),
        helper.make_node(
            "ArgMax", ["row_proj"], ["r_min_i"],
            axis=2, keepdims=1, name="r_min"),
        helper.make_node(
            "ArgMax", ["col_proj"], ["c_min_i"],
            axis=3, keepdims=1, name="c_min"),
        helper.make_node(
            "Cast", ["r_min_i"], ["r_min_f"], to=TensorProto.FLOAT,
            name="cast_r_min"),
        helper.make_node(
            "Cast", ["c_min_i"], ["c_min_f"], to=TensorProto.FLOAT,
            name="cast_c_min"),
        helper.make_node(
            "Sub", ["r_max_f", "r_min_f"], ["h_m1"], name="h_m1_op"),
        helper.make_node(
            "Sub", ["c_max_f", "c_min_f"], ["w_m1"], name="w_m1_op"),
        helper.make_node(
            "Add", ["h_m1", "one_f"], ["h_f"], name="h_add"),
        helper.make_node(
            "Add", ["w_m1", "one_f"], ["w_f"], name="w_add"),
        # Shift indices.
        helper.make_node(
            "Reshape", ["arange_h", "shape4_h"], ["arange_h_4d_i"],
            name="resh_ah_i"),
        helper.make_node(
            "Reshape", ["arange_w", "shape4_w"], ["arange_w_4d_i"],
            name="resh_aw_i"),
        helper.make_node(
            "Add", ["arange_h_4d_f", "r_min_f"], ["shift_h_4d_f"],
            name="shift_h_add"),
        helper.make_node(
            "Add", ["arange_w_4d_f", "c_min_f"], ["shift_w_4d_f"],
            name="shift_w_add"),
        helper.make_node(
            "Cast", ["shift_h_4d_f"], ["shift_h_4d_i"],
            to=TensorProto.INT64, name="shift_h_cast"),
        helper.make_node(
            "Cast", ["shift_w_4d_f"], ["shift_w_4d_i"],
            to=TensorProto.INT64, name="shift_w_cast"),
        helper.make_node(
            "Reshape", ["shift_h_4d_i", "flat_h"], ["shift_h_1d"],
            name="flat_h_idx"),
        helper.make_node(
            "Reshape", ["shift_w_4d_i", "flat_w"], ["shift_w_1d"],
            name="flat_w_idx"),
        # Mod wrap indices to [0, HEIGHT-1] and [0, WIDTH-1]
        helper.make_node(
            "Mod", ["shift_h_1d", "height_i"], ["mod_h_1d"],
            name="mod_h"),
        helper.make_node(
            "Mod", ["shift_w_1d", "width_i"], ["mod_w_1d"],
            name="mod_w"),
        helper.make_node(
            "Gather", ["input", "mod_h_1d"], ["shifted_rows"],
            axis=2, name="gather_h"),
        helper.make_node(
            "Gather", ["shifted_rows", "mod_w_1d"], ["shifted"],
            axis=3, name="gather_w"),
        # In-bbox mask.
        helper.make_node(
            "Less", ["arange_h_4d_f", "h_f"], ["h_le_b"], name="h_le_op"),
        helper.make_node(
            "Less", ["arange_w_4d_f", "w_f"], ["w_le_b"], name="w_le_op"),
        helper.make_node(
            "Cast", ["h_le_b"], ["h_le_f"], to=TensorProto.FLOAT,
            name="h_le_cast"),
        helper.make_node(
            "Cast", ["w_le_b"], ["w_le_f"], to=TensorProto.FLOAT,
            name="w_le_cast"),
        helper.make_node(
            "Mul", ["h_le_f", "w_le_f"], ["bbox_mask"], name="bbox_mask_op"),
        helper.make_node(
            "Mul", ["shifted", "bbox_mask"], ["output"], name="apply_mask"),
    ]

    inputs = [helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, CHANNELS, HEIGHT, WIDTH])]
    outputs = [helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, CHANNELS, HEIGHT, WIDTH])]
    graph = helper.make_graph(
        nodes, "bbox_color_extract", inputs, outputs,
        initializer=init, value_info=value_info)
    return helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
        ir_version=IR_VERSION)


def solve_bbox_color_extract(task: dict) -> Optional[onnx.ModelProto]:
    rule = _detect(task)
    if rule is None:
        return None
    return _build(rule)
