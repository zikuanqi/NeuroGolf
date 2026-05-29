"""Solver: output is a constant H×W rectangle filled with the input's
majority non-background color.

Only fires when (a) every example's output shape is the same (H, W) and
(b) the input's `Counter` of non-zero cells has a strict winner that
matches every output cell.

Pipeline (opset 11):

  counts_all   = ReduceSum(input, axes=[2, 3])            # (1, 10, 1, 1)
  counts_nobg  = Slice(counts_all, channels 1..10)        # (1, 9, 1, 1)
  flat         = Reshape(counts_nobg, [9])
  argmax       = ArgMax(flat) + 1                          # color in [1, 9]
  onehot_chan  = OneHot(argmax, depth=10, [0, 1], axis=0)  # (10,)
  onehot_4d    = Reshape(onehot_chan, [1, 10, 1, 1])
  fill         = Mul(onehot_4d, ones_HxW)                  # (1, 10, H, W)
  output       = Pad(fill) to (1, 10, 30, 30)
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


def _detect(task: dict) -> tuple[int, int, int] | None:
    """Find (H, W, threshold) where the rule is "color of top count if its
    count >= threshold and strictly beats the runner-up; else color 0",
    consistent with every example."""
    examples = list(all_examples(task))
    if not examples:
        return None
    shapes = {(len(e["output"]), len(e["output"][0])) for e in examples}
    if len(shapes) != 1:
        return None
    H, W = shapes.pop()
    if H == 0 or W == 0 or H > HEIGHT or W > WIDTH:
        return None
    # Output must be uniformly colored per example (single color).
    for ex in examples:
        colors = {c for row in ex["output"] for c in row}
        if len(colors) != 1:
            return None
    # Try thresholds 1..30 (counts can be up to 30*30 = 900 but reasonable
    # ARC counts are small).
    for threshold in range(1, 31):
        ok = True
        for ex in examples:
            cnt = Counter()
            for row in ex["input"]:
                for c in row:
                    if c != 0:
                        cnt[c] += 1
            ranked = sorted(cnt.items(), key=lambda x: (-x[1], x[0]))
            has_unique_max = ranked and (
                len(ranked) < 2 or ranked[0][1] > ranked[1][1])
            threshold_met = ranked and ranked[0][1] >= threshold
            expected = ranked[0][0] if (has_unique_max and threshold_met) else 0
            actual = ex["output"][0][0]
            if expected != actual:
                ok = False
                break
        if ok:
            return H, W, threshold
    return None


def _build(H: int, W: int, threshold: int) -> onnx.ModelProto:
    def i64(name: str, vals) -> onnx.TensorProto:
        return numpy_helper.from_array(np.array(vals, dtype=np.int64), name)

    def f32(name: str, vals) -> onnx.TensorProto:
        return numpy_helper.from_array(np.array(vals, dtype=np.float32), name)

    # Use the threshold as an exclusive lower bound for Greater: a count
    # `c` qualifies when c > (threshold - 1).
    init = [
        # Slice channels 1..10 (drop channel 0 = bg).
        i64("ch_start", [1]),
        i64("ch_end", [10]),
        i64("ch_axes", [1]),
        i64("flat9_shape", [9]),
        i64("chan_shape", [1, 10, 1, 1]),
        # OneHot params.
        i64("depth", [10]),
        f32("values", [0.0, 1.0]),
        i64("one_idx", [1]),
        i64("zero_idx", [0]),
        i64("k_two", [2]),
        f32("thresh_minus_1", float(threshold - 1)),
        # Spatial 1s for the broadcast tile, and pad bounds.
        f32("ones_spatial", np.ones((1, 1, H, W), dtype=np.float32)),
        i64("pads", [0, 0, 0, 0, 0, 0, HEIGHT - H, WIDTH - W]),
    ]

    value_info = [
        helper.make_tensor_value_info(
            "counts_all", TensorProto.FLOAT, [1, CHANNELS, 1, 1]),
        helper.make_tensor_value_info(
            "counts_nobg", TensorProto.FLOAT, [1, 9, 1, 1]),
        helper.make_tensor_value_info(
            "counts_flat", TensorProto.FLOAT, [9]),
        helper.make_tensor_value_info(
            "top2_vals", TensorProto.FLOAT, [2]),
        helper.make_tensor_value_info(
            "top2_idx", TensorProto.INT64, [2]),
        helper.make_tensor_value_info(
            "v0", TensorProto.FLOAT, [1]),
        helper.make_tensor_value_info(
            "v1", TensorProto.FLOAT, [1]),
        helper.make_tensor_value_info(
            "idx0", TensorProto.INT64, [1]),
        helper.make_tensor_value_info(
            "unique_b", TensorProto.BOOL, [1]),
        helper.make_tensor_value_info(
            "threshold_b", TensorProto.BOOL, [1]),
        helper.make_tensor_value_info(
            "valid_b", TensorProto.BOOL, [1]),
        helper.make_tensor_value_info(
            "color_plus_1", TensorProto.INT64, [1]),
        helper.make_tensor_value_info(
            "final_color", TensorProto.INT64, [1]),
        helper.make_tensor_value_info(
            "onehot10", TensorProto.FLOAT, [1, 10]),
        helper.make_tensor_value_info(
            "onehot_4d", TensorProto.FLOAT, [1, CHANNELS, 1, 1]),
        helper.make_tensor_value_info(
            "fill", TensorProto.FLOAT, [1, CHANNELS, H, W]),
    ]

    nodes = [
        helper.make_node(
            "ReduceSum", ["input"], ["counts_all"],
            axes=[2, 3], keepdims=1, name="reduce_spatial"),
        helper.make_node(
            "Slice", ["counts_all", "ch_start", "ch_end", "ch_axes"],
            ["counts_nobg"], name="drop_bg"),
        helper.make_node(
            "Reshape", ["counts_nobg", "flat9_shape"], ["counts_flat"],
            name="flat_counts"),
        # TopK along axis 0, k=2.
        helper.make_node(
            "TopK", ["counts_flat", "k_two"], ["top2_vals", "top2_idx"],
            axis=0, name="top2"),
        # Split top2 into v0/v1 via Slice [0..1] and [1..2].
        helper.make_node(
            "Slice", ["top2_vals", "zero_idx", "one_idx"], ["v0"],
            name="get_v0"),
        helper.make_node(
            "Slice", ["top2_vals", "one_idx", "k_two"], ["v1"],
            name="get_v1"),
        helper.make_node(
            "Slice", ["top2_idx", "zero_idx", "one_idx"], ["idx0"],
            name="get_idx0"),
        # Unique max iff v0 > v1; threshold iff v0 > (T-1).
        helper.make_node(
            "Greater", ["v0", "v1"], ["unique_b"], name="unique_op"),
        helper.make_node(
            "Greater", ["v0", "thresh_minus_1"], ["threshold_b"],
            name="threshold_op"),
        helper.make_node(
            "And", ["unique_b", "threshold_b"], ["valid_b"],
            name="valid_op"),
        # color = idx0 + 1 (channel shift past bg). Final = Where(valid, color, 0).
        helper.make_node(
            "Add", ["idx0", "one_idx"], ["color_plus_1"],
            name="color_shift"),
        helper.make_node(
            "Where", ["valid_b", "color_plus_1", "zero_idx"], ["final_color"],
            name="select_color"),
        # OneHot: indices [1] -> (1, 10) at axis 1.
        helper.make_node(
            "OneHot", ["final_color", "depth", "values"], ["onehot10"],
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
        nodes, "majority_fill", inputs, outputs,
        initializer=init, value_info=value_info)
    return helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
        ir_version=IR_VERSION)


def solve_majority_fill(task: dict) -> Optional[onnx.ModelProto]:
    spec = _detect(task)
    if spec is None:
        return None
    return _build(*spec)
