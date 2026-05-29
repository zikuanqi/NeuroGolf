"""Solver: column-label.

Task 010: columns containing a marker color (5) are labelled 1,2,3,...
in order of the top-most row where the marker appears in that column.
Earlier top-row -> lower label. Columns without the marker stay color-0.

Input: only colors {0, 5}. Output replaces 5s with consecutive labels.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from ..grids import CHANNELS, HEIGHT, WIDTH

OPSET = 11
IR_VERSION = 8
MARKER_COLOR = 5


def _detect(task: dict) -> Optional[dict]:
    examples = list(task.get("train", [])) + list(task.get("test", []))
    if not examples:
        return None

    max_label = 0
    for ex in examples:
        inp, out = ex["input"], ex["output"]
        if len(inp) != len(out) or (inp and len(inp[0]) != len(out[0])):
            return None
        H, W = len(inp), len(inp[0])
        for r in range(H):
            for c in range(W):
                if inp[r][c] not in (0, MARKER_COLOR):
                    return None
                if inp[r][c] == MARKER_COLOR and out[r][c] == 0:
                    return None
                if inp[r][c] != MARKER_COLOR and out[r][c] != 0:
                    return None
                if out[r][c] > max_label:
                    max_label = out[r][c]
        # Each column's label must be consistent
        col_labels = {}
        for c in range(W):
            vals = set(out[r][c] for r in range(H) if inp[r][c] == MARKER_COLOR)
            if len(vals) > 1:
                return None
            if vals:
                col_labels[c] = next(iter(vals))

    if max_label < 1 or max_label > 9:
        return None

    return {"marker": MARKER_COLOR, "max_label": max_label}


def _build(config: dict) -> onnx.ModelProto:
    def f32(name, vals):
        return numpy_helper.from_array(np.array(vals, dtype=np.float32), name)

    def i64(name, vals):
        return numpy_helper.from_array(np.array(vals, dtype=np.int64), name)

    max_label = config["max_label"]
    W = WIDTH
    H = HEIGHT

    init = [
        i64("col_idx_s", np.arange(W, dtype=np.int64).reshape(1, 1, 1, W)),
        i64("col_idx_b", np.arange(W, dtype=np.int64).reshape(1, 1, W, 1)),
        f32("zero", [0.0]),
        f32("one", [1.0]),
        f32("large", [float(H + 10)]),
        i64("s5", [MARKER_COLOR]),
        i64("e6", [MARKER_COLOR + 1]),
        i64("axes_c", [1]),
        i64("step_1", [1]),
        i64("shape_w1", [1, 1, W, 1]),
        i64("shape_1w", [1, 1, 1, W]),
    ]

    # Label constants for one-hot encoding
    for k in range(1, max_label + 1):
        init.append(f32(f"label_{k}", [float(k)]))

    nodes = []

    # Step 1: Extract channel 5 -> (1,1,H,W)
    nodes.append(helper.make_node(
        "Slice", ["input", "s5", "e6", "axes_c", "step_1"], ["ch5"],
    ))

    # Step 2: Which columns have 5? col_has_5 = ReduceSum(ch5, axis=2) > 0 -> (1,1,1,W)
    nodes.append(helper.make_node(
        "ReduceSum", ["ch5"], ["col_sum"], axes=[2], keepdims=1,
    ))
    nodes.append(helper.make_node(
        "Greater", ["col_sum", "zero"], ["col_has_5_bool"],
    ))
    nodes.append(helper.make_node(
        "Cast", ["col_has_5_bool"], ["col_has_5"], to=TensorProto.FLOAT,
    ))

    # Step 3: top_row = ArgMax(ch5, axis=2, keepdims=1) -> (1,1,1,W) int64
    nodes.append(helper.make_node(
        "ArgMax", ["ch5"], ["top_row"], axis=2, keepdims=1,
    ))

    # Step 4: Push non-5 columns to large values
    # penalty = (1 - col_has_5) * (H+10)
    nodes.append(helper.make_node(
        "Sub", ["one", "col_has_5"], ["inv_mask"],
    ))
    nodes.append(helper.make_node(
        "Mul", ["inv_mask", "large"], ["penalty"],
    ))
    nodes.append(helper.make_node(
        "Cast", ["top_row"], ["top_row_f"], to=TensorProto.FLOAT,
    ))
    nodes.append(helper.make_node(
        "Add", ["top_row_f", "penalty"], ["top_eff"],
    ))

    # Step 5: Pairwise ranking
    # Reshape top_eff for broadcasting: big=(1,1,W,1), small=(1,1,1,W)
    top_big = "top_eff_big"
    top_small = "top_eff_small"

    nodes.append(helper.make_node(
        "Reshape", ["top_eff", "shape_w1"], [top_big],
    ))
    nodes.append(helper.make_node(
        "Reshape", ["top_eff", "shape_1w"], [top_small],
    ))

    # Less: top_small[c2] < top_big[c1] -> (1,1,W,W)
    nodes.append(helper.make_node(
        "Less", [top_small, top_big], ["less_than"],
    ))

    # Tie-break: equal top_row AND smaller column index
    nodes.append(helper.make_node(
        "Equal", [top_small, top_big], ["equal"],
    ))
    nodes.append(helper.make_node(
        "Less", ["col_idx_s", "col_idx_b"], ["col_less"],
    ))
    nodes.append(helper.make_node(
        "And", ["equal", "col_less"], ["tie_break"],
    ))

    # before = less_than OR tie_break
    nodes.append(helper.make_node(
        "Or", ["less_than", "tie_break"], ["before"],
    ))

    # Mask by col_has_5 on axis c2 (the "small" side)
    # Expand col_has_5 to (1,1,1,W) for broadcast along c2
    nodes.append(helper.make_node(
        "Reshape", ["col_has_5", "shape_1w"], ["col_has_5_small"],
    ))
    nodes.append(helper.make_node(
        "Cast", ["before"], ["before_f"], to=TensorProto.FLOAT,
    ))
    nodes.append(helper.make_node(
        "Mul", ["before_f", "col_has_5_small"], ["before_valid"],
    ))

    # rank = ReduceSum(before_valid, axis=3, keepdims=1) + 1 -> (1,1,W,1)
    nodes.append(helper.make_node(
        "ReduceSum", ["before_valid"], ["rank_sum"], axes=[3], keepdims=1,
    ))
    nodes.append(helper.make_node(
        "Add", ["rank_sum", "one"], ["rank"],  # rank values: 1..K for 5-columns
    ))  # shape: (1,1,W,1)

    # Step 6: Create label_map (1,1,H,W): broadcast rank * col_has_5 into H dimension
    # First reshape rank to (1,1,1,W)
    nodes.append(helper.make_node(
        "Reshape", ["rank", "shape_1w"], ["rank_1d"],
    ))  # (1,1,1,W)

    # label_map = ch5 * rank_1d  (broadcast: (1,1,H,W) * (1,1,1,W) -> (1,1,H,W))
    nodes.append(helper.make_node(
        "Mul", ["ch5", "rank_1d"], ["label_map"],
    ))  # ch5 is 1 where color=5, 0 elsewhere. label_map has label value or 0.

    # Step 7: One-hot encode label_map into output channels
    # For each label k in 1..max_label:
    #   mask_k = Equal(label_map, k) -> Cast to float
    mask_names = []
    for k in range(1, max_label + 1):
        mk = f"mask_{k}"
        nodes.append(helper.make_node(
            "Equal", ["label_map", f"label_{k}"], [mk + "_bool"],
        ))
        nodes.append(helper.make_node(
            "Cast", [mk + "_bool"], [mk], to=TensorProto.FLOAT,
        ))
        mask_names.append(mk)

    # Step 8: Gather original channel 0
    nodes.append(helper.make_node(
        "Gather", ["input", "zero_i"], ["ch0"], axis=1,
    ))
    init.append(i64("zero_i", [0]))

    # Step 9: Reassemble output: ch0, mask_1..mask_K, ch5_zero, ch6..ch9
    parts = ["ch0"]
    parts.extend(mask_names)

    # Channel 5: zeros
    zero_hw = "ch5_zero"
    nodes.append(helper.make_node(
        "Mul", ["ch5", "zero"], [zero_hw],
    ))
    parts.append(zero_hw)

    # If max_label < 5: fill unused channels 1..max_label are already masks
    # but we still have gap between max_label+1 and 4 that need to be zero
    for k in range(max_label + 1, 5):
        zk = f"gap_zero_{k}"
        nodes.append(helper.make_node(
            "Mul", ["ch5", "zero"], [zk],
        ))
        parts.append(zk)

    # Channels 6-9: zeros (or keep as input, but safer to zero)
    for k in range(6, CHANNELS):
        zk = f"ch{k}_zero"
        nodes.append(helper.make_node(
            "Mul", ["ch5", "zero"], [zk],
        ))
        parts.append(zk)

    nodes.append(helper.make_node("Concat", parts, ["output"], axis=1))

    inputs = [helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, CHANNELS, HEIGHT, WIDTH])]
    outputs = [helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, CHANNELS, HEIGHT, WIDTH])]
    graph = helper.make_graph(nodes, "column_label", inputs, outputs,
                              initializer=init)
    return helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
        ir_version=IR_VERSION)


def solve_column_label(task: dict) -> Optional[onnx.ModelProto]:
    config = _detect(task)
    if config is None:
        return None
    return _build(config)
