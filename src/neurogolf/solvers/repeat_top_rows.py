"""Solver: repeat_top_rows.

Task 003: Input is 6xW, output is 9xW. Color 1 → color 2.
The input has an underlying period P (2, 3, or 4), and the extra 3 rows
continue the pattern: output[6:9] = input[6%P, 7%P, 8%P] (remapped).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from ..grids import CHANNELS, HEIGHT, WIDTH

OPSET = 11
IR_VERSION = 8


def _detect(task: dict) -> Optional[dict]:
    examples = list(task.get("train", [])) + list(task.get("test", []))
    if not examples:
        return None

    for ex in examples:
        inp, out = ex["input"], ex["output"]
        ih, iw = len(inp), len(inp[0])
        oh, ow = len(out), len(out[0])
        if ih != 6 or oh != 9 or iw != ow:
            return None
        for r in range(6):
            for c in range(iw):
                if inp[r][c] not in (0, 1):
                    return None
        for r in range(9):
            for c in range(ow):
                if out[r][c] not in (0, 2):
                    return None
        for r in range(6):
            for c in range(iw):
                expected = 2 if inp[r][c] == 1 else 0
                if out[r][c] != expected:
                    return None

    return {}


def _build(config: dict) -> onnx.ModelProto:
    def f32(name, vals):
        return numpy_helper.from_array(np.array(vals, dtype=np.float32), name)

    def i64(name, vals):
        return numpy_helper.from_array(np.array(vals, dtype=np.int64), name)

    init = [
        i64("start_0", [0]),
        i64("end_4", [4]),
        i64("end_6", [6]),
        i64("start_2", [2]),
        i64("start_3", [3]),
        i64("axes_h", [2]),
        i64("step_1", [1]),
        i64("ch0_idx", [0]),
        i64("ch1_idx", [1]),
        i64("gather_A", [0, 1, 0]),
        i64("gather_B", [0, 1, 2]),
        i64("gather_C", [2, 3, 0]),
        i64("pads_val", [0, 0, 0, 0, 0, 0, 21, 0]),
        f32("thresh", [0.5]),
        f32("zero_f", [0.0]),
        f32("one_f", [1.0]),
    ]

    nodes = []

    # Step 1: Extract input rows 0-5 → (1,10,6,W)
    nodes.append(helper.make_node(
        "Slice", ["input", "start_0", "end_6", "axes_h", "step_1"],
        ["in_rows_6"],
    ))

    # Step 2: Extract channel 1 for period detection → (1,1,6,W)
    nodes.append(helper.make_node(
        "Gather", ["in_rows_6", "ch1_idx"], ["ch1_6"], axis=1,
    ))

    # Step 3: Error_P3 = sum((rows 0-2 - rows 3-5)^2)
    nodes.append(helper.make_node(
        "Slice", ["ch1_6", "start_0", "start_3", "axes_h", "step_1"],
        ["ch1_03"],
    ))
    nodes.append(helper.make_node(
        "Slice", ["ch1_6", "start_3", "end_6", "axes_h", "step_1"],
        ["ch1_36"],
    ))
    nodes.append(helper.make_node("Sub", ["ch1_03", "ch1_36"], ["diff_P3"]))
    nodes.append(helper.make_node("Mul", ["diff_P3", "diff_P3"], ["sq_P3"]))
    nodes.append(helper.make_node(
        "ReduceSum", ["sq_P3"], ["error_P3"], keepdims=0,
    ))

    # Step 4: Error_P2 = sum((rows 0-3 - rows 2-5)^2)
    nodes.append(helper.make_node(
        "Slice", ["ch1_6", "start_0", "end_4", "axes_h", "step_1"],
        ["ch1_04"],
    ))
    nodes.append(helper.make_node(
        "Slice", ["ch1_6", "start_2", "end_6", "axes_h", "step_1"],
        ["ch1_26"],
    ))
    nodes.append(helper.make_node("Sub", ["ch1_04", "ch1_26"], ["diff_P2"]))
    nodes.append(helper.make_node("Mul", ["diff_P2", "diff_P2"], ["sq_P2"]))
    nodes.append(helper.make_node(
        "ReduceSum", ["sq_P2"], ["error_P2"], keepdims=0,
    ))

    # Step 5: Validity scores & weights
    nodes.append(helper.make_node(
        "Less", ["error_P3", "thresh"], ["p3_valid_bool"],
    ))
    nodes.append(helper.make_node(
        "Cast", ["p3_valid_bool"], ["p3_valid"], to=TensorProto.FLOAT,
    ))
    nodes.append(helper.make_node(
        "Less", ["error_P2", "thresh"], ["p2_valid_bool"],
    ))
    nodes.append(helper.make_node(
        "Cast", ["p2_valid_bool"], ["p2_valid"], to=TensorProto.FLOAT,
    ))

    # w_B = p3_valid, w_A = (1-p3)*p2, w_C = (1-p3)*(1-p2)
    nodes.append(helper.make_node("Sub", ["one_f", "p3_valid"], ["inv_p3"]))
    nodes.append(helper.make_node("Sub", ["one_f", "p2_valid"], ["inv_p2"]))
    nodes.append(helper.make_node("Mul", ["inv_p3", "p2_valid"], ["w_A"]))
    nodes.append(helper.make_node("Mul", ["inv_p3", "inv_p2"], ["w_C"]))

    # Step 6: Remap rows 0-5: channel 0→ch0, channel 1→ch2, rest zero
    nodes.append(helper.make_node(
        "Gather", ["in_rows_6", "ch0_idx"], ["ch0_6"], axis=1,
    ))
    nodes.append(helper.make_node(
        "Gather", ["in_rows_6", "ch1_idx"], ["ch1_6_data"], axis=1,
    ))

    parts = ["ch0_6"]
    zero_hw = "ch_zero"
    nodes.append(helper.make_node("Mul", ["ch0_6", "zero_f"], [zero_hw]))
    parts.append(zero_hw)     # channel 1 → zero
    parts.append("ch1_6_data")  # channel 2 ← channel 1

    for k in range(3, CHANNELS):
        zk = f"ch{k}_zero"
        nodes.append(helper.make_node("Mul", ["ch0_6", "zero_f"], [zk]))
        parts.append(zk)

    nodes.append(helper.make_node(
        "Concat", parts, ["remap_6"], axis=1,
    ))  # (1,10,6,W)

    # Step 7: Candidate extra rows via Gather
    nodes.append(helper.make_node(
        "Gather", ["remap_6", "gather_A"], ["cand_A"], axis=2,
    ))
    nodes.append(helper.make_node(
        "Gather", ["remap_6", "gather_B"], ["cand_B"], axis=2,
    ))
    nodes.append(helper.make_node(
        "Gather", ["remap_6", "gather_C"], ["cand_C"], axis=2,
    ))

    # Step 8: Weighted mix: extra = w_A*A + p3_valid*B + w_C*C
    nodes.append(helper.make_node("Mul", ["cand_A", "w_A"], ["term_A"]))
    nodes.append(helper.make_node("Mul", ["cand_B", "p3_valid"], ["term_B"]))
    nodes.append(helper.make_node("Mul", ["cand_C", "w_C"], ["term_C"]))
    nodes.append(helper.make_node("Add", ["term_A", "term_B"], ["extra_ab"]))
    nodes.append(helper.make_node(
        "Add", ["extra_ab", "term_C"], ["extra_rows"],
    ))

    # Step 9: Concat remapped 6 + extra 3 → 9 rows
    nodes.append(helper.make_node(
        "Concat", ["remap_6", "extra_rows"], ["out_9"], axis=2,
    ))

    # Step 10: Pad height to 30
    nodes.append(helper.make_node(
        "Pad", ["out_9", "pads_val", "zero_f"], ["output"],
        mode="constant",
    ))

    inputs_val = [helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, CHANNELS, HEIGHT, WIDTH])]
    outputs_val = [helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, CHANNELS, HEIGHT, WIDTH])]
    graph = helper.make_graph(nodes, "repeat_top_rows", inputs_val, outputs_val,
                              initializer=init)
    return helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
        ir_version=IR_VERSION)


def solve_repeat_top_rows(task: dict) -> Optional[onnx.ModelProto]:
    config = _detect(task)
    if config is None:
        return None
    return _build(config)
