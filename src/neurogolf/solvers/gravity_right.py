"""Solver: gravity to the right - each row's cells slide right until blocked.
Each non-zero cell moves right until it hits another cell or the right edge.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from ..grids import CHANNELS, HEIGHT, WIDTH, all_examples

OPSET = 11
IR_VERSION = 8


def _detect(task: dict) -> bool:
    """Return True if every example follows gravity-right pattern."""
    examples = list(all_examples(task))
    if not examples:
        return False
    
    for ex in examples:
        inp, out = ex["input"], ex["output"]
        h, w = len(inp), len(inp[0])
        
        if len(out) != h or len(out[0]) != w:
            return False
        
        # For each row, simulate gravity right
        for r in range(h):
            # Count non-zero cells in input row
            nonzeros = [c for c in inp[r] if c != 0]
            # In output, they should be at the rightmost positions
            expected = [0] * (w - len(nonzeros)) + nonzeros
            if out[r] != expected:
                return False
    return True


def _build() -> onnx.ModelProto:
    """Build ONNX for gravity to the right."""
    def i64(name: str, vals) -> onnx.TensorProto:
        return numpy_helper.from_array(np.array(vals, dtype=np.int64), name)
    
    def f32(name: str, vals) -> onnx.TensorProto:
        return numpy_helper.from_array(np.array(vals, dtype=np.float32), name)
    
    # We'll use a per-row approach:
    # 1. For each row, count non-zero cells per channel
    # 2. Create a mask where non-zero cells are at the rightmost positions
    
    init = [
        i64("zero", [0]),
        i64("one", [1]),
        i64("two", [2]),
        i64("three", [3]),
        f32("zero_f", [0.0]),
        f32("one_f", [1.0]),
        i64("axes_3", [3]),  # column axis
        i64("axes_23", [2, 3]),
    ]
    
    # Create arange for column indices
    col_indices = np.arange(WIDTH, dtype=np.float32)
    init.append(f32("col_indices_f", col_indices))
    init.append(i64("col_indices_i64", col_indices.astype(np.int64)))
    
    nodes = []
    
    # Get input shape
    nodes.append(helper.make_node(
        "Shape", ["input"], ["shape"], name="get_shape"))
    
    # Extract W
    nodes.append(helper.make_node(
        "Gather", ["shape", "three"], ["W_i64"], axis=0, name="gather_W"))
    nodes.append(helper.make_node(
        "Cast", ["W_i64"], ["W_f"], to=TensorProto.FLOAT, name="cast_W"))
    
    # Create column indices for this width
    nodes.append(helper.make_node(
        "Range", ["zero_f", "W_f", "one_f"], ["arange_w_f"], name="range_w"))
    nodes.append(helper.make_node(
        "Cast", ["arange_w_f"], ["arange_w_i64"], to=TensorProto.INT64, name="cast_arange_w"))
    
    # For each channel, count non-zero cells per row
    # We'll do this by thresholding at > 0.0 and summing across columns
    nodes.append(helper.make_node(
        "Greater", ["input", "zero_f"], ["nonzero_mask"], name="nonzero_mask"))
    nodes.append(helper.make_node(
        "Cast", ["nonzero_mask"], ["nonzero_mask_f"], to=TensorProto.FLOAT, name="cast_mask"))
    
    # Sum across columns to get count per row per channel
    nodes.append(helper.make_node(
        "ReduceSum", ["nonzero_mask_f"], ["counts_per_row"], axes=[3], keepdims=1, name="sum_cols"))
    
    # For each row, we need to shift cells right by (W - count)
    nodes.append(helper.make_node(
        "Sub", ["W_f", "counts_per_row"], ["shift_amount_f"], name="calc_shift"))
    nodes.append(helper.make_node(
        "Cast", ["shift_amount_f"], ["shift_amount_i64"], to=TensorProto.INT64, name="cast_shift"))
    
    # Add shift to column indices
    nodes.append(helper.make_node(
        "Add", ["arange_w_i64", "shift_amount_i64"], ["shifted_indices_i64"], name="add_shift"))
    
    # Mod wrap to handle wrap-around (though gravity-right shouldn't wrap)
    nodes.append(helper.make_node(
        "Mod", ["shifted_indices_i64", "W_i64"], ["mod_indices_i64"], name="mod_indices"))
    
    # Gather using shifted indices
    nodes.append(helper.make_node(
        "Gather", ["input", "mod_indices_i64"], ["output"], axis=3, name="gather_cols"))
    
    inputs = [helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, CHANNELS, HEIGHT, WIDTH])]
    outputs = [helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, CHANNELS, HEIGHT, WIDTH])]
    
    graph = helper.make_graph(
        nodes, "gravity_right", inputs, outputs, initializer=init)
    
    return helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
        ir_version=IR_VERSION)


def solve_gravity_right(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()