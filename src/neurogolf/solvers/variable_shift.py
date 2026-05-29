"""Solver: output is input shifted by fixed offset, with zero-fill.
Handles variable input shapes by using dynamic Slice and Pad.
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
    """Return (dr, dc) if every example follows the same shift pattern."""
    examples = list(all_examples(task))
    if not examples:
        return None
    
    # Check all examples have same shift
    dr_set, dc_set = set(), set()
    for ex in examples:
        inp, out = ex["input"], ex["output"]
        ih, iw = len(inp), len(inp[0])
        oh, ow = len(out), len(out[0])
        
        if ih != oh or iw != ow:
            return None  # must be same shape
        
        # Find shift by comparing patterns
        found = False
        for dr in range(-(ih-1), ih):
            for dc in range(-(iw-1), iw):
                if dr == 0 and dc == 0:
                    continue
                ok = True
                for r in range(ih):
                    for c in range(iw):
                        sr, sc = r - dr, c - dc
                        if 0 <= sr < ih and 0 <= sc < iw:
                            if out[r][c] != inp[sr][sc]:
                                ok = False
                                break
                        else:
                            if out[r][c] != 0:
                                ok = False
                                break
                    if not ok:
                        break
                if ok:
                    dr_set.add(dr)
                    dc_set.add(dc)
                    found = True
                    break
            if found:
                break
        if not found:
            return None
    
    if len(dr_set) != 1 or len(dc_set) != 1:
        return None
    
    return (next(iter(dr_set)), next(iter(dc_set)))


def _build(dr: int, dc: int) -> onnx.ModelProto:
    """Build ONNX for shift by (dr, dc) with zero-fill."""
    def i64(name: str, vals) -> onnx.TensorProto:
        return numpy_helper.from_array(np.array(vals, dtype=np.int64), name)
    
    def f32(name: str, vals) -> onnx.TensorProto:
        return numpy_helper.from_array(np.array(vals, dtype=np.float32), name)
    
    # For dynamic shapes, we need to compute slice indices based on input shape
    # Input shape: (1, 10, H, W)
    # Output shape: (1, 10, H, W)
    
    # We'll use Gather to extract shape dimensions
    init = [
        i64("zero", [0]),
        i64("one", [1]),
        i64("two", [2]),
        i64("three", [3]),
        i64("axes_23", [2, 3]),
        i64("axes_01", [0, 1]),
        f32("zero_f", [0.0]),
        f32("one_f", [1.0]),
    ]
    
    # If dr > 0: shift down, fill top rows with color 0
    # If dr < 0: shift up, fill bottom rows with color 0
    # If dc > 0: shift right, fill left columns with color 0
    # If dc < 0: shift left, fill right columns with color 0
    
    nodes = []
    
    # Get input shape dimensions
    nodes.append(helper.make_node(
        "Shape", ["input"], ["shape"], name="get_shape"))
    
    # Extract H and W
    nodes.append(helper.make_node(
        "Gather", ["shape", "two"], ["H_i64"], axis=0, name="gather_H"))
    nodes.append(helper.make_node(
        "Gather", ["shape", "three"], ["W_i64"], axis=0, name="gather_W"))
    
    # Convert to float for calculations
    nodes.append(helper.make_node(
        "Cast", ["H_i64"], ["H_f"], to=TensorProto.FLOAT, name="cast_H"))
    nodes.append(helper.make_node(
        "Cast", ["W_i64"], ["W_f"], to=TensorProto.FLOAT, name="cast_W"))
    
    # Create arange for rows and columns
    nodes.append(helper.make_node(
        "Range", ["zero_f", "H_f", "one_f"], ["arange_h_f"], name="range_h"))
    nodes.append(helper.make_node(
        "Range", ["zero_f", "W_f", "one_f"], ["arange_w_f"], name="range_w"))
    
    # Create shift indices
    dr_f = np.array([float(dr)], dtype=np.float32)
    dc_f = np.array([float(dc)], dtype=np.float32)
    init.append(f32("dr_f", dr_f))
    init.append(f32("dc_f", dc_f))
    
    nodes.append(helper.make_node(
        "Add", ["arange_h_f", "dr_f"], ["shift_h_f"], name="add_dr"))
    nodes.append(helper.make_node(
        "Add", ["arange_w_f", "dc_f"], ["shift_w_f"], name="add_dc"))
    
    # Convert back to int64 for Gather
    nodes.append(helper.make_node(
        "Cast", ["shift_h_f"], ["shift_h_i64"], to=TensorProto.INT64, name="cast_shift_h"))
    nodes.append(helper.make_node(
        "Cast", ["shift_w_f"], ["shift_w_i64"], to=TensorProto.INT64, name="cast_shift_w"))
    
    # Apply Mod to wrap indices
    nodes.append(helper.make_node(
        "Mod", ["shift_h_i64", "H_i64"], ["mod_h_i64"], name="mod_h"))
    nodes.append(helper.make_node(
        "Mod", ["shift_w_i64", "W_i64"], ["mod_w_i64"], name="mod_w"))
    
    # Gather shifted rows and columns
    nodes.append(helper.make_node(
        "Gather", ["input", "mod_h_i64"], ["shifted_rows"], axis=2, name="gather_rows"))
    nodes.append(helper.make_node(
        "Gather", ["shifted_rows", "mod_w_i64"], ["shifted"], axis=3, name="gather_cols"))
    
    # Create mask for positions that should be color 0 (wrapped around)
    # Position is wrapped if shift index != original index (i.e., if we wrapped around)
    nodes.append(helper.make_node(
        "Equal", ["shift_h_i64", "mod_h_i64"], ["h_wrap_bool"], name="h_wrap_eq"))
    nodes.append(helper.make_node(
        "Equal", ["shift_w_i64", "mod_w_i64"], ["w_wrap_bool"], name="w_wrap_eq"))
    
    nodes.append(helper.make_node(
        "Or", ["h_wrap_bool", "w_wrap_bool"], ["wrap_bool"], name="or_wrap"))
    
    nodes.append(helper.make_node(
        "Where", ["wrap_bool", "zero_f", "one_f"], ["wrap_mask"], name="wrap_mask"))
    
    # Reshape mask to (1, 1, H, W)
    nodes.append(helper.make_node(
        "Reshape", ["wrap_mask", "shape"], ["mask_4d"], name="reshape_mask"))
    
    # Apply mask: wrapped positions become color 0
    nodes.append(helper.make_node(
        "Mul", ["shifted", "mask_4d"], ["output"], name="apply_mask"))
    
    inputs = [helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, CHANNELS, HEIGHT, WIDTH])]
    outputs = [helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, CHANNELS, HEIGHT, WIDTH])]
    
    graph = helper.make_graph(
        nodes, f"variable_shift_{dr}_{dc}", inputs, outputs,
        initializer=init)
    
    return helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
        ir_version=IR_VERSION)


def solve_variable_shift(task: dict) -> Optional[onnx.ModelProto]:
    shift = _detect(task)
    if shift is None:
        return None
    dr, dc = shift
    return _build(dr, dc)