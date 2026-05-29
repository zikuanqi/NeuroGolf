"""Detect scaling patterns: N× nearest-neighbor upscale or downscale."""
from __future__ import annotations

from typing import Optional

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from ..grids import CHANNELS, HEIGHT, WIDTH, all_examples

OPSET = 11
IR_VERSION = 8


def _detect(task: dict) -> tuple[int, int] | None:
    """Return (scale_h, scale_w) if all examples follow the same scaling ratio.
    
    For upscale: output = input repeated scale_h×scale_w times per pixel.
    For downscale: output = input sampled every scale_h/scale_w pixels.
    """
    examples = list(all_examples(task))
    if not examples:
        return None
    
    # Collect all ratios
    ratios = set()
    for ex in examples:
        inp, out = ex["input"], ex["output"]
        ih, iw = len(inp), len(inp[0]) if inp else 0
        oh, ow = len(out), len(out[0]) if out else 0
        
        if ih == 0 or iw == 0 or oh == 0 or ow == 0:
            return None
        
        # Check if ratio is integer
        if oh % ih == 0 and ow % iw == 0:
            sh, sw = oh // ih, ow // iw
            ratios.add((sh, sw))
        elif ih % oh == 0 and iw % ow == 0:
            sh, sw = ih // oh, iw // ow
            ratios.add((1/sh, 1/sw) if sh > 0 and sw > 0 else None)
        else:
            return None
    
    if len(ratios) != 1:
        return None
    
    scale_h, scale_w = next(iter(ratios))
    
    # Verify the pattern holds
    for ex in examples:
        inp, out = ex["input"], ex["output"]
        ih, iw = len(inp), len(inp[0])
        oh, ow = len(out), len(out[0])
        
        if scale_h > 1 and scale_w > 1:  # Upscale
            for ro in range(oh):
                for co in range(ow):
                    ri, ci = ro // scale_h, co // scale_w
                    if out[ro][co] != inp[ri][ci]:
                        return None
        elif scale_h < 1 and scale_w < 1:  # Downscale
            scale_h_inv, scale_w_inv = int(1/scale_h), int(1/scale_w)
            for ro in range(oh):
                for co in range(ow):
                    ri, ci = ro * scale_h_inv, co * scale_w_inv
                    if out[ro][co] != inp[ri][ci]:
                        return None
        else:
            return None
    
    return (scale_h, scale_w)


def _build(scale_h: int, scale_w: int) -> onnx.ModelProto:
    """Build ONNX for upscale or downscale."""
    if scale_h > 1 and scale_w > 1:
        # Upscale via Resize with nearest neighbor
        scales = np.array([1.0, 1.0, float(scale_h), float(scale_w)], dtype=np.float32)
        init = [
            numpy_helper.from_array(scales, "scales"),
            numpy_helper.from_array(np.array([], dtype=np.float32), "empty"),
        ]
        
        nodes = [
            helper.make_node(
                "Resize", ["input", "empty", "scales"], ["output"],
                mode="nearest", name="resize_upscale"),
        ]
    else:
        # Downscale via strided slice
        stride_h, stride_w = int(1/scale_h), int(1/scale_w)
        starts = np.array([0, 0, 0, 0], dtype=np.int64)
        ends = np.array([1, CHANNELS, HEIGHT, WIDTH], dtype=np.int64)
        strides = np.array([1, 1, stride_h, stride_w], dtype=np.int64)
        
        init = [
            numpy_helper.from_array(starts, "starts"),
            numpy_helper.from_array(ends, "ends"),
            numpy_helper.from_array(strides, "strides"),
        ]
        
        nodes = [
            helper.make_node(
                "Slice", ["input", "starts", "ends", "strides"], ["output"],
                name="slice_downscale"),
        ]
    
    inputs = [helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, CHANNELS, HEIGHT, WIDTH])]
    outputs = [helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, CHANNELS, HEIGHT, WIDTH])]
    
    graph = helper.make_graph(
        nodes, f"scale_{scale_h}x{scale_w}", inputs, outputs,
        initializer=init)
    
    return helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
        ir_version=IR_VERSION)


def solve_scale_detector(task: dict) -> Optional[onnx.ModelProto]:
    scale = _detect(task)
    if scale is None:
        return None
    scale_h, scale_w = scale
    return _build(scale_h, scale_w)