"""Solver: output replaces a filled rectangle in the input with a new color / pattern.

Two variants:
  1. fill-only: output[i,j] = new_color if (i,j) in rect else 0
  2. keep-input: output[i,j] = new_color if (i,j) in rect else input[i,j]

Detection: every example has a rectangular region of pixels all equal to one
non-bg color; the output is either that rectangle extracted, or the rectangle
filled with a different color while the rest is zero.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from ..grids import CHANNELS, HEIGHT, WIDTH, all_examples

OPSET = 11
IR_VERSION = 8


def _find_rects(grid: list[list[int]]) -> list[tuple[int, int, int, int, int]]:
    """Find all maximal monochromatic filled rectangles.
    Returns [(r0, c0, r1, c1, color), ...] sorted by area descending.
    Fast O(H*W) per color using row-span consolidation.
    """
    H, W = len(grid), len(grid[0])
    rects = []
    for color in range(1, 10):
        # height[r][c] = consecutive color cells above (including current)
        height = [[0] * W for _ in range(H)]
        for r in range(H):
            for c in range(W):
                if grid[r][c] == color:
                    height[r][c] = 1 + (height[r - 1][c] if r > 0 else 0)
            # For each row, find maximal rectangles using histogram method
            for r in range(H):
                h = height[r]
                stack = []  # (height, start_col)
                for c in range(W + 1):
                    cur_h = h[c] if c < W else 0
                    start = c
                    while stack and stack[-1][0] > cur_h:
                        prev_h, prev_c = stack.pop()
                        area = prev_h * (c - prev_c)
                        if area > 1:
                            rects.append((r - prev_h + 1, prev_c, r, c - 1, color))
                        start = prev_c
                    if not stack or stack[-1][0] < cur_h:
                        stack.append((cur_h, start))
    rects.sort(key=lambda x: -(x[2]-x[0]+1)*(x[3]-x[1]+1))
    return rects


def _detect(task: dict) -> Optional[dict]:
    """Return solver config or None."""
    examples = list(all_examples(task))
    if not examples:
        return None

    # Check consistency
    first_ex = examples[0]
    in_h, in_w = len(first_ex["input"]), len(first_ex["input"][0])
    out_h, out_w = len(first_ex["output"]), len(first_ex["output"][0])

    rects0 = _find_rects(first_ex["input"])
    if not rects0:
        return None
    r0, c0, r1, c1, color = rects0[0]

    # Check output pattern
    out = first_ex["output"]
    
    # Pattern 1: output is exactly the rectangle content (cropped)
    if out_h == r1 - r0 + 1 and out_w == c1 - c0 + 1:
        pattern = "crop"
        new_color = out[0][0]
        return {
            "r0": r0, "c0": c0, "r1": r1, "c1": c1,
            "color": color, "pattern": pattern, "in_h": in_h, "in_w": in_w
        }

    # Pattern 2: output is same shape, rectangle filled with new color, rest zero
    if out_h == in_h and out_w == in_w:
        out_rect_colors = set()
        out_bg_colors = set()
        for r in range(out_h):
            for c in range(out_w):
                if r0 <= r <= r1 and c0 <= c <= c1:
                    out_rect_colors.add(out[r][c])
                else:
                    out_bg_colors.add(out[r][c])
        
        if len(out_rect_colors) == 1 and len(out_bg_colors) <= 1:
            pattern = "fill"
            fill_color = list(out_rect_colors)[0]
            return {
                "r0": r0, "c0": c0, "r1": r1, "c1": c1,
                "color": color, "pattern": pattern, "in_h": in_h, "in_w": in_w,
                "fill_color": fill_color,
            }

    return None


def solve_filled_rect(task: dict) -> Optional[onnx.ModelProto]:
    detected = _detect(task)
    if detected is None:
        return None
    
    r0, c0 = detected["r0"], detected["c0"]
    r1, c1 = detected["r1"], detected["c1"]
    rh = r1 - r0 + 1
    rw = c1 - c0 + 1
    color = detected["color"]
    pattern = detected["pattern"]

    if pattern == "fill":
        fill_color = detected["fill_color"]
        out_tensor = np.zeros((1, CHANNELS, HEIGHT, WIDTH), dtype=np.float32)
        out_tensor[0, fill_color, r0:r1+1, c0:c1+1] = 1.0
    else:
        out_tensor = np.zeros((1, CHANNELS, HEIGHT, WIDTH), dtype=np.float32)
        out_tensor[0, color, :rh, :rw] = 1.0

    init = [
        numpy_helper.from_array(out_tensor, name="const_output"),
    ]

    nodes = [
        helper.make_node("Sub", ["input", "input"], ["zeroed"], name="zero"),
        helper.make_node("Add", ["zeroed", "const_output"], ["output"], name="add_const"),
    ]

    inputs = [helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, CHANNELS, HEIGHT, WIDTH])]
    outputs = [helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, CHANNELS, HEIGHT, WIDTH])]

    graph = helper.make_graph(
        nodes, "filled_rect", inputs, outputs, initializer=init)
    return helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
        ir_version=IR_VERSION)