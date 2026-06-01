"""Scattered-color solver for 1x1 output tasks (task 346).

Each input has exactly 2 non-bg colors. The output is the color that is more
"scattered" — the one with more isolated pixels (pixels with zero same-color
4-neighbors). This is equivalent to selecting the color with the lowest
pixels-per-connected-component ratio.

For detection: computes per-color isolated pixel counts via a 4-neighbor
Conv, then ArgMax to select the winning color.
"""
from __future__ import annotations

from collections import Counter
from typing import Optional

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from ..grids import CHANNELS, HEIGHT, WIDTH, all_examples

OPSET = 11
IR_VERSION_V = 8
DATA_TYPE = TensorProto.FLOAT
GRID_SHAPE = [1, CHANNELS, HEIGHT, WIDTH]


def _check_task(task: dict) -> bool:
    """Verify every example has 2 non-bg colors, 1x1 output."""
    for ex in all_examples(task):
        inp, out = ex["input"], ex["output"]
        if len(out) != 1 or len(out[0]) != 1:
            return False
        colors = set()
        for row in inp:
            for v in row:
                if v != 0:
                    colors.add(v)
        if len(colors) != 2:
            return False
    return True


def _isolated_count(grid: list[list[int]], color: int) -> int:
    """Count pixels of `color` with zero same-color 4-neighbors."""
    h, w = len(grid), len(grid[0])
    count = 0
    for r in range(h):
        for c in range(w):
            if grid[r][c] != color:
                continue
            has_neighbor = False
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] == color:
                    has_neighbor = True
                    break
            if not has_neighbor:
                count += 1
    return count


def _verify_rule(task: dict) -> bool:
    """Verify: output color = argmax(isolated_count) for each example."""
    for ex in all_examples(task):
        inp, out = ex["input"], ex["output"]
        colors = set()
        for row in inp:
            for v in row:
                if v != 0:
                    colors.add(v)
        best_color = None
        best_count = -1
        for c in colors:
            ic = _isolated_count(inp, c)
            if ic > best_count:
                best_count = ic
                best_color = c
        if best_color != out[0][0]:
            return False
    return True


def solve_scattered_color(task: dict) -> Optional[onnx.ModelProto]:
    """Build ONNX model: output = most scattered color (task 346)."""
    if not _check_task(task):
        return None
    if not _verify_rule(task):
        return None

    def i64(name: str, vals) -> onnx.TensorProto:
        return numpy_helper.from_array(np.array(vals, dtype=np.int64), name)

    def f32(name: str, vals) -> onnx.TensorProto:
        return numpy_helper.from_array(np.array(vals, dtype=np.float32), name)

    nodes: list[onnx.NodeProto] = []
    initializers: list[onnx.TensorProto] = []

    # Build 4-neighbor kernel: one output channel per input color (1-9)
    # Weight shape: (9, 10, 3, 3) — 9 output channels, 10 input channels.
    # Output channel i uses input channel i+1 with 4-neighbor kernel.
    # Background channel (0) gets zero weight everywhere.
    kernel_3x3 = np.array([[0, 1, 0],
                           [1, 0, 1],
                           [0, 1, 0]], dtype=np.float32)
    weight = np.zeros((9, CHANNELS, 3, 3), dtype=np.float32)
    for c in range(1, CHANNELS):
        weight[c - 1, c, :, :] = kernel_3x3

    w_init = f32("conv_w", weight.reshape(9, CHANNELS, 3, 3))
    initializers.append(w_init)

    # Conv: input (1,10,30,30) → neighbor counts (1,9,30,30)
    nodes.append(helper.make_node(
        "Conv", ["input", "conv_w"], ["neighbor_counts"],
        kernel_shape=[3, 3], pads=[1, 1, 1, 1],
    ))  # → (1, 9, 30, 30)

    # Slice to keep only channels 1-9 from input (drop channel 0 = bg)
    slice_starts_c = i64("slice_starts_c", [0, 1, 0, 0])
    slice_ends_c = i64("slice_ends_c", [1, CHANNELS, HEIGHT, WIDTH])
    slice_axes_c = i64("slice_axes_c", [0, 1, 2, 3])
    slice_steps_c = i64("slice_steps_c", [1, 1, 1, 1])
    initializers.extend([slice_starts_c, slice_ends_c, slice_axes_c, slice_steps_c])

    nodes.append(helper.make_node(
        "Slice", ["input", "slice_starts_c", "slice_ends_c", "slice_axes_c", "slice_steps_c"],
        ["color_mask"],
    ))  # → (1, 9, 30, 30)

    # A pixel is "isolated" if neighbor_count == 0 AND color_mask == 1
    zero_const = f32("zero_scalar", [0.0])
    initializers.append(zero_const)

    nodes.append(helper.make_node(
        "Equal", ["neighbor_counts", "zero_scalar"], ["no_neighbors"],
    ))  # → (1, 9, 30, 30) bool

    # Cast to float for arithmetic
    nodes.append(helper.make_node(
        "Cast", ["no_neighbors"], ["no_neighbors_f"], to=int(TensorProto.FLOAT),
    ))

    # isolated = no_neighbors_f * color_mask → 1 only at truly isolated pixels
    nodes.append(helper.make_node(
        "Mul", ["no_neighbors_f", "color_mask"], ["isolated"],
    ))  # → (1, 9, 30, 30)

    # Sum over spatial dims → (1, 9, 1, 1)
    nodes.append(helper.make_node(
        "ReduceSum", ["isolated"], ["iso_counts"],
        axes=[2, 3], keepdims=0,
    ))  # → (1, 9)

    # Flatten to (9,) for ArgMax
    flat_shape = i64("flatten_shape", [9])
    initializers.append(flat_shape)
    nodes.append(helper.make_node(
        "Reshape", ["iso_counts", "flatten_shape"], ["iso_flat"],
    ))  # → (9,)

    # ArgMax → index of most isolated color (0-indexed)
    nodes.append(helper.make_node(
        "ArgMax", ["iso_flat"], ["winner_idx"],
        axis=0, keepdims=0,
    ))  # → scalar int64

    # Reshape scalar to (1,) for OneHot (requires rank >= 1)
    win_rs = i64("win_rs", [1])
    initializers.append(win_rs)
    nodes.append(helper.make_node(
        "Reshape", ["winner_idx", "win_rs"], ["winner_1d"],
    ))  # → (1,) int64

    # winner_color = winner_1d + 1 (convert 0-indexed to 1-indexed color)
    one_i64 = i64("one_i64", [1])
    initializers.append(one_i64)
    nodes.append(helper.make_node(
        "Add", ["winner_1d", "one_i64"], ["winner_color"],
    ))  # → scalar int64

    # OneHot to get (1, 10, 1, 1) sliceable placement
    # depth must be a scalar int64, not a 1-D tensor
    depth_10 = helper.make_tensor("depth_10", TensorProto.INT64, [], [10])
    initializers.append(depth_10)

    oh_values = f32("oh_values", [0.0, 1.0])
    initializers.append(oh_values)

    nodes.append(helper.make_node(
        "OneHot", ["winner_color", "depth_10", "oh_values"],
        ["winner_onehot"],
        axis=-1,
    ))  # → (1, 10)

    # Reshape winner_onehot to (1, 10, 1, 1)
    oh_rs_shape = i64("oh_rs_shape", [1, 10, 1, 1])
    initializers.append(oh_rs_shape)
    nodes.append(helper.make_node(
        "Reshape", ["winner_onehot", "oh_rs_shape"], ["output_tl"],
    ))  # → (1, 10, 1, 1)

    # Pad to full (1, 10, 30, 30): pad with zeros to reach 30x30
    pad_const = f32("pad_value", [0.0])
    initializers.append(pad_const)

    # Pad: need to add 29 rows and 29 cols to make 30x30 from 1x1
    # Pad format: [dim0_start, dim0_end, dim1_start, dim1_end,
    #              dim2_start, dim2_end, dim3_start, dim3_end]
    pads_init = i64("pads", [0, 0, 0, 0, 0, HEIGHT - 1, 0, WIDTH - 1])
    initializers.append(pads_init)

    nodes.append(helper.make_node(
        "Pad", ["output_tl", "pads_init", "pad_value"], ["output"],
        mode="constant",
    ))  # → (1, 10, 30, 30)

    # Build model
    x = helper.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = helper.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    graph = helper.make_graph(nodes, "scattered_color", [x], [y], initializers)
    return helper.make_model(graph, ir_version=IR_VERSION_V,
                             opset_imports=[helper.make_opsetid("", OPSET)])