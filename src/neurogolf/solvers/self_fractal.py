"""Solver: self-similar fractal upscale.

A square N x N input maps to an (N*N) x (N*N) output laid out as an N x N grid
of N x N blocks. Block (R, C) is a full copy of the input where the selector
mask holds, and an all-zero block otherwise:

    output[R*N + r, C*N + c] = input[r, c]  if input[R, C] == selector_colour
                             = 0             otherwise

Two selector rules are supported, chosen by whichever explains every example:
  * a fixed colour (task 315), and
  * the most frequent non-background colour of the input (task 304).

The graph is `tiled_input * upscaled_mask`:
  * tiled_input: crop the real N x N grid and `Tile` it N x N times so every
    block is a copy of the input;
  * upscaled_mask: build the N x N selector mask, then expand each cell into an
    N x N block with two `Gather`s (a Kronecker upscale), matching the block
    layout of the tiled input.
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


def _most_nonzero(grid) -> Optional[int]:
    flat = [v for row in grid for v in row if v != 0]
    if not flat:
        return None
    return Counter(flat).most_common(1)[0][0]


def _expected(grid, n, color) -> list:
    out = [[0] * (n * n) for _ in range(n * n)]
    for R in range(n):
        for C in range(n):
            if grid[R][C] == color:
                for r in range(n):
                    for c in range(n):
                        out[R * n + r][C * n + c] = grid[r][c]
    return out


def _detect(task: dict):
    """Return (n, rule) where rule is ('fixed', colour) or ('most', None)."""
    examples = list(all_examples(task))
    if not examples:
        return None

    first = examples[0]
    n = len(first["input"])
    if n < 2 or n > 5:                      # n*n cells, (n*n)^2 <= 30*30 => n<=5
        return None
    for ex in examples:
        inp, out = ex["input"], ex["output"]
        if len(inp) != n or len(inp[0]) != n:
            return None
        if len(out) != n * n or len(out[0]) != n * n:
            return None

    # Try the most-frequent-non-background rule first, then each fixed colour.
    if all(
        (mc := _most_nonzero(ex["input"])) is not None
        and _expected(ex["input"], n, mc) == ex["output"]
        for ex in examples
    ):
        return n, ("most", None)

    for color in range(1, 10):
        if all(_expected(ex["input"], n, color) == ex["output"]
               for ex in examples):
            return n, ("fixed", color)
    return None


def _build(n: int, rule) -> onnx.ModelProto:
    def i64(name, data):
        return numpy_helper.from_array(np.array(data, dtype=np.int64), name)

    def f32(name, data):
        return numpy_helper.from_array(np.array(data, dtype=np.float32), name)

    size = n * n
    init = [
        i64("crop_starts", [0, 0, 0, 0]),
        i64("crop_ends", [1, CHANNELS, n, n]),
        i64("crop_axes", [0, 1, 2, 3]),
        i64("tile_reps", [1, 1, n, n]),
        # Kronecker upscale indices: repeat each row/col index n times.
        i64("row_idx", [r for r in range(n) for _ in range(n)]),
        i64("col_idx", [c for c in range(n) for _ in range(n)]),
        # keep only colour channels 1..9 when rebuilding background
        i64("c1", [1]), i64("c10", [CHANNELS]), i64("ax1", [1]), i64("st1", [1]),
        f32("one_f", [1.0]),
        # pad the (n*n)x(n*n) colour block out to 30x30
        i64("pad_colors", [0, 0, 0, 0, 0, 0, HEIGHT - size, WIDTH - size]),
        # background channel: 1 inside the n*n grid, then pad with 1s outside
        i64("bg_pad", [0, 0, 0, 0, 0, 0, HEIGHT - size, WIDTH - size]),
    ]

    nodes = [
        helper.make_node("Slice", ["input", "crop_starts", "crop_ends",
                                   "crop_axes"], ["crop"]),
        # tiled_input: every N x N block is a copy of the cropped input
        helper.make_node("Tile", ["crop", "tile_reps"], ["tiled"]),
    ]

    # selector mask over the cropped grid -> [1,1,n,n]
    if rule[0] == "fixed":
        color = rule[1]
        init.append(i64("color_idx", [color]))
        nodes.append(helper.make_node("Gather", ["crop", "color_idx"],
                                      ["mask_nn"], axis=1))
    else:  # most frequent non-background colour
        init.append(i64("one_i", [1]))
        init.append(i64("shape1", [1]))
        nodes += [
            helper.make_node("ReduceSum", ["crop"], ["chan_counts"],
                             axes=[2, 3], keepdims=1),          # [1,10,1,1]
            helper.make_node("Slice", ["chan_counts", "c1", "c10", "ax1",
                                       "st1"], ["counts_nz"]),  # [1,9,1,1]
            helper.make_node("ArgMax", ["counts_nz"], ["amax"], axis=1,
                             keepdims=1),                       # [1,1,1,1]
            helper.make_node("Add", ["amax", "one_i"], ["color_sel"]),
            helper.make_node("Reshape", ["color_sel", "shape1"],
                             ["color_sel_1d"]),                 # [1]
            helper.make_node("Gather", ["crop", "color_sel_1d"], ["mask_nn"],
                             axis=1),                           # [1,1,n,n]
        ]

    nodes += [
        # Kronecker-upscale the mask: each cell -> n x n block
        helper.make_node("Gather", ["mask_nn", "row_idx"], ["mask_rows"],
                         axis=2),
        helper.make_node("Gather", ["mask_rows", "col_idx"], ["mask_big"],
                         axis=3),                               # [1,1,size,size]
        # colour channels only; channel 0 is rebuilt as the complement
        helper.make_node("Slice", ["tiled", "c1", "c10", "ax1", "st1"],
                         ["tiled_colors"]),                     # [1,9,size,size]
        helper.make_node("Mul", ["tiled_colors", "mask_big"], ["frac_colors"]),
        # background inside the grid = 1 - sum(colour channels)
        helper.make_node("ReduceSum", ["frac_colors"], ["color_sum"], axes=[1],
                         keepdims=1),                           # [1,1,size,size]
        helper.make_node("Sub", ["one_f", "color_sum"], ["bg_block"]),
        # pad everything with 0 outside the n*n grid: the scorer crops trailing
        # all-zero rows/cols, recovering the true (n*n)x(n*n) output shape.
        helper.make_node("Pad", ["frac_colors", "pad_colors"], ["colors_full"],
                         mode="constant"),
        helper.make_node("Pad", ["bg_block", "bg_pad"], ["bg_full"],
                         mode="constant"),
        helper.make_node("Concat", ["bg_full", "colors_full"], ["output"],
                         axis=1),
    ]

    inputs = [helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, CHANNELS, HEIGHT, WIDTH])]
    outputs = [helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, CHANNELS, HEIGHT, WIDTH])]
    value_info = [
        helper.make_tensor_value_info("crop", TensorProto.FLOAT,
                                      [1, CHANNELS, n, n]),
        helper.make_tensor_value_info("tiled", TensorProto.FLOAT,
                                      [1, CHANNELS, size, size]),
        helper.make_tensor_value_info("mask_nn", TensorProto.FLOAT,
                                      [1, 1, n, n]),
        helper.make_tensor_value_info("mask_rows", TensorProto.FLOAT,
                                      [1, 1, size, n]),
        helper.make_tensor_value_info("mask_big", TensorProto.FLOAT,
                                      [1, 1, size, size]),
        helper.make_tensor_value_info("tiled_colors", TensorProto.FLOAT,
                                      [1, CHANNELS - 1, size, size]),
        helper.make_tensor_value_info("frac_colors", TensorProto.FLOAT,
                                      [1, CHANNELS - 1, size, size]),
        helper.make_tensor_value_info("color_sum", TensorProto.FLOAT,
                                      [1, 1, size, size]),
        helper.make_tensor_value_info("bg_block", TensorProto.FLOAT,
                                      [1, 1, size, size]),
        helper.make_tensor_value_info("colors_full", TensorProto.FLOAT,
                                      [1, CHANNELS - 1, HEIGHT, WIDTH]),
        helper.make_tensor_value_info("bg_full", TensorProto.FLOAT,
                                      [1, 1, HEIGHT, WIDTH]),
    ]
    graph = helper.make_graph(nodes, "self_fractal", inputs, outputs,
                              initializer=init, value_info=value_info)
    return helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
        ir_version=IR_VERSION)


def solve_self_fractal(task: dict) -> Optional[onnx.ModelProto]:
    spec = _detect(task)
    if spec is None:
        return None
    return _build(*spec)
