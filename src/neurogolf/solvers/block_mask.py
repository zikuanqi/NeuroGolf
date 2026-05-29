"""Solver: block_mask.

Task 001 variant: Input is N×N, output is N²×N².
The input's non-zero cells define a mask. The output places a full copy
of the input pattern into every block position where the mask cell is non-zero.

For all examples, N=3 (3×3 → 9×9).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from ..grids import CHANNELS, HEIGHT, WIDTH, all_examples

OPSET = 11
IR_VERSION = 8


def _detect(task: dict) -> Optional[int]:
    """Return N if all examples follow the block-mask pattern (N×N → N²×N²)."""
    examples = list(all_examples(task))
    if not examples:
        return None

    # All inputs must have same shape N×N
    ih_set = {len(ex["input"]) for ex in examples}
    iw_set = {len(ex["input"][0]) for ex in examples if ex["input"]}
    if len(ih_set) != 1 or len(iw_set) != 1:
        return None
    N = ih_set.pop()
    iw = iw_set.pop()
    if N != iw or N < 2 or N > 10:
        return None
    if N * N > HEIGHT or N * N > WIDTH:
        return None

    for ex in examples:
        inp, out = ex["input"], ex["output"]
        if len(out) != N * N or len(out[0]) != N * N:
            return None

        # Compute block-level mask
        mask = np.array(inp) != 0

        # Verify: output[r,c] = input[r%N, c%N] if mask[r//N, c//N] else 0
        for r in range(N * N):
            br, pr = divmod(r, N)
            for c in range(N * N):
                bc, pc = divmod(c, N)
                expected = inp[pr][pc] if mask[br][bc] else 0
                if out[r][c] != expected:
                    return None

    return N


def _build(N: int) -> onnx.ModelProto:
    def i64(name, data):
        return numpy_helper.from_array(np.array(data, dtype=np.int64), name)

    def f32(name, data):
        return numpy_helper.from_array(np.array(data, dtype=np.float32), name)

    out_size = N * N  # 9

    # Initializers
    starts = i64("starts", [0, 0, 0, 0])
    ends = i64("ends", [1, CHANNELS, N, N])
    axes_all = i64("axes_all", [0, 1, 2, 3])

    # Row/col gather indices for tiling input: cycle through all N rows/cols N times
    row_idx = i64("row_idx", [r for _ in range(N) for r in range(N)])
    col_idx = i64("col_idx", [c for _ in range(N) for c in range(N)])

    # Mask reshape/tile shapes
    mask_shape_in = i64("mask_shape_in", [1, 1, N, N])
    mask_shape_inter = i64("mask_shape_inter", [1, 1, N, 1, N, 1])
    mask_shape_out = i64("mask_shape_out", [1, 1, out_size, out_size])
    mask_tile_rp = i64("mask_tile_rp", [1, 1, 1, N, 1, N])

    # Constants
    zero_f = f32("zero_f", [0.0])
    one_f = f32("one_f", [1.0])

    # For channel 1-9 slice
    ch1_start = i64("ch1_start", [0, 1, 0, 0])
    ch1_end = i64("ch1_end", [1, CHANNELS, N, N])

    # Pad to canvas
    pads = i64("pads", [0, 0, 0, 0, 0, 0,
                        HEIGHT - out_size, WIDTH - out_size])

    init = [
        starts, ends, axes_all, row_idx, col_idx,
        ch1_start, ch1_end,
        mask_shape_inter, mask_shape_out, mask_tile_rp,
        zero_f, one_f, pads,
    ]

    nodes = []

    # 1. Slice input to (1,10,N,N)
    nodes.append(helper.make_node(
        "Slice", ["input", "starts", "ends", "axes_all"], ["crop"],
    ))

    # 2. Compute mask: any non-zero (channels 1-9) → float mask (1,1,N,N)
    # Slice to exclude channel 0 (background)
    nodes.append(helper.make_node(
        "Slice", ["crop", "ch1_start", "ch1_end", "axes_all"], ["crop_ch1_9"],
    ))  # (1,9,N,N)
    nodes.append(helper.make_node(
        "ReduceSum", ["crop_ch1_9"], ["sum_across_ch"],
        axes=[1], keepdims=1,
    ))  # (1,1,N,N) - sum of channels 1-9
    nodes.append(helper.make_node(
        "Greater", ["sum_across_ch", "zero_f"], ["mask_bool"],
    ))
    nodes.append(helper.make_node(
        "Cast", ["mask_bool"], ["mask_f"], to=TensorProto.FLOAT,
    ))  # (1,1,N,N)

    # 3. Block-level tile the mask: kron(mask, ones(N,N))
    # Reshape (1,1,N,N) → (1,1,N,1,N,1)
    nodes.append(helper.make_node(
        "Reshape", ["mask_f", "mask_shape_inter"], ["mask_6d"],
    ))
    # Tile to (1,1,N,N,N,N)
    nodes.append(helper.make_node(
        "Tile", ["mask_6d", "mask_tile_rp"], ["mask_tiled_6d"],
    ))
    # Reshape to (1,1,N²,N²)
    nodes.append(helper.make_node(
        "Reshape", ["mask_tiled_6d", "mask_shape_out"], ["mask_9x9"],
    ))

    # 4. Tile input: (1,10,N,N) → (1,10,N²,N²)
    nodes.append(helper.make_node(
        "Gather", ["crop", "row_idx"], ["rows_expanded"],
        axis=2,
    ))  # (1,10,N²,N)
    nodes.append(helper.make_node(
        "Gather", ["rows_expanded", "col_idx"], ["expanded"],
        axis=3,
    ))  # (1,10,N²,N²)

    # 5. Apply mask: output = expanded * mask + ch0_onehot * (1-mask)
    #    ch0_onehot is a (1,10,1,1) tensor with 1 at ch0 and 0 elsewhere,
    #    tiled to (1,10,9,9).
    ch0_data = np.zeros((1, CHANNELS, 1, 1), dtype=np.float32)
    ch0_data[0, 0, 0, 0] = 1.0
    ch0_onehot = f32("ch0_onehot", ch0_data)
    init.append(ch0_onehot)

    ch0_tile = i64("ch0_tile", [1, 1, out_size, out_size])
    init.append(ch0_tile)

    nodes.append(helper.make_node(
        "Tile", ["ch0_onehot", "ch0_tile"], ["ch0_broadcast"],
    ))  # (1,10,9,9)

    # inv_mask_broadcast = (1 - mask_9x9) broadcast to (1,10,9,9)
    nodes.append(helper.make_node(
        "Sub", ["one_f", "mask_9x9"], ["inv_mask"],
    ))  # (1,1,9,9)

    # ch0_bg = ch0_broadcast * inv_mask → (1,10,9,9), only ch0 has values
    nodes.append(helper.make_node(
        "Mul", ["ch0_broadcast", "inv_mask"], ["ch0_bg"],
    ))

    # masked = expanded * mask_9x9 (broadcast applies mask to all channels)
    nodes.append(helper.make_node(
        "Mul", ["expanded", "mask_9x9"], ["masked"],
    ))  # (1,10,9,9)

    # out_9x9 = masked + ch0_bg (restores ch0 in masked-out blocks)
    nodes.append(helper.make_node(
        "Add", ["masked", "ch0_bg"], ["out_9x9"],
    ))

    # 6. Pad to 30x30
    nodes.append(helper.make_node(
        "Pad", ["out_9x9", "pads", "zero_f"], ["output"], mode="constant",
    ))

    inputs_val = [helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, CHANNELS, HEIGHT, WIDTH])]
    outputs_val = [helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, CHANNELS, HEIGHT, WIDTH])]
    graph = helper.make_graph(nodes, "block_mask", inputs_val, outputs_val,
                              initializer=init)
    return helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
        ir_version=IR_VERSION,
    )


def solve_block_mask(task: dict) -> Optional[onnx.ModelProto]:
    N = _detect(task)
    if N is None:
        return None
    return _build(N)
