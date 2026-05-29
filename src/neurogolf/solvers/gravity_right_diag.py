"""Solver: gravity_right_diag.

Task 004: Shift pixels right by 1 within each connected component,
except the bottom row and rightmost column of each component stay in place.
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
    """Detect if all examples follow the gravity-right-diag pattern."""
    import numpy as np
    examples = list(all_examples(task))
    if not examples:
        return False

    for ex in examples:
        inp = np.array(ex["input"])
        out = np.array(ex["output"])
        if inp.shape != out.shape:
            return False
        h, w = inp.shape
        if h < 2 or w < 2:
            return False

        visited = np.zeros((h, w), dtype=bool)
        pred = np.zeros_like(inp)

        for color in range(1, 10):
            ys, xs = np.where(inp == color)
            for y, x in zip(ys, xs):
                if visited[y, x]:
                    continue
                # BFS for connected component
                comp = []
                stack = [(y, x)]
                visited[y, x] = True
                while stack:
                    cy, cx = stack.pop()
                    comp.append((cy, cx))
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            ny, nx = cy + dy, cx + dx
                            if 0 <= ny < h and 0 <= nx < w \
                               and not visited[ny, nx] and inp[ny, nx] == color:
                                visited[ny, nx] = True
                                stack.append((ny, nx))
                if len(comp) < 2:
                    continue

                max_row = max(r for r, c in comp)
                max_col = max(c for r, c in comp)

                for r, c in comp:
                    if r == max_row:
                        pred[r, c] = color
                    elif c == max_col:
                        pred[r, c] = color
                    else:
                        nc = c + 1
                        if nc <= max_col:
                            pred[r, nc] = color
                        else:
                            pred[r, c] = color

        if not np.array_equal(pred, out):
            return False

    return True


def _build() -> onnx.ModelProto:
    def i64(name, data):
        return numpy_helper.from_array(np.array(data, dtype=np.int64), name)

    def f32(name, data):
        return numpy_helper.from_array(np.array(data, dtype=np.float32), name)

    # Row index weights for ArgMax: [0,1,2,...,29]
    row_weights_data = np.arange(HEIGHT, dtype=np.float32)
    col_weights_data = np.arange(WIDTH, dtype=np.float32)

    row_weights = f32("row_weights", row_weights_data.reshape(1, 1, HEIGHT, 1))
    col_weights = f32("col_weights", col_weights_data.reshape(1, 1, 1, WIDTH))
    zero_f = f32("zero_f", [0.0])
    one_f = f32("one_f", [1.0])

    # Padding for shift-right
    axes_4d = i64("axes_4d", [0, 1, 2, 3])

    init = [
        row_weights, col_weights, zero_f, one_f,
        axes_4d,
    ]

    nodes = []

    # 1. Per-channel row activity: ReduceMax over W → (1,10,30,1)
    nodes.append(helper.make_node(
        "ReduceMax", ["input"], ["row_activity"],
        axes=[3], keepdims=1,
    ))

    # 2. Weighted row activity → max_row per channel
    #    row_activity * row_weights → ArgMax over H → (1,10,1,1)
    nodes.append(helper.make_node(
        "Mul", ["row_activity", "row_weights"], ["weighted_rows"],
    ))
    nodes.append(helper.make_node(
        "ArgMax", ["weighted_rows"], ["max_row_idx"],
        axis=2, keepdims=1,
    ))  # (1,10,1,1) int64

    # 3. Per-channel col activity: ReduceMax over H → (1,10,1,30)
    nodes.append(helper.make_node(
        "ReduceMax", ["input"], ["col_activity"],
        axes=[2], keepdims=1,
    ))

    # 4. Weighted col activity → max_col per channel
    nodes.append(helper.make_node(
        "Mul", ["col_activity", "col_weights"], ["weighted_cols"],
    ))
    nodes.append(helper.make_node(
        "ArgMax", ["weighted_cols"], ["max_col_idx"],
        axis=3, keepdims=1,
    ))  # (1,10,1,1) int64

    # 5. Create row index grid (1,1,30,1)
    row_grid_f = f32("row_grid", row_weights_data.reshape(1, 1, HEIGHT, 1))
    col_grid_f = f32("col_grid", col_weights_data.reshape(1, 1, 1, WIDTH))
    init.append(row_grid_f)
    init.append(col_grid_f)

    # 6. Cast max indices to float
    nodes.append(helper.make_node(
        "Cast", ["max_row_idx"], ["max_row_f"], to=TensorProto.FLOAT,
    ))
    nodes.append(helper.make_node(
        "Cast", ["max_col_idx"], ["max_col_f"], to=TensorProto.FLOAT,
    ))

    # 7. bottom_mask[c,r,*] = (r == max_row[c])
    nodes.append(helper.make_node(
        "Equal", ["row_grid", "max_row_f"], ["bottom_mask_2d"],
    ))  # (1,10,30,1) bool

    # 8. right_mask[c,*,c] = (c == max_col[c])
    nodes.append(helper.make_node(
        "Equal", ["col_grid", "max_col_f"], ["right_mask_2d"],
    ))  # (1,10,1,30) bool

    # 9. Cast to float
    nodes.append(helper.make_node(
        "Cast", ["bottom_mask_2d"], ["bottom_f"], to=TensorProto.FLOAT,
    ))  # (1,10,30,1)
    nodes.append(helper.make_node(
        "Cast", ["right_mask_2d"], ["right_f"], to=TensorProto.FLOAT,
    ))  # (1,10,1,30)

    # 10. stay_mask = max(bottom_f, right_f) → (1,10,30,30)
    nodes.append(helper.make_node(
        "Max", ["bottom_f", "right_f"], ["stay_mask"],
    ))  # broadcast: (1,10,30,1) vs (1,10,1,30) → (1,10,30,30)

    # 11. shift_mask = 1 - stay_mask
    nodes.append(helper.make_node(
        "Sub", ["one_f", "stay_mask"], ["shift_mask"],
    ))

    # 12. Move non-stay pixels right by 1:
    #     move_source = input * shift_mask (at SOURCE positions)
    #     shifted = shift_right(move_source) (padding left with 0)
    #     output = input * stay_mask + shifted
    nodes.append(helper.make_node(
        "Mul", ["input", "shift_mask"], ["move_source"],
    ))  # (1,10,30,30) - pixels to move, at their source positions

    # Slice move_source[:,:,:,:-1] then Pad left with 0
    slice_w = i64("slice_w", [0, 0, 0, 0])
    slice_w_end = i64("slice_w_end", [1, CHANNELS, HEIGHT, WIDTH - 1])
    w_pad_left = i64("w_pad_left", [0, 0, 0, 1, 0, 0, 0, 0])
    init.extend([slice_w, slice_w_end, w_pad_left])

    nodes.append(helper.make_node(
        "Slice", ["move_source", "slice_w", "slice_w_end", "axes_4d"], ["move_no_last"],
    ))  # (1,10,30,29)
    nodes.append(helper.make_node(
        "Pad", ["move_no_last", "w_pad_left", "zero_f"], ["shifted"],
        mode="constant",
    ))  # (1,10,30,30)

    # 13. output_raw = input * stay_mask + shifted
    nodes.append(helper.make_node(
        "Mul", ["input", "stay_mask"], ["kept"],
    ))
    nodes.append(helper.make_node(
        "Add", ["kept", "shifted"], ["output_raw"],
    ))  # (1,10,30,30) - ch0 may have non-1 values

    # 14. Derive channel 0 from channels 1-9:
    #     fg_sum = ReduceSum(output_raw[:,1:], axis=1) → (1,1,30,30)
    #     ch0 = 1 - min(fg_sum, 1)
    ch1_start_sl = i64("ch1_start_sl", [0, 1, 0, 0])
    ch_end_sl = i64("ch_end_sl", [1, CHANNELS, HEIGHT, WIDTH])
    init.extend([ch1_start_sl, ch_end_sl])
    nodes.append(helper.make_node(
        "Slice", ["output_raw", "ch1_start_sl", "ch_end_sl", "axes_4d"], ["output_fg"],
    ))  # (1,9,30,30)
    nodes.append(helper.make_node(
        "ReduceSum", ["output_fg"], ["fg_sum"],
        axes=[1], keepdims=1,
    ))  # (1,1,30,30)
    nodes.append(helper.make_node(
        "Min", ["fg_sum", "one_f"], ["fg_clipped"],
    ))  # clip to [0,1)
    nodes.append(helper.make_node(
        "Sub", ["one_f", "fg_clipped"], ["ch0_new"],
    ))  # (1,1,30,30)
    # Compute valid-region mask from input (1 where input has any non-zero)
    nodes.append(helper.make_node(
        "ReduceSum", ["input"], ["input_mask"],
        axes=[1], keepdims=1,
    ))  # (1,1,30,30): 1 at valid positions, 0 at padding
    # Zero out channel 0 and foreground in padded regions
    nodes.append(helper.make_node(
        "Mul", ["ch0_new", "input_mask"], ["ch0_final"],
    ))
    nodes.append(helper.make_node(
        "Mul", ["output_fg", "input_mask"], ["output_fg_final"],
    ))
    nodes.append(helper.make_node(
        "Concat", ["ch0_final", "output_fg_final"], ["output"],
        axis=1,
    ))

    inputs_val = [helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, CHANNELS, HEIGHT, WIDTH])]
    outputs_val = [helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, CHANNELS, HEIGHT, WIDTH])]
    graph = helper.make_graph(nodes, "gravity_right_diag", inputs_val, outputs_val,
                              initializer=init)
    return helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
        ir_version=IR_VERSION,
    )


def solve_gravity_right_diag(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
