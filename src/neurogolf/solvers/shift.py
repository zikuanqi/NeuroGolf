"""Solver: output is input translated by a fixed (dr, dc) with zero-fill.

For constant-shape tasks the source rectangle and the "shifted-out" gap are
both known at solve-time, so the graph is:

  src      = Slice(input, source rows/cols)
  fill_*   = (1, 10, gap_h, gap_w) initializer — channel 0 set to 1.0 to
             encode "color 0" cells (NOT padding-empty cells; the scorer
             distinguishes those in the full-canvas comparison).
  framed   = Concat(fill, src, axis=H_or_W)
  output   = Pad(framed)  # zero-fill outside the H x W output grid.

We support pure horizontal or pure vertical shifts. Mixed (dr ≠ 0, dc ≠ 0)
would need two concats; not currently observed in the dataset.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from ..grids import CHANNELS, HEIGHT, WIDTH, all_examples

OPSET = 11
IR_VERSION = 8


def _detect(task: dict) -> tuple[int, int, int, int] | None:
    """Return (H, W, dr, dc) if every example is the same shift; else None."""
    examples = list(all_examples(task))
    if not examples:
        return None
    shapes_in = {(len(ex["input"]), len(ex["input"][0])) for ex in examples}
    shapes_out = {(len(ex["output"]), len(ex["output"][0])) for ex in examples}
    if len(shapes_in) != 1 or len(shapes_out) != 1:
        return None
    H, W = shapes_in.pop()
    OH, OW = shapes_out.pop()
    if (H, W) != (OH, OW) or H == 0 or W == 0:
        return None
    if H > HEIGHT or W > WIDTH:
        return None

    for dr in range(-(H - 1), H):
        for dc in range(-(W - 1), W):
            if dr == 0 and dc == 0:
                continue
            # Currently emit either a pure row or a pure column shift.
            if dr != 0 and dc != 0:
                continue
            ok = True
            for ex in examples:
                inp, out = ex["input"], ex["output"]
                for r in range(H):
                    for c in range(W):
                        sr, sc = r - dr, c - dc
                        if 0 <= sr < H and 0 <= sc < W:
                            if out[r][c] != inp[sr][sc]:
                                ok = False
                                break
                        else:
                            if out[r][c] != 0:
                                ok = False
                                break
                    if not ok:
                        break
                if not ok:
                    break
            if ok:
                return H, W, dr, dc
    return None


def _build(H: int, W: int, dr: int, dc: int) -> onnx.ModelProto:
    def i64(name: str, vals) -> onnx.TensorProto:
        return numpy_helper.from_array(np.array(vals, dtype=np.int64), name)

    # Source rectangle inside the input canvas.
    src_r_lo = max(0, -dr)
    src_r_hi = H - max(0, dr)
    src_c_lo = max(0, -dc)
    src_c_hi = W - max(0, dc)
    src_h = src_r_hi - src_r_lo
    src_w = src_c_hi - src_c_lo

    # Fill rectangle dimensions.
    if dr != 0:
        # Vertical shift: fill is a strip of rows.
        gap = abs(dr)
        fill_shape = [1, CHANNELS, gap, W]
        concat_axis = 2
        fill_before = dr > 0  # shift-down → fill on top.
    else:
        gap = abs(dc)
        fill_shape = [1, CHANNELS, H, gap]
        concat_axis = 3
        fill_before = dc > 0  # shift-right → fill on the left.

    fill_data = np.zeros(fill_shape, dtype=np.float32)
    fill_data[0, 0, :, :] = 1.0  # color-0 cells in the gap.
    fill_init = numpy_helper.from_array(fill_data, "fill")

    starts = i64("starts", [0, 0, src_r_lo, src_c_lo])
    ends = i64("ends", [1, CHANNELS, src_r_hi, src_c_hi])
    axes = i64("axes", [0, 1, 2, 3])

    # After concat: (1, CHANNELS, H, W) covering the full output grid.
    pads = i64("pads",
               [0, 0, 0, 0, 0, 0, HEIGHT - H, WIDTH - W])

    inputs_to_concat = (["fill", "src"] if fill_before else ["src", "fill"])
    nodes = [
        helper.make_node(
            "Slice", ["input", "starts", "ends", "axes"], ["src"],
            name="slice_src"),
        helper.make_node(
            "Concat", inputs_to_concat, ["framed"], axis=concat_axis,
            name="concat_fill"),
        helper.make_node(
            "Pad", ["framed", "pads"], ["output"], mode="constant",
            name="pad_to_canvas"),
    ]

    inputs = [helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, CHANNELS, HEIGHT, WIDTH])]
    outputs = [helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, CHANNELS, HEIGHT, WIDTH])]
    value_info = [
        helper.make_tensor_value_info(
            "src", TensorProto.FLOAT, [1, CHANNELS, src_h, src_w]),
        helper.make_tensor_value_info(
            "framed", TensorProto.FLOAT, [1, CHANNELS, H, W]),
    ]
    graph = helper.make_graph(
        nodes, "shift", inputs, outputs,
        initializer=[starts, ends, axes, pads, fill_init],
        value_info=value_info)
    return helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
        ir_version=IR_VERSION)


def solve_shift(task: dict) -> Optional[onnx.ModelProto]:
    spec = _detect(task)
    if spec is None:
        return None
    return _build(*spec)
