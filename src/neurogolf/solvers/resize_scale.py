"""Solver: nearest-neighbor upscaling with variable input shapes.

Companion to `kron_scale`. That one bakes the input geometry into the graph
via fixed `Gather` indices, so every example must share the same H, W. This
solver instead uses `Resize` with constant scales, which works for any
content size that fits inside the canvas after scaling.

Pipeline (for scale N):

  Slice(input, 0:HEIGHT/N, 0:WIDTH/N) → (1, 10, HEIGHT/N, WIDTH/N)
  Resize(..., scales=[1, 1, N, N])    → (1, 10, HEIGHT, WIDTH)

Padding cells (input has nothing past the actual H, W) stay zero through
both ops, so the canvas alignment is preserved.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from ..grids import CHANNELS, HEIGHT, WIDTH, all_examples

OPSET = 11
IR_VERSION = 8


def _detect(task: dict) -> int | None:
    """Return N if every example is an N-times nearest upscale; else None."""
    examples = list(all_examples(task))
    if not examples:
        return None
    for N in (2, 3):
        if HEIGHT % N != 0 or WIDTH % N != 0:
            continue
        ok = True
        for ex in examples:
            i, o = ex["input"], ex["output"]
            if not i or not o:
                ok = False
                break
            ih, iw = len(i), len(i[0])
            if len(o) != N * ih or len(o[0]) != N * iw:
                ok = False
                break
            if N * ih > HEIGHT or N * iw > WIDTH:
                ok = False
                break
            for r in range(N * ih):
                row_in = i[r // N]
                row_out = o[r]
                for c in range(N * iw):
                    if row_out[c] != row_in[c // N]:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            return N
    return None


def _build(N: int) -> onnx.ModelProto:
    src = HEIGHT // N

    def i64(name: str, vals: list[int]) -> onnx.TensorProto:
        return numpy_helper.from_array(np.array(vals, dtype=np.int64), name)

    starts_src = i64("starts_src", [0, 0, 0, 0])
    ends_src = i64("ends_src", [1, CHANNELS, src, src])
    axes_full = i64("axes_full", [0, 1, 2, 3])
    scales = numpy_helper.from_array(
        np.array([1.0, 1.0, float(N), float(N)], dtype=np.float32), "scales")
    # `roi` is ignored when coordinate_transformation_mode != "tf_crop_and_resize",
    # but ORT still requires a non-empty input tensor in that slot. Spec wants
    # shape [2 * rank], so we provide an identity ROI [0,0,0,0,1,1,1,1]. The 8
    # floats add 8 params; the values don't affect output.
    roi = numpy_helper.from_array(
        np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.float32), "roi")

    nodes = [
        helper.make_node(
            "Slice",
            ["input", "starts_src", "ends_src", "axes_full"],
            ["src_patch"], name="slice_src",
        ),
        helper.make_node(
            "Resize", ["src_patch", "roi", "scales"], ["output"],
            mode="nearest", coordinate_transformation_mode="asymmetric",
            nearest_mode="floor", name="upscale",
        ),
    ]
    inputs = [helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, CHANNELS, HEIGHT, WIDTH])]
    outputs = [helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, CHANNELS, HEIGHT, WIDTH])]
    value_info = [helper.make_tensor_value_info(
        "src_patch", TensorProto.FLOAT, [1, CHANNELS, src, src])]
    graph = helper.make_graph(
        nodes, "resize_scale", inputs, outputs,
        initializer=[starts_src, ends_src, axes_full, scales, roi],
        value_info=value_info,
    )
    return helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
        ir_version=IR_VERSION,
    )


def solve_resize_scale(task: dict) -> Optional[onnx.ModelProto]:
    N = _detect(task)
    if N is None:
        return None
    return _build(N)
