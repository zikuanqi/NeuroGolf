"""Solver: output is a constant-size, constant-offset slice of the input.

When every example has output equal to `input[r0:r0+H, c0:c0+W]` for fixed
`(r0, c0, H, W)`, we just `Slice` that region and `Pad` it back to the 30x30
canvas. No shape detection or marker logic is needed; the bounds are
compile-time constants.
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
    """Return (r0, c0, H, W) if every example agrees on a fixed crop."""
    examples = list(all_examples(task))
    if not examples:
        return None

    # Output must have a constant size across examples.
    h = len(examples[0]["output"])
    if h == 0 or h > HEIGHT:
        return None
    w = len(examples[0]["output"][0])
    if w == 0 or w > WIDTH:
        return None
    for ex in examples:
        if len(ex["output"]) != h or len(ex["output"][0]) != w:
            return None

    # Search every (r0, c0) that fits inside the smallest input.
    min_ih = min(len(ex["input"]) for ex in examples)
    min_iw = min(len(ex["input"][0]) for ex in examples if ex["input"])
    if min_ih < h or min_iw < w:
        return None

    for r0 in range(min_ih - h + 1):
        for c0 in range(min_iw - w + 1):
            ok = True
            for ex in examples:
                for r in range(h):
                    for c in range(w):
                        if ex["input"][r0 + r][c0 + c] != ex["output"][r][c]:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if ok:
                return r0, c0, h, w
    return None


def _build(r0: int, c0: int, h: int, w: int) -> onnx.ModelProto:
    def i64(name, data):
        return numpy_helper.from_array(np.array(data, dtype=np.int64), name)

    starts = i64("starts", [r0, c0])
    ends = i64("ends", [r0 + h, c0 + w])
    axes = i64("axes", [2, 3])
    pads = i64("pads", [0, 0, 0, 0, 0, 0, HEIGHT - h, WIDTH - w])

    nodes = [
        helper.make_node(
            "Slice", ["input", "starts", "ends", "axes"], ["crop"],
            name="crop_slice",
        ),
        helper.make_node(
            "Pad", ["crop", "pads"], ["output"], mode="constant",
            name="pad_to_canvas",
        ),
    ]

    inputs = [helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, CHANNELS, HEIGHT, WIDTH])]
    outputs = [helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, CHANNELS, HEIGHT, WIDTH])]
    value_info = [helper.make_tensor_value_info(
        "crop", TensorProto.FLOAT, [1, CHANNELS, h, w])]
    graph = helper.make_graph(
        nodes, "static_crop", inputs, outputs,
        initializer=[starts, ends, axes, pads],
        value_info=value_info,
    )
    return helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
        ir_version=IR_VERSION,
    )


def solve_static_crop(task: dict) -> Optional[onnx.ModelProto]:
    spec = _detect(task)
    if spec is None:
        return None
    return _build(*spec)
