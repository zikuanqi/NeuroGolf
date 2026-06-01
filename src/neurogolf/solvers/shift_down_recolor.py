"""Solver: shift down + recolor (8→2) using Conv2d + background fix.

Three-step pipeline:
  1. Conv3x3 shifts color-8 pixels down and recolors to 2
  2. Constant provides background (channel 0 = 1 everywhere)
  3. Add → combined (overlap: both ch0 and ch2 active at shift positions)
  4. Conv1x1 fix: ch0 -= ch2, canceling background where shifted pixels exist
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from ..grids import CHANNELS, HEIGHT, WIDTH, all_examples

OPSET = 10
IR_VERSION = 10


def _to_init(name: str, arr: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(arr.astype(np.float32), name)


def solve_shift_down_recolor(task: dict) -> Optional[onnx.ModelProto]:
    """Detect: shift color 8 down by 1, output color 2."""
    examples = list(all_examples(task))
    if not examples:
        return None

    for ex in examples:
        inp, out = ex["input"], ex["output"]
        h, w = len(inp), len(inp[0])
        pred = [[0] * w for _ in range(h)]
        for r in range(h):
            for c in range(w):
                if inp[r][c] == 8 and r + 1 < h:
                    pred[r + 1][c] = 2
        if pred != out:
            return None

    # Conv shift: 8 → 2, shifted down
    w_shift = np.zeros((CHANNELS, CHANNELS, 3, 3), dtype=np.float32)
    w_shift[2, 8, 0, 1] = 1.0

    # Background constant: channel 0 = 1 everywhere on 30x30 canvas
    bg = np.zeros((1, CHANNELS, HEIGHT, WIDTH), dtype=np.float32)
    bg[0, 0, :, :] = 1.0

    # Conv fix (1x1): channel 0 = channel_0 - channel_2; channel 2 = channel_2
    w_fix = np.zeros((CHANNELS, CHANNELS, 1, 1), dtype=np.float32)
    w_fix[0, 0, 0, 0] = 1.0
    w_fix[0, 2, 0, 0] = -1.0
    w_fix[2, 2, 0, 0] = 1.0

    nodes = [
        helper.make_node(
            "Conv", ["input", "w_shift"], ["shifted"],
            kernel_shape=[3, 3], pads=[1, 1, 1, 1],
        ),
        helper.make_node(
            "Add", ["shifted", "bg"], ["combined"],
        ),
        helper.make_node(
            "Conv", ["combined", "w_fix"], ["output"],
            kernel_shape=[1, 1],
        ),
    ]

    inits = [_to_init("w_shift", w_shift),
             _to_init("bg", bg),
             _to_init("w_fix", w_fix)]

    x = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, CHANNELS, HEIGHT, WIDTH])
    y = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, CHANNELS, HEIGHT, WIDTH])

    graph = helper.make_graph(nodes, "shift_down_recolor", [x], [y], inits)
    return helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", OPSET)],
        ir_version=IR_VERSION)