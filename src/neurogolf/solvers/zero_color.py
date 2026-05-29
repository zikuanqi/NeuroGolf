"""Solver: output = input with a specific color zeroed out.

For tasks where every example has the same shape and the output is identical
to the input except all cells of a fixed color C become color 0.

This is a simple channel-wise mask: output = input * mask, where mask[C] = 0
and mask[i] = 1 for i != C.
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
    """Return the color C to zero out, or None."""
    examples = list(all_examples(task))
    if not examples:
        return None
    target = None
    for ex in examples:
        inp, out = ex["input"], ex["output"]
        # Must have same shape
        if len(inp) != len(out) or len(inp[0]) != len(out[0]):
            return None
        diff = {}
        for r in range(len(inp)):
            for c in range(len(inp[0])):
                if inp[r][c] != out[r][c]:
                    if inp[r][c] not in diff:
                        diff[inp[r][c]] = out[r][c]
                    elif diff[inp[r][c]] != out[r][c]:
                        return None
        # Must be exactly one color→0 mapping
        if len(diff) != 1:
            return None
        removed = list(diff.keys())[0]
        if diff[removed] != 0:
            return None
        if target is None:
            target = removed
        elif target != removed:
            return None
    return target


def _build(target_color: int) -> onnx.ModelProto:
    mask = np.ones((1, CHANNELS, 1, 1), dtype=np.float32)
    mask[0, target_color, 0, 0] = 0.0

    init = [
        numpy_helper.from_array(mask, name="mask"),
    ]

    nodes = [
        helper.make_node(
            "Mul", ["input", "mask"], ["output"],
            name="zero_color"),
    ]

    inputs = [helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, CHANNELS, HEIGHT, WIDTH])]
    outputs = [helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, CHANNELS, HEIGHT, WIDTH])]
    graph = helper.make_graph(
        nodes, "zero_color", inputs, outputs, initializer=init)
    return helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
        ir_version=IR_VERSION)


def solve_zero_color(task: dict) -> Optional[onnx.ModelProto]:
    target = _detect(task)
    if target is None:
        return None
    return _build(target)