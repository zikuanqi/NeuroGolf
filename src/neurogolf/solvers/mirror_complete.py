"""Solver: complete a grid by mirroring it across a horizontal or vertical axis.

Half the grid holds a pattern and the other half is erased to background 0; the
output restores the missing half as the mirror image of the present one
(tasks 113 and 385):

    output[cell] = input[cell]            if input[cell] != 0
                 = input[mirror(cell)]    otherwise

The mirror is the shape-aware flip from `shape_aware_flip` (so it works on
variable, top-left-aligned grids), and the merge keeps the original colour
wherever the input is non-background, filling only the erased cells.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from ..grids import CHANNELS, HEIGHT, WIDTH, all_examples
from .shape_aware_flip import _flip_along

OPSET = 11
IR_VERSION = 8


def _mirror_fill(grid, axis: int):
    g = np.array(grid)
    h, w = g.shape
    t = np.flipud(g) if axis == 2 else np.fliplr(g)
    out = np.where(g != 0, g, t)
    return out.tolist()


def _detect(task: dict) -> Optional[int]:
    """Return spatial axis (2 = vertical mirror, 3 = horizontal) or None."""
    examples = list(all_examples(task))
    if not examples:
        return None
    for axis in (2, 3):
        ok = True
        filled = False
        for ex in examples:
            inp, out = ex["input"], ex["output"]
            if len(inp) != len(out) or len(inp[0]) != len(out[0]):
                ok = False
                break
            # non-background input cells must be preserved
            preserved = all(
                inp[r][c] == 0 or inp[r][c] == out[r][c]
                for r in range(len(inp)) for c in range(len(inp[0])))
            if not preserved:
                ok = False
                break
            if _mirror_fill(inp, axis) != out:
                ok = False
                break
            if any(inp[r][c] == 0 and out[r][c] != 0
                   for r in range(len(inp)) for c in range(len(inp[0]))):
                filled = True
        if ok and filled:
            return axis
    return None


def _build(axis: int) -> onnx.ModelProto:
    def f32(name, arr):
        return numpy_helper.from_array(arr.astype(np.float32), name)

    init: list = []
    value_info: list = []
    # shape-aware flip of the input -> "flipped"
    nodes = _flip_along(axis, "m", "input", "flipped", init, value_info)

    # keep input where it is non-background, else take the flipped value.
    # erased cells are exactly where the input's background channel (0) is on.
    nodes += [
        helper.make_node("Slice", ["input", "c0_s", "c0_e", "c_ax", "c_st"],
                         ["bg_ch"]),
        helper.make_node("Sub", ["one_c", "bg_ch"], ["occ"]),     # [1,1,H,W]
        # input restricted to occupied cells (drops the background bit there)
        helper.make_node("Mul", ["input", "occ"], ["kept"]),
        # flip restricted to the erased cells
        helper.make_node("Mul", ["flipped", "bg_ch"], ["fill"]),
        helper.make_node("Add", ["kept", "fill"], ["output"]),
    ]
    init += [
        numpy_helper.from_array(np.array([0], dtype=np.int64), "c0_s"),
        numpy_helper.from_array(np.array([1], dtype=np.int64), "c0_e"),
        numpy_helper.from_array(np.array([1], dtype=np.int64), "c_ax"),
        numpy_helper.from_array(np.array([1], dtype=np.int64), "c_st"),
        f32("one_c", np.array([1.0])),
    ]
    value_info += [
        helper.make_tensor_value_info("bg_ch", TensorProto.FLOAT,
                                      [1, 1, HEIGHT, WIDTH]),
        helper.make_tensor_value_info("occ", TensorProto.FLOAT,
                                      [1, 1, HEIGHT, WIDTH]),
        helper.make_tensor_value_info("kept", TensorProto.FLOAT,
                                      [1, CHANNELS, HEIGHT, WIDTH]),
        helper.make_tensor_value_info("fill", TensorProto.FLOAT,
                                      [1, CHANNELS, HEIGHT, WIDTH]),
    ]

    inputs = [helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, CHANNELS, HEIGHT, WIDTH])]
    outputs = [helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, CHANNELS, HEIGHT, WIDTH])]
    graph = helper.make_graph(nodes, "mirror_complete", inputs, outputs,
                              initializer=init, value_info=value_info)
    return helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
        ir_version=IR_VERSION)


def solve_mirror_complete(task: dict) -> Optional[onnx.ModelProto]:
    axis = _detect(task)
    if axis is None:
        return None
    return _build(axis)
