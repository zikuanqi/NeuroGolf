"""Solvers for tasks that are pure spatial reorderings of the input.

Only `Transpose` is currently emitted as a network. Flips and 180° rotation
detect their pattern in the example data but cannot be implemented as a simple
30-wide `Gather` because input grids are top-left-aligned inside the 30x30
canvas - reversing all 30 positions shifts the content to the opposite edge
instead of flipping it in place. Implementing a shape-aware flip would need
either dynamic shapes (forbidden) or a per-width gather selector chain. Left
for a later pass.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import onnx
from onnx import TensorProto, helper

from ..grids import HEIGHT, WIDTH, all_examples
from ..onnx_ops import OPSET_IMPORTS, IR_VERSION, finalize

REVERSE_30 = np.arange(29, -1, -1, dtype=np.int64)


def _check_same_shape(task: dict) -> bool:
    for ex in all_examples(task):
        if (not ex["input"] or not ex["output"] or
                len(ex["input"]) != len(ex["output"]) or
                len(ex["input"][0]) != len(ex["output"][0])):
            return False
    return True


def _check_transpose_shape(task: dict) -> bool:
    for ex in all_examples(task):
        i, o = ex["input"], ex["output"]
        if not i or not o:
            return False
        if len(i) != len(o[0]) or len(i[0]) != len(o):
            return False
    return True


def _is_flip_h(task: dict) -> bool:
    if not _check_same_shape(task):
        return False
    saw = False
    for ex in all_examples(task):
        for r, row in enumerate(ex["input"]):
            w = len(row)
            for c, v in enumerate(row):
                if ex["output"][r][w - 1 - c] != v:
                    return False
        saw = True
    return saw


def _is_flip_v(task: dict) -> bool:
    if not _check_same_shape(task):
        return False
    saw = False
    for ex in all_examples(task):
        h = len(ex["input"])
        for r, row in enumerate(ex["input"]):
            for c, v in enumerate(row):
                if ex["output"][h - 1 - r][c] != v:
                    return False
        saw = True
    return saw


def _is_rot180(task: dict) -> bool:
    if not _check_same_shape(task):
        return False
    saw = False
    for ex in all_examples(task):
        h, w = len(ex["input"]), len(ex["input"][0])
        for r, row in enumerate(ex["input"]):
            for c, v in enumerate(row):
                if ex["output"][h - 1 - r][w - 1 - c] != v:
                    return False
        saw = True
    return saw


def _is_transpose(task: dict) -> bool:
    if not _check_transpose_shape(task):
        return False
    saw = False
    for ex in all_examples(task):
        for r, row in enumerate(ex["input"]):
            for c, v in enumerate(row):
                if ex["output"][c][r] != v:
                    return False
        saw = True
    return saw


def _gather_reverse(axis: int) -> onnx.ModelProto:
    """Build a single-Gather model that reverses positions along `axis`."""
    indices = helper.make_tensor(
        "idx", TensorProto.INT64, [HEIGHT if axis == 2 else WIDTH],
        REVERSE_30.tolist(),
    )
    node = helper.make_node("Gather", ["input", "idx"], ["output"], axis=axis)
    return finalize([node], [indices])


def _gather_reverse_both() -> onnx.ModelProto:
    """Reverse rows then columns. Two Gathers with one shared index initializer."""
    indices = helper.make_tensor(
        "idx", TensorProto.INT64, [HEIGHT], REVERSE_30.tolist())
    nodes = [
        helper.make_node("Gather", ["input", "idx"], ["flipped_r"], axis=2),
        helper.make_node("Gather", ["flipped_r", "idx"], ["output"], axis=3),
    ]
    # Need to declare the intermediate value with known shape.
    flipped = helper.make_tensor_value_info(
        "flipped_r", TensorProto.FLOAT, [1, 10, HEIGHT, WIDTH])
    x = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 10, HEIGHT, WIDTH])
    y = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 10, HEIGHT, WIDTH])
    graph = helper.make_graph(
        nodes, "rot180", [x], [y], [indices], value_info=[flipped])
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=OPSET_IMPORTS)


def _transpose_model() -> onnx.ModelProto:
    node = helper.make_node("Transpose", ["input"], ["output"],
                            perm=[0, 1, 3, 2])
    return finalize([node], [])


def solve_flip_h(task: dict) -> Optional[onnx.ModelProto]:
    if _is_flip_h(task):
        return _gather_reverse(axis=3)
    return None


def solve_flip_v(task: dict) -> Optional[onnx.ModelProto]:
    if _is_flip_v(task):
        return _gather_reverse(axis=2)
    return None


def solve_rot180(task: dict) -> Optional[onnx.ModelProto]:
    if _is_rot180(task):
        return _gather_reverse_both()
    return None


def solve_transpose(task: dict) -> Optional[onnx.ModelProto]:
    if _is_transpose(task):
        return _transpose_model()
    return None
