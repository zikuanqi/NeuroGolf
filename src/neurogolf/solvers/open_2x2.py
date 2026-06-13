"""Solver: keep only cells inside a solid 2x2+ block (task 193).

A single colour appears both as solid rectangular blocks and as scattered
single-cell noise.  The output keeps every cell that belongs to a filled 2x2
square and removes the rest (back to background)::

    . X . . X X X        . . . . X X X
    . . . . X X X   ->   . . . . X X X     (isolated X removed,
    . . X X . . .        . . X X . . .      2x2 blocks kept)
    . . X X . . .        . . X X . . .

This is a morphological **opening** with a 2x2 structuring element: erosion
``full2 = -MaxPool(-mask)`` (a 2x2 AND, with bottom-right zero-padding) then
dilation ``keep = MaxPool(full2)`` (a 2x2 OR, with top-left zero-padding).
Removed colour cells become background; padding stays empty.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from ..grids import CHANNELS, HEIGHT, WIDTH, all_examples

OPSET = 11
IR_VERSION = 8
F = TensorProto.FLOAT


def _ref(g: np.ndarray) -> Optional[np.ndarray]:
    H, W = g.shape
    m = (g != 0).astype(int)
    full = np.zeros((H, W), int)
    full[:H - 1, :W - 1] = m[:H - 1, :W - 1] & m[1:, :W - 1] & m[:H - 1, 1:] & m[1:, 1:]
    keep = np.zeros((H, W), bool)
    for dr in (0, 1):
        for dc in (0, 1):
            src = np.zeros((H, W), int)
            src[dr:, dc:] = full[:H - dr if dr else H, :W - dc if dc else W]
            keep |= src > 0
    out = np.where(keep, g, 0)
    return out if not np.array_equal(out, g) else None


def _detect(task: dict) -> bool:
    saw = False
    for ex in all_examples(task):
        i, o = ex["input"], ex["output"]
        if not i or not i[0] or len(i) > HEIGHT or len(i[0]) > WIDTH:
            continue
        r = _ref(np.array(i))
        if r is None or not np.array_equal(r, np.array(o)):
            return False
        saw = True
    return saw


def _build() -> onnx.ModelProto:
    n = helper.make_node
    e0 = np.zeros((1, CHANNELS, 1, 1), np.float32); e0[0, 0] = 1.0
    init = [
        numpy_helper.from_array(e0, "e0"),
        numpy_helper.from_array(np.array([0], np.int64), "c0s"),
        numpy_helper.from_array(np.array([1], np.int64), "c1e"),
        numpy_helper.from_array(np.array([1], np.int64), "ax1"),
        numpy_helper.from_array(np.array([0, 0, 0, 0, 0, 0, 1, 1], np.int64), "padBR"),
        numpy_helper.from_array(np.array([0, 0, 1, 1, 0, 0, 0, 0], np.int64), "padTL"),
    ]
    nodes = [
        n("ReduceSum", ["input"], ["occ"], axes=[1], keepdims=1),
        n("Slice", ["input", "c0s", "c1e", "ax1"], ["is0"]),
        n("Sub", ["occ", "is0"], ["mask"]),
        # erosion: 2x2 AND via -MaxPool(-mask), padded bottom-right so output stays 30x30
        n("Pad", ["mask", "padBR"], ["maskBR"], mode="constant"),
        n("Neg", ["maskBR"], ["nmask"]),
        n("MaxPool", ["nmask"], ["mp1"], kernel_shape=[2, 2], strides=[1, 1]),
        n("Neg", ["mp1"], ["full2"]),
        # dilation: 2x2 OR via MaxPool, padded top-left
        n("Pad", ["full2", "padTL"], ["fullTL"], mode="constant"),
        n("MaxPool", ["fullTL"], ["keep"], kernel_shape=[2, 2], strides=[1, 1]),
        # reassemble: kept colours stay, removed colours -> background
        n("Mul", ["is0", "e0"], ["is0e0"]),
        n("Sub", ["input", "is0e0"], ["nonbg"]),
        n("Mul", ["nonbg", "keep"], ["keptC"]),
        n("Mul", ["mask", "keep"], ["mk"]),
        n("Sub", ["occ", "mk"], ["ch0v"]),
        n("Mul", ["ch0v", "e0"], ["ch0"]),
        n("Add", ["keptC", "ch0"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "open_2x2",
                              [helper.make_tensor_value_info("input", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              [helper.make_tensor_value_info("output", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_open_2x2(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
