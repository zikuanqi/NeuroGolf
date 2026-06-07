"""Solver: complete an L-tromino into a 2x2 square (task 81).

Each shape is three cells of a 2x2 block of a single colour; the missing fourth
corner is filled with ``1``::

    8 .      ->   8 1
    8 8           8 8

Rule (purely local, radius 1): a background cell becomes ``1`` iff, for one of the
four diagonal directions, its vertical neighbour, horizontal neighbour and the
diagonal cell between them are all non-background (the other three corners of a
2x2 square).

Build: ``nz`` = real non-background mask; four Pad+Slice shifts (above/below/
left/right) plus four diagonal compositions; OR the four corner conjunctions;
paint ``e_1`` where a background cell meets the completion mask.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from ..grids import CHANNELS, HEIGHT, WIDTH, all_examples

OPSET = 11
IR_VERSION = 8
FULL = [1, CHANNELS, HEIGHT, WIDTH]


def _ref(g: np.ndarray) -> Optional[np.ndarray]:
    nz = g != 0
    H, W = g.shape

    def sh(m, dr, dc):
        r = np.zeros_like(m)
        rs = slice(max(0, dr), H + min(0, dr)); rd = slice(max(0, -dr), H + min(0, -dr))
        cs = slice(max(0, dc), W + min(0, dc)); cd = slice(max(0, -dc), W + min(0, -dc))
        r[rd, cd] = m[rs, cs]; return r

    comp = np.zeros((H, W), bool)
    for dr, dc in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        comp |= sh(nz, dr, 0) & sh(nz, 0, dc) & sh(nz, dr, dc)
    out = g.copy()
    out[(g == 0) & comp] = 1
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
    F = TensorProto.FLOAT
    n = helper.make_node

    e1 = np.zeros((1, CHANNELS, 1, 1), np.float32); e1[0, 1] = 1.0
    init = [
        numpy_helper.from_array(e1, "e1"),
        numpy_helper.from_array(np.array(1.0, np.float32), "one"),
        numpy_helper.from_array(np.array([0], np.int64), "c0s"),
        numpy_helper.from_array(np.array([1], np.int64), "c1e"),
        numpy_helper.from_array(np.array([1], np.int64), "ax1"),
        numpy_helper.from_array(np.array([2], np.int64), "ax2"),
        numpy_helper.from_array(np.array([3], np.int64), "ax3"),
        numpy_helper.from_array(np.array([0], np.int64), "z0"),
        numpy_helper.from_array(np.array([1], np.int64), "one_i"),
        numpy_helper.from_array(np.array([HEIGHT], np.int64), "rH"),
        numpy_helper.from_array(np.array([HEIGHT + 1], np.int64), "rH1"),
        numpy_helper.from_array(np.array([WIDTH], np.int64), "wW"),
        numpy_helper.from_array(np.array([WIDTH + 1], np.int64), "wW1"),
        numpy_helper.from_array(np.array([0, 0, 1, 0, 0, 0, 0, 0], np.int64), "pad_top"),
        numpy_helper.from_array(np.array([0, 0, 0, 0, 0, 0, 1, 0], np.int64), "pad_bot"),
        numpy_helper.from_array(np.array([0, 0, 0, 1, 0, 0, 0, 0], np.int64), "pad_left"),
        numpy_helper.from_array(np.array([0, 0, 0, 0, 0, 0, 0, 1], np.int64), "pad_right"),
    ]

    def shift_above(x, o):  # o[r]=x[r-1]
        return [n("Pad", [x, "pad_top"], [o + "_p"], mode="constant"),
                n("Slice", [o + "_p", "z0", "rH", "ax2"], [o])]

    def shift_below(x, o):  # o[r]=x[r+1]
        return [n("Pad", [x, "pad_bot"], [o + "_p"], mode="constant"),
                n("Slice", [o + "_p", "one_i", "rH1", "ax2"], [o])]

    def shift_left(x, o):   # o[c]=x[c-1]
        return [n("Pad", [x, "pad_left"], [o + "_p"], mode="constant"),
                n("Slice", [o + "_p", "z0", "wW", "ax3"], [o])]

    def shift_right(x, o):  # o[c]=x[c+1]
        return [n("Pad", [x, "pad_right"], [o + "_p"], mode="constant"),
                n("Slice", [o + "_p", "one_i", "wW1", "ax3"], [o])]

    nodes = [
        n("ReduceSum", ["input"], ["content"], axes=[1], keepdims=1),
        n("Slice", ["input", "c0s", "c1e", "ax1"], ["bg"]),
        n("Sub", ["content", "bg"], ["nz"]),
    ]
    nodes += shift_above("nz", "A")
    nodes += shift_below("nz", "B")
    nodes += shift_left("nz", "L")
    nodes += shift_right("nz", "R")
    # diagonals by composing
    nodes += shift_right("B", "BR")
    nodes += shift_left("B", "BL")
    nodes += shift_right("A", "AR")
    nodes += shift_left("A", "AL")
    # four corner conjunctions
    nodes += [
        n("Mul", ["B", "R"], ["c1a"]), n("Mul", ["c1a", "BR"], ["c1"]),
        n("Mul", ["B", "L"], ["c2a"]), n("Mul", ["c2a", "BL"], ["c2"]),
        n("Mul", ["A", "R"], ["c3a"]), n("Mul", ["c3a", "AR"], ["c3"]),
        n("Mul", ["A", "L"], ["c4a"]), n("Mul", ["c4a", "AL"], ["c4"]),
        n("Max", ["c1", "c2", "c3", "c4"], ["comp"]),
        n("Mul", ["bg", "comp"], ["paint"]),
        n("Sub", ["one", "paint"], ["inv"]),
        n("Mul", ["input", "inv"], ["kept"]),
        n("Mul", ["e1", "paint"], ["addc"]),
        n("Add", ["kept", "addc"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "square_complete",
                              [helper.make_tensor_value_info("input", F, FULL)],
                              [helper.make_tensor_value_info("output", F, FULL)],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_square_complete(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
