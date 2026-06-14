"""Solver: replace each 2 with four diagonal-neighbour colours (task 266).

Every ``2`` is erased and its four diagonal neighbours are painted by a fixed
map (clipped to the grid)::

    up-left = 3   up-right = 6
    down-left = 8 down-right = 7

Each stamp is a diagonal shift of the ``2`` mask, confined to the grid by the
occupancy mask.
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
_MAP = {(-1, -1): 3, (-1, 1): 6, (1, -1): 8, (1, 1): 7}


def _ref(g: np.ndarray) -> Optional[np.ndarray]:
    H, W = g.shape
    out = g.copy()
    ys, xs = np.where(g == 2)
    if len(ys) == 0:
        return None
    for r, c in zip(ys, xs):
        out[r, c] = 0
        for (dr, dc), col in _MAP.items():
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W:
                out[nr, nc] = col
    return out


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


def _ev(k):
    v = np.zeros((1, CHANNELS, 1, 1), np.float32); v[0, k] = 1.0
    return v


def _build() -> onnx.ModelProto:
    n = helper.make_node
    e0me2 = np.zeros((1, CHANNELS, 1, 1), np.float32); e0me2[0, 0] = 1.0; e0me2[0, 2] = -1.0
    init = [
        numpy_helper.from_array(e0me2, "e0me2"),
        numpy_helper.from_array(_ev(3), "e3"),
        numpy_helper.from_array(_ev(6), "e6"),
        numpy_helper.from_array(_ev(8), "e8"),
        numpy_helper.from_array(_ev(7), "e7"),
        numpy_helper.from_array(np.array([2], np.int64), "c2"),
        numpy_helper.from_array(np.array([3], np.int64), "c3"),
        numpy_helper.from_array(np.array([1], np.int64), "ax1"),
        numpy_helper.from_array(np.array([2, 3], np.int64), "ax23"),
        # diagonal pads (only one corner padded per direction so a 30x30 slice realigns)
        numpy_helper.from_array(np.array([0, 0, 0, 0, 0, 0, 1, 1], np.int64), "padUL"),
        numpy_helper.from_array(np.array([0, 0, 0, 1, 0, 0, 1, 0], np.int64), "padUR"),
        numpy_helper.from_array(np.array([0, 0, 1, 0, 0, 0, 0, 1], np.int64), "padDL"),
        numpy_helper.from_array(np.array([0, 0, 1, 1, 0, 0, 0, 0], np.int64), "padDR"),
        numpy_helper.from_array(np.array([1, 1], np.int64), "stUL"), numpy_helper.from_array(np.array([HEIGHT + 1, WIDTH + 1], np.int64), "enUL"),
        numpy_helper.from_array(np.array([1, 0], np.int64), "stUR"), numpy_helper.from_array(np.array([HEIGHT + 1, WIDTH], np.int64), "enUR"),
        numpy_helper.from_array(np.array([0, 1], np.int64), "stDL"), numpy_helper.from_array(np.array([HEIGHT, WIDTH + 1], np.int64), "enDL"),
        numpy_helper.from_array(np.array([0, 0], np.int64), "stDR"), numpy_helper.from_array(np.array([HEIGHT, WIDTH], np.int64), "enDR"),
    ]
    nodes = [
        n("ReduceSum", ["input"], ["occ"], axes=[1], keepdims=1),
        n("Slice", ["input", "c2", "c3", "ax1"], ["is2"]),
        # remove the 2s
        n("Mul", ["is2", "e0me2"], ["clr"]),
        n("Add", ["input", "clr"], ["base"]),
    ]
    for tag, pad, st, en, col in (("UL", "padUL", "stUL", "enUL", "e3"),
                                  ("UR", "padUR", "stUR", "enUR", "e6"),
                                  ("DL", "padDL", "stDL", "enDL", "e8"),
                                  ("DR", "padDR", "stDR", "enDR", "e7")):
        nodes += [
            n("Pad", ["is2", pad], [f"p{tag}"]),
            n("Slice", [f"p{tag}", st, en, "ax23"], [f"sh{tag}"]),
            n("Mul", [f"sh{tag}", "occ"], [f"m{tag}"]),
            n("Mul", [f"m{tag}", col], [f"L{tag}"]),
        ]
    nodes += [
        n("Add", ["mUL", "mUR"], ["a1"]), n("Add", ["mDL", "mDR"], ["a2"]),
        n("Add", ["a1", "a2"], ["stampAny"]),
        n("Add", ["LUL", "LUR"], ["c1"]), n("Add", ["LDL", "LDR"], ["c2c"]),
        n("Add", ["c1", "c2c"], ["stampColor"]),
        n("Mul", ["base", "stampAny"], ["overlap"]),
        n("Sub", ["base", "overlap"], ["baseClr"]),  # base*(1-stampAny)
        n("Add", ["baseClr", "stampColor"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "diag_corner_stamp",
                              [helper.make_tensor_value_info("input", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              [helper.make_tensor_value_info("output", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_diag_corner_stamp(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
