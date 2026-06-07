"""Solver: downward-opening staple drops a marker to the floor (task 126).

Each shape is a ``∪`` rotated to open downward -- a top bar with two legs::

    6 6 6
    6 . 6

i.e. a background cell whose up / left / right neighbours are all non-background.
For every column that contains such a "pocket", a ``4`` is written into the
bottom row of the grid (the marker falls straight down onto the floor).

Build: ``nz`` = real non-background mask; ``bg`` = channel-0 (real background);
``pocket = bg * up(nz) * left(nz) * right(nz)`` via Pad+Slice shifts; reduce over
rows for the per-column flag, intersect with a one-hot of the last real row, and
paint ``e_4`` there.
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
    H, W = g.shape
    nz = g != 0
    up = np.zeros_like(nz); up[1:] = nz[:-1]
    left = np.zeros_like(nz); left[:, 1:] = nz[:, :-1]
    right = np.zeros_like(nz); right[:, :-1] = nz[:, 1:]
    pocket = (g == 0) & up & left & right
    if not pocket.any():
        return None
    colhas = pocket.any(axis=0)
    out = g.copy()
    out[H - 1, colhas] = 4
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

    row_idx = np.arange(HEIGHT, dtype=np.float32).reshape(1, 1, HEIGHT, 1)
    e4 = np.zeros((1, CHANNELS, 1, 1), np.float32); e4[0, 4] = 1.0
    init = [
        numpy_helper.from_array(row_idx, "row_idx"),
        numpy_helper.from_array(e4, "e4"),
        numpy_helper.from_array(np.array(0.5, np.float32), "half"),
        numpy_helper.from_array(np.array(1.0, np.float32), "one"),
        numpy_helper.from_array(np.array([0], np.int64), "c0s"),
        numpy_helper.from_array(np.array([1], np.int64), "c1e"),
        numpy_helper.from_array(np.array([1], np.int64), "ax1"),
        # pad specs (NCHW begin+end): top / left / right shifts
        numpy_helper.from_array(np.array([0, 0, 1, 0, 0, 0, 0, 0], np.int64), "pad_top"),
        numpy_helper.from_array(np.array([0, 0, 0, 1, 0, 0, 0, 0], np.int64), "pad_left"),
        numpy_helper.from_array(np.array([0, 0, 0, 0, 0, 0, 0, 1], np.int64), "pad_right"),
        numpy_helper.from_array(np.array([0], np.int64), "z0"),
        numpy_helper.from_array(np.array([HEIGHT], np.int64), "rH"),
        numpy_helper.from_array(np.array([WIDTH], np.int64), "wW"),
        numpy_helper.from_array(np.array([1], np.int64), "one_i"),
        numpy_helper.from_array(np.array([WIDTH + 1], np.int64), "rW1"),
        numpy_helper.from_array(np.array([2], np.int64), "ax2"),
        numpy_helper.from_array(np.array([3], np.int64), "ax3"),
    ]

    nodes = [
        # content = real-cell mask (1 where any channel set); ch0 = real background
        n("ReduceSum", ["input"], ["content"], axes=[1], keepdims=1),
        n("Slice", ["input", "c0s", "c1e", "ax1"], ["bg"]),               # (1,1,H,W)
        n("Sub", ["content", "bg"], ["nz"]),                              # real non-bg
        # neighbour shifts of nz
        n("Pad", ["nz", "pad_top"], ["nz_pt"], mode="constant"),
        n("Slice", ["nz_pt", "z0", "rH", "ax2"], ["up_nz"]),              # up neighbour
        n("Pad", ["nz", "pad_left"], ["nz_pl"], mode="constant"),
        n("Slice", ["nz_pl", "z0", "wW", "ax3"], ["left_nz"]),            # left neighbour
        n("Pad", ["nz", "pad_right"], ["nz_pr"], mode="constant"),
        n("Slice", ["nz_pr", "one_i", "rW1", "ax3"], ["right_nz"]),       # right neighbour
        # pocket = bg & up & left & right
        n("Mul", ["bg", "up_nz"], ["p1"]),
        n("Mul", ["p1", "left_nz"], ["p2"]),
        n("Mul", ["p2", "right_nz"], ["pocket"]),
        n("ReduceMax", ["pocket"], ["colhas"], axes=[2], keepdims=1),     # (1,1,1,W)
        # bottom real row one-hot
        n("ReduceMax", ["content"], ["realrow"], axes=[3], keepdims=1),   # (1,1,H,1)
        n("Mul", ["row_idx", "realrow"], ["rrow"]),
        n("ReduceMax", ["rrow"], ["rlast"], axes=[2], keepdims=1),        # (1,1,1,1)
        n("Sub", ["row_idx", "rlast"], ["rd"]),
        n("Abs", ["rd"], ["ard"]),
        n("Less", ["ard", "half"], ["isb_b"]),
        n("Cast", ["isb_b"], ["isbottom"], to=F),                         # (1,1,H,1)
        n("Mul", ["isbottom", "colhas"], ["marker"]),                     # (1,1,H,W)
        # output = input*(1-marker) + e4*marker
        n("Sub", ["one", "marker"], ["inv"]),
        n("Mul", ["input", "inv"], ["kept"]),
        n("Mul", ["e4", "marker"], ["addc"]),
        n("Add", ["kept", "addc"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "pocket_drop",
                              [helper.make_tensor_value_info("input", F, FULL)],
                              [helper.make_tensor_value_info("output", F, FULL)],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_pocket_drop(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
