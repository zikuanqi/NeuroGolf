"""Solver: surround every 2 with a 3x3 halo of 1s (task 352).

Each ``2`` paints its eight neighbouring background cells ``1`` (overlapping
halos merge); the 2s and all other colours are unchanged.  A 3x3 ``MaxPool``
dilation of the ``2`` mask, kept on background cells.
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
    is2 = (g == 2).astype(int)
    if is2.sum() == 0:
        return None
    dil = np.zeros_like(is2)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            ys = slice(max(0, dy), H + min(0, dy)); xs = slice(max(0, dx), W + min(0, dx))
            yt = slice(max(0, -dy), H + min(0, -dy)); xt = slice(max(0, -dx), W + min(0, -dx))
            dil[yt, xt] = np.maximum(dil[yt, xt], is2[ys, xs])
    ring = dil * (g == 0)
    if ring.sum() == 0:
        return None
    out = g.copy()
    out[ring.astype(bool)] = 1
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


def _build() -> onnx.ModelProto:
    n = helper.make_node
    e1me0 = np.zeros((1, CHANNELS, 1, 1), np.float32); e1me0[0, 1] = 1.0; e1me0[0, 0] = -1.0
    init = [
        numpy_helper.from_array(e1me0, "e1me0"),
        numpy_helper.from_array(np.array([0], np.int64), "c0"),
        numpy_helper.from_array(np.array([1], np.int64), "c1"),
        numpy_helper.from_array(np.array([2], np.int64), "c2"),
        numpy_helper.from_array(np.array([3], np.int64), "c3"),
        numpy_helper.from_array(np.array([1], np.int64), "ax1"),
    ]
    nodes = [
        n("Slice", ["input", "c0", "c1", "ax1"], ["is0"]),
        n("Slice", ["input", "c2", "c3", "ax1"], ["is2"]),
        n("MaxPool", ["is2"], ["dil"], kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]),
        n("Mul", ["dil", "is0"], ["ring"]),
        n("Mul", ["ring", "e1me0"], ["delta"]),
        n("Add", ["input", "delta"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "two_halo_ones",
                              [helper.make_tensor_value_info("input", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              [helper.make_tensor_value_info("output", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_two_halo_ones(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
