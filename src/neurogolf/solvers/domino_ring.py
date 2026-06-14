"""Solver: frame each adjacent 2-pair with a 3 ring (task 278).

A ``2`` that has an orthogonally-adjacent ``2`` (i.e. belongs to a domino/group)
gets its 3x3 surround painted ``3``; isolated ``2`` cells are left alone::

    . . .        3 3 3
    . 2 .   ->   3 2 3     (only when the 2 has a 2 neighbour)
    . 2 .        3 2 3
    . . .        3 3 3

Group membership = the ``2`` mask intersected with its orthogonal-neighbour
``2`` mask; the ring = a 3x3 ``MaxPool`` dilation of the group, kept on
background cells only.
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
    nb = np.zeros_like(is2)
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        ys = slice(max(0, dy), H + min(0, dy)); xs = slice(max(0, dx), W + min(0, dx))
        yt = slice(max(0, -dy), H + min(0, -dy)); xt = slice(max(0, -dx), W + min(0, -dx))
        nb[yt, xt] = np.maximum(nb[yt, xt], is2[ys, xs])
    group = is2 * nb
    dil = np.zeros_like(is2)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            ys = slice(max(0, dy), H + min(0, dy)); xs = slice(max(0, dx), W + min(0, dx))
            yt = slice(max(0, -dy), H + min(0, -dy)); xt = slice(max(0, -dx), W + min(0, -dx))
            dil[yt, xt] = np.maximum(dil[yt, xt], group[ys, xs])
    ring = dil * (g == 0)
    if ring.sum() == 0:
        return None
    out = g.copy()
    out[ring.astype(bool)] = 3
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
    e3me0 = np.zeros((1, CHANNELS, 1, 1), np.float32); e3me0[0, 3] = 1.0; e3me0[0, 0] = -1.0
    init = [
        numpy_helper.from_array(e3me0, "e3me0"),
        numpy_helper.from_array(np.array([0], np.int64), "c0"),
        numpy_helper.from_array(np.array([1], np.int64), "c1"),
        numpy_helper.from_array(np.array([2], np.int64), "c2"),
        numpy_helper.from_array(np.array([3], np.int64), "c3"),
        numpy_helper.from_array(np.array([1], np.int64), "ax1"),
        numpy_helper.from_array(np.array([2], np.int64), "ax2"),
        numpy_helper.from_array(np.array([3], np.int64), "ax3"),
        numpy_helper.from_array(np.array([0], np.int64), "s0"),
        numpy_helper.from_array(np.array([1], np.int64), "s1"),
        numpy_helper.from_array(np.array([HEIGHT], np.int64), "sN"),
        numpy_helper.from_array(np.array([HEIGHT + 1], np.int64), "sN1"),
        numpy_helper.from_array(np.array([0, 0, 1, 0, 0, 0, 0, 0], np.int64), "padT"),
        numpy_helper.from_array(np.array([0, 0, 0, 0, 0, 0, 1, 0], np.int64), "padB"),
        numpy_helper.from_array(np.array([0, 0, 0, 1, 0, 0, 0, 0], np.int64), "padL"),
        numpy_helper.from_array(np.array([0, 0, 0, 0, 0, 0, 0, 1], np.int64), "padR"),
    ]
    nodes = [
        n("Slice", ["input", "c0", "c1", "ax1"], ["is0"]),
        n("Slice", ["input", "c2", "c3", "ax1"], ["is2"]),
        n("Pad", ["is2", "padT"], ["pT"]), n("Slice", ["pT", "s0", "sN", "ax2"], ["up"]),
        n("Pad", ["is2", "padB"], ["pB"]), n("Slice", ["pB", "s1", "sN1", "ax2"], ["dn"]),
        n("Pad", ["is2", "padL"], ["pL"]), n("Slice", ["pL", "s0", "sN", "ax3"], ["lf"]),
        n("Pad", ["is2", "padR"], ["pR"]), n("Slice", ["pR", "s1", "sN1", "ax3"], ["rt"]),
        n("Max", ["up", "dn"], ["m1"]), n("Max", ["lf", "rt"], ["m2"]),
        n("Max", ["m1", "m2"], ["orthNb"]),
        n("Mul", ["is2", "orthNb"], ["group"]),
        n("MaxPool", ["group"], ["dil"], kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]),
        n("Mul", ["dil", "is0"], ["ring"]),
        n("Mul", ["ring", "e3me0"], ["delta"]),
        n("Add", ["input", "delta"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "domino_ring",
                              [helper.make_tensor_value_info("input", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              [helper.make_tensor_value_info("output", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_domino_ring(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
