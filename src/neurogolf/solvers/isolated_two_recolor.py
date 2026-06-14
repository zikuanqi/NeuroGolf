"""Solver: recolour 4-isolated 2-cells to 1 (task 272).

Every cell of colour ``2`` that has no orthogonally-adjacent ``2`` is repainted
``1``; clustered 2s stay.  The neighbour test is the max of the four
single-step shifts of the ``2`` mask.
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
    out = g.copy()
    changed = False
    for r in range(H):
        for c in range(W):
            if g[r, c] != 2:
                continue
            iso = True
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and g[nr, nc] == 2:
                    iso = False
                    break
            if iso:
                out[r, c] = 1
                changed = True
    return out if changed else None


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
    e1me2 = np.zeros((1, CHANNELS, 1, 1), np.float32); e1me2[0, 1] = 1.0; e1me2[0, 2] = -1.0
    init = [
        numpy_helper.from_array(e1me2, "e1me2"),
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
        n("Slice", ["input", "c2", "c3", "ax1"], ["is2"]),
        # up neighbour value (is2[r-1,c]) = shift content down
        n("Pad", ["is2", "padT"], ["pT"]), n("Slice", ["pT", "s0", "sN", "ax2"], ["up"]),
        # down neighbour (is2[r+1,c]) = shift content up
        n("Pad", ["is2", "padB"], ["pB"]), n("Slice", ["pB", "s1", "sN1", "ax2"], ["dn"]),
        # left neighbour (is2[r,c-1]) = shift content right
        n("Pad", ["is2", "padL"], ["pL"]), n("Slice", ["pL", "s0", "sN", "ax3"], ["lf"]),
        # right neighbour (is2[r,c+1]) = shift content left
        n("Pad", ["is2", "padR"], ["pR"]), n("Slice", ["pR", "s1", "sN1", "ax3"], ["rt"]),
        n("Max", ["up", "dn"], ["m1"]), n("Max", ["lf", "rt"], ["m2"]),
        n("Max", ["m1", "m2"], ["nbr"]),
        n("Sub", ["is2", "nbr"], ["isoRaw"]),
        n("Relu", ["isoRaw"], ["iso"]),  # is2 * (1 - nbr), both 0/1
        n("Mul", ["iso", "e1me2"], ["delta"]),
        n("Add", ["input", "delta"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "isolated_two_recolor",
                              [helper.make_tensor_value_info("input", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              [helper.make_tensor_value_info("output", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_isolated_two_recolor(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
