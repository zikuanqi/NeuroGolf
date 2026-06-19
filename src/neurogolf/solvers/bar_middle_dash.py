"""Solver: dash the middle row of each solid 3-row bar (task 85).

Each solid horizontal bar keeps its top and bottom rows but its middle row is
dashed: every other cell (from the bar's left edge) is cleared to background.

A middle-row cell is a non-background cell whose neighbours directly above and
below share its colour.  Within each row the kept/cleared parity comes from an
inclusive column ``CumSum`` of the middle mask: cells at an odd running count
(1st, 3rd, ...) survive, the even ones revert to background.
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
    o = g.copy()
    saw = False
    for r in range(H):
        for c in range(W):
            v = g[r, c]
            if v != 0 and 0 < r < H - 1 and g[r - 1, c] == v and g[r + 1, c] == v:
                cs = c
                while (cs - 1 >= 0 and g[r, cs - 1] == v
                       and g[r - 1, cs - 1] == v and g[r + 1, cs - 1] == v):
                    cs -= 1
                if (c - cs) % 2 == 1:
                    o[r, c] = 0; saw = True
    if not saw:
        return None
    return o


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
    notbg = np.ones((1, CHANNELS, 1, 1), np.float32); notbg[0, 0] = 0.0
    e0 = np.zeros((1, CHANNELS, 1, 1), np.float32); e0[0, 0] = 1.0
    init = [
        numpy_helper.from_array(notbg, "notbg"),
        numpy_helper.from_array(e0, "e0"),
        numpy_helper.from_array(np.array(1.0, np.float32), "one"),
        numpy_helper.from_array(np.array(0.5, np.float32), "half"),
        numpy_helper.from_array(np.array(3, np.int64), "ax3v"),
        numpy_helper.from_array(np.array([0], np.int64), "z0"),
        numpy_helper.from_array(np.array([1], np.int64), "z1"),
        numpy_helper.from_array(np.array([HEIGHT], np.int64), "zH"),
        numpy_helper.from_array(np.array([HEIGHT + 1], np.int64), "zH1"),
        numpy_helper.from_array(np.array([2], np.int64), "ax2"),
        numpy_helper.from_array(np.array([0, 0, 1, 0, 0, 0, 0, 0], np.int64), "padTop"),
        numpy_helper.from_array(np.array([0, 0, 0, 0, 0, 0, 1, 0], np.int64), "padBot"),
    ]
    nodes = [
        # cell above / below
        n("Pad", ["input", "padTop"], ["padU"]), n("Slice", ["padU", "z0", "zH", "ax2"], ["inUp"]),
        n("Pad", ["input", "padBot"], ["padD"]), n("Slice", ["padD", "z1", "zH1", "ax2"], ["inDn"]),
        # middle-row bar cell: same colour above and below, non-background
        n("Mul", ["input", "inUp"], ["iu"]), n("Mul", ["iu", "inDn"], ["iud"]),
        n("Mul", ["iud", "notbg"], ["midOH"]),
        n("ReduceSum", ["midOH"], ["midMask"], axes=[1], keepdims=1),     # (1,1,H,W)
        # run parity via inclusive column cumsum
        n("CumSum", ["midMask", "ax3v"], ["cs"]),
        n("Mul", ["cs", "half"], ["csh"]), n("Floor", ["csh"], ["csf"]),
        n("Add", ["csf", "csf"], ["cs2"]), n("Sub", ["cs", "cs2"], ["par"]),  # cs mod 2
        n("Sub", ["one", "par"], ["evenc"]), n("Mul", ["midMask", "evenc"], ["clearMask"]),
        # clear even cells to background
        n("Sub", ["one", "clearMask"], ["keepM"]), n("Mul", ["input", "keepM"], ["kept"]),
        n("Mul", ["clearMask", "e0"], ["bgL"]), n("Add", ["kept", "bgL"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "bar_middle_dash",
                              [helper.make_tensor_value_info("input", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              [helper.make_tensor_value_info("output", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_bar_middle_dash(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
