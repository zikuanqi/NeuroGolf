"""Solver: extend markers as rays around a 5-divider (task 212).

A full row of colour 5 splits the grid.  Each ``2`` marker grows a vertical
line **toward** the divider (up to the cell adjacent to it); each ``1`` marker
grows a vertical line **away** from the divider (to the grid edge)::

    . 2 . 1 .          . 2 . 1 .
    . . . . .    ->    . 2 . 1 .      (2 -> down toward divider,
    5 5 5 5 5          5 5 5 5 5        1 -> up to the edge)

Each direction is a single ``CumSum`` along the row axis, masked to the half
it belongs to: 2-above = forward cumsum, 2-below = reverse, 1-above = reverse
(outward), 1-below = forward; ``> 0`` marks every cell between a marker and its
target.
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
    drows = [r for r in range(H) if np.all(g[r] == 5)]
    if len(drows) != 1:
        return None
    d = drows[0]
    R = np.arange(H)[:, None]
    top = R < d
    bot = R > d
    is2 = g == 2
    is1 = g == 1
    f2t = (np.cumsum(is2 * top, axis=0) > 0) & top
    f2b = (np.cumsum((is2 * bot)[::-1], axis=0)[::-1] > 0) & bot
    f1t = (np.cumsum((is1 * top)[::-1], axis=0)[::-1] > 0) & top
    f1b = (np.cumsum(is1 * bot, axis=0) > 0) & bot
    out = g.copy()
    out[(f2t | f2b) & (g == 0)] = 2
    out[(f1t | f1b) & (g == 0)] = 1
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
    e2me0 = np.zeros((1, CHANNELS, 1, 1), np.float32); e2me0[0, 2] = 1.0; e2me0[0, 0] = -1.0
    e1me0 = np.zeros((1, CHANNELS, 1, 1), np.float32); e1me0[0, 1] = 1.0; e1me0[0, 0] = -1.0
    init = [
        numpy_helper.from_array(e2me0, "e2me0"),
        numpy_helper.from_array(e1me0, "e1me0"),
        numpy_helper.from_array(np.arange(HEIGHT, dtype=np.float32).reshape(1, 1, HEIGHT, 1), "ah"),
        numpy_helper.from_array(np.array(0.5, np.float32), "half"),
        numpy_helper.from_array(np.array(2, np.int64), "axis2"),
        numpy_helper.from_array(np.array([0], np.int64), "c0s"),
        numpy_helper.from_array(np.array([1], np.int64), "c1"),
        numpy_helper.from_array(np.array([2], np.int64), "c2"),
        numpy_helper.from_array(np.array([3], np.int64), "c3"),
        numpy_helper.from_array(np.array([5], np.int64), "c5"),
        numpy_helper.from_array(np.array([6], np.int64), "c6"),
        numpy_helper.from_array(np.array([1], np.int64), "ax1"),
    ]

    def fill(seedmask, halfmask, reverse, p):
        sm, cs, gb, cf, out = f"{p}sm", f"{p}cs", f"{p}gb", f"{p}cf", f"{p}out"
        return [
            n("Mul", [seedmask, halfmask], [sm]),
            n("CumSum", [sm, "axis2"], [cs], exclusive=0, reverse=reverse),
            n("Greater", [cs, "half"], [gb]), n("Cast", [gb], [cf], to=F),
            n("Mul", [cf, halfmask], [out]),
        ]

    nodes = [
        n("Slice", ["input", "c0s", "c1", "ax1"], ["is0"]),
        n("Slice", ["input", "c1", "c2", "ax1"], ["is1"]),
        n("Slice", ["input", "c2", "c3", "ax1"], ["is2"]),
        n("Slice", ["input", "c5", "c6", "ax1"], ["is5"]),
        n("ReduceMax", ["is5"], ["drow"], axes=[3], keepdims=1),
        n("ArgMax", ["drow"], ["d_i"], axis=2, keepdims=1), n("Cast", ["d_i"], ["d"], to=F),
        n("Less", ["ah", "d"], ["tb"]), n("Cast", ["tb"], ["top"], to=F),
        n("Greater", ["ah", "d"], ["bb"]), n("Cast", ["bb"], ["bot"], to=F),
    ]
    nodes += fill("is2", "top", 0, "a")     # 2 above -> down toward divider
    nodes += fill("is2", "bot", 1, "b")     # 2 below -> up toward divider
    nodes += fill("is1", "top", 1, "c")     # 1 above -> up to edge
    nodes += fill("is1", "bot", 0, "d")     # 1 below -> down to edge
    nodes += [
        n("Add", ["aout", "bout"], ["fill2"]),
        n("Add", ["cout", "dout"], ["fill1"]),
        n("Mul", ["fill2", "is0"], ["nf2"]),
        n("Mul", ["fill1", "is0"], ["nf1"]),
        n("Mul", ["nf2", "e2me0"], ["add2"]),
        n("Mul", ["nf1", "e1me0"], ["add1"]),
        n("Add", ["input", "add2"], ["t1"]),
        n("Add", ["t1", "add1"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "divider_rays",
                              [helper.make_tensor_value_info("input", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              [helper.make_tensor_value_info("output", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_divider_rays(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
