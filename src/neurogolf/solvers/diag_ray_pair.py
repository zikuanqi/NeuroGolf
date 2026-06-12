"""Solver: two blocks each emit a fixed diagonal ray (task 136).

The grid holds a colour-1 2x2 block and a colour-2 2x2 block.  Each emits a
single-cell diagonal ray to the grid edge: the ``1`` block shoots up-left from
its top-left corner, the ``2`` block shoots down-right from its bottom-right
corner::

    . . . .          1 . . .
    . 1 1 .   ->     . 1 1 .
    . 1 1 .          . 1 1 .
    . . . 2          . . . 2
    . . . . 2        . . . 2 .   (wait: 2 ray continues down-right)

A point ``(r,c)`` on an up-left ray from corner ``(mr,mc)`` satisfies
``r - c == mr - mc`` and ``r < mr``; the down-right ray is symmetric with
``r > maxr2``.  Both rays are constant-diagonal masks against ``r - c``.
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
    if not (g == 1).any() or not (g == 2).any():
        return None
    R = np.arange(H)[:, None]
    C = np.arange(W)[None, :]
    RmC = R - C
    y1, x1 = np.where(g == 1); d1 = y1.min() - x1.min()
    y2, x2 = np.where(g == 2); d2 = y2.max() - x2.max()
    out = g.copy()
    out[(RmC == d1) & (R < y1.min())] = 1
    out[(RmC == d2) & (R > y2.max())] = 2
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
    e1 = np.zeros((1, CHANNELS, 1, 1), np.float32); e1[0, 1] = 1.0
    e2 = np.zeros((1, CHANNELS, 1, 1), np.float32); e2[0, 2] = 1.0
    idx = np.arange(HEIGHT)
    RmC = (idx[:, None] - idx[None, :]).astype(np.float32).reshape(1, 1, HEIGHT, WIDTH)
    init = [
        numpy_helper.from_array(e0, "e0"),
        numpy_helper.from_array(e1, "e1"),
        numpy_helper.from_array(e2, "e2"),
        numpy_helper.from_array(RmC, "RmC"),
        numpy_helper.from_array(np.arange(HEIGHT, dtype=np.float32).reshape(1, 1, HEIGHT, 1), "ah"),
        numpy_helper.from_array(np.arange(WIDTH, dtype=np.float32).reshape(1, 1, 1, WIDTH), "aw"),
        numpy_helper.from_array(np.array(0.5, np.float32), "half"),
        numpy_helper.from_array(np.array([1], np.int64), "c1"),
        numpy_helper.from_array(np.array([2], np.int64), "c2"),
        numpy_helper.from_array(np.array([3], np.int64), "c3"),
        numpy_helper.from_array(np.array([1], np.int64), "ax1"),
    ]
    nodes = [
        n("ReduceSum", ["input"], ["occ"], axes=[1], keepdims=1),
        n("Slice", ["input", "c1", "c2", "ax1"], ["is1"]),
        n("Slice", ["input", "c2", "c3", "ax1"], ["is2"]),
        # colour-1 top-left corner (min row, min col)
        n("ReduceMax", ["is1"], ["row1"], axes=[3], keepdims=1),
        n("ReduceMax", ["is1"], ["col1"], axes=[2], keepdims=1),
        n("ArgMax", ["row1"], ["mr1i"], axis=2, keepdims=1), n("Cast", ["mr1i"], ["mr1"], to=F),
        n("ArgMax", ["col1"], ["mc1i"], axis=3, keepdims=1), n("Cast", ["mc1i"], ["mc1"], to=F),
        n("Sub", ["mr1", "mc1"], ["d1"]),
        # colour-2 bottom-right corner (max row, max col)
        n("ReduceMax", ["is2"], ["row2"], axes=[3], keepdims=1),
        n("ReduceMax", ["is2"], ["col2"], axes=[2], keepdims=1),
        n("Mul", ["row2", "ah"], ["r2pos"]), n("ReduceMax", ["r2pos"], ["Mr2"], axes=[2], keepdims=1),
        n("Mul", ["col2", "aw"], ["c2pos"]), n("ReduceMax", ["c2pos"], ["Mc2"], axes=[3], keepdims=1),
        n("Sub", ["Mr2", "Mc2"], ["d2"]),
        # ray1 = (RmC == d1) & (row < mr1) & in-grid
        n("Sub", ["RmC", "d1"], ["df1"]), n("Abs", ["df1"], ["ad1"]),
        n("Less", ["ad1", "half"], ["eq1_b"]), n("Cast", ["eq1_b"], ["eq1"], to=F),
        n("Less", ["ah", "mr1"], ["ab_b"]), n("Cast", ["ab_b"], ["ab"], to=F),
        n("Mul", ["eq1", "ab"], ["r1a"]), n("Mul", ["r1a", "occ"], ["ray1"]),
        # ray2 = (RmC == d2) & (row > Mr2) & in-grid
        n("Sub", ["RmC", "d2"], ["df2"]), n("Abs", ["df2"], ["ad2"]),
        n("Less", ["ad2", "half"], ["eq2_b"]), n("Cast", ["eq2_b"], ["eq2"], to=F),
        n("Greater", ["ah", "Mr2"], ["bl_b"]), n("Cast", ["bl_b"], ["bl"], to=F),
        n("Mul", ["eq2", "bl"], ["r2a"]), n("Mul", ["r2a", "occ"], ["ray2"]),
        # paint
        n("Mul", ["ray1", "e1"], ["add1"]),
        n("Mul", ["ray2", "e2"], ["add2"]),
        n("Add", ["ray1", "ray2"], ["rsum"]), n("Mul", ["rsum", "e0"], ["sub0"]),
        n("Add", ["input", "add1"], ["t1"]),
        n("Add", ["t1", "add2"], ["t2"]),
        n("Sub", ["t2", "sub0"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "diag_ray_pair",
                              [helper.make_tensor_value_info("input", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              [helper.make_tensor_value_info("output", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_diag_ray_pair(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
