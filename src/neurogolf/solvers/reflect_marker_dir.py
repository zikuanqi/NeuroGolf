"""Solver: mirror the 8-shape, direction set by the 4-marker (task 181).

An 8-shape sits above a 4-marker.  The 8-shape is reflected horizontally and
the mirror copy is appended directly beside it; whether it goes left or right
is decided by the 4-marker, whose top arm is displaced to one side::

    . 8 . 8        8 . 8 8 . 8       (4-marker's top cell is on the right
    . . 8 8   ->   . . 8 8 . .        -> mirror appended on the right)
    . 4 . .        . 4 . .
    4 4 4 .        4 4 4 .   (4 unchanged)
    . 4 . .        . 4 . .

The reflection is a runtime column matrix ``S[c, nc] = 1 iff c + nc == axis``,
where ``axis = 2*c1+1`` (right edge) or ``2*c0-1`` (left edge) of the 8-bbox,
selected by ``dirright = top4col > centre4col``.
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
I64 = TensorProto.INT64


def _ref(g: np.ndarray) -> Optional[np.ndarray]:
    H, W = g.shape
    ys8, xs8 = np.where(g == 8)
    ys4, xs4 = np.where(g == 4)
    if len(ys8) == 0 or len(ys4) == 0:
        return None
    c0, c1 = xs8.min(), xs8.max()
    topr = ys4.min()
    topcols = xs4[ys4 == topr]
    cc = (xs4.min() + xs4.max()) / 2
    dirright = topcols.mean() > cc
    out = g.copy()
    for r, c in zip(ys8, xs8):
        nc = (2 * c1 + 1 - c) if dirright else (2 * c0 - 1 - c)
        if 0 <= nc < W:
            out[r, nc] = 8
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
    e8 = np.zeros((1, CHANNELS, 1, 1), np.float32); e8[0, 8] = 1.0
    e0 = np.zeros((1, CHANNELS, 1, 1), np.float32); e0[0, 0] = 1.0
    idx = np.arange(WIDTH)
    Dsum = (idx[:, None] + idx[None, :]).astype(np.float32).reshape(1, 1, WIDTH, WIDTH)
    init = [
        numpy_helper.from_array(e8, "e8"),
        numpy_helper.from_array(e0, "e0"),
        numpy_helper.from_array(Dsum, "Dsum"),
        numpy_helper.from_array(np.arange(WIDTH, dtype=np.float32).reshape(1, 1, 1, WIDTH), "aw"),
        numpy_helper.from_array(np.array(0.5, np.float32), "half"),
        numpy_helper.from_array(np.array(1.0, np.float32), "one"),
        numpy_helper.from_array(np.array(2.0, np.float32), "two"),
        numpy_helper.from_array(np.array([0], np.int64), "c0s"),
        numpy_helper.from_array(np.array([1], np.int64), "c1e"),
        numpy_helper.from_array(np.array([4], np.int64), "c4s"),
        numpy_helper.from_array(np.array([5], np.int64), "c5e"),
        numpy_helper.from_array(np.array([8], np.int64), "c8s"),
        numpy_helper.from_array(np.array([9], np.int64), "c9e"),
        numpy_helper.from_array(np.array([1], np.int64), "ax1"),
        numpy_helper.from_array(np.array([1], np.int64), "shape1"),
    ]
    nodes = [
        n("Slice", ["input", "c0s", "c1e", "ax1"], ["is0"]),
        n("Slice", ["input", "c4s", "c5e", "ax1"], ["is4"]),
        n("Slice", ["input", "c8s", "c9e", "ax1"], ["is8"]),
        # 8 bbox columns
        n("ReduceMax", ["is8"], ["col8"], axes=[2], keepdims=1),
        n("ArgMax", ["col8"], ["c0i"], axis=3, keepdims=1), n("Cast", ["c0i"], ["c0"], to=F),
        n("Mul", ["col8", "aw"], ["c8p"]), n("ReduceMax", ["c8p"], ["c1"], axes=[3], keepdims=1),
        # 4 marker: top row col, centre col
        n("ReduceMax", ["is4"], ["row4"], axes=[3], keepdims=1),
        n("ArgMax", ["row4"], ["tr_i"], axis=2, keepdims=1),
        n("Reshape", ["tr_i", "shape1"], ["tr1d"]),
        n("Gather", ["is4", "tr1d"], ["toprow"], axis=2),          # (1,1,1,30)
        n("ArgMax", ["toprow"], ["tc_i"], axis=3, keepdims=1), n("Cast", ["tc_i"], ["topcol"], to=F),
        n("ReduceMax", ["is4"], ["col4"], axes=[2], keepdims=1),
        n("ArgMax", ["col4"], ["mc4i"], axis=3, keepdims=1), n("Cast", ["mc4i"], ["minc4"], to=F),
        n("Mul", ["col4", "aw"], ["c4p"]), n("ReduceMax", ["c4p"], ["maxc4"], axes=[3], keepdims=1),
        n("Add", ["minc4", "maxc4"], ["sum4"]), n("Mul", ["sum4", "half"], ["centre4"]),
        n("Greater", ["topcol", "centre4"], ["dr_b"]), n("Cast", ["dr_b"], ["dirR"], to=F),
        # axis = dirR*(2c1+1) + (1-dirR)*(2c0-1)
        n("Mul", ["c1", "two"], ["c1x2"]), n("Add", ["c1x2", "one"], ["rc"]),
        n("Mul", ["c0", "two"], ["c0x2"]), n("Sub", ["c0x2", "one"], ["lc"]),
        n("Mul", ["dirR", "rc"], ["t1"]),
        n("Sub", ["one", "dirR"], ["dirL"]), n("Mul", ["dirL", "lc"], ["t2"]),
        n("Add", ["t1", "t2"], ["axisc"]),
        # reflection matrix S[c,nc] = 1 iff c+nc == axisc
        n("Sub", ["Dsum", "axisc"], ["dd"]), n("Abs", ["dd"], ["da"]),
        n("Less", ["da", "half"], ["S_b"]), n("Cast", ["S_b"], ["S"], to=F),
        n("MatMul", ["is8", "S"], ["refl"]),
        n("Mul", ["refl", "is0"], ["addm"]),
        n("Mul", ["addm", "e8"], ["a8"]),
        n("Mul", ["addm", "e0"], ["s0"]),
        n("Add", ["input", "a8"], ["t3"]),
        n("Sub", ["t3", "s0"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "reflect_marker_dir",
                              [helper.make_tensor_value_info("input", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              [helper.make_tensor_value_info("output", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_reflect_marker_dir(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
