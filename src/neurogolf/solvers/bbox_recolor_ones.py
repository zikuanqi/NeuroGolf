"""Solver: recolor 1 -> 3 inside the bounding box of the 8-shape (task 70).

The grid carries a periodic 1/0 texture plus one shape drawn in colour 8.
Every ``1`` that lies within the 8-shape's bounding box is recoloured to ``3``;
the 8-cells and everything outside the box are left untouched::

    . 8 8 8 .          . 8 8 8 .
    1 1 1 1 1   ->     1 3 3 3 1      (the 1s inside the 8-bbox become 3)
    . 8 8 8 .          . 8 8 8 .

The 8-bbox bounds come from row/col presence projections (ArgMax for the min,
``max(presence * index)`` for the max); the box mask is an ``arange`` band test.
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
    ys, xs = np.where(g == 8)
    if len(ys) == 0:
        return None
    r0, r1, c0, c1 = ys.min(), ys.max(), xs.min(), xs.max()
    out = g.copy()
    sub = out[r0:r1 + 1, c0:c1 + 1]
    sub[sub == 1] = 3
    out[r0:r1 + 1, c0:c1 + 1] = sub
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
    e3me1 = np.zeros((1, CHANNELS, 1, 1), np.float32); e3me1[0, 3] = 1.0; e3me1[0, 1] = -1.0
    init = [
        numpy_helper.from_array(e3me1, "e3me1"),
        numpy_helper.from_array(np.arange(HEIGHT, dtype=np.float32).reshape(1, 1, HEIGHT, 1), "ah"),
        numpy_helper.from_array(np.arange(WIDTH, dtype=np.float32).reshape(1, 1, 1, WIDTH), "aw"),
        numpy_helper.from_array(np.array(0.5, np.float32), "half"),
        numpy_helper.from_array(np.array([1], np.int64), "c1"),
        numpy_helper.from_array(np.array([2], np.int64), "c2"),
        numpy_helper.from_array(np.array([8], np.int64), "c8"),
        numpy_helper.from_array(np.array([9], np.int64), "c9"),
        numpy_helper.from_array(np.array([1], np.int64), "ax1"),
    ]
    nodes = [
        n("Slice", ["input", "c8", "c9", "ax1"], ["is8"]),
        n("Slice", ["input", "c1", "c2", "ax1"], ["is1"]),
        n("ReduceMax", ["is8"], ["row8"], axes=[3], keepdims=1),
        n("ReduceMax", ["is8"], ["col8"], axes=[2], keepdims=1),
        n("ArgMax", ["row8"], ["minr_i"], axis=2, keepdims=1),
        n("ArgMax", ["col8"], ["minc_i"], axis=3, keepdims=1),
        n("Cast", ["minr_i"], ["minr"], to=F),
        n("Cast", ["minc_i"], ["minc"], to=F),
        n("Mul", ["row8", "ah"], ["rpos"]),
        n("ReduceMax", ["rpos"], ["maxr"], axes=[2], keepdims=1),
        n("Mul", ["col8", "aw"], ["cpos"]),
        n("ReduceMax", ["cpos"], ["maxc"], axes=[3], keepdims=1),
        n("Sub", ["minr", "half"], ["minr_m"]),
        n("Greater", ["ah", "minr_m"], ["rge"]), n("Cast", ["rge"], ["rge_f"], to=F),
        n("Add", ["maxr", "half"], ["maxr_p"]),
        n("Less", ["ah", "maxr_p"], ["rle"]), n("Cast", ["rle"], ["rle_f"], to=F),
        n("Mul", ["rge_f", "rle_f"], ["rowIn"]),
        n("Sub", ["minc", "half"], ["minc_m"]),
        n("Greater", ["aw", "minc_m"], ["cge"]), n("Cast", ["cge"], ["cge_f"], to=F),
        n("Add", ["maxc", "half"], ["maxc_p"]),
        n("Less", ["aw", "maxc_p"], ["cle"]), n("Cast", ["cle"], ["cle_f"], to=F),
        n("Mul", ["cge_f", "cle_f"], ["colIn"]),
        n("Mul", ["rowIn", "colIn"], ["bbox"]),
        n("Mul", ["bbox", "is1"], ["recolor"]),
        n("Mul", ["recolor", "e3me1"], ["addc"]),
        n("Add", ["input", "addc"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "bbox_recolor_ones",
                              [helper.make_tensor_value_info("input", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              [helper.make_tensor_value_info("output", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_bbox_recolor_ones(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
