"""Solver: crop the 3x3 shape centred on the lone 8 and recolour it (task 121).

Several small shapes sit in the grid; exactly one has a colour-8 cell at its
centre.  The output is that shape's 3x3 block, with the central 8 repainted in
the shape's own colour::

    . 4 .                 . 4 .
    4 8 4   (the 8 one)   4 4 4
    . 4 .                 . 4 .

The 8 is located by argmax of its channel; a clamped two-axis ``Gather`` brings
its 3x3 neighbourhood to the top-left; the shape colour is the only non-bg,
non-8 channel present, and the 8 cell is overwritten with it.
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
    ys, xs = np.where(g == 8)
    if len(ys) != 1:
        return None
    r, c = ys[0], xs[0]
    if r < 1 or c < 1 or r + 2 > g.shape[0] or c + 2 > g.shape[1]:
        return None
    crop = g[r - 1:r + 2, c - 1:c + 2].copy()
    sc = [v for v in np.unique(crop) if v not in (0, 8)]
    if len(sc) != 1:
        return None
    crop[crop == 8] = sc[0]
    return crop


def _detect(task: dict) -> bool:
    saw = False
    for ex in all_examples(task):
        i, o = ex["input"], ex["output"]
        if not i or not i[0] or len(i) > HEIGHT or len(i[0]) > WIDTH:
            continue
        r = _ref(np.array(i))
        if r is None or r.shape != np.array(o).shape or not np.array_equal(r, np.array(o)):
            return False
        saw = True
    return saw


def _build() -> onnx.ModelProto:
    n = helper.make_node
    e8 = np.zeros((1, CHANNELS, 1, 1), np.float32); e8[0, 8] = 1.0
    keepc = np.ones((1, CHANNELS, 1, 1), np.float32); keepc[0, 0] = 0.0; keepc[0, 8] = 0.0
    init = [
        numpy_helper.from_array(e8, "e8"),
        numpy_helper.from_array(keepc, "keepc"),
        numpy_helper.from_array(np.arange(HEIGHT, dtype=np.float32).reshape(1, 1, HEIGHT, 1), "ah"),
        numpy_helper.from_array(np.arange(WIDTH, dtype=np.float32).reshape(1, 1, 1, WIDTH), "aw"),
        numpy_helper.from_array(np.array(1.0, np.float32), "one"),
        numpy_helper.from_array(np.array(0.5, np.float32), "half"),
        numpy_helper.from_array(np.array(3.0, np.float32), "three"),
        numpy_helper.from_array(np.array(0.0, np.float32), "cmin"),
        numpy_helper.from_array(np.array(float(HEIGHT - 1), np.float32), "cmax"),
        numpy_helper.from_array(np.array([8], np.int64), "ch8s"),
        numpy_helper.from_array(np.array([9], np.int64), "ch8e"),
        numpy_helper.from_array(np.array([1], np.int64), "ax1"),
        numpy_helper.from_array(np.array([HEIGHT], np.int64), "h30"),
        numpy_helper.from_array(np.array([WIDTH], np.int64), "w30"),
    ]
    nodes = [
        n("Slice", ["input", "ch8s", "ch8e", "ax1"], ["is8"]),
        n("ReduceMax", ["is8"], ["row8"], axes=[3], keepdims=1),
        n("ReduceMax", ["is8"], ["col8"], axes=[2], keepdims=1),
        n("ArgMax", ["row8"], ["r8i"], axis=2, keepdims=1), n("Cast", ["r8i"], ["r8"], to=F),
        n("ArgMax", ["col8"], ["c8i"], axis=3, keepdims=1), n("Cast", ["c8i"], ["c8"], to=F),
        n("Sub", ["r8", "one"], ["r8m1"]), n("Sub", ["c8", "one"], ["c8m1"]),
        n("Add", ["ah", "r8m1"], ["addr"]),
        n("Clip", ["addr", "cmin", "cmax"], ["clr"]),
        n("Cast", ["clr"], ["ri"], to=I64), n("Reshape", ["ri", "h30"], ["r1d"]),
        n("Add", ["aw", "c8m1"], ["addc"]),
        n("Clip", ["addc", "cmin", "cmax"], ["clc"]),
        n("Cast", ["clc"], ["ci"], to=I64), n("Reshape", ["ci", "w30"], ["c1d"]),
        n("Gather", ["input", "r1d"], ["gr"], axis=2),
        n("Gather", ["gr", "c1d"], ["gathered"], axis=3),
        n("Less", ["ah", "three"], ["rlt_b"]), n("Cast", ["rlt_b"], ["rlt"], to=F),
        n("Less", ["aw", "three"], ["clt_b"]), n("Cast", ["clt_b"], ["clt"], to=F),
        n("Mul", ["rlt", "clt"], ["mask3"]),
        n("Mul", ["gathered", "mask3"], ["crop3"]),
        n("ReduceSum", ["crop3"], ["counts"], axes=[2, 3], keepdims=1),
        n("Mul", ["counts", "keepc"], ["masked"]),
        n("Greater", ["masked", "half"], ["sv_b"]), n("Cast", ["sv_b"], ["shapeVec"], to=F),
        n("Slice", ["crop3", "ch8s", "ch8e", "ax1"], ["is8crop"]),
        n("Sub", ["shapeVec", "e8"], ["smc"]),
        n("Mul", ["is8crop", "smc"], ["addterm"]),
        n("Add", ["crop3", "addterm"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "eight_center_crop",
                              [helper.make_tensor_value_info("input", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              [helper.make_tensor_value_info("output", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_eight_center_crop(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
