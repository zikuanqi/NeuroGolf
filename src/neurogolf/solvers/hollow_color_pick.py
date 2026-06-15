"""Solver: output the colour of the one hollow rectangle as a 1x1 grid (task 291).

Among several solid rectangular blocks, exactly one is a hollow frame (its cell
count is smaller than its bounding-box area).  The output is a single cell of
that colour.

Per colour: ``count`` from a spatial ``ReduceSum``; bounding-box extents from
``ReduceMax``/``ReduceMin`` of the row/column-presence index; hollow iff
``count < extent_r * extent_c``.
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
    best = None
    for c in range(1, 10):
        ys, xs = np.where(g == c)
        if len(ys) == 0:
            continue
        area = (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)
        if len(ys) < area:
            if best is not None:
                return None
            best = c
    return np.array([[best]]) if best else None


def _detect(task: dict) -> bool:
    saw = False
    for ex in all_examples(task):
        i, o = ex["input"], ex["output"]
        if not i or not i[0] or len(i) > HEIGHT or len(i[0]) > WIDTH:
            continue
        r = _ref(np.array(i))
        if r is None or r.tolist() != o:
            return False
        saw = True
    return saw


def _build() -> onnx.ModelProto:
    n = helper.make_node
    rowidx = np.arange(HEIGHT, dtype=np.float32).reshape(1, 1, HEIGHT, 1)
    colidx = np.arange(WIDTH, dtype=np.float32).reshape(1, 1, 1, WIDTH)
    notbg = np.ones((1, CHANNELS, 1, 1), np.float32); notbg[0, 0] = 0.0
    corner = np.zeros((1, 1, HEIGHT, WIDTH), np.float32); corner[0, 0, 0, 0] = 1.0
    init = [
        numpy_helper.from_array(rowidx, "rowidx"),
        numpy_helper.from_array(colidx, "colidx"),
        numpy_helper.from_array(notbg, "notbg"),
        numpy_helper.from_array(corner, "corner"),
        numpy_helper.from_array(np.array(1.0, np.float32), "one"),
        numpy_helper.from_array(np.array(0.5, np.float32), "half"),
        numpy_helper.from_array(np.array(1e4, np.float32), "big"),
    ]
    nodes = [
        n("ReduceSum", ["input"], ["cnt"], axes=[2, 3], keepdims=1),          # (1,10,1,1)
        n("ReduceMax", ["input"], ["rowHas"], axes=[3], keepdims=1),          # (1,10,30,1)
        n("ReduceMax", ["input"], ["colHas"], axes=[2], keepdims=1),          # (1,10,1,30)
        # row extent
        n("Mul", ["rowHas", "rowidx"], ["rIdxM"]),
        n("ReduceMax", ["rIdxM"], ["maxR"], axes=[2], keepdims=1),
        n("Sub", ["one", "rowHas"], ["rowInv"]),
        n("Mul", ["rowInv", "big"], ["rowInvB"]),
        n("Mul", ["rowHas", "rowidx"], ["rIdxM2"]),
        n("Add", ["rIdxM2", "rowInvB"], ["rMinCand"]),
        n("ReduceMin", ["rMinCand"], ["minR"], axes=[2], keepdims=1),
        n("Sub", ["maxR", "minR"], ["rSpan"]),
        n("Add", ["rSpan", "one"], ["rExt"]),
        # col extent
        n("Mul", ["colHas", "colidx"], ["cIdxM"]),
        n("ReduceMax", ["cIdxM"], ["maxC"], axes=[3], keepdims=1),
        n("Sub", ["one", "colHas"], ["colInv"]),
        n("Mul", ["colInv", "big"], ["colInvB"]),
        n("Mul", ["colHas", "colidx"], ["cIdxM2"]),
        n("Add", ["cIdxM2", "colInvB"], ["cMinCand"]),
        n("ReduceMin", ["cMinCand"], ["minC"], axes=[3], keepdims=1),
        n("Sub", ["maxC", "minC"], ["cSpan"]),
        n("Add", ["cSpan", "one"], ["cExt"]),
        # area, hollow test
        n("Mul", ["rExt", "cExt"], ["area"]),
        n("Sub", ["area", "cnt"], ["deficit"]),
        n("Greater", ["deficit", "half"], ["hol_b"]), n("Cast", ["hol_b"], ["hol"], to=F),
        n("Greater", ["cnt", "half"], ["pres_b"]), n("Cast", ["pres_b"], ["pres"], to=F),
        n("Mul", ["hol", "pres"], ["h1"]),
        n("Mul", ["h1", "notbg"], ["pick"]),       # (1,10,1,1) one-hot colour
        n("Mul", ["pick", "corner"], ["output"]),  # place at (0,0)
    ]
    graph = helper.make_graph(nodes, "hollow_color_pick",
                              [helper.make_tensor_value_info("input", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              [helper.make_tensor_value_info("output", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_hollow_color_pick(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
