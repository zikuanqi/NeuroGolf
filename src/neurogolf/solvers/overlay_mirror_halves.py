"""Solver: overlay the left half with the mirrored right half (task 360).

A full column of ``5`` splits the grid into equal halves.  The output (the left
half's width) is the left half with the horizontally-mirrored right half laid on
top; the two halves never collide, so it is a disjoint union.

Fixed geometry: a 9-wide grid with the divider at column 4, halves of width 4.
``out[:, c] = nonbg(in[:, c]) + nonbg(in[:, 8-c])`` for ``c in 0..3``.
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


def _ref(g: np.ndarray):
    H, W = g.shape
    divs = [c for c in range(W) if np.all(g[:, c] == 5)]
    if len(divs) != 1:
        return None
    d = divs[0]
    left = g[:, :d]; right = g[:, d + 1:]
    if left.shape[1] != right.shape[1] or left.shape[1] == 0:
        return None
    rmir = right[:, ::-1]
    out = left.copy()
    for r in range(H):
        for c in range(left.shape[1]):
            if out[r, c] == 0:
                out[r, c] = rmir[r, c]
            elif rmir[r, c] != 0 and rmir[r, c] != out[r, c]:
                return None  # collision -> not this rule
    return out, d


def _detect(task: dict) -> bool:
    saw = False
    for ex in all_examples(task):
        i, o = ex["input"], ex["output"]
        if not i or not i[0] or len(i) > HEIGHT or len(i[0]) > WIDTH:
            continue
        a = np.array(i)
        res = _ref(a)
        if res is None:
            return False
        out, d = res
        # build is hardcoded for a 9-wide grid, divider at column 4
        if a.shape[1] != 9 or d != 4:
            return False
        if out.tolist() != o:
            return False
        saw = True
    return saw


def _build() -> onnx.ModelProto:
    n = helper.make_node
    e0 = np.zeros((1, CHANNELS, 1, 1), np.float32); e0[0, 0] = 1.0
    notbg = np.ones((1, CHANNELS, 1, 1), np.float32); notbg[0, 0] = 0.0
    colmask = np.zeros((1, 1, 1, WIDTH), np.float32); colmask[0, 0, 0, :4] = 1.0
    idxmir = np.zeros(WIDTH, np.int64)
    for c in range(4):
        idxmir[c] = 8 - c
    init = [
        numpy_helper.from_array(e0, "e0"),
        numpy_helper.from_array(notbg, "notbg"),
        numpy_helper.from_array(colmask, "colmask"),
        numpy_helper.from_array(idxmir, "idxmir"),
    ]
    nodes = [
        n("ReduceSum", ["input"], ["occ"], axes=[1], keepdims=1),
        n("Mul", ["occ", "colmask"], ["region"]),
        n("Mul", ["input", "notbg"], ["inNonbg"]),
        n("Mul", ["inNonbg", "colmask"], ["leftPart"]),
        n("Gather", ["inNonbg", "idxmir"], ["mirAll"], axis=3),
        n("Mul", ["mirAll", "colmask"], ["mirPart"]),
        n("Add", ["leftPart", "mirPart"], ["colorLayer"]),
        n("ReduceSum", ["colorLayer"], ["nonbgCnt"], axes=[1], keepdims=1),
        n("Sub", ["region", "nonbgCnt"], ["bgv"]),
        n("Mul", ["bgv", "e0"], ["bgLayer"]),
        n("Add", ["colorLayer", "bgLayer"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "overlay_mirror_halves",
                              [helper.make_tensor_value_info("input", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              [helper.make_tensor_value_info("output", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_overlay_mirror_halves(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
