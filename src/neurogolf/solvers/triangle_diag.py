"""Solver: diagonal triangle around a left-anchored 2-segment (task 256).

A row r0 holds a left-anchored run of colour 2 ending at column c1.  The output
fills the left-anchored region under the diagonal c <= c1 + (r0 - r): colour 3
above r0, colour 2 on it, colour 1 below; everything else is background.

Build: r0 and c1 come from index-ramp reductions of channel 2; an index mask
`col <= c1 + r0 - row` gives the region, split by row vs r0 into the three
colours and OR-ed with background over the grid.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from ..grids import CHANNELS, HEIGHT, WIDTH, all_examples

OPSET = 11
IR_VERSION = 8
FULL = [1, CHANNELS, HEIGHT, WIDTH]


def _ref(g: np.ndarray) -> Optional[np.ndarray]:
    H, W = g.shape
    ys, xs = np.where(g == 2)
    if len(ys) == 0 or len(set(ys.tolist())) != 1:
        return None
    r0 = int(ys[0]); c1 = int(xs.max())
    out = np.zeros_like(g)
    for r in range(H):
        for c in range(W):
            if 0 <= c <= c1 + r0 - r:
                out[r, c] = 3 if r < r0 else (2 if r == r0 else 1)
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
    F = TensorProto.FLOAT
    n = helper.make_node

    def cf(b, name):
        return n("Cast", [b], [name], to=F)

    row_idx = np.arange(HEIGHT, dtype=np.float32).reshape(1, 1, HEIGHT, 1)
    col_idx = np.arange(WIDTH, dtype=np.float32).reshape(1, 1, 1, WIDTH)
    init = [
        numpy_helper.from_array(row_idx, "row_idx"),
        numpy_helper.from_array(col_idx, "col_idx"),
        numpy_helper.from_array(np.array(0.5, np.float32), "half"),
        numpy_helper.from_array(np.array([0, 2, 0, 0], np.int64), "s2"),
        numpy_helper.from_array(np.array([1, 3, HEIGHT, WIDTH], np.int64), "e2s"),
        numpy_helper.from_array(np.array([0, 1, 2, 3], np.int64), "ax4"),
    ]
    for k in (0, 1, 2, 3):
        v = np.zeros((1, CHANNELS, 1, 1), np.float32); v[0, k] = 1.0
        init.append(numpy_helper.from_array(v, f"e{k}"))
    nodes = [
        n("Slice", ["input", "s2", "e2s", "ax4"], ["ch2"]),               # (1,1,H,W)
        n("ReduceMax", ["ch2"], ["rowhas2"], axes=[3], keepdims=1),       # (1,1,H,1)
        n("Mul", ["rowhas2", "row_idx"], ["rr"]), n("ReduceMax", ["rr"], ["r0"], keepdims=0),
        n("Mul", ["ch2", "col_idx"], ["cc"]), n("ReduceMax", ["cc"], ["c1"], keepdims=0),
        n("Add", ["c1", "r0"], ["c1r0"]),
        n("Sub", ["c1r0", "row_idx"], ["thr"]),                           # (1,1,H,1)
        n("Add", ["thr", "half"], ["thr_h"]),
        n("Less", ["col_idx", "thr_h"], ["reg_b"]), cf("reg_b", "region"),  # (1,1,H,W)
        n("ReduceSum", ["input"], ["grid"], axes=[1], keepdims=1),
        n("Mul", ["region", "grid"], ["regiong"]),
        n("Less", ["row_idx", "r0"], ["ab_b"]), cf("ab_b", "above"),
        n("Greater", ["row_idx", "r0"], ["be_b"]), cf("be_b", "below"),
        n("Sub", ["row_idx", "r0"], ["dr"]), n("Abs", ["dr"], ["adr"]),
        n("Less", ["adr", "half"], ["on_b"]), cf("on_b", "on"),
        n("Mul", ["regiong", "above"], ["r_ab"]),
        n("Mul", ["regiong", "on"], ["r_on"]),
        n("Mul", ["regiong", "below"], ["r_be"]),
        n("Add", ["r_ab", "r_on"], ["c01"]), n("Add", ["c01", "r_be"], ["colored"]),
        n("Sub", ["grid", "colored"], ["bg"]),
        n("Mul", ["e3", "r_ab"], ["p3"]),
        n("Mul", ["e2", "r_on"], ["p2"]),
        n("Mul", ["e1", "r_be"], ["p1"]),
        n("Mul", ["e0", "bg"], ["pbg"]),
        n("Add", ["p3", "p2"], ["q1"]), n("Add", ["p1", "pbg"], ["q2"]),
        n("Add", ["q1", "q2"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "triangle_diag",
                              [helper.make_tensor_value_info("input", F, FULL)],
                              [helper.make_tensor_value_info("output", F, FULL)],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_triangle_diag(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
