"""Solver: a vertical 7-line expands into a 7/8 triangle (task 348).

A single column of ``7`` (rows ``0..L-1``) becomes a downward triangle: row ``r``
spans columns ``c0 +/- (L-1-r)``, coloured ``7`` where the column offset from the
line is even and ``8`` where it is odd.

``r + |c - c0| <= L-1`` selects the triangle; ``c0`` and ``L`` come from the
``7`` mask's column/row projections.
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
    ys, xs = np.where(g == 7)
    if len(ys) == 0 or not np.all(xs == xs[0]):
        return None
    c0 = xs[0]
    L = ys.max() + 1
    if set(ys.tolist()) != set(range(L)):
        return None
    out = np.zeros_like(g)
    for r in range(H):
        for c in range(W):
            d = abs(c - c0)
            if r + d <= L - 1:
                out[r, c] = 7 if d % 2 == 0 else 8
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


def _ev(k):
    v = np.zeros((1, CHANNELS, 1, 1), np.float32); v[0, k] = 1.0
    return v


def _build() -> onnx.ModelProto:
    n = helper.make_node
    rowidx = np.arange(HEIGHT, dtype=np.float32).reshape(1, 1, HEIGHT, 1)
    colidx = np.arange(WIDTH, dtype=np.float32).reshape(1, 1, 1, WIDTH)
    init = [
        numpy_helper.from_array(_ev(0), "e0"),
        numpy_helper.from_array(_ev(7), "e7"),
        numpy_helper.from_array(_ev(8), "e8"),
        numpy_helper.from_array(rowidx, "rowidx"),
        numpy_helper.from_array(colidx, "colidx"),
        numpy_helper.from_array(np.array(0.5, np.float32), "half"),
        numpy_helper.from_array(np.array(2.0, np.float32), "two"),
        numpy_helper.from_array(np.array([7], np.int64), "c7s"),
        numpy_helper.from_array(np.array([8], np.int64), "c8e"),
        numpy_helper.from_array(np.array([1], np.int64), "ax1"),
    ]
    nodes = [
        n("ReduceSum", ["input"], ["occ"], axes=[1], keepdims=1),
        n("Slice", ["input", "c7s", "c8e", "ax1"], ["is7"]),
        n("ReduceMax", ["is7"], ["colC0"], axes=[2], keepdims=1),       # (1,1,1,W)
        n("Mul", ["colC0", "colidx"], ["c0w"]),
        n("ReduceSum", ["c0w"], ["c0"], axes=[3], keepdims=1),          # (1,1,1,1)
        n("Mul", ["is7", "rowidx"], ["row7"]),
        n("ReduceMax", ["row7"], ["maxRow"], axes=[2, 3], keepdims=1),  # L-1
        n("Sub", ["colidx", "c0"], ["cdiff"]),
        n("Abs", ["cdiff"], ["distCol"]),                                # (1,1,1,W)
        n("Add", ["rowidx", "distCol"], ["triVal"]),                     # (1,1,H,W)
        n("Add", ["maxRow", "half"], ["thr"]),
        n("Less", ["triVal", "thr"], ["mask_b"]), n("Cast", ["mask_b"], ["mask"], to=F),
        n("Mul", ["mask", "occ"], ["maskG"]),
        # parity of distCol
        n("Mul", ["distCol", "half"], ["dh"]), n("Floor", ["dh"], ["dfl"]),
        n("Mul", ["dfl", "two"], ["dbl"]), n("Sub", ["distCol", "dbl"], ["rem"]),
        n("Less", ["rem", "half"], ["even_b"]), n("Cast", ["even_b"], ["evenCol"], to=F),
        n("Mul", ["maskG", "evenCol"], ["c7cells"]),
        n("Sub", ["maskG", "c7cells"], ["c8cells"]),
        n("Mul", ["c7cells", "e7"], ["L7"]),
        n("Mul", ["c8cells", "e8"], ["L8"]),
        n("Sub", ["occ", "maskG"], ["bgv"]),
        n("Mul", ["bgv", "e0"], ["L0"]),
        n("Add", ["L7", "L8"], ["t1"]),
        n("Add", ["t1", "L0"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "line_triangle_expand",
                              [helper.make_tensor_value_info("input", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              [helper.make_tensor_value_info("output", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_line_triangle_expand(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
