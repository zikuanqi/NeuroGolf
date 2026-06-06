"""Solver: swap which line is on top at a crossing (task 293).

A horizontal line of colour A and a vertical line of colour B overlap; in the
input one line is drawn over the other at the crossing.  The output swaps them:
every crossing cell showing A becomes B and vice versa, so the occluded line is
the one now visible.

Build (all colours detected at runtime): A is the non-bg channel with the widest
column span (a full-width row band), B the other non-bg colour.  The crossing is
(rows where A appears) x (cols where B appears); there A<->B are swapped via
`(e_B - e_A)` on the A cells and `(e_A - e_B)` on the B cells.
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
    cols = [int(c) for c in np.unique(g) if c != 0]
    if len(cols) != 2:
        return None
    span = {c: int((g == c).any(axis=0).sum()) for c in cols}      # column span
    A = max(cols, key=lambda c: span[c])
    B = min(cols, key=lambda c: span[c])
    if span[A] == span[B]:
        return None
    Arows = (g == A).any(axis=1)
    Bcols = (g == B).any(axis=0)
    out = g.copy()
    for r in np.where(Arows)[0]:
        for c in np.where(Bcols)[0]:
            if g[r, c] == A:
                out[r, c] = B
            elif g[r, c] == B:
                out[r, c] = A
    return out


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
    note0 = np.ones((1, CHANNELS, 1, 1), np.float32); note0[0, 0] = 0.0
    init = [
        numpy_helper.from_array(note0, "note0"),
        numpy_helper.from_array(np.array(0.5, np.float32), "half"),
        numpy_helper.from_array(np.array(1.0, np.float32), "one"),
    ]
    nodes = [
        n("ReduceMax", ["input"], ["colpres"], axes=[2], keepdims=1),   # (1,10,1,W)
        n("ReduceSum", ["colpres"], ["colspan"], axes=[3], keepdims=1), # (1,10,1,1)
        n("Mul", ["colspan", "note0"], ["span_nb"]),
        n("ReduceMax", ["span_nb"], ["maxspan"], axes=[1], keepdims=1), # (1,1,1,1)
        n("Sub", ["maxspan", "half"], ["max_mh"]),
        n("Greater", ["span_nb", "max_mh"], ["eA_b"]),
        n("Cast", ["eA_b"], ["eA"], to=F),                              # (1,10,1,1) horizontal colour
        n("Greater", ["span_nb", "half"], ["pres_b"]),
        n("Cast", ["pres_b"], ["pres"], to=F),
        n("Sub", ["one", "eA"], ["notA"]),
        n("Mul", ["pres", "notA"], ["eB"]),                            # the other colour
        n("Mul", ["input", "eA"], ["inA_c"]),
        n("ReduceSum", ["inA_c"], ["inA"], axes=[1], keepdims=1),       # (1,1,H,W)
        n("Mul", ["input", "eB"], ["inB_c"]),
        n("ReduceSum", ["inB_c"], ["inB"], axes=[1], keepdims=1),
        n("ReduceMax", ["inA"], ["rowA"], axes=[3], keepdims=1),        # (1,1,H,1)
        n("ReduceMax", ["inB"], ["colB"], axes=[2], keepdims=1),        # (1,1,1,W)
        n("Mul", ["rowA", "colB"], ["inter"]),                         # (1,1,H,W) crossing
        n("Mul", ["inA", "inter"], ["cellA"]),
        n("Mul", ["inB", "inter"], ["cellB"]),
        n("Sub", ["eB", "eA"], ["eBmA"]),
        n("Sub", ["eA", "eB"], ["eAmB"]),
        n("Mul", ["eBmA", "cellA"], ["dA"]),                           # (1,10,H,W)
        n("Mul", ["eAmB", "cellB"], ["dB"]),
        n("Add", ["input", "dA"], ["s1"]),
        n("Add", ["s1", "dB"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "line_cross_swap",
                              [helper.make_tensor_value_info("input", F, FULL)],
                              [helper.make_tensor_value_info("output", F, FULL)],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_line_cross_swap(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
