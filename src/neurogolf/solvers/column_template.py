"""Solver: fill rows from a template's repeated-column pattern (task 197).

One row is a full-width "template"; the others are seeds (a short prefix, rest
blank).  Every row is rewritten so column c takes the colour the row has at the
first column that shares the template's colour at c.  For the template this is a
no-op; for a seed it tiles/recolours its prefix into the template's pattern.

Build: the template is the row with the most non-background cells.  A pairwise
column-equality matrix (one-hot dot via `MatMul`) gives, per column, the first
column with the same template colour (`ReduceMin` of a masked index ramp); the
whole grid is then `Gather`-ed along columns by those indices.
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
    nb = [(g[r] != 0).sum() for r in range(H)]
    tr = int(np.argmax(nb))
    tmpl = g[tr]
    if (tmpl == 0).any():
        return None
    firstcol = np.zeros(W, int)
    for c in range(W):
        for cp in range(W):
            if tmpl[cp] == tmpl[c]:
                firstcol[c] = cp
                break
    out = g.copy()
    for r in range(H):
        if nb[r] == 0:
            continue
        for c in range(W):
            out[r, c] = g[r, firstcol[c]]
    if np.array_equal(out, g):
        return None
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
    col_ramp = np.arange(WIDTH, dtype=np.float32).reshape(1, WIDTH)
    init = [
        numpy_helper.from_array(note0, "note0"),
        numpy_helper.from_array(col_ramp, "col_ramp"),
        numpy_helper.from_array(np.arange(WIDTH, dtype=np.float32), "col_flat"),
        numpy_helper.from_array(np.array(1.0, np.float32), "one"),
        numpy_helper.from_array(np.array(1000.0, np.float32), "big"),
        numpy_helper.from_array(np.array([CHANNELS, WIDTH], np.int64), "rs_tmpl"),
        numpy_helper.from_array(np.array([1], np.int64), "rs1"),
    ]
    nodes = [
        n("Mul", ["input", "note0"], ["inb"]),
        n("ReduceSum", ["inb"], ["rowcount"], axes=[1, 3], keepdims=1),     # (1,1,H,1)
        n("ArgMax", ["rowcount"], ["tr_i"], axis=2, keepdims=1),            # (1,1,1,1)
        n("Reshape", ["tr_i", "rs1"], ["tr_1d"]),                           # (1,)
        n("Gather", ["input", "tr_1d"], ["template"], axis=2),             # (1,10,1,W)
        n("Reshape", ["template", "rs_tmpl"], ["tmpl2d"]),                 # (10,W)
        n("Transpose", ["tmpl2d"], ["tmplT"], perm=[1, 0]),               # (W,10)
        n("MatMul", ["tmplT", "tmpl2d"], ["equal"]),                       # (W,W) 1 if same colour
        n("Sub", ["one", "equal"], ["neq"]),
        n("Mul", ["big", "neq"], ["bigterm"]),
        n("Mul", ["col_ramp", "equal"], ["colterm"]),
        n("Add", ["colterm", "bigterm"], ["masked"]),                      # c' if equal else BIG
        n("ReduceMin", ["masked"], ["firstcol_raw"], axes=[1], keepdims=0),    # (W,)
        n("Min", ["firstcol_raw", "col_flat"], ["firstcol"]),              # clamp padding cols to self
        n("Cast", ["firstcol"], ["firstcol_i"], to=TensorProto.INT64),
        n("Gather", ["input", "firstcol_i"], ["output"], axis=3),         # (1,10,H,W)
    ]
    graph = helper.make_graph(nodes, "column_template",
                              [helper.make_tensor_value_info("input", F, FULL)],
                              [helper.make_tensor_value_info("output", F, FULL)],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_column_template(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
