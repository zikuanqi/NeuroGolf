"""Solver: move a full cross down-left by the marker count (task 362).

A colour-C cross (one full row and one full column) sits in the grid alongside
some colour-5 markers.  The output is just that cross, moved down by k rows and
left by k columns, where k is the number of 5 markers (everything else cleared).

Build: C is the channel owning a full row and a full column (content-aware, and
excluding background and the marker colour 5); the cross position (cr, cc) and
k = sum(channel 5) give the new row cr+k and column cc-k, where colour C is
painted along `row == cr+k` or `col == cc-k`.
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
MARK = 5


def _ref(g: np.ndarray) -> Optional[np.ndarray]:
    H, W = g.shape
    C = cr = cx = None
    for c in np.unique(g):
        if c == 0 or c == MARK:
            continue
        rr = [r for r in range(H) if (g[r] == c).all()]
        cc = [x for x in range(W) if (g[:, x] == c).all()]
        if rr and cc:
            C, cr, cx = int(c), rr[0], cc[0]
    if C is None:
        return None
    k = int((g == MARK).sum())
    nr, nc = cr + k, cx - k
    out = np.zeros_like(g)
    if 0 <= nr < H:
        out[nr, :] = C
    if 0 <= nc < W:
        out[:, nc] = C
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
    row_idx = np.arange(HEIGHT, dtype=np.float32).reshape(1, 1, HEIGHT, 1)
    col_idx = np.arange(WIDTH, dtype=np.float32).reshape(1, 1, 1, WIDTH)
    note05 = np.ones((1, CHANNELS, 1, 1), np.float32); note05[0, 0] = 0.0; note05[0, MARK] = 0.0
    e0 = np.zeros((1, CHANNELS, 1, 1), np.float32); e0[0, 0] = 1.0
    init = [
        numpy_helper.from_array(row_idx, "row_idx"),
        numpy_helper.from_array(col_idx, "col_idx"),
        numpy_helper.from_array(note05, "note05"),
        numpy_helper.from_array(e0, "e0"),
        numpy_helper.from_array(np.array(0.5, np.float32), "half"),
        numpy_helper.from_array(np.array(1.0, np.float32), "one"),
        numpy_helper.from_array(np.array([0, MARK, 0, 0], np.int64), "m_s"),
        numpy_helper.from_array(np.array([1, MARK + 1, HEIGHT, WIDTH], np.int64), "m_e"),
        numpy_helper.from_array(np.array([0, 1, 2, 3], np.int64), "ax4"),
    ]

    def cf(b, name):
        return n("Cast", [b], [name], to=F)

    nodes = [
        n("ReduceSum", ["input"], ["content"], axes=[1], keepdims=1),
        n("ReduceSum", ["content"], ["con_r"], axes=[3], keepdims=1),    # (1,1,H,1)
        n("ReduceSum", ["content"], ["con_c"], axes=[2], keepdims=1),    # (1,1,1,W)
        n("ReduceSum", ["input"], ["cnt_r"], axes=[3], keepdims=1),      # (1,10,H,1)
        n("Sub", ["con_r", "cnt_r"], ["dr"]),
        n("Less", ["dr", "half"], ["fr_b"]), cf("fr_b", "fr0"),
        n("Greater", ["con_r", "half"], ["rpos_b"]), cf("rpos_b", "rpos"),
        n("Mul", ["fr0", "rpos"], ["fr"]),
        n("ReduceMax", ["fr"], ["hfr"], axes=[2], keepdims=1),           # (1,10,1,1)
        n("ReduceSum", ["input"], ["cnt_c"], axes=[2], keepdims=1),
        n("Sub", ["con_c", "cnt_c"], ["dc"]),
        n("Less", ["dc", "half"], ["fc_b"]), cf("fc_b", "fc0"),
        n("Greater", ["con_c", "half"], ["cpos_b"]), cf("cpos_b", "cpos"),
        n("Mul", ["fc0", "cpos"], ["fc"]),
        n("ReduceMax", ["fc"], ["hfc"], axes=[3], keepdims=1),
        n("Mul", ["hfr", "hfc"], ["hrc"]),
        n("Mul", ["hrc", "note05"], ["eC"]),                             # (1,10,1,1) cross colour
        # cross position
        n("Mul", ["input", "eC"], ["Cc"]),
        n("ReduceSum", ["Cc"], ["Cmask"], axes=[1], keepdims=1),         # (1,1,H,W)
        n("ReduceSum", ["Cmask"], ["Crow"], axes=[3], keepdims=1),
        n("Sub", ["con_r", "Crow"], ["drr"]),
        n("Less", ["drr", "half"], ["iscr_b"]), cf("iscr_b", "iscr0"),
        n("Mul", ["iscr0", "rpos"], ["iscr"]),
        n("Mul", ["iscr", "row_idx"], ["crw"]), n("ReduceSum", ["crw"], ["cr"]),
        n("ReduceSum", ["Cmask"], ["Ccol"], axes=[2], keepdims=1),
        n("Sub", ["con_c", "Ccol"], ["dcc"]),
        n("Less", ["dcc", "half"], ["iscc_b"]), cf("iscc_b", "iscc0"),
        n("Mul", ["iscc0", "cpos"], ["iscc"]),
        n("Mul", ["iscc", "col_idx"], ["ccw"]), n("ReduceSum", ["ccw"], ["cc"]),
        # marker count k
        n("Slice", ["input", "m_s", "m_e", "ax4"], ["ch5"]),
        n("ReduceSum", ["ch5"], ["k"]),                                  # scalar
        n("Add", ["cr", "k"], ["nr"]),
        n("Sub", ["cc", "k"], ["nc"]),
        # cross mask at (nr, nc)
        n("Sub", ["row_idx", "nr"], ["rd0"]), n("Abs", ["rd0"], ["rda"]),
        n("Less", ["rda", "half"], ["rowsel_b"]), cf("rowsel_b", "rowsel"),   # (1,1,H,1)
        n("Sub", ["col_idx", "nc"], ["cd0"]), n("Abs", ["cd0"], ["cda"]),
        n("Less", ["cda", "half"], ["colsel_b"]), cf("colsel_b", "colsel"),   # (1,1,1,W)
        n("Max", ["rowsel", "colsel"], ["cross0"]),                      # (1,1,H,W)
        n("Mul", ["cross0", "content"], ["cross"]),
        # output
        n("Mul", ["eC", "cross"], ["pC"]),
        n("Sub", ["one", "cross"], ["ncross"]),
        n("Mul", ["content", "ncross"], ["bg"]),
        n("Mul", ["e0", "bg"], ["pBg"]),
        n("Add", ["pC", "pBg"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "cross_move",
                              [helper.make_tensor_value_info("input", F, FULL)],
                              [helper.make_tensor_value_info("output", F, FULL)],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_cross_move(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
