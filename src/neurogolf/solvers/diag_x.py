"""Solver: draw a diagonal X from a single marker cell (task 141).

The input is a blank grid with one coloured marker; the output draws the full
4-way diagonal cross through it (every cell with |r-mr| == |c-mc|) in the
marker's colour, out to the grid edges.

Build: the marker colour one-hot is the present non-background channel; its
position (mr, mc) comes from a weighted `ReduceSum`; the X mask is
`abs(|row-mr| - |col-mc|) < 0.5` intersected with the in-grid content, and the
marker colour is painted there.
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
    nz = np.argwhere(g != 0)
    if len(nz) != 1:
        return None
    mr, mc = nz[0]
    M = g[mr, mc]
    out = g.copy()
    for r in range(H):
        for c in range(W):
            if abs(r - mr) == abs(c - mc):
                out[r, c] = M
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
    B = TensorProto.BOOL
    n = helper.make_node
    row_idx = np.arange(HEIGHT, dtype=np.float32).reshape(1, 1, HEIGHT, 1)
    col_idx = np.arange(WIDTH, dtype=np.float32).reshape(1, 1, 1, WIDTH)
    note0 = np.ones((1, CHANNELS, 1, 1), np.float32); note0[0, 0] = 0.0
    init = [
        numpy_helper.from_array(row_idx, "row_idx"),
        numpy_helper.from_array(col_idx, "col_idx"),
        numpy_helper.from_array(note0, "note0"),
        numpy_helper.from_array(np.array(0.5, np.float32), "half"),
        numpy_helper.from_array(np.array(1.0, np.float32), "one"),
    ]
    s1 = [1, 1, HEIGHT, WIDTH]; chan = [1, CHANNELS, 1, 1]
    rc = [1, 1, HEIGHT, 1]; cc = [1, 1, 1, WIDTH]; sc = [1, 1, 1, 1]
    nodes = [
        n("ReduceSum", ["input"], ["tot"], axes=[2, 3], keepdims=1),     # (1,10,1,1)
        n("Greater", ["tot", "half"], ["pres_b"]), n("Cast", ["pres_b"], ["present"], to=F),
        n("Mul", ["present", "note0"], ["eM"]),                          # (1,10,1,1)
        n("Mul", ["input", "eM"], ["mk_c"]),
        n("ReduceSum", ["mk_c"], ["marker"], axes=[1], keepdims=1),      # (1,1,H,W)
        n("Mul", ["marker", "row_idx"], ["mkr"]),
        n("ReduceSum", ["mkr"], ["mr"]),                                 # scalar
        n("Mul", ["marker", "col_idx"], ["mkc"]),
        n("ReduceSum", ["mkc"], ["mc"]),
        n("Sub", ["row_idx", "mr"], ["dr0"]), n("Abs", ["dr0"], ["dr"]),  # (1,1,H,1)
        n("Sub", ["col_idx", "mc"], ["dc0"]), n("Abs", ["dc0"], ["dc"]),  # (1,1,1,W)
        n("Sub", ["dr", "dc"], ["dd"]), n("Abs", ["dd"], ["dda"]),        # (1,1,H,W)
        n("Less", ["dda", "half"], ["mx_b"]), n("Cast", ["mx_b"], ["maskX"], to=F),
        n("ReduceSum", ["input"], ["content"], axes=[1], keepdims=1),
        n("Mul", ["maskX", "content"], ["mg"]),
        n("Sub", ["one", "mg"], ["keep"]),
        n("Mul", ["input", "keep"], ["kept"]),
        n("Mul", ["eM", "mg"], ["paint"]),
        n("Add", ["kept", "paint"], ["output"]),
    ]
    vi = [
        helper.make_tensor_value_info("tot", F, chan),
        helper.make_tensor_value_info("pres_b", B, chan),
        helper.make_tensor_value_info("present", F, chan),
        helper.make_tensor_value_info("eM", F, chan),
        helper.make_tensor_value_info("marker", F, s1),
        helper.make_tensor_value_info("mr", F, sc),
        helper.make_tensor_value_info("mc", F, sc),
        helper.make_tensor_value_info("dr", F, rc),
        helper.make_tensor_value_info("dc", F, cc),
        helper.make_tensor_value_info("dda", F, s1),
        helper.make_tensor_value_info("mx_b", B, s1),
        helper.make_tensor_value_info("maskX", F, s1),
        helper.make_tensor_value_info("content", F, s1),
        helper.make_tensor_value_info("mg", F, s1),
    ]
    graph = helper.make_graph(nodes, "diag_x",
                              [helper.make_tensor_value_info("input", F, FULL)],
                              [helper.make_tensor_value_info("output", F, FULL)],
                              initializer=init, value_info=vi)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_diag_x(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
