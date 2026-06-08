"""Solver: reverse the colours of concentric square rings (task 203).

The grid is a set of nested square rings, each a solid colour.  The output keeps
the ring geometry but reverses the colour order -- the outermost ring takes the
centre colour and vice versa::

    4 4 4 4          8 8 8 8
    4 2 2 4    ->    8 5 5 8
    4 2 2 4          8 5 5 8
    4 4 4 4          8 8 8 8        (rings 4,2,..  ->  reversed)

With ``d(r,c) = min(r, c, H-1-r, W-1-c)`` the ring depth and ``D`` the centre
depth, ``out[r,c] = in[D-d, D-d]`` (the diagonal corner of the mirror ring).

Build: extract the diagonal colour palette ``pal[k] = in[k,k]`` (via an identity
mask + ``ReduceSum``); compute ``d`` from the real-grid extent and ``e = D-d``;
``Gather`` the palette by ``e`` per cell, masked to the real grid.
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
    R = np.arange(H).reshape(-1, 1); C = np.arange(W).reshape(1, -1)
    d = np.minimum(np.minimum(R, H - 1 - R), np.minimum(C, W - 1 - C))
    D = int(d.max())
    out = g.copy()
    for r in range(H):
        for c in range(W):
            e = D - d[r, c]
            out[r, c] = g[e, e]
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
    I = TensorProto.INT64
    n = helper.make_node

    eye = np.eye(HEIGHT, WIDTH, dtype=np.float32).reshape(1, 1, HEIGHT, WIDTH)
    row_idx = np.arange(HEIGHT, dtype=np.float32).reshape(1, 1, HEIGHT, 1)
    col_idx = np.arange(WIDTH, dtype=np.float32).reshape(1, 1, 1, WIDTH)
    init = [
        numpy_helper.from_array(eye, "eye"),
        numpy_helper.from_array(row_idx, "row_idx"),
        numpy_helper.from_array(col_idx, "col_idx"),
        numpy_helper.from_array(np.array(0.0, np.float32), "clo"),
        numpy_helper.from_array(np.array(float(HEIGHT - 1), np.float32), "chi"),
        numpy_helper.from_array(np.array([HEIGHT, WIDTH], np.int64), "shp"),
    ]
    nodes = [
        n("ReduceSum", ["input"], ["content"], axes=[1], keepdims=1),       # (1,1,H,W)
        # diagonal palette pal[k] = input[:,:,k,k]
        n("Mul", ["input", "eye"], ["idiag"]),
        n("ReduceSum", ["idiag"], ["pal"], axes=[3], keepdims=0),           # (1,C,H)
        # real-grid extent
        n("ReduceMax", ["content"], ["realrow"], axes=[3], keepdims=1),     # (1,1,H,1)
        n("Mul", ["row_idx", "realrow"], ["rr"]),
        n("ReduceMax", ["rr"], ["rlast"], axes=[2], keepdims=1),            # (1,1,1,1)
        n("ReduceMax", ["content"], ["realcol"], axes=[2], keepdims=1),     # (1,1,1,W)
        n("Mul", ["col_idx", "realcol"], ["cc"]),
        n("ReduceMax", ["cc"], ["clast"], axes=[3], keepdims=1),            # (1,1,1,1)
        # ring depth d = min(r, c, rlast-r, clast-c)
        n("Sub", ["rlast", "row_idx"], ["botd"]),
        n("Sub", ["clast", "col_idx"], ["rightd"]),
        n("Min", ["row_idx", "botd"], ["m1"]),
        n("Min", ["col_idx", "rightd"], ["m2"]),
        n("Min", ["m1", "m2"], ["d"]),                                     # (1,1,H,W)
        n("Mul", ["d", "content"], ["dc"]),
        n("ReduceMax", ["dc"], ["D"], axes=[2, 3], keepdims=1),            # (1,1,1,1)
        n("Sub", ["D", "d"], ["e"]),
        n("Clip", ["e", "clo", "chi"], ["ecl"]),
        n("Reshape", ["ecl", "shp"], ["e2d"]),                             # (H,W)
        n("Cast", ["e2d"], ["eidx"], to=I),
        n("Gather", ["pal", "eidx"], ["gathered"], axis=2),                # (1,C,H,W)
        n("Mul", ["gathered", "content"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "ring_reverse",
                              [helper.make_tensor_value_info("input", F, FULL)],
                              [helper.make_tensor_value_info("output", F, FULL)],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_ring_reverse(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
