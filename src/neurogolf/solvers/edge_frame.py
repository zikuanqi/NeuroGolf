"""Solver: edge-replicate the grid by one cell, corners blanked (task 114).

The output is the input grown to ``(H+2) x (W+2)``: the original grid sits in
the centre, each side is extended outward by replicating the adjacent edge
row/column, and the four corners are left as background::

    1 2          0 1 2 0
    3 8    ->    1 1 2 2
                 3 3 8 8
                 0 3 8 0

Implemented as a clamped two-axis ``Gather`` (index ``clamp(i-1, 0, dim-1)``
gives the edge replication), masked to the ``(H+2) x (W+2)`` window, with the
four corner cells overwritten by the background channel.
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
    H, W = g.shape
    if H + 2 > HEIGHT or W + 2 > WIDTH:
        return None
    out = np.zeros((H + 2, W + 2), int)
    for r in range(H + 2):
        for c in range(W + 2):
            out[r, c] = g[min(max(r - 1, 0), H - 1), min(max(c - 1, 0), W - 1)]
    out[0, 0] = out[0, W + 1] = out[H + 1, 0] = out[H + 1, W + 1] = 0
    return out


def _detect(task: dict) -> bool:
    saw = False
    for ex in all_examples(task):
        i, o = ex["input"], ex["output"]
        if not i or not i[0] or len(i) + 2 > HEIGHT or len(i[0]) + 2 > WIDTH:
            continue
        r = _ref(np.array(i))
        if r is None or r.shape != np.array(o).shape or not np.array_equal(r, np.array(o)):
            return False
        saw = True
    return saw


def _build() -> onnx.ModelProto:
    n = helper.make_node
    e0 = np.zeros((1, CHANNELS, 1, 1), np.float32); e0[0, 0] = 1.0
    init = [
        numpy_helper.from_array(e0, "e0"),
        numpy_helper.from_array(np.arange(HEIGHT, dtype=np.float32).reshape(1, 1, HEIGHT, 1), "ah"),
        numpy_helper.from_array(np.arange(WIDTH, dtype=np.float32).reshape(1, 1, 1, WIDTH), "aw"),
        numpy_helper.from_array(np.array(0.5, np.float32), "half"),
        numpy_helper.from_array(np.array(1.0, np.float32), "one"),
        numpy_helper.from_array(np.array(2.0, np.float32), "two"),
        numpy_helper.from_array(np.array(0.0, np.float32), "zero"),
        numpy_helper.from_array(np.array([HEIGHT], np.int64), "h30"),
        numpy_helper.from_array(np.array([WIDTH], np.int64), "w30"),
    ]
    nodes = [
        n("ReduceSum", ["input"], ["occ"], axes=[1], keepdims=1),
        n("ReduceMax", ["occ"], ["rowData"], axes=[3], keepdims=1),
        n("ReduceMax", ["occ"], ["colData"], axes=[2], keepdims=1),
        n("ReduceSum", ["rowData"], ["H"], axes=[2], keepdims=1),
        n("ReduceSum", ["colData"], ["W"], axes=[3], keepdims=1),
        n("Sub", ["H", "one"], ["Hm1"]), n("Sub", ["W", "one"], ["Wm1"]),
        # row gather index = clamp(arange-1, 0, H-1)
        n("Sub", ["ah", "one"], ["amh"]),
        n("Max", ["amh", "zero"], ["loh"]),
        n("Min", ["loh", "Hm1"], ["rsf"]),
        n("Cast", ["rsf"], ["rsi"], to=I64),
        n("Reshape", ["rsi", "h30"], ["rs1d"]),
        n("Sub", ["aw", "one"], ["amw"]),
        n("Max", ["amw", "zero"], ["low"]),
        n("Min", ["low", "Wm1"], ["csf"]),
        n("Cast", ["csf"], ["csi"], to=I64),
        n("Reshape", ["csi", "w30"], ["cs1d"]),
        n("Gather", ["input", "rs1d"], ["gr"], axis=2),
        n("Gather", ["gr", "cs1d"], ["gathered"], axis=3),
        # valid window = (arange_h < H+2) & (arange_w < W+2)
        n("Add", ["H", "two"], ["Hp2"]), n("Add", ["W", "two"], ["Wp2"]),
        n("Less", ["ah", "Hp2"], ["rv_b"]), n("Cast", ["rv_b"], ["rv"], to=F),
        n("Less", ["aw", "Wp2"], ["cv_b"]), n("Cast", ["cv_b"], ["cvm"], to=F),
        n("Mul", ["rv", "cvm"], ["valid"]),
        # corner = (r==0 | r==H+1) & (c==0 | c==W+1)
        n("Add", ["H", "one"], ["Hp1"]), n("Sub", ["Hp1", "half"], ["Hp1h"]),
        n("Less", ["ah", "half"], ["r0_b"]), n("Cast", ["r0_b"], ["r0"], to=F),
        n("Greater", ["ah", "Hp1h"], ["re_b"]), n("Cast", ["re_b"], ["re"], to=F),
        n("Add", ["r0", "re"], ["rEdge"]),
        n("Add", ["W", "one"], ["Wp1"]), n("Sub", ["Wp1", "half"], ["Wp1h"]),
        n("Less", ["aw", "half"], ["c0_b"]), n("Cast", ["c0_b"], ["c0"], to=F),
        n("Greater", ["aw", "Wp1h"], ["ce_b"]), n("Cast", ["ce_b"], ["ce"], to=F),
        n("Add", ["c0", "ce"], ["cEdge"]),
        n("Mul", ["rEdge", "cEdge"], ["corner"]),
        # assemble
        n("Sub", ["one", "corner"], ["invc"]),
        n("Mul", ["valid", "invc"], ["ncv"]),
        n("Mul", ["valid", "corner"], ["cv"]),
        n("Mul", ["gathered", "ncv"], ["a"]),
        n("Mul", ["e0", "cv"], ["b"]),
        n("Add", ["a", "b"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "edge_frame",
                              [helper.make_tensor_value_info("input", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              [helper.make_tensor_value_info("output", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_edge_frame(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
