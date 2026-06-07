"""Solver: connect each colour's marker pair with a line (task 92).

Every non-background colour appears as exactly two aligned markers.  The output
draws a straight line of that colour between them (horizontal if same row,
vertical if same column).  Where a vertical and a horizontal line cross, the
vertical one is drawn on top.

Build (per channel, background excluded): "between first and last marker in a
row" is `prefix-has AND suffix-has` via forward/reverse `CumSum` along the width;
likewise along the height for vertical.  Vertical fills take precedence over
horizontal, and remaining in-grid cells stay background.
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
    out = g.copy()
    hf, vf = [], []
    for c in np.unique(g):
        if c == 0:
            continue
        ys, xs = np.where(g == c)
        if len(ys) != 2:
            continue
        if ys[0] == ys[1]:
            hf.append((ys[0], min(xs), max(xs), c))
        elif xs[0] == xs[1]:
            vf.append((xs[0], min(ys), max(ys), c))
        else:
            return None
    if not hf and not vf:
        return None
    for (r, c0, c1, c) in hf:
        out[r, c0:c1 + 1] = c
    for (cc, r0, r1, c) in vf:
        out[r0:r1 + 1, cc] = c
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

    def cf(b, name):
        return n("Cast", [b], [name], to=F)

    note0 = np.ones((1, CHANNELS, 1, 1), np.float32); note0[0, 0] = 0.0
    e0 = np.zeros((1, CHANNELS, 1, 1), np.float32); e0[0, 0] = 1.0
    init = [
        numpy_helper.from_array(note0, "note0"),
        numpy_helper.from_array(e0, "e0"),
        numpy_helper.from_array(np.array(0.5, np.float32), "half"),
        numpy_helper.from_array(np.array(1.0, np.float32), "one"),
        numpy_helper.from_array(np.array(3, np.int64), "ax3"),
        numpy_helper.from_array(np.array(2, np.int64), "ax2"),
    ]
    nodes = [
        n("Mul", ["input", "note0"], ["inb"]),                          # drop background channel
        # horizontal: between first and last marker in each row
        n("CumSum", ["inb", "ax3"], ["pfh"]),
        n("CumSum", ["inb", "ax3"], ["sfh"], reverse=1),
        n("Greater", ["pfh", "half"], ["pfh_b"]), cf("pfh_b", "pfhf"),
        n("Greater", ["sfh", "half"], ["sfh_b"]), cf("sfh_b", "sfhf"),
        n("Mul", ["pfhf", "sfhf"], ["hfill"]),                          # (1,10,H,W)
        # vertical
        n("CumSum", ["inb", "ax2"], ["pfv"]),
        n("CumSum", ["inb", "ax2"], ["sfv"], reverse=1),
        n("Greater", ["pfv", "half"], ["pfv_b"]), cf("pfv_b", "pfvf"),
        n("Greater", ["sfv", "half"], ["sfv_b"]), cf("sfv_b", "sfvf"),
        n("Mul", ["pfvf", "sfvf"], ["vfill"]),
        # vertical on top
        n("ReduceSum", ["vfill"], ["vany"], axes=[1], keepdims=1),      # (1,1,H,W)
        n("Sub", ["one", "vany"], ["nvf"]),
        n("Mul", ["hfill", "nvf"], ["hmask"]),
        n("Add", ["vfill", "hmask"], ["paint"]),                        # (1,10,H,W)
        # background for remaining grid cells
        n("ReduceSum", ["paint"], ["pany"], axes=[1], keepdims=1),
        n("ReduceSum", ["input"], ["grid"], axes=[1], keepdims=1),
        n("Sub", ["grid", "pany"], ["bg"]),
        n("Mul", ["e0", "bg"], ["pbg"]),
        n("Add", ["paint", "pbg"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "connect_pairs",
                              [helper.make_tensor_value_info("input", F, FULL)],
                              [helper.make_tensor_value_info("output", F, FULL)],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_connect_pairs(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
