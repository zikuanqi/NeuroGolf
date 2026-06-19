"""Solver: each top-row marker draws a vertical zigzag downward (task 82).

A marker in the top row paints a zigzag down its column: the marker colour sits
on the centre column at even rows and on the two side columns at odd rows
(``|c'-c| == r mod 2``).  Markers act independently.

Built from the top row alone: a centre layer (top row gated to even rows) plus a
left+right shifted layer (gated to odd rows), confined to the real grid.
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
    o = np.zeros_like(g)
    saw = False
    for c in range(W):
        v = g[0, c]
        if v != 0:
            saw = True
            for r in range(H):
                if r % 2 == 0:
                    o[r, c] = v
                else:
                    if c - 1 >= 0:
                        o[r, c - 1] = v
                    if c + 1 < W:
                        o[r, c + 1] = v
    if not saw or np.array_equal(o, g):
        return None
    return o


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
    n = helper.make_node
    notbg = np.ones((1, CHANNELS, 1, 1), np.float32); notbg[0, 0] = 0.0
    e0 = np.zeros((1, CHANNELS, 1, 1), np.float32); e0[0, 0] = 1.0
    even = np.zeros((1, 1, HEIGHT, 1), np.float32)
    odd = np.zeros((1, 1, HEIGHT, 1), np.float32)
    for r in range(HEIGHT):
        if r % 2 == 0:
            even[0, 0, r, 0] = 1.0
        else:
            odd[0, 0, r, 0] = 1.0
    init = [
        numpy_helper.from_array(notbg, "notbg"),
        numpy_helper.from_array(e0, "e0"),
        numpy_helper.from_array(even, "even"),
        numpy_helper.from_array(odd, "odd"),
        numpy_helper.from_array(np.array([0], np.int64), "z0"),
        numpy_helper.from_array(np.array([1], np.int64), "z1"),
        numpy_helper.from_array(np.array([WIDTH], np.int64), "zW"),
        numpy_helper.from_array(np.array([WIDTH + 1], np.int64), "zW1"),
        numpy_helper.from_array(np.array([2], np.int64), "ax2"),
        numpy_helper.from_array(np.array([3], np.int64), "ax3"),
        numpy_helper.from_array(np.array([0, 0, 0, 0, 0, 0, 0, 1], np.int64), "padR"),
        numpy_helper.from_array(np.array([0, 0, 0, 1, 0, 0, 0, 0], np.int64), "padL"),
    ]
    nodes = [
        n("Slice", ["input", "z0", "z1", "ax2"], ["topRow"]),       # (1,10,1,W)
        n("Mul", ["topRow", "notbg"], ["topNB"]),
        n("Mul", ["topNB", "even"], ["center"]),                    # (1,10,H,W)
        # shift content left: out[c'] = top[c'+1]
        n("Pad", ["topNB", "padR"], ["padRr"]),
        n("Slice", ["padRr", "z1", "zW1", "ax3"], ["sl"]),
        # shift content right: out[c'] = top[c'-1]
        n("Pad", ["topNB", "padL"], ["padLl"]),
        n("Slice", ["padLl", "z0", "zW", "ax3"], ["sr"]),
        n("Add", ["sl", "sr"], ["sides"]),
        n("Mul", ["sides", "odd"], ["oddLayer"]),                   # (1,10,H,W)
        n("Add", ["center", "oddLayer"], ["colour"]),
        # confine to the real grid; fill background elsewhere
        n("ReduceSum", ["input"], ["occ"], axes=[1], keepdims=1),
        n("Mul", ["colour", "occ"], ["colourC"]),
        n("ReduceSum", ["colourC"], ["cAny"], axes=[1], keepdims=1),
        n("Sub", ["occ", "cAny"], ["bgv"]), n("Mul", ["bgv", "e0"], ["bgL"]),
        n("Add", ["colourC", "bgL"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "top_marker_zigzag",
                              [helper.make_tensor_value_info("input", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              [helper.make_tensor_value_info("output", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_top_marker_zigzag(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
