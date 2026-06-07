"""Solver: mark each 2x2 block's diagonal corners 1/2/3/4 (task 230).

Every 2x2 block of colour 5 gets four single markers one cell diagonally out
from each corner: top-left -> 1 (up-left), top-right -> 2 (up-right), bottom-left
-> 3 (down-left), bottom-right -> 4 (down-right).  The blocks stay.

Build (all constant shifts): the four cells of each block are classified by which
orthogonal neighbours are also foreground (top-left = has-below & has-right, ...);
each is shifted one cell diagonally outward and painted its marker colour, then
OR-ed over the kept input.
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
    out = g.copy()
    found = False
    for r in range(H - 1):
        for c in range(W - 1):
            if g[r, c] == 5 and g[r, c + 1] == 5 and g[r + 1, c] == 5 and g[r + 1, c + 1] == 5:
                found = True
                for (rr, cc, col) in [(r - 1, c - 1, 1), (r - 1, c + 2, 2),
                                      (r + 2, c - 1, 3), (r + 2, c + 2, 4)]:
                    if 0 <= rr < H and 0 <= cc < W:
                        out[rr, cc] = col
    return out if found else None


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
    init = [numpy_helper.from_array(note0, "note0"),
            numpy_helper.from_array(np.array(1.0, np.float32), "one"),
            numpy_helper.from_array(np.array([0.0], np.float32), "zero")]
    for k in (1, 2, 3, 4):
        v = np.zeros((1, CHANNELS, 1, 1), np.float32); v[0, k] = 1.0
        init.append(numpy_helper.from_array(v, f"e{k}"))
    nodes = []

    def shift(src, dr, dc, tag):
        hb, he = max(dr, 0), max(-dr, 0)
        wb, we = max(dc, 0), max(-dc, 0)
        init.append(numpy_helper.from_array(np.array([0, 0, hb, wb, 0, 0, he, we], np.int64), f"pad_{tag}"))
        init.append(numpy_helper.from_array(np.array([he, we], np.int64), f"ss_{tag}"))
        init.append(numpy_helper.from_array(np.array([he + HEIGHT, we + WIDTH], np.int64), f"se_{tag}"))
        if "ax23" not in [i.name for i in init]:
            init.append(numpy_helper.from_array(np.array([2, 3], np.int64), "ax23"))
        nodes.append(n("Pad", [src, f"pad_{tag}", "zero"], [f"p_{tag}"]))
        nodes.append(n("Slice", [f"p_{tag}", f"ss_{tag}", f"se_{tag}", "ax23"], [f"o_{tag}"]))
        return f"o_{tag}"

    nodes += [
        n("Mul", ["input", "note0"], ["nb_in"]),
        n("ReduceSum", ["nb_in"], ["fg"], axes=[1], keepdims=1),     # (1,1,H,W)
    ]
    below = shift("fg", -1, 0, "bel")
    right = shift("fg", 0, -1, "rgt")
    above = shift("fg", 1, 0, "abv")
    left = shift("fg", 0, 1, "lft")
    nodes += [
        n("Mul", ["fg", below], ["fb"]), n("Mul", ["fb", right], ["tl"]),
        n("Mul", ["fb", left], ["tr"]),
        n("Mul", ["fg", above], ["fa"]), n("Mul", ["fa", right], ["bl"]),
        n("Mul", ["fa", left], ["br"]),
    ]
    tl_m = shift("tl", -1, -1, "tlm")
    tr_m = shift("tr", -1, 1, "trm")
    bl_m = shift("bl", 1, -1, "blm")
    br_m = shift("br", 1, 1, "brm")
    nodes += [
        n("Mul", ["e1", tl_m], ["p1"]),
        n("Mul", ["e2", tr_m], ["p2"]),
        n("Mul", ["e3", bl_m], ["p3"]),
        n("Mul", ["e4", br_m], ["p4"]),
        n("Add", ["p1", "p2"], ["pa"]), n("Add", ["p3", "p4"], ["pb"]),
        n("Add", ["pa", "pb"], ["markers"]),                         # (1,10,H,W)
        n("ReduceSum", ["markers"], ["mmask"], axes=[1], keepdims=1),
        n("Sub", ["one", "mmask"], ["keep"]),
        n("Mul", ["input", "keep"], ["kept"]),
        n("Add", ["kept", "markers"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "diagonal_markers",
                              [helper.make_tensor_value_info("input", F, FULL)],
                              [helper.make_tensor_value_info("output", F, FULL)],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_diagonal_markers(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
