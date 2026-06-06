"""Solver: merge an adjacent 3-2 pair into an 8 (task 344).

Wherever a colour-3 cell is 4-adjacent to a colour-2 cell, the 3 becomes 8 and
the 2 is erased; cells with no such neighbour are unchanged.

Build: slice the colour-3 and colour-2 channels, find each cell's neighbours by
shifting the other channel four ways (`Pad`+`Slice`) and taking the `Max`; the
3-cells with a 2-neighbour get `e_8 - e_3`, the 2-cells with a 3-neighbour get
`e_0 - e_2`.
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
A, M, T = 3, 2, 8


def _ref(g: np.ndarray) -> Optional[np.ndarray]:
    H, W = g.shape
    out = g.copy()
    changed = False
    for r in range(H):
        for c in range(W):
            nb = [(r + dr, c + dc) for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
                  if 0 <= r + dr < H and 0 <= c + dc < W]
            if g[r, c] == A and any(g[nr, nc] == M for nr, nc in nb):
                out[r, c] = T; changed = True
            elif g[r, c] == M and any(g[nr, nc] == A for nr, nc in nb):
                out[r, c] = 0; changed = True
    return out if changed else None


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
    d3 = np.zeros((1, CHANNELS, 1, 1), np.float32); d3[0, T] = 1.0; d3[0, A] = -1.0
    d2 = np.zeros((1, CHANNELS, 1, 1), np.float32); d2[0, 0] = 1.0; d2[0, M] = -1.0
    init = [
        numpy_helper.from_array(np.array([1], np.int64), "st1"),
        numpy_helper.from_array(np.array([0.0], np.float32), "zero"),
        numpy_helper.from_array(np.array([0, A, 0, 0], np.int64), "a_s"),
        numpy_helper.from_array(np.array([1, A + 1, HEIGHT, WIDTH], np.int64), "a_e"),
        numpy_helper.from_array(np.array([0, M, 0, 0], np.int64), "m_s"),
        numpy_helper.from_array(np.array([1, M + 1, HEIGHT, WIDTH], np.int64), "m_e"),
        numpy_helper.from_array(np.array([0, 1, 2, 3], np.int64), "ax4"),
        numpy_helper.from_array(d3, "d3"),
        numpy_helper.from_array(d2, "d2"),
    ]
    shifts = {
        "u": ([0, 0, 1, 0, 0, 0, 0, 0], [0], [HEIGHT], 2),
        "d": ([0, 0, 0, 0, 0, 0, 1, 0], [1], [HEIGHT + 1], 2),
        "l": ([0, 0, 0, 1, 0, 0, 0, 0], [0], [WIDTH], 3),
        "r": ([0, 0, 0, 0, 0, 0, 0, 1], [1], [WIDTH + 1], 3),
    }
    nodes = [
        n("Slice", ["input", "a_s", "a_e", "ax4"], ["ch3"]),
        n("Slice", ["input", "m_s", "m_e", "ax4"], ["ch2"]),
    ]

    def neighbours(src, tag):
        outs = []
        for name, (pads, slo, shi, ax) in shifts.items():
            t = f"{tag}{name}"
            init.append(numpy_helper.from_array(np.array(pads, np.int64), t + "p"))
            init.append(numpy_helper.from_array(np.array(slo, np.int64), t + "lo"))
            init.append(numpy_helper.from_array(np.array(shi, np.int64), t + "hi"))
            init.append(numpy_helper.from_array(np.array([ax], np.int64), t + "ax"))
            nodes.append(n("Pad", [src, t + "p", "zero"], [t + "pp"]))
            nodes.append(n("Slice", [t + "pp", t + "lo", t + "hi", t + "ax", "st1"], [t]))
            outs.append(t)
        nodes.append(n("Max", [outs[0], outs[1]], [tag + "_m1"]))
        nodes.append(n("Max", [outs[2], outs[3]], [tag + "_m2"]))
        nodes.append(n("Max", [tag + "_m1", tag + "_m2"], [tag + "_any"]))
        return tag + "_any"

    has2 = neighbours("ch2", "n2")    # cell has a 2-neighbour
    has3 = neighbours("ch3", "n3")    # cell has a 3-neighbour
    nodes += [
        n("Mul", ["ch3", has2], ["active3"]),     # 3 with a 2 neighbour -> 8
        n("Mul", ["ch2", has3], ["active2"]),     # 2 with a 3 neighbour -> 0
        n("Mul", ["d3", "active3"], ["delta3"]),
        n("Mul", ["d2", "active2"], ["delta2"]),
        n("Add", ["input", "delta3"], ["p1"]),
        n("Add", ["p1", "delta2"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "merge_pair",
                              [helper.make_tensor_value_info("input", F, FULL)],
                              [helper.make_tensor_value_info("output", F, FULL)],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_merge_pair(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
