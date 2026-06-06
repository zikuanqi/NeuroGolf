"""Solver: recolour solid rectangles by orthogonal-neighbour count (task 283).

Every solid block is repainted per cell by how many of its 4 orthogonal
neighbours are also foreground: a corner (2 neighbours) -> colour 1, an edge
cell (3) -> colour 4, an interior cell (4) -> colour 2.  The rule is purely
local, so any number of rectangles are handled at once; background is untouched.

Build: a foreground mask is shifted in the four directions (`Pad`+`Slice`) and
summed to a neighbour count; equality masks at 2/3/4 select the three target
colours, restricted to foreground cells.
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
MAP = {2: 1, 3: 4, 4: 2}     # neighbour-count -> output colour


def _ref(g: np.ndarray) -> Optional[np.ndarray]:
    H, W = g.shape
    fg = (g != 0).astype(int)
    out = g.copy()
    changed = False
    for r in range(H):
        for c in range(W):
            if not fg[r, c]:
                continue
            nf = sum(1 for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
                     if 0 <= r + dr < H and 0 <= c + dc < W and fg[r + dr, c + dc])
            if nf in MAP:
                out[r, c] = MAP[nf]
                changed = True
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
    note0 = np.ones((1, CHANNELS, 1, 1), np.float32); note0[0, 0] = 0.0
    e1 = np.zeros((1, CHANNELS, 1, 1), np.float32); e1[0, 1] = 1.0
    e2 = np.zeros((1, CHANNELS, 1, 1), np.float32); e2[0, 2] = 1.0
    e4 = np.zeros((1, CHANNELS, 1, 1), np.float32); e4[0, 4] = 1.0
    init = [
        numpy_helper.from_array(note0, "note0"),
        numpy_helper.from_array(e1, "e1"),
        numpy_helper.from_array(e2, "e2"),
        numpy_helper.from_array(e4, "e4"),
        numpy_helper.from_array(np.array(0.5, np.float32), "half"),
        numpy_helper.from_array(np.array(1.0, np.float32), "one"),
        numpy_helper.from_array(np.array(2.0, np.float32), "two"),
        numpy_helper.from_array(np.array(3.0, np.float32), "three"),
        numpy_helper.from_array(np.array(4.0, np.float32), "four"),
        numpy_helper.from_array(np.array([1], np.int64), "st1"),
        numpy_helper.from_array(np.array([0.0], np.float32), "zero"),
    ]
    shifts = {
        "up":    ([0, 0, 1, 0, 0, 0, 0, 0], [0], [HEIGHT], 2),
        "down":  ([0, 0, 0, 0, 0, 0, 1, 0], [1], [HEIGHT + 1], 2),
        "left":  ([0, 0, 0, 1, 0, 0, 0, 0], [0], [WIDTH], 3),
        "right": ([0, 0, 0, 0, 0, 0, 0, 1], [1], [WIDTH + 1], 3),
    }
    nodes = [
        n("ReduceSum", ["input"], ["content"], axes=[1], keepdims=1),
        n("Mul", ["input", "note0"], ["nb_in"]),
        n("ReduceSum", ["nb_in"], ["fg"], axes=[1], keepdims=1),   # (1,1,H,W) non-bg mask
    ]
    sums = []
    for name, (pads, slo, shi, ax) in shifts.items():
        init.append(numpy_helper.from_array(np.array(pads, np.int64), f"pad_{name}"))
        init.append(numpy_helper.from_array(np.array(slo, np.int64), f"slo_{name}"))
        init.append(numpy_helper.from_array(np.array(shi, np.int64), f"shi_{name}"))
        init.append(numpy_helper.from_array(np.array([ax], np.int64), f"ax_{name}"))
        nodes.append(n("Pad", ["fg", f"pad_{name}", "zero"], [f"p_{name}"]))
        nodes.append(n("Slice", [f"p_{name}", f"slo_{name}", f"shi_{name}",
                                 f"ax_{name}", "st1"], [f"nb_{name}"]))
        sums.append(f"nb_{name}")
    nodes += [
        n("Add", [sums[0], sums[1]], ["s01"]),
        n("Add", [sums[2], sums[3]], ["s23"]),
        n("Add", ["s01", "s23"], ["nf"]),                          # neighbour count 0..4
        # equality masks (|nf - k| < 0.5)
        n("Sub", ["nf", "two"], ["d2"]), n("Abs", ["d2"], ["a2"]),
        n("Less", ["a2", "half"], ["m2b"]), n("Cast", ["m2b"], ["m2"], to=F),
        n("Sub", ["nf", "three"], ["d3"]), n("Abs", ["d3"], ["a3"]),
        n("Less", ["a3", "half"], ["m3b"]), n("Cast", ["m3b"], ["m3"], to=F),
        n("Sub", ["nf", "four"], ["d4"]), n("Abs", ["d4"], ["a4"]),
        n("Less", ["a4", "half"], ["m4b"]), n("Cast", ["m4b"], ["m4"], to=F),
        n("Mul", ["m2", "fg"], ["m2f"]),
        n("Mul", ["m3", "fg"], ["m3f"]),
        n("Mul", ["m4", "fg"], ["m4f"]),
        n("Add", ["m2f", "m3f"], ["rc0"]), n("Add", ["rc0", "m4f"], ["recol"]),
        n("Sub", ["one", "recol"], ["keep"]),
        n("Mul", ["input", "keep"], ["kept"]),
        n("Mul", ["e1", "m2f"], ["p1"]),
        n("Mul", ["e4", "m3f"], ["p4"]),
        n("Mul", ["e2", "m4f"], ["p2"]),
        n("Add", ["kept", "p1"], ["q1"]),
        n("Add", ["q1", "p4"], ["q2"]),
        n("Add", ["q2", "p2"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "ring_recolor",
                              [helper.make_tensor_value_info("input", F, FULL)],
                              [helper.make_tensor_value_info("output", F, FULL)],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_ring_recolor(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
