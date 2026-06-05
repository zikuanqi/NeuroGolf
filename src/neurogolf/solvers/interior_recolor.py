"""Solver: recolour the interior of each solid shape to colour 8 (task 120).

A non-background cell whose four orthogonal neighbours all share its colour is
an interior cell (the one-cell erosion of the shape); every interior cell is
recoloured to 8 while borders keep their colour.

Build: the one-hot input is shifted one step in each direction with `Pad`+
`Slice`; a `ReduceSum` of `input * neighbour` over the channel axis is 1 exactly
when the neighbour shares the cell's colour; the product of the four gives the
interior mask, restricted to non-background, and `e_8` is painted there.
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
TARGET = 8


def _ref(g: np.ndarray) -> Optional[np.ndarray]:
    H, W = g.shape
    out = g.copy()
    changed = False
    for r in range(H):
        for c in range(W):
            if g[r, c] == 0:
                continue
            if all(0 <= r + dr < H and 0 <= c + dc < W and g[r + dr, c + dc] == g[r, c]
                   for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))):
                out[r, c] = TARGET
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
    e8 = np.zeros((1, CHANNELS, 1, 1), np.float32); e8[0, TARGET] = 1.0
    init = [
        numpy_helper.from_array(e8, "e8"),
        numpy_helper.from_array(np.array(1.0, np.float32), "one"),
        numpy_helper.from_array(np.array([1], np.int64), "st1"),
        numpy_helper.from_array(np.array([0.0], np.float32), "zero"),
        numpy_helper.from_array(np.array([0, 0, 0, 0], np.int64), "c0s"),
        numpy_helper.from_array(np.array([1, 1, HEIGHT, WIDTH], np.int64), "c0e"),
        numpy_helper.from_array(np.array([0, 1, 2, 3], np.int64), "ax4"),
    ]
    # neighbour shifts: (pads, slice_starts, slice_ends, axis)
    shifts = {
        "up":    ([0, 0, 1, 0, 0, 0, 0, 0], [0], [HEIGHT], 2),       # input[r-1]
        "down":  ([0, 0, 0, 0, 0, 0, 1, 0], [1], [HEIGHT + 1], 2),   # input[r+1]
        "left":  ([0, 0, 0, 1, 0, 0, 0, 0], [0], [WIDTH], 3),        # input[c-1]
        "right": ([0, 0, 0, 0, 0, 0, 0, 1], [1], [WIDTH + 1], 3),    # input[c+1]
    }
    nodes = []
    sames = []
    for name, (pads, slo, shi, ax) in shifts.items():
        init.append(numpy_helper.from_array(np.array(pads, np.int64), f"pad_{name}"))
        init.append(numpy_helper.from_array(np.array(slo, np.int64), f"slo_{name}"))
        init.append(numpy_helper.from_array(np.array(shi, np.int64), f"shi_{name}"))
        init.append(numpy_helper.from_array(np.array([ax], np.int64), f"ax_{name}"))
        nodes.append(n("Pad", ["input", f"pad_{name}", "zero"], [f"p_{name}"]))
        nodes.append(n("Slice", [f"p_{name}", f"slo_{name}", f"shi_{name}",
                                 f"ax_{name}", "st1"], [f"nb_{name}"]))
        nodes.append(n("Mul", ["input", f"nb_{name}"], [f"m_{name}"]))
        nodes.append(n("ReduceSum", [f"m_{name}"], [f"same_{name}"], axes=[1], keepdims=1))
        sames.append(f"same_{name}")
    nodes += [
        n("Mul", [sames[0], sames[1]], ["i1"]),
        n("Mul", [sames[2], sames[3]], ["i2"]),
        n("Mul", ["i1", "i2"], ["interior"]),                  # (1,1,H,W)
        n("ReduceSum", ["input"], ["content"], axes=[1], keepdims=1),
        n("Slice", ["input", "c0s", "c0e", "ax4"], ["ch0"]),
        n("Sub", ["content", "ch0"], ["nonbg"]),               # colours 1-9 present
        n("Mul", ["interior", "nonbg"], ["intf"]),             # (1,1,H,W)
        n("Sub", ["one", "intf"], ["keep"]),
        n("Mul", ["input", "keep"], ["kept"]),                 # zero interior cells
        n("Mul", ["e8", "intf"], ["paint"]),                   # channel 8 at interior
        n("Add", ["kept", "paint"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "interior_recolor",
                              [helper.make_tensor_value_info("input", F, FULL)],
                              [helper.make_tensor_value_info("output", F, FULL)],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_interior_recolor(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
