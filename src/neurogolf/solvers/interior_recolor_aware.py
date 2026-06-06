"""Solver: recolour the interior of each solid shape to a detected colour.

Same erosion rule as `interior_recolor` (task 120) but the fill colour is
inferred from the examples instead of being fixed at 8 — e.g. task 294 fills the
one-cell-eroded interior with colour 2.

Build: identical to `interior_recolor` with the target channel parameterised.
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


def _ref(g: np.ndarray, target: int) -> Optional[np.ndarray]:
    H, W = g.shape
    out = g.copy()
    changed = False
    for r in range(H):
        for c in range(W):
            if g[r, c] == 0:
                continue
            if all(0 <= r + dr < H and 0 <= c + dc < W and g[r + dr, c + dc] == g[r, c]
                   for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))):
                out[r, c] = target
                changed = True
    return out if changed else None


def _detect(task: dict) -> Optional[int]:
    target = None
    saw = False
    for ex in all_examples(task):
        i, o = ex["input"], ex["output"]
        if not i or not i[0] or len(i) > HEIGHT or len(i[0]) > WIDTH:
            continue
        gi, go = np.array(i), np.array(o)
        if gi.shape != go.shape:
            return None
        diff = gi != go
        if diff.any():
            ks = set(go[diff].tolist())
            if len(ks) != 1:
                return None
            k = ks.pop()
            if target is None:
                target = k
            elif target != k:
                return None
        if target is None:
            continue
        if not np.array_equal(_ref(gi, target) if _ref(gi, target) is not None else gi, go):
            return None
        saw = True
    return target if (saw and target is not None) else None


def _build(target: int) -> onnx.ModelProto:
    F = TensorProto.FLOAT
    n = helper.make_node
    eT = np.zeros((1, CHANNELS, 1, 1), np.float32); eT[0, target] = 1.0
    init = [
        numpy_helper.from_array(eT, "eT"),
        numpy_helper.from_array(np.array(1.0, np.float32), "one"),
        numpy_helper.from_array(np.array([1], np.int64), "st1"),
        numpy_helper.from_array(np.array([0.0], np.float32), "zero"),
        numpy_helper.from_array(np.array([0, 0, 0, 0], np.int64), "c0s"),
        numpy_helper.from_array(np.array([1, 1, HEIGHT, WIDTH], np.int64), "c0e"),
        numpy_helper.from_array(np.array([0, 1, 2, 3], np.int64), "ax4"),
    ]
    shifts = {
        "up":    ([0, 0, 1, 0, 0, 0, 0, 0], [0], [HEIGHT], 2),
        "down":  ([0, 0, 0, 0, 0, 0, 1, 0], [1], [HEIGHT + 1], 2),
        "left":  ([0, 0, 0, 1, 0, 0, 0, 0], [0], [WIDTH], 3),
        "right": ([0, 0, 0, 0, 0, 0, 0, 1], [1], [WIDTH + 1], 3),
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
        n("Mul", ["i1", "i2"], ["interior"]),
        n("ReduceSum", ["input"], ["content"], axes=[1], keepdims=1),
        n("Slice", ["input", "c0s", "c0e", "ax4"], ["ch0"]),
        n("Sub", ["content", "ch0"], ["nonbg"]),
        n("Mul", ["interior", "nonbg"], ["intf"]),
        n("Sub", ["one", "intf"], ["keep"]),
        n("Mul", ["input", "keep"], ["kept"]),
        n("Mul", ["eT", "intf"], ["paint"]),
        n("Add", ["kept", "paint"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "interior_recolor_aware",
                              [helper.make_tensor_value_info("input", F, FULL)],
                              [helper.make_tensor_value_info("output", F, FULL)],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_interior_recolor_aware(task: dict) -> Optional[onnx.ModelProto]:
    target = _detect(task)
    if target is None:
        return None
    return _build(target)
