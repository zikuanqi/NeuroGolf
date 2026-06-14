"""Solver: flood-fill 1 from existing 1-cells through the 0-region (task 243).

A noisy grid already contains some ``1`` cells.  Every background ``0`` that is
4-connected to a ``1`` through other ``0``s becomes ``1``; isolated 0-pockets
that no ``1`` can reach stay ``0``; all other colours are walls and unchanged.

Standard 4-connected ``MaxPool`` flood: seed the ``1`` mask, dilate (max of a
1x3 and a 3x1 ``MaxPool``) and re-mask by the passable set (0-cells or 1-cells)
for N steps.  The deepest reachable cell is ~28 steps in, so N=32 is ample.
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
NITER = 32


def _ref(g: np.ndarray) -> Optional[np.ndarray]:
    from collections import deque
    H, W = g.shape
    reach = np.zeros((H, W), bool)
    q = deque()
    for r in range(H):
        for c in range(W):
            if g[r, c] == 1:
                reach[r, c] = True
                q.append((r, c))
    if not q:
        return None
    while q:
        y, x = q.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            yy, xx = y + dr, x + dc
            if 0 <= yy < H and 0 <= xx < W and g[yy, xx] == 0 and not reach[yy, xx]:
                reach[yy, xx] = True
                q.append((yy, xx))
    out = g.copy()
    out[(g == 0) & reach] = 1
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
    n = helper.make_node
    e1me0 = np.zeros((1, CHANNELS, 1, 1), np.float32); e1me0[0, 1] = 1.0; e1me0[0, 0] = -1.0
    init = [
        numpy_helper.from_array(e1me0, "e1me0"),
        numpy_helper.from_array(np.array([0], np.int64), "c0s"),
        numpy_helper.from_array(np.array([1], np.int64), "c1e"),
        numpy_helper.from_array(np.array([2], np.int64), "c2e"),
        numpy_helper.from_array(np.array([1], np.int64), "ax1"),
    ]
    nodes = [
        n("Slice", ["input", "c0s", "c1e", "ax1"], ["is0"]),
        n("Slice", ["input", "c1e", "c2e", "ax1"], ["is1"]),
        n("Add", ["is0", "is1"], ["pass"]),   # passable = 0-cells or 1-cells
    ]
    cur = "is1"
    for k in range(NITER):
        h, v, d, nx = f"h{k}", f"v{k}", f"d{k}", f"reach{k + 1}"
        nodes += [
            n("MaxPool", [cur], [h], kernel_shape=[1, 3], strides=[1, 1], pads=[0, 1, 0, 1]),
            n("MaxPool", [cur], [v], kernel_shape=[3, 1], strides=[1, 1], pads=[1, 0, 1, 0]),
            n("Max", [h, v], [d]),
            n("Mul", [d, "pass"], [nx]),
        ]
        cur = nx
    nodes += [
        n("Mul", [cur, "is0"], ["reachZero"]),
        n("Mul", ["reachZero", "e1me0"], ["delta"]),
        n("Add", ["input", "delta"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "flood_ones",
                              [helper.make_tensor_value_info("input", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              [helper.make_tensor_value_info("output", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_flood_ones(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
