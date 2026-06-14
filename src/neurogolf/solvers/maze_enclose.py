"""Solver: colour maze regions outside / inside the walls (task 187).

A single wall colour draws a maze of corridors and enclosed chambers on a
background of 0.  Background reachable from the grid border (the "outside")
becomes ``3``; background sealed off by walls becomes ``2``; the walls keep
their colour::

    W . . W . W            W 3 3 W 2 W
    W . W W . W     ->      W 3 W W 2 W
    W . . . . W            W 3 3 3 3 W

Border reachability is a 4-connected flood: seed the 30x30 frame, then for N
steps dilate (``max`` of a 1x3 and a 3x1 ``MaxPool``) and re-mask by the
passable set (background **or** out-of-grid padding).  The deepest reachable
cell sits only ~15 steps in, so N=24 is ample.
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
NITER = 24


def _ref(g: np.ndarray) -> Optional[np.ndarray]:
    from collections import deque
    H, W = g.shape
    nz = [c for c in np.unique(g) if c != 0]
    if len(nz) != 1:
        return None
    wall = nz[0]
    passable = g != wall
    reach = np.zeros((H, W), bool)
    q = deque()
    for r in range(H):
        for c in (0, W - 1):
            if passable[r, c] and not reach[r, c]:
                reach[r, c] = True; q.append((r, c))
    for c in range(W):
        for r in (0, H - 1):
            if passable[r, c] and not reach[r, c]:
                reach[r, c] = True; q.append((r, c))
    while q:
        y, x = q.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            yy, xx = y + dr, x + dc
            if 0 <= yy < H and 0 <= xx < W and passable[yy, xx] and not reach[yy, xx]:
                reach[yy, xx] = True; q.append((yy, xx))
    out = g.copy()
    out[(g == 0) & reach] = 3
    out[(g == 0) & ~reach] = 2
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
    e3me0 = np.zeros((1, CHANNELS, 1, 1), np.float32); e3me0[0, 3] = 1.0; e3me0[0, 0] = -1.0
    e2me0 = np.zeros((1, CHANNELS, 1, 1), np.float32); e2me0[0, 2] = 1.0; e2me0[0, 0] = -1.0
    frame = np.zeros((1, 1, HEIGHT, WIDTH), np.float32)
    frame[0, 0, 0, :] = 1; frame[0, 0, HEIGHT - 1, :] = 1
    frame[0, 0, :, 0] = 1; frame[0, 0, :, WIDTH - 1] = 1
    init = [
        numpy_helper.from_array(e3me0, "e3me0"),
        numpy_helper.from_array(e2me0, "e2me0"),
        numpy_helper.from_array(frame, "frame"),
        numpy_helper.from_array(np.array(1.0, np.float32), "one"),
        numpy_helper.from_array(np.array([0], np.int64), "c0s"),
        numpy_helper.from_array(np.array([1], np.int64), "c1e"),
        numpy_helper.from_array(np.array([1], np.int64), "ax1"),
    ]
    nodes = [
        n("ReduceSum", ["input"], ["occ"], axes=[1], keepdims=1),
        n("Slice", ["input", "c0s", "c1e", "ax1"], ["is0"]),
        n("Sub", ["one", "occ"], ["invocc"]),
        n("Add", ["is0", "invocc"], ["pass"]),          # passable = bg or padding
        n("Mul", ["pass", "frame"], ["reach0"]),
    ]
    cur = "reach0"
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
        n("Mul", [cur, "is0"], ["reachBg"]),
        n("Sub", ["is0", "reachBg"], ["enclBg"]),
        n("Mul", ["reachBg", "e3me0"], ["addr"]),
        n("Mul", ["enclBg", "e2me0"], ["adde"]),
        n("Add", ["input", "addr"], ["t1"]),
        n("Add", ["t1", "adde"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "maze_enclose",
                              [helper.make_tensor_value_info("input", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              [helper.make_tensor_value_info("output", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_maze_enclose(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
