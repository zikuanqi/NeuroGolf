"""Solver: fill each hollow box's enclosed interior with 3 (task 338).

Background cells sealed off from the grid border (enclosed by a frame of any
colour) become ``3``; the frames themselves and all border-reachable background
revert to background.

Border reachability is a 4-connected flood: seed the 30x30 frame, dilate with a
1x3 / 3x1 ``MaxPool`` max and re-mask by the passable set (background or
out-of-grid padding); enclosed background = bg minus reached.
"""
from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from ..grids import CHANNELS, HEIGHT, WIDTH, all_examples

OPSET = 11
IR_VERSION = 8
F = TensorProto.FLOAT
NITER = 30


def _ref(g: np.ndarray) -> Optional[np.ndarray]:
    H, W = g.shape
    bg = (g == 0)
    reach = np.zeros((H, W), bool)
    q = deque()
    for r in range(H):
        for c in (0, W - 1):
            if bg[r, c] and not reach[r, c]:
                reach[r, c] = True; q.append((r, c))
    for c in range(W):
        for r in (0, H - 1):
            if bg[r, c] and not reach[r, c]:
                reach[r, c] = True; q.append((r, c))
    while q:
        y, x = q.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            yy, xx = y + dr, x + dc
            if 0 <= yy < H and 0 <= xx < W and bg[yy, xx] and not reach[yy, xx]:
                reach[yy, xx] = True; q.append((yy, xx))
    interior = bg & ~reach
    out = np.zeros_like(g)
    out[interior] = 3
    return out


def _detect(task: dict) -> bool:
    saw_fill = False
    saw = False
    for ex in all_examples(task):
        i, o = ex["input"], ex["output"]
        if not i or not i[0] or len(i) > HEIGHT or len(i[0]) > WIDTH:
            continue
        r = _ref(np.array(i))
        if not np.array_equal(r, np.array(o)):
            return False
        saw = True
        if (r == 3).any():
            saw_fill = True
    return saw and saw_fill


def _build() -> onnx.ModelProto:
    n = helper.make_node
    e3 = np.zeros((1, CHANNELS, 1, 1), np.float32); e3[0, 3] = 1.0
    e0 = np.zeros((1, CHANNELS, 1, 1), np.float32); e0[0, 0] = 1.0
    frame = np.zeros((1, 1, HEIGHT, WIDTH), np.float32)
    frame[0, 0, 0, :] = 1; frame[0, 0, HEIGHT - 1, :] = 1
    frame[0, 0, :, 0] = 1; frame[0, 0, :, WIDTH - 1] = 1
    init = [
        numpy_helper.from_array(e3, "e3"),
        numpy_helper.from_array(e0, "e0"),
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
        n("Add", ["is0", "invocc"], ["pass"]),
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
        n("Mul", ["enclBg", "e3"], ["L3"]),
        n("Sub", ["occ", "enclBg"], ["bgv"]),
        n("Mul", ["bgv", "e0"], ["L0"]),
        n("Add", ["L3", "L0"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "box_interior_fill",
                              [helper.make_tensor_value_info("input", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              [helper.make_tensor_value_info("output", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_box_interior_fill(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
