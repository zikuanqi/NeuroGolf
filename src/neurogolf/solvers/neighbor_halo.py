"""Solver: stamp a fixed halo around single-cell markers (task 15).

Every isolated marker cell spawns a coloured halo on the background cells
around it, the pattern depending on the marker colour::

    a ``1`` paints ``7`` on its 4 orthogonal neighbours (a plus),
    a ``2`` paints ``4`` on its 4 diagonal neighbours (an X),

other colours (``6``, ``8``, ...) are left untouched.  A halo cell is only
written where the grid is currently background (``0``); the markers themselves
stay.  This is a pure neighbourhood stamp - a plus-kernel convolution over the
1-mask and an X-kernel convolution over the 2-mask, each masked by background.
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
F = TensorProto.FLOAT

_ORTH = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_DIAG = [(-1, -1), (-1, 1), (1, -1), (1, 1)]


def _ref(g: np.ndarray) -> Optional[np.ndarray]:
    H, W = g.shape
    out = g.copy()
    for r in range(H):
        for c in range(W):
            v = g[r, c]
            if v == 1:
                offs, col = _ORTH, 7
            elif v == 2:
                offs, col = _DIAG, 4
            else:
                continue
            for dr, dc in offs:
                rr, cc = r + dr, c + dc
                if 0 <= rr < H and 0 <= cc < W and g[rr, cc] == 0:
                    out[rr, cc] = col
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


def _chan(idx: int) -> np.ndarray:
    e = np.zeros((1, CHANNELS, 1, 1), np.float32)
    e[0, idx] = 1.0
    return e


def _build() -> onnx.ModelProto:
    n = helper.make_node
    plus = np.array([[[[0, 1, 0], [1, 0, 1], [0, 1, 0]]]], np.float32)
    cross = np.array([[[[1, 0, 1], [0, 0, 0], [1, 0, 1]]]], np.float32)
    init = [
        numpy_helper.from_array(plus, "plusK"),
        numpy_helper.from_array(cross, "xK"),
        numpy_helper.from_array(_chan(7), "e7"),
        numpy_helper.from_array(_chan(4), "e4"),
        numpy_helper.from_array(_chan(0), "e0"),
        numpy_helper.from_array(np.array(0.5, np.float32), "half"),
        numpy_helper.from_array(np.array([0], np.int64), "c0"),
        numpy_helper.from_array(np.array([1], np.int64), "c1"),
        numpy_helper.from_array(np.array([2], np.int64), "c2"),
        numpy_helper.from_array(np.array([3], np.int64), "c3"),
        numpy_helper.from_array(np.array([1], np.int64), "ax1"),
    ]
    nodes = [
        n("Slice", ["input", "c0", "c1", "ax1"], ["is0"]),
        n("Slice", ["input", "c1", "c2", "ax1"], ["is1"]),
        n("Slice", ["input", "c2", "c3", "ax1"], ["is2"]),
        n("Conv", ["is1", "plusK"], ["cnt1"], pads=[1, 1, 1, 1]),
        n("Conv", ["is2", "xK"], ["cnt2"], pads=[1, 1, 1, 1]),
        n("Greater", ["cnt1", "half"], ["h1b"]), n("Cast", ["h1b"], ["h1"], to=F),
        n("Greater", ["cnt2", "half"], ["h2b"]), n("Cast", ["h2b"], ["h2"], to=F),
        n("Mul", ["h1", "is0"], ["halo7"]),
        n("Mul", ["h2", "is0"], ["halo4"]),
        n("Mul", ["halo7", "e7"], ["add7"]),
        n("Mul", ["halo4", "e4"], ["add4"]),
        n("Add", ["halo7", "halo4"], ["halosum"]),
        n("Mul", ["halosum", "e0"], ["sub0"]),
        n("Add", ["input", "add7"], ["t1"]),
        n("Add", ["t1", "add4"], ["t2"]),
        n("Sub", ["t2", "sub0"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "neighbor_halo",
                              [helper.make_tensor_value_info("input", F, FULL)],
                              [helper.make_tensor_value_info("output", F, FULL)],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_neighbor_halo(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
