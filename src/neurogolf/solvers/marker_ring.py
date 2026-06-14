"""Solver: draw a 3x3 ring around each marker (task 220).

Each isolated marker stays put and is surrounded by a 3x3 ring whose colour is
fixed by the marker colour (``2->1``, ``3->6``, ``8->4``)::

    . . . . .          . 6 6 6 .
    . . 3 . .   ->     . 6 3 6 .
    . . . . .          . 6 6 6 .

Per marker colour the ring is a 3x3 ``MaxPool`` dilation masked to background
cells; the marker itself (non-background) is untouched.
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
_MAP = {2: 1, 3: 6, 8: 4}


def _ref(g: np.ndarray) -> Optional[np.ndarray]:
    H, W = g.shape
    out = g.copy()
    ys, xs = np.where(g != 0)
    if len(ys) == 0:
        return None
    for r, c in zip(ys, xs):
        cm = int(g[r, c])
        if cm not in _MAP:
            return None
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (dr, dc) == (0, 0):
                    continue
                rr, cc = r + dr, c + dc
                if 0 <= rr < H and 0 <= cc < W and g[rr, cc] == 0:
                    out[rr, cc] = _MAP[cm]
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


def _chan(idx):
    e = np.zeros((1, CHANNELS, 1, 1), np.float32); e[0, idx] = 1.0
    return e


def _build() -> onnx.ModelProto:
    n = helper.make_node
    init = [
        numpy_helper.from_array(_chan(0), "e0"),
        numpy_helper.from_array(_chan(1), "e1"),
        numpy_helper.from_array(_chan(4), "e4"),
        numpy_helper.from_array(_chan(6), "e6"),
        numpy_helper.from_array(np.array([1], np.int64), "ax1"),
    ]
    for k in (0, 2, 3, 8):
        init.append(numpy_helper.from_array(np.array([k], np.int64), f"s{k}"))
        init.append(numpy_helper.from_array(np.array([k + 1], np.int64), f"e{k}_"))
    nodes = [n("Slice", ["input", "s0", "e0_", "ax1"], ["is0"])]
    for k, ec in ((2, "e1"), (3, "e6"), (8, "e4")):
        nodes += [
            n("Slice", ["input", f"s{k}", f"e{k}_", "ax1"], [f"is{k}"]),
            n("MaxPool", [f"is{k}"], [f"d{k}"], kernel_shape=[3, 3], strides=[1, 1], pads=[1, 1, 1, 1]),
            n("Mul", [f"d{k}", "is0"], [f"ring{k}"]),
            n("Mul", [f"ring{k}", ec], [f"add{k}"]),
        ]
    nodes += [
        n("Add", ["ring2", "ring3"], ["rs0"]), n("Add", ["rs0", "ring8"], ["rsum"]),
        n("Mul", ["rsum", "e0"], ["sub0"]),
        n("Add", ["add2", "add3"], ["a01"]), n("Add", ["a01", "add8"], ["addall"]),
        n("Add", ["input", "addall"], ["t1"]),
        n("Sub", ["t1", "sub0"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "marker_ring",
                              [helper.make_tensor_value_info("input", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              [helper.make_tensor_value_info("output", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_marker_ring(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
