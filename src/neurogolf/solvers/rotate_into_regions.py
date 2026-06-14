"""Solver: fill two 5-divided regions with rotations of the key (task 214).

A 3x3 colour key sits in the left columns; two colour-5 divider columns mark
off two empty 3x3 regions, which are filled with the key rotated 90 degrees
clockwise and 180 degrees::

    K K K 5 . . . 5 . . .        K K K 5 R R R 5 S S S
    K K K 5 . . . 5 . . .   ->   K K K 5 R R R 5 S S S
    K K K 5 . . . 5 . . .        K K K 5 R R R 5 S S S
                                 R = rot90cw(K)   S = rot180(K)

Every grid is 3x11 with dividers at columns 3 and 7, so the whole map is a
constant cell permutation: flatten the spatial axes and ``Gather`` with a
precomputed 900-index array (rotations are just index reflections).
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


def _ref(g: np.ndarray) -> Optional[np.ndarray]:
    if g.shape != (3, 11) or not np.all(g[:, 3] == 5) or not np.all(g[:, 7] == 5):
        return None
    key = g[:, 0:3]
    out = g.copy()
    out[:, 4:7] = np.rot90(key, -1)
    out[:, 8:11] = np.rot90(key, 2)
    return out if not np.array_equal(out, g) else None


def _detect(task: dict) -> bool:
    saw = False
    for ex in all_examples(task):
        i, o = ex["input"], ex["output"]
        if not i or not i[0]:
            continue
        r = _ref(np.array(i))
        if r is None or not np.array_equal(r, np.array(o)):
            return False
        saw = True
    return saw


def _build() -> onnx.ModelProto:
    n = helper.make_node
    N = HEIGHT * WIDTH
    idx = np.arange(N, dtype=np.int64)
    for r in range(3):
        for c in range(4, 7):
            idx[r * WIDTH + c] = (6 - c) * WIDTH + r          # rot90 CW
        for c in range(8, 11):
            idx[r * WIDTH + c] = (2 - r) * WIDTH + (10 - c)   # rot180
    init = [
        numpy_helper.from_array(idx, "idx"),
        numpy_helper.from_array(np.array([1, CHANNELS, N], np.int64), "flat_shape"),
        numpy_helper.from_array(np.array([1, CHANNELS, HEIGHT, WIDTH], np.int64), "grid_shape"),
    ]
    nodes = [
        n("Reshape", ["input", "flat_shape"], ["flat"]),
        n("Gather", ["flat", "idx"], ["gathered"], axis=2),
        n("Reshape", ["gathered", "grid_shape"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "rotate_into_regions",
                              [helper.make_tensor_value_info("input", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              [helper.make_tensor_value_info("output", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_rotate_into_regions(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
