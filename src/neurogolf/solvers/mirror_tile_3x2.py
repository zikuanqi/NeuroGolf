"""Solver: reflect-tile a 3x2 grid into a 9x4 pattern (task 211).

Every input is 3x2 and every output is 9x4, built by reflecting the input::

    A          = [ flip(input) , input ]   (3x4, mirror across the centre col)
    output     = [ A , flipV(A) , A ]      (9x4)

Because the sizes are fixed, this is a constant index map
``output[r,c] = input[rowmap[r], colmap[c]]`` with
``rowmap = [2,1,0, 0,1,2, 2,1,0]`` and ``colmap = [1,0,0,1]`` — two ``Gather``s
plus a 9x4 validity mask.
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

_ROWMAP = [2, 1, 0, 0, 1, 2, 2, 1, 0]
_COLMAP = [1, 0, 0, 1]


def _ref(g: np.ndarray) -> Optional[np.ndarray]:
    if g.shape != (3, 2):
        return None
    return np.array([[g[_ROWMAP[a], _COLMAP[b]] for b in range(4)] for a in range(9)])


def _detect(task: dict) -> bool:
    saw = False
    for ex in all_examples(task):
        i, o = ex["input"], ex["output"]
        if not i or not i[0]:
            continue
        r = _ref(np.array(i))
        if r is None or r.shape != np.array(o).shape or not np.array_equal(r, np.array(o)):
            return False
        saw = True
    return saw


def _build() -> onnx.ModelProto:
    n = helper.make_node
    rowidx = np.array(_ROWMAP + [0] * (HEIGHT - len(_ROWMAP)), np.int64)
    colidx = np.array(_COLMAP + [0] * (WIDTH - len(_COLMAP)), np.int64)
    vmask = np.zeros((1, 1, HEIGHT, WIDTH), np.float32)
    vmask[0, 0, :9, :4] = 1.0
    init = [
        numpy_helper.from_array(rowidx, "rowidx"),
        numpy_helper.from_array(colidx, "colidx"),
        numpy_helper.from_array(vmask, "vmask"),
    ]
    nodes = [
        n("Gather", ["input", "rowidx"], ["gr"], axis=2),
        n("Gather", ["gr", "colidx"], ["gc"], axis=3),
        n("Mul", ["gc", "vmask"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "mirror_tile_3x2",
                              [helper.make_tensor_value_info("input", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              [helper.make_tensor_value_info("output", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_mirror_tile_3x2(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
