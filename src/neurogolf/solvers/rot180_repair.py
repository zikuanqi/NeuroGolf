"""Solver: restore a rot180-symmetric image occluded by one colour (task 287).

The grid is point-symmetric about its centre.  A solid rectangle of a single
"occluder" colour C hides part of it; each hidden cell is restored from its
180-degree partner.  All examples share one fixed square size S, so the
rotation is baked: reverse the top-left S x S block along both axes with two
`Gather`s (reversed index vector), pad it back to the 30x30 canvas, and select
the rotated value exactly on the channel-C cells.
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


def _rot180(g: np.ndarray) -> np.ndarray:
    return g[::-1, ::-1]


def _params(task: dict) -> Optional[tuple[int, int]]:
    size = None
    occ = None
    saw = False
    for ex in all_examples(task):
        i, o = ex["input"], ex["output"]
        if not i or not i[0]:
            continue
        H, W = len(i), len(i[0])
        if H > HEIGHT or W > WIDTH:
            continue
        if H != W:
            return None
        if size is None:
            size = H
        elif size != H:
            return None  # must be a single baked size
        g = np.array(i); out = np.array(o)
        if g.shape != out.shape:
            return None
        diff = g != out
        if not diff.any():
            return None
        cvals = set(g[diff].tolist())
        if len(cvals) != 1:
            return None
        c = cvals.pop()
        if occ is None:
            occ = c
        elif occ != c:
            return None
        pred = np.where(g == c, _rot180(g), g)
        if not np.array_equal(pred, out):
            return None
        saw = True
    if not saw or size is None or occ is None:
        return None
    return size, occ


def _build(size: int, occ: int) -> onnx.ModelProto:
    F = TensorProto.FLOAT
    n = helper.make_node
    rev = np.arange(size - 1, -1, -1, dtype=np.int64)
    pad_end = WIDTH - size
    pads = np.array([0, 0, 0, 0, 0, 0, pad_end, pad_end], np.int64)
    init = [
        numpy_helper.from_array(rev, "rev"),
        numpy_helper.from_array(pads, "pads"),
        numpy_helper.from_array(np.array(1.0, np.float32), "one"),
        numpy_helper.from_array(np.array([0, occ, 0, 0], np.int64), "c_s"),
        numpy_helper.from_array(np.array([1, occ + 1, HEIGHT, WIDTH], np.int64), "c_e"),
        numpy_helper.from_array(np.array([0, 1, 2, 3], np.int64), "ax4"),
    ]
    nodes = [
        n("Slice", ["input", "c_s", "c_e", "ax4"], ["maskC"]),       # (1,1,H,W)
        n("Gather", ["input", "rev"], ["g2"], axis=2),               # (1,10,S,W)
        n("Gather", ["g2", "rev"], ["rotSS"], axis=3),               # (1,10,S,S)
        n("Pad", ["rotSS", "pads"], ["rotpad"], mode="constant"),    # (1,10,H,W)
        n("Sub", ["one", "maskC"], ["inv"]),                         # (1,1,H,W)
        n("Mul", ["input", "inv"], ["keep"]),                        # (1,10,H,W)
        n("Mul", ["rotpad", "maskC"], ["fill"]),                     # (1,10,H,W)
        n("Add", ["keep", "fill"], ["output"]),
    ]
    s1 = [1, 1, HEIGHT, WIDTH]
    vi = [
        helper.make_tensor_value_info("maskC", F, s1),
        helper.make_tensor_value_info("g2", F, [1, CHANNELS, size, WIDTH]),
        helper.make_tensor_value_info("rotSS", F, [1, CHANNELS, size, size]),
        helper.make_tensor_value_info("rotpad", F, FULL),
        helper.make_tensor_value_info("inv", F, s1),
        helper.make_tensor_value_info("keep", F, FULL),
        helper.make_tensor_value_info("fill", F, FULL),
    ]
    graph = helper.make_graph(nodes, "rot180_repair",
                              [helper.make_tensor_value_info("input", F, FULL)],
                              [helper.make_tensor_value_info("output", F, FULL)],
                              initializer=init, value_info=vi)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_rot180_repair(task: dict) -> Optional[onnx.ModelProto]:
    p = _params(task)
    if p is None:
        return None
    return _build(*p)
