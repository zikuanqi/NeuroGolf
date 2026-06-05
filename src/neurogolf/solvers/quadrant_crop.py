"""Solver: crop the top-left quadrant of the content bounding box (task 39).

The only content is a single 2h x 2h block; the output is its top-left h x h
quadrant.  h is fixed per task, so it is baked.  At runtime the bounding box
position is found with a CumSum (number of empty leading rows/cols = rmin /
cmin), then `Gather` lifts the h rows starting at rmin and the h cols starting
at cmin, and `Pad` returns the h x h crop to the 30x30 canvas top-left.
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


def _quadrant(g: np.ndarray) -> Optional[tuple[np.ndarray, int]]:
    nz = np.argwhere(g != 0)
    if len(nz) == 0:
        return None
    r0, c0 = nz.min(0); r1, c1 = nz.max(0)
    H, W = r1 - r0 + 1, c1 - c0 + 1
    if H != W or H % 2 != 0:
        return None
    h = H // 2
    return g[r0:r0 + h, c0:c0 + h], h


def _params(task: dict) -> Optional[int]:
    half = None
    saw = False
    for ex in all_examples(task):
        i, o = ex["input"], ex["output"]
        if not i or not i[0]:
            continue
        if len(i) > HEIGHT or len(i[0]) > WIDTH:
            continue
        res = _quadrant(np.array(i))
        if res is None:
            return None
        quad, h = res
        if quad.shape != np.array(o).shape or not np.array_equal(quad, np.array(o)):
            return None
        if half is None:
            half = h
        elif half != h:
            return None
        saw = True
    if not saw or half is None or half < 1 or 2 * half > min(HEIGHT, WIDTH):
        return None
    return half


def _build(half: int) -> onnx.ModelProto:
    half = int(half)
    F = TensorProto.FLOAT
    I = TensorProto.INT64
    n = helper.make_node
    ar = np.arange(half, dtype=np.int64)
    pad_end = WIDTH - half
    init = [
        numpy_helper.from_array(np.array(2, np.int64), "ax2"),
        numpy_helper.from_array(np.array(3, np.int64), "ax3"),
        numpy_helper.from_array(np.array(0.5, np.float32), "half_f"),
        numpy_helper.from_array(ar, "ar"),
        numpy_helper.from_array(np.array([1], np.int64), "shp1"),
        numpy_helper.from_array(np.array([0, 0, 0, 0, 0, 0, pad_end, pad_end], np.int64), "pads"),
        numpy_helper.from_array(np.array([0, 1, 0, 0], np.int64), "ch_s"),
        numpy_helper.from_array(np.array([1, CHANNELS, HEIGHT, WIDTH], np.int64), "ch_e"),
        numpy_helper.from_array(np.array([0, 1, 2, 3], np.int64), "ax4"),
    ]
    rc = [1, 1, HEIGHT, 1]; cc = [1, 1, 1, WIDTH]; s1 = [1, 1, HEIGHT, WIDTH]
    B = TensorProto.BOOL
    nodes = [
        n("Slice", ["input", "ch_s", "ch_e", "ax4"], ["nz"]),               # (1,9,H,W) non-bg
        n("ReduceSum", ["nz"], ["content"], axes=[1], keepdims=1),          # (1,1,H,W)
        n("ReduceMax", ["content"], ["row_has"], axes=[3], keepdims=1),     # (1,1,H,1)
        n("ReduceMax", ["content"], ["col_has"], axes=[2], keepdims=1),     # (1,1,1,W)
        n("CumSum", ["row_has", "ax2"], ["crow"]),
        n("CumSum", ["col_has", "ax3"], ["ccol"]),
        n("Less", ["crow", "half_f"], ["rlead_b"]), n("Cast", ["rlead_b"], ["rlead"], to=F),
        n("Less", ["ccol", "half_f"], ["clead_b"]), n("Cast", ["clead_b"], ["clead"], to=F),
        n("ReduceSum", ["rlead"], ["rmin_f"]),                              # scalar (1,1,1,1)
        n("ReduceSum", ["clead"], ["cmin_f"]),
        n("Reshape", ["rmin_f", "shp1"], ["rmin1_f"]),                      # (1,)
        n("Reshape", ["cmin_f", "shp1"], ["cmin1_f"]),
        n("Cast", ["rmin1_f"], ["rmin1"], to=I),
        n("Cast", ["cmin1_f"], ["cmin1"], to=I),
        n("Add", ["rmin1", "ar"], ["idx_r"]),                              # (half,)
        n("Add", ["cmin1", "ar"], ["idx_c"]),                              # (half,)
        n("Gather", ["input", "idx_r"], ["g_r"], axis=2),                  # (1,10,half,W)
        n("Gather", ["g_r", "idx_c"], ["quad"], axis=3),                   # (1,10,half,half)
        n("Pad", ["quad", "pads"], ["output"], mode="constant"),
    ]
    vi = [
        helper.make_tensor_value_info("nz", F, [1, CHANNELS - 1, HEIGHT, WIDTH]),
        helper.make_tensor_value_info("content", F, s1),
        helper.make_tensor_value_info("row_has", F, rc),
        helper.make_tensor_value_info("col_has", F, cc),
        helper.make_tensor_value_info("crow", F, rc),
        helper.make_tensor_value_info("ccol", F, cc),
        helper.make_tensor_value_info("rlead_b", B, rc),
        helper.make_tensor_value_info("rlead", F, rc),
        helper.make_tensor_value_info("clead_b", B, cc),
        helper.make_tensor_value_info("clead", F, cc),
        helper.make_tensor_value_info("rmin_f", F, [1, 1, 1, 1]),
        helper.make_tensor_value_info("cmin_f", F, [1, 1, 1, 1]),
        helper.make_tensor_value_info("rmin1_f", F, [1]),
        helper.make_tensor_value_info("cmin1_f", F, [1]),
        helper.make_tensor_value_info("rmin1", I, [1]),
        helper.make_tensor_value_info("cmin1", I, [1]),
        helper.make_tensor_value_info("idx_r", I, [half]),
        helper.make_tensor_value_info("idx_c", I, [half]),
        helper.make_tensor_value_info("g_r", F, [1, CHANNELS, half, WIDTH]),
        helper.make_tensor_value_info("quad", F, [1, CHANNELS, half, half]),
    ]
    graph = helper.make_graph(nodes, "quadrant_crop",
                              [helper.make_tensor_value_info("input", F, FULL)],
                              [helper.make_tensor_value_info("output", F, FULL)],
                              initializer=init, value_info=vi)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_quadrant_crop(task: dict) -> Optional[onnx.ModelProto]:
    half = _params(task)
    if half is None:
        return None
    return _build(half)
