"""Solver: crop the bounding box and swap its two colours (task 290).

The single shape is a filled rectangle with a border colour and an interior
colour.  The output is the tight bounding box cropped to the top-left, with the
two colours exchanged.

* colour swap — a runtime 10x10 permutation matrix that swaps the two present
  non-background colours, applied as a channel ``MatMul``;
* crop — the standard ``bbox_strip`` clamped two-axis ``Gather`` to the top-left.
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
NCELL = HEIGHT * WIDTH


def _ref(g: np.ndarray) -> Optional[np.ndarray]:
    ys, xs = np.where(g != 0)
    if len(ys) == 0:
        return None
    sub = g[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    cols = sorted(set(sub[sub != 0].tolist()))
    if len(cols) != 2:
        return None
    a, b = cols
    out = sub.copy()
    out[sub == a] = b
    out[sub == b] = a
    return out


def _detect(task: dict) -> bool:
    saw = False
    for ex in all_examples(task):
        i, o = ex["input"], ex["output"]
        if not i or not i[0] or len(i) > HEIGHT or len(i[0]) > WIDTH:
            continue
        r = _ref(np.array(i))
        if r is None or r.tolist() != o:
            return False
        saw = True
    return saw


def _build() -> onnx.ModelProto:
    n = helper.make_node
    notbg = np.ones((1, CHANNELS, 1, 1), np.float32); notbg[0, 0] = 0.0
    init = [
        numpy_helper.from_array(notbg, "notbg"),
        numpy_helper.from_array(np.eye(CHANNELS, dtype=np.float32), "I10"),
        numpy_helper.from_array(np.array(1.0, np.float32), "one"),
        numpy_helper.from_array(np.array([CHANNELS, 1], np.int64), "to_col10"),
        numpy_helper.from_array(np.array([1, CHANNELS], np.int64), "to_row10"),
        numpy_helper.from_array(np.array([CHANNELS, NCELL], np.int64), "to_flat"),
        numpy_helper.from_array(np.array([1, CHANNELS, HEIGHT, WIDTH], np.int64), "to_grid"),
        # crop constants
        numpy_helper.from_array(np.array([0], np.int64), "c0s"),
        numpy_helper.from_array(np.array([1], np.int64), "c0e"),
        numpy_helper.from_array(np.array([1], np.int64), "ax1"),
        numpy_helper.from_array(np.arange(HEIGHT, dtype=np.float32), "arangeH"),
        numpy_helper.from_array(np.arange(WIDTH, dtype=np.float32), "arangeW"),
        numpy_helper.from_array(np.array([1, 1, HEIGHT, 1], np.int64), "shH"),
        numpy_helper.from_array(np.array([1, 1, 1, WIDTH], np.int64), "shW"),
        numpy_helper.from_array(np.array([HEIGHT], np.int64), "idxH"),
        numpy_helper.from_array(np.array([WIDTH], np.int64), "idxW"),
        numpy_helper.from_array(np.array(0.0, np.float32), "clipMin"),
        numpy_helper.from_array(np.array(float(HEIGHT - 1), np.float32), "clipMax"),
    ]
    nodes = [
        # ---- build swap permutation P ----
        n("ReduceMax", ["input"], ["present"], axes=[2, 3], keepdims=1),
        n("Mul", ["present", "notbg"], ["v"]),
        n("Reshape", ["v", "to_col10"], ["vr"]),     # (10,1)
        n("Reshape", ["v", "to_row10"], ["vc"]),     # (1,10)
        n("MatMul", ["vr", "vc"], ["vOuter"]),       # (10,10)
        n("Mul", ["I10", "vr"], ["diagV"]),
        n("Sub", ["one", "vr"], ["oneMinusVr"]),
        n("Mul", ["I10", "oneMinusVr"], ["diag1mv"]),
        n("Add", ["diag1mv", "vOuter"], ["pTmp"]),
        n("Sub", ["pTmp", "diagV"], ["P"]),
        # ---- apply swap ----
        n("Reshape", ["input", "to_flat"], ["inFlat"]),
        n("MatMul", ["P", "inFlat"], ["swFlat"]),
        n("Reshape", ["swFlat", "to_grid"], ["swapped"]),
        # ---- crop bbox of non-bg to top-left (bg = 0) ----
        n("ReduceSum", ["input"], ["allsum"], axes=[1], keepdims=1),
        n("Slice", ["input", "c0s", "c0e", "ax1"], ["bgonly"]),
        n("Sub", ["allsum", "bgonly"], ["nonbg"]),
        n("ReduceMax", ["nonbg"], ["rowProj"], axes=[3], keepdims=1),
        n("ReduceMax", ["nonbg"], ["colProj"], axes=[2], keepdims=1),
        n("Reshape", ["arangeH", "shH"], ["arH"]),
        n("Reshape", ["arangeW", "shW"], ["arW"]),
        n("Mul", ["rowProj", "arH"], ["rowPos"]),
        n("Mul", ["colProj", "arW"], ["colPos"]),
        n("ReduceMax", ["rowPos"], ["rMax"], axes=[2], keepdims=1),
        n("ReduceMax", ["colPos"], ["cMax"], axes=[3], keepdims=1),
        n("ArgMax", ["rowProj"], ["rMinI"], axis=2, keepdims=1),
        n("ArgMax", ["colProj"], ["cMinI"], axis=3, keepdims=1),
        n("Cast", ["rMinI"], ["rMin"], to=F),
        n("Cast", ["cMinI"], ["cMin"], to=F),
        n("Sub", ["rMax", "rMin"], ["hm1"]),
        n("Sub", ["cMax", "cMin"], ["wm1"]),
        n("Add", ["arH", "rMin"], ["shRowF"]),
        n("Clip", ["shRowF", "clipMin", "clipMax"], ["shRowC"]),
        n("Cast", ["shRowC"], ["shRowI"], to=TensorProto.INT64),
        n("Reshape", ["shRowI", "idxH"], ["shRow1d"]),
        n("Add", ["arW", "cMin"], ["shColF"]),
        n("Clip", ["shColF", "clipMin", "clipMax"], ["shColC"]),
        n("Cast", ["shColC"], ["shColI"], to=TensorProto.INT64),
        n("Reshape", ["shColI", "idxW"], ["shCol1d"]),
        n("Gather", ["swapped", "shRow1d"], ["gRows"], axis=2),
        n("Gather", ["gRows", "shCol1d"], ["gCrop"], axis=3),
        n("Add", ["hm1", "one"], ["hf"]),
        n("Add", ["wm1", "one"], ["wf"]),
        n("Less", ["arH", "hf"], ["hle_b"]), n("Cast", ["hle_b"], ["hle"], to=F),
        n("Less", ["arW", "wf"], ["wle_b"]), n("Cast", ["wle_b"], ["wle"], to=F),
        n("Mul", ["hle", "wle"], ["bboxMask"]),
        n("Mul", ["gCrop", "bboxMask"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "crop_swap_pair",
                              [helper.make_tensor_value_info("input", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              [helper.make_tensor_value_info("output", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_crop_swap_pair(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
