"""Solver: recolour a fixed two-region frame template (task 28).

Every example is a 10x10 grid with exactly two single markers (one in an upper
row, one in a lower row). The output is a constant frame template: an upper
region painted the upper marker's colour and a lower region painted the lower
marker's colour. Only the two colours vary, so the whole transform is: read the
two marker colours and paint two baked masks.
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


def _params(task: dict):
    """Return (maskA, maskB, rA, rB) or None: constant region masks + the input
    rows holding the upper/lower markers."""
    maskA = maskB = None
    rA = rB = None
    saw = False
    for ex in all_examples(task):
        i, o = np.array(ex["input"]), np.array(ex["output"])
        if i.shape != o.shape or i.shape[0] > HEIGHT or i.shape[1] > WIDTH:
            return None
        nz = np.argwhere(i != 0)
        if len(nz) != 2:
            return None
        (r_a, c_a), (r_b, c_b) = sorted(nz.tolist())
        A, Bc = int(i[r_a, c_a]), int(i[r_b, c_b])
        if A == 0 or Bc == 0 or A == Bc:
            return None
        mA = np.zeros((HEIGHT, WIDTH), bool)
        mB = np.zeros((HEIGHT, WIDTH), bool)
        H, W = o.shape
        mA[:H, :W] = (o == A)
        mB[:H, :W] = (o == Bc)
        # every coloured output cell must be A or B
        if not np.array_equal((o != 0), (o == A) | (o == Bc)):
            return None
        if maskA is None:
            maskA, maskB, rA, rB = mA, mB, r_a, r_b
        elif not (np.array_equal(mA, maskA) and np.array_equal(mB, maskB)
                  and rA == r_a and rB == r_b):
            return None
        saw = True
    if not saw or maskA is None or (not maskA.any()) or (not maskB.any()):
        return None
    return maskA, maskB, rA, rB


def _build(maskA, maskB, rA, rB) -> onnx.ModelProto:
    F = TensorProto.FLOAT
    n = helper.make_node
    e_nonbg = np.ones((1, CHANNELS, 1, 1), np.float32); e_nonbg[0, 0] = 0.0
    e0 = np.zeros((1, CHANNELS, 1, 1), np.float32); e0[0, 0] = 1.0
    init = [
        numpy_helper.from_array(e_nonbg, "e_nonbg"),
        numpy_helper.from_array(e0, "e0"),
        numpy_helper.from_array(maskA.astype(np.float32).reshape(1, 1, HEIGHT, WIDTH), "maskA"),
        numpy_helper.from_array(maskB.astype(np.float32).reshape(1, 1, HEIGHT, WIDTH), "maskB"),
        numpy_helper.from_array(np.array([0, 0, rA, 0], np.int64), "rA_s"),
        numpy_helper.from_array(np.array([1, CHANNELS, rA + 1, WIDTH], np.int64), "rA_e"),
        numpy_helper.from_array(np.array([0, 0, rB, 0], np.int64), "rB_s"),
        numpy_helper.from_array(np.array([1, CHANNELS, rB + 1, WIDTH], np.int64), "rB_e"),
        numpy_helper.from_array(np.array([0, 1, 2, 3], np.int64), "ax4"),
    ]
    nodes = [
        n("Slice", ["input", "rA_s", "rA_e", "ax4"], ["rowA"]),
        n("ReduceMax", ["rowA"], ["maxA"], axes=[2, 3], keepdims=1),
        n("Mul", ["maxA", "e_nonbg"], ["Aoh"]),
        n("Slice", ["input", "rB_s", "rB_e", "ax4"], ["rowB"]),
        n("ReduceMax", ["rowB"], ["maxB"], axes=[2, 3], keepdims=1),
        n("Mul", ["maxB", "e_nonbg"], ["Boh"]),
        n("Mul", ["Aoh", "maskA"], ["partA"]),
        n("Mul", ["Boh", "maskB"], ["partB"]),
        # background = real grid (content) minus the two masks, on channel 0
        n("ReduceSum", ["input"], ["content"], axes=[1], keepdims=1),
        n("Sub", ["content", "maskA"], ["cm1"]),
        n("Sub", ["cm1", "maskB"], ["bgmask"]),
        n("Mul", ["e0", "bgmask"], ["partBg"]),
        n("Add", ["partA", "partB"], ["ab"]),
        n("Add", ["ab", "partBg"], ["output"]),
    ]
    c11 = [1, CHANNELS, 1, 1]; s1 = [1, 1, HEIGHT, WIDTH]
    vi = [
        helper.make_tensor_value_info("rowA", F, [1, CHANNELS, 1, WIDTH]),
        helper.make_tensor_value_info("maxA", F, c11),
        helper.make_tensor_value_info("Aoh", F, c11),
        helper.make_tensor_value_info("rowB", F, [1, CHANNELS, 1, WIDTH]),
        helper.make_tensor_value_info("maxB", F, c11),
        helper.make_tensor_value_info("Boh", F, c11),
        helper.make_tensor_value_info("partA", F, FULL),
        helper.make_tensor_value_info("partB", F, FULL),
        helper.make_tensor_value_info("content", F, s1),
        helper.make_tensor_value_info("cm1", F, s1),
        helper.make_tensor_value_info("bgmask", F, s1),
        helper.make_tensor_value_info("partBg", F, FULL),
        helper.make_tensor_value_info("ab", F, FULL),
    ]
    graph = helper.make_graph(nodes, "framed_regions",
                              [helper.make_tensor_value_info("input", F, FULL)],
                              [helper.make_tensor_value_info("output", F, FULL)],
                              initializer=init, value_info=vi)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_framed_regions(task: dict) -> Optional[onnx.ModelProto]:
    p = _params(task)
    if p is None:
        return None
    return _build(*p)
