"""Solver: output a 1xN horizontal bar whose length counts the markers.

The grid holds some cells of a single non-background colour; the output is a
single row of that colour whose length equals the number of marker cells
(task 339).

One-hot channel arithmetic on the (1, 10, 30, 30) tensor:
  N        = number of marker cells (sum of non-background)
  bar_mask = row 0, columns 0..N-1
  M        = one-hot of the (single) marker colour
  output   = M painted on the bar mask
"""
from __future__ import annotations

from collections import Counter
from typing import Optional

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from ..grids import CHANNELS, HEIGHT, WIDTH, all_examples

OPSET = 11
IR_VERSION = 8


def _transform(grid):
    cnt = Counter(int(x) for row in grid for x in row if x != 0)
    if len(cnt) != 1:
        return None
    color, n = next(iter(cnt.items()))
    return [[color] * n]


def _detect(task: dict) -> bool:
    examples = list(all_examples(task))
    if not examples:
        return False
    for ex in examples:
        inp, out = ex["input"], ex["output"]
        if len(inp) > 30 or len(inp[0]) > 30:
            continue
        if _transform(inp) != out:
            return False
    return True


def _build() -> onnx.ModelProto:
    F = TensorProto.FLOAT
    B = TensorProto.BOOL

    def f32(name, arr):
        return numpy_helper.from_array(arr.astype(np.float32), name)

    e_nonbg = np.ones((1, CHANNELS, 1, 1), dtype=np.float32)
    e_nonbg[0, 0, 0, 0] = 0.0
    colramp = np.arange(WIDTH, dtype=np.float32).reshape(1, 1, 1, WIDTH)
    e_row0 = np.zeros((1, 1, HEIGHT, 1), dtype=np.float32)
    e_row0[0, 0, 0, 0] = 1.0

    init = [
        f32("e_nonbg", e_nonbg), f32("colramp", colramp), f32("e_row0", e_row0),
        f32("half", np.array([[[[0.5]]]])),
        numpy_helper.from_array(np.array([0], dtype=np.int64), "idx0"),
    ]

    n = helper.make_node
    nodes = [
        n("ReduceSum", ["input"], ["content"], axes=[1], keepdims=1),
        n("Gather", ["input", "idx0"], ["ch0"], axis=1),
        n("Sub", ["content", "ch0"], ["nonbg"]),
        n("ReduceSum", ["nonbg"], ["ncount"], axes=[2, 3], keepdims=1),
        n("Less", ["colramp", "ncount"], ["lt_b"]),
        n("Cast", ["lt_b"], ["barcol"], to=F),
        n("Mul", ["e_row0", "barcol"], ["bar_mask"]),
        n("ReduceSum", ["input"], ["counts"], axes=[2, 3], keepdims=1),
        n("Mul", ["counts", "e_nonbg"], ["counts_nz"]),
        n("Greater", ["counts_nz", "half"], ["m_b"]),
        n("Cast", ["m_b"], ["m_onehot"], to=F),
        n("Mul", ["m_onehot", "bar_mask"], ["output"]),
    ]

    def vi(name, shape, dt=F):
        return helper.make_tensor_value_info(name, dt, shape)

    g4 = [1, CHANNELS, HEIGHT, WIDTH]
    s1 = [1, 1, HEIGHT, WIDTH]
    c11 = [1, CHANNELS, 1, 1]
    value_info = [
        vi("content", s1), vi("ch0", s1), vi("nonbg", s1),
        vi("ncount", [1, 1, 1, 1]),
        vi("lt_b", [1, 1, 1, WIDTH], B), vi("barcol", [1, 1, 1, WIDTH]),
        vi("bar_mask", s1),
        vi("counts", c11), vi("counts_nz", c11),
        vi("m_b", c11, B), vi("m_onehot", c11),
    ]

    inputs = [vi("input", g4)]
    outputs = [vi("output", g4)]
    graph = helper.make_graph(nodes, "count_bar", inputs, outputs,
                              initializer=init, value_info=value_info)
    return helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
        ir_version=IR_VERSION)


def solve_count_bar(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
