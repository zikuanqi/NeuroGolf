"""Solver: keep the most-frequent colour, recolour every other marker to 5.

The non-background colour that occurs most often is left untouched; all other
non-background cells are repainted colour 5; background stays background
(task 29).

One-hot channel arithmetic on the (1, 10, 30, 30) tensor:
  counts   = per-channel cell count (channel 0 zeroed)
  M        = one-hot of the argmax channel (the most-frequent colour)
  is_M     = cells whose colour is M
  output   = M on the M cells + colour 5 on the other markers + background
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from ..grids import CHANNELS, HEIGHT, WIDTH, all_examples

OPSET = 11
IR_VERSION = 8
OTHER_COLOR = 5


def _most_freq(grid):
    from collections import Counter
    cnt = Counter(int(x) for row in grid for x in row if x != 0)
    if not cnt:
        return None, False
    items = sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))
    tie = len(items) > 1 and items[0][1] == items[1][1]
    return items[0][0], tie


def _transform(grid):
    m, tie = _most_freq(grid)
    if m is None or tie:
        return None
    return [[0 if x == 0 else (m if x == m else OTHER_COLOR) for x in row]
            for row in grid]


def _detect(task: dict) -> bool:
    examples = list(all_examples(task))
    if not examples:
        return False
    changed = False
    for ex in examples:
        inp, out = ex["input"], ex["output"]
        if len(inp) > 30 or len(inp[0]) > 30:
            continue
        if len(inp) != len(out) or len(inp[0]) != len(out[0]):
            return False
        if _transform(inp) != out:
            return False
        if inp != out:
            changed = True
    return changed


def _build() -> onnx.ModelProto:
    F = TensorProto.FLOAT
    B = TensorProto.BOOL

    def f32(name, arr):
        return numpy_helper.from_array(arr.astype(np.float32), name)

    e_nonbg = np.ones((1, CHANNELS, 1, 1), dtype=np.float32)
    e_nonbg[0, 0, 0, 0] = 0.0
    e5 = np.zeros((1, CHANNELS, 1, 1), dtype=np.float32)
    e5[0, OTHER_COLOR, 0, 0] = 1.0
    e0 = np.zeros((1, CHANNELS, 1, 1), dtype=np.float32)
    e0[0, 0, 0, 0] = 1.0

    init = [
        f32("e_nonbg", e_nonbg), f32("e5", e5), f32("e0", e0),
        numpy_helper.from_array(np.array([0], dtype=np.int64), "idx0"),
    ]

    n = helper.make_node
    nodes = [
        n("ReduceSum", ["input"], ["content"], axes=[1], keepdims=1),
        n("Gather", ["input", "idx0"], ["ch0"], axis=1),
        n("Sub", ["content", "ch0"], ["nonbg"]),
        n("ReduceSum", ["input"], ["counts"], axes=[2, 3], keepdims=1),
        n("Mul", ["counts", "e_nonbg"], ["counts_nz"]),
        n("ReduceMax", ["counts_nz"], ["maxc"], axes=[1], keepdims=1),
        n("Equal", ["counts_nz", "maxc"], ["eq_b"]),
        n("Cast", ["eq_b"], ["m_raw"], to=F),
        n("Mul", ["m_raw", "e_nonbg"], ["m_onehot"]),
        n("Mul", ["input", "m_onehot"], ["m_input"]),
        n("ReduceSum", ["m_input"], ["is_m"], axes=[1], keepdims=1),
        n("Mul", ["m_onehot", "is_m"], ["keep_m"]),
        n("Sub", ["nonbg", "is_m"], ["other_mask"]),
        n("Mul", ["e5", "other_mask"], ["out_5"]),
        n("Mul", ["e0", "ch0"], ["out_bg"]),
        n("Add", ["keep_m", "out_5"], ["tmp"]),
        n("Add", ["tmp", "out_bg"], ["output"]),
    ]

    def vi(name, shape, dt=F):
        return helper.make_tensor_value_info(name, dt, shape)

    g4 = [1, CHANNELS, HEIGHT, WIDTH]
    s1 = [1, 1, HEIGHT, WIDTH]
    c11 = [1, CHANNELS, 1, 1]
    value_info = [
        vi("content", s1), vi("ch0", s1), vi("nonbg", s1),
        vi("counts", c11), vi("counts_nz", c11), vi("maxc", [1, 1, 1, 1]),
        vi("eq_b", c11, B), vi("m_raw", c11), vi("m_onehot", c11),
        vi("m_input", g4), vi("is_m", s1), vi("keep_m", g4),
        vi("other_mask", s1), vi("out_5", g4), vi("out_bg", g4), vi("tmp", g4),
    ]

    inputs = [vi("input", g4)]
    outputs = [vi("output", g4)]
    graph = helper.make_graph(nodes, "keep_majority", inputs, outputs,
                              initializer=init, value_info=value_info)
    return helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
        ir_version=IR_VERSION)


def solve_keep_majority(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
