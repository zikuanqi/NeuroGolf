"""Solver: repaint the majority blob with the lone key colour, clear the rest.

The grid holds exactly two non-background colours: a frequent "blob" colour A
and a rarer "key" colour B. The output paints every A cell with colour B and
turns everything else (the key cell and the background) into background
(task 267).

One-hot channel arithmetic on the (1, 10, 30, 30) tensor:
  A = argmax channel of the per-colour counts   (the blob colour)
  B = the other present non-background channel   (present - A)
  is_A = cells whose colour is A
  output = B on the A cells + background everywhere else
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from ..grids import CHANNELS, HEIGHT, WIDTH, all_examples

OPSET = 11
IR_VERSION = 8


def _transform(grid):
    from collections import Counter
    cnt = Counter(int(x) for row in grid for x in row if x != 0)
    if len(cnt) != 2:
        return None
    items = sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))
    if items[0][1] == items[1][1]:
        return None
    A, B = items[0][0], items[1][0]
    return [[B if x == A else 0 for x in row] for row in grid]


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
    e0 = np.zeros((1, CHANNELS, 1, 1), dtype=np.float32)
    e0[0, 0, 0, 0] = 1.0

    init = [
        f32("e_nonbg", e_nonbg), f32("e0", e0),
        f32("half", np.array([[[[0.5]]]])),
    ]

    n = helper.make_node
    nodes = [
        n("ReduceSum", ["input"], ["content"], axes=[1], keepdims=1),
        n("ReduceSum", ["input"], ["counts"], axes=[2, 3], keepdims=1),
        n("Mul", ["counts", "e_nonbg"], ["counts_nz"]),
        n("ReduceMax", ["counts_nz"], ["maxc"], axes=[1], keepdims=1),
        n("Equal", ["counts_nz", "maxc"], ["eq_b"]),
        n("Cast", ["eq_b"], ["a_raw"], to=F),
        n("Mul", ["a_raw", "e_nonbg"], ["a_onehot"]),
        n("Greater", ["counts_nz", "half"], ["present_b"]),
        n("Cast", ["present_b"], ["present"], to=F),
        n("Sub", ["present", "a_onehot"], ["b_onehot"]),
        n("Mul", ["input", "a_onehot"], ["a_input"]),
        n("ReduceSum", ["a_input"], ["is_a"], axes=[1], keepdims=1),
        n("Mul", ["b_onehot", "is_a"], ["out_blob"]),
        n("Sub", ["content", "is_a"], ["not_a_real"]),
        n("Mul", ["e0", "not_a_real"], ["out_bg"]),
        n("Add", ["out_blob", "out_bg"], ["output"]),
    ]

    def vi(name, shape, dt=F):
        return helper.make_tensor_value_info(name, dt, shape)

    g4 = [1, CHANNELS, HEIGHT, WIDTH]
    s1 = [1, 1, HEIGHT, WIDTH]
    c11 = [1, CHANNELS, 1, 1]
    value_info = [
        vi("content", s1), vi("counts", c11), vi("counts_nz", c11),
        vi("maxc", [1, 1, 1, 1]), vi("eq_b", c11, B), vi("a_raw", c11),
        vi("a_onehot", c11), vi("present_b", c11, B), vi("present", c11),
        vi("b_onehot", c11), vi("a_input", g4), vi("is_a", s1),
        vi("out_blob", g4), vi("not_a_real", s1), vi("out_bg", g4),
    ]

    inputs = [vi("input", g4)]
    outputs = [vi("output", g4)]
    graph = helper.make_graph(nodes, "blob_recolor", inputs, outputs,
                              initializer=init, value_info=value_info)
    return helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
        ir_version=IR_VERSION)


def solve_blob_recolor(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
