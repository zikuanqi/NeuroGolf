"""Solver: expand every marker into a filled 3x3 block of colour 1.

Any cell within the 3x3 neighbourhood of a non-background cell (including the
markers themselves) becomes colour 1; everything else becomes background. The
original colours are discarded (task 317).

One-hot channel arithmetic on the (1, 10, 30, 30) tensor:
  nonbg  = content - channel-0          (marker cells)
  cnt    = 3x3 box convolution of nonbg
  on     = (cnt > 0) restricted to real cells   -> colour 1
  output = colour 1 on the dilated cells + background elsewhere
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from ..grids import CHANNELS, HEIGHT, WIDTH, all_examples

OPSET = 11
IR_VERSION = 8
FILL_COLOR = 1


def _transform(grid):
    H = len(grid); W = len(grid[0])
    out = [[0] * W for _ in range(H)]
    for r in range(H):
        for c in range(W):
            near = any(grid[r + dr][c + dc] != 0
                       for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                       if 0 <= r + dr < H and 0 <= c + dc < W)
            if near:
                out[r][c] = FILL_COLOR
    return out


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

    e0 = np.zeros((1, CHANNELS, 1, 1), dtype=np.float32)
    e0[0, 0, 0, 0] = 1.0
    e1 = np.zeros((1, CHANNELS, 1, 1), dtype=np.float32)
    e1[0, FILL_COLOR, 0, 0] = 1.0
    ones_k = np.ones((1, 1, 3, 3), dtype=np.float32)

    init = [
        f32("e0", e0), f32("e1", e1), f32("ones_k", ones_k),
        f32("half", np.array([[[[0.5]]]])),
        numpy_helper.from_array(np.array([0], dtype=np.int64), "idx0"),
    ]

    n = helper.make_node
    nodes = [
        n("ReduceSum", ["input"], ["content"], axes=[1], keepdims=1),
        n("Gather", ["input", "idx0"], ["ch0"], axis=1),
        n("Sub", ["content", "ch0"], ["nonbg"]),
        n("Conv", ["nonbg", "ones_k"], ["cnt"],
          kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        n("Greater", ["cnt", "half"], ["dil_b"]),
        n("Cast", ["dil_b"], ["dilated"], to=F),
        n("Mul", ["dilated", "content"], ["on"]),
        n("Sub", ["content", "on"], ["off"]),
        n("Mul", ["e1", "on"], ["out_1"]),
        n("Mul", ["e0", "off"], ["out_0"]),
        n("Add", ["out_1", "out_0"], ["output"]),
    ]

    def vi(name, shape, dt=F):
        return helper.make_tensor_value_info(name, dt, shape)

    g4 = [1, CHANNELS, HEIGHT, WIDTH]
    s1 = [1, 1, HEIGHT, WIDTH]
    value_info = [
        vi("content", s1), vi("ch0", s1), vi("nonbg", s1), vi("cnt", s1),
        vi("dil_b", s1, B), vi("dilated", s1), vi("on", s1), vi("off", s1),
        vi("out_1", g4), vi("out_0", g4),
    ]

    inputs = [vi("input", g4)]
    outputs = [vi("output", g4)]
    graph = helper.make_graph(nodes, "dilate_ones", inputs, outputs,
                              initializer=init, value_info=value_info)
    return helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
        ir_version=IR_VERSION)


def solve_dilate_ones(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
