"""Solver: denoise by removing isolated single cells.

A cell is kept only if it has at least one 8-neighbour of the same colour;
lone specks are erased to background (task 97).

Per colour channel the same-colour neighbour count is a 3x3 convolution with a
hollow all-ones kernel (centre = 0). A cell survives where it is that colour
AND its neighbour count is positive. The convolution is depthwise (group = 9)
so colours never mix, and the background channel is rebuilt inside the grid so
the output stays a valid one-hot.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from ..grids import CHANNELS, HEIGHT, WIDTH, all_examples

OPSET = 11
IR_VERSION = 8
NC = CHANNELS - 1   # colour channels (1..9)


def _denoise(grid):
    g = np.array(grid)
    H, W = g.shape
    out = np.zeros((H, W), dtype=int)
    for k in range(1, 10):
        m = (g == k).astype(float)
        pad = np.pad(m, 1)
        nb = np.zeros((H, W))
        for dr in range(3):
            for dc in range(3):
                if not (dr == 1 and dc == 1):
                    nb += pad[dr:dr + H, dc:dc + W]
        out = np.where((m > 0) & (nb > 0), k, out)
    return out.tolist()


def _detect(task: dict) -> bool:
    examples = list(all_examples(task))
    if not examples:
        return False
    changed = False
    for ex in examples:
        inp, out = ex["input"], ex["output"]
        if len(inp) != len(out) or len(inp[0]) != len(out[0]):
            return False
        if _denoise(inp) != out:
            return False
        if inp != out:
            changed = True
    return changed


def _build() -> onnx.ModelProto:
    def i64(name, vals):
        return numpy_helper.from_array(np.array(vals, dtype=np.int64), name)

    def f32(name, arr):
        return numpy_helper.from_array(arr.astype(np.float32), name)

    # depthwise hollow 3x3 kernel: shape [NC, 1, 3, 3], centre 0, rest 1
    kernel = np.ones((NC, 1, 3, 3), dtype=np.float32)
    kernel[:, :, 1, 1] = 0.0

    init = [
        i64("c1", [1]), i64("c10", [CHANNELS]), i64("ax1", [1]), i64("st1", [1]),
        f32("kernel", kernel),
        f32("zero", np.array([0.0])), f32("one_f", np.array([1.0])),
    ]

    nodes = [
        helper.make_node("Slice", ["input", "c1", "c10", "ax1", "st1"],
                         ["colors"]),                          # [1,9,H,W]
        # same-colour neighbour count, depthwise (each colour independent)
        helper.make_node("Conv", ["colors", "kernel"], ["nb"],
                         kernel_shape=[3, 3], pads=[1, 1, 1, 1], group=NC),
        helper.make_node("Greater", ["nb", "zero"], ["has_nb_b"]),
        helper.make_node("Cast", ["has_nb_b"], ["has_nb"],
                         to=TensorProto.FLOAT),
        helper.make_node("Mul", ["colors", "has_nb"], ["kept"]),   # [1,9,H,W]
        # rebuild background inside the real grid: bg = in_grid AND not a colour
        helper.make_node("ReduceSum", ["input"], ["content"], axes=[1],
                         keepdims=1),
        helper.make_node("Greater", ["content", "zero"], ["in_grid_b"]),
        helper.make_node("Cast", ["in_grid_b"], ["in_grid"],
                         to=TensorProto.FLOAT),
        helper.make_node("ReduceSum", ["kept"], ["csum"], axes=[1],
                         keepdims=1),
        helper.make_node("Sub", ["one_f", "csum"], ["not_color"]),
        helper.make_node("Mul", ["in_grid", "not_color"], ["bg"]),
        helper.make_node("Concat", ["bg", "kept"], ["output"], axis=1),
    ]

    inputs = [helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, CHANNELS, HEIGHT, WIDTH])]
    outputs = [helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, CHANNELS, HEIGHT, WIDTH])]
    value_info = [
        helper.make_tensor_value_info("colors", TensorProto.FLOAT,
                                      [1, NC, HEIGHT, WIDTH]),
        helper.make_tensor_value_info("nb", TensorProto.FLOAT,
                                      [1, NC, HEIGHT, WIDTH]),
        helper.make_tensor_value_info("kept", TensorProto.FLOAT,
                                      [1, NC, HEIGHT, WIDTH]),
        helper.make_tensor_value_info("bg", TensorProto.FLOAT,
                                      [1, 1, HEIGHT, WIDTH]),
    ]
    graph = helper.make_graph(nodes, "denoise", inputs, outputs,
                              initializer=init, value_info=value_info)
    return helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
        ir_version=IR_VERSION)


def solve_denoise(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
