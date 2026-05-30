"""Solver: gravity up - each column's cells rise to the top edge.

Every non-background cell slides straight up, preserving the relative order of
the cells within its column (a stable rise, unlike per-channel sinking). This
handles columns that mix several colours (task 078).

The whole column is reordered by a per-cell sort key, then every channel is
permuted by the *same* indices so the result stays a valid one-hot tensor:

    key = row                      for non-background cells   (rise, in order)
        = H + row                  for in-grid background     (fill below them)
        = 2H + row                 for the all-zero padding   (stay outside)

Because each cell's key is unique (row is part of every branch), a `TopK`
ascending sort along the row axis is fully determined - no stable-sort caveat.
The sort indices are tiled across the 10 channels and applied with
`GatherElements`.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from ..grids import CHANNELS, HEIGHT, WIDTH, all_examples

OPSET = 11
IR_VERSION = 8


def _detect(task: dict) -> bool:
    examples = list(all_examples(task))
    if not examples:
        return False
    for ex in examples:
        inp, out = ex["input"], ex["output"]
        h, w = len(inp), len(inp[0])
        if len(out) != h or len(out[0]) != w:
            return False
        for c in range(w):
            vals = [inp[r][c] for r in range(h) if inp[r][c] != 0]
            col = vals + [0] * (h - len(vals))
            for r in range(h):
                if out[r][c] != col[r]:
                    return False
    return True


def _build() -> onnx.ModelProto:
    def i64(name, data):
        return numpy_helper.from_array(np.array(data, dtype=np.int64), name)

    def f32(name, data):
        return numpy_helper.from_array(np.array(data, dtype=np.float32), name)

    rows = np.arange(HEIGHT, dtype=np.float32).reshape(1, 1, HEIGHT, 1)
    init = [
        f32("rows", rows),
        f32("zero", [0.0]),
        f32("H_f", [float(HEIGHT)]),
        f32("twoH_f", [2.0 * HEIGHT]),
        i64("k_rows", [HEIGHT]),
        i64("tile_ch", [1, CHANNELS, 1, 1]),
        i64("c1", [1]),
        i64("c10", [CHANNELS]),
        i64("ax1", [1]),
        i64("st1", [1]),
    ]

    nodes = [
        # content mask: 1 inside the real grid, 0 in the all-zero padding
        helper.make_node("ReduceSum", ["input"], ["content"], axes=[1],
                         keepdims=1),
        # non-background mask: sum of colour channels 1..9
        helper.make_node("Slice", ["input", "c1", "c10", "ax1", "st1"],
                         ["colors"]),
        helper.make_node("ReduceSum", ["colors"], ["nonbg"], axes=[1],
                         keepdims=1),
        helper.make_node("Greater", ["nonbg", "zero"], ["nonbg_b"]),
        helper.make_node("Cast", ["nonbg_b"], ["nonbg_f"], to=TensorProto.FLOAT),
        helper.make_node("Greater", ["content", "zero"], ["content_b"]),
        helper.make_node("Cast", ["content_b"], ["content_f"],
                         to=TensorProto.FLOAT),
        # key = 2H + row - content*H - nonbg*H
        #   non-bg cell  -> row;  in-grid bg -> H+row;  padding -> 2H+row
        helper.make_node("Mul", ["content_f", "H_f"], ["content_H"]),
        helper.make_node("Mul", ["nonbg_f", "H_f"], ["nonbg_H"]),
        helper.make_node("Add", ["rows", "twoH_f"], ["base"]),
        helper.make_node("Sub", ["base", "content_H"], ["key1"]),
        helper.make_node("Sub", ["key1", "nonbg_H"], ["key"]),
        # ascending sort of each column -> reorder indices [1,1,H,W]
        helper.make_node("TopK", ["key", "k_rows"], ["sorted_key", "order"],
                         axis=2, largest=0, sorted=1),
        # same indices for every channel -> valid one-hot after the gather
        helper.make_node("Tile", ["order", "tile_ch"], ["order_ch"]),
        helper.make_node("GatherElements", ["input", "order_ch"], ["output"],
                         axis=2),
    ]

    inputs = [helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, CHANNELS, HEIGHT, WIDTH])]
    outputs = [helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, CHANNELS, HEIGHT, WIDTH])]
    value_info = [
        helper.make_tensor_value_info("content", TensorProto.FLOAT,
                                      [1, 1, HEIGHT, WIDTH]),
        helper.make_tensor_value_info("colors", TensorProto.FLOAT,
                                      [1, CHANNELS - 1, HEIGHT, WIDTH]),
        helper.make_tensor_value_info("nonbg", TensorProto.FLOAT,
                                      [1, 1, HEIGHT, WIDTH]),
        helper.make_tensor_value_info("key", TensorProto.FLOAT,
                                      [1, 1, HEIGHT, WIDTH]),
        helper.make_tensor_value_info("sorted_key", TensorProto.FLOAT,
                                      [1, 1, HEIGHT, WIDTH]),
        helper.make_tensor_value_info("order", TensorProto.INT64,
                                      [1, 1, HEIGHT, WIDTH]),
        helper.make_tensor_value_info("order_ch", TensorProto.INT64,
                                      [1, CHANNELS, HEIGHT, WIDTH]),
    ]
    graph = helper.make_graph(nodes, "gravity_up", inputs, outputs,
                              initializer=init, value_info=value_info)
    return helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
        ir_version=IR_VERSION)


def solve_gravity_up(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
