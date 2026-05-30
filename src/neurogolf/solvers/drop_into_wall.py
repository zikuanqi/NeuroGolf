"""Solver: colour-1 cells drop into a full colour-5 wall row.

A solid horizontal wall of colour 5 spans the grid; above it sit some colour-1
cells. Each column that holds a 1 above the wall has that 1 removed and the
wall cell in that column repainted from 5 to 1 (task 73).

Channel arithmetic on the one-hot tensor:
  wall_row  : rows where channel-5 covers the whole real width
  col_has1  : columns with any channel-1 cell
  place     : wall_row x col_has1  (outer product) - where the 1s land
  channel 1 := place ;  channel 5 -= place ;  channel 0 += (old 1s - place)
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
    g = np.array(grid)
    H, W = g.shape
    wall = -1
    for r in range(H):
        if all(g[r][c] == 5 for c in range(W)):
            wall = r
    if wall < 0:
        return None
    out = g.copy()
    for c in range(W):
        if any(g[r][c] == 1 for r in range(wall)):
            for r in range(wall):
                if out[r][c] == 1:
                    out[r][c] = 0
            out[wall][c] = 1
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
        res = _transform(inp)
        if res is None or res != out:
            return False
        if inp != out:
            changed = True
    return changed


def _build() -> onnx.ModelProto:
    def i64(name, vals):
        return numpy_helper.from_array(np.array(vals, dtype=np.int64), name)

    def f32(name, arr):
        return numpy_helper.from_array(arr.astype(np.float32), name)

    init = [
        i64("idx1", [1]), i64("idx5", [5]), i64("idx0", [0]),
        f32("zero", np.array([0.0])), f32("one_f", np.array([1.0])),
        i64("c1", [1]), i64("c10", [CHANNELS]), i64("ax1", [1]), i64("st1", [1]),
        i64("eps", [0]),
    ]

    nodes = [
        helper.make_node("Gather", ["input", "idx1"], ["ch1"], axis=1),  # [1,1,H,W]
        helper.make_node("Gather", ["input", "idx5"], ["ch5"], axis=1),
        helper.make_node("Gather", ["input", "idx0"], ["ch0"], axis=1),
        # real width = max over rows of (sum of content along width)... simpler:
        # content present per column, count columns -> Wreal
        helper.make_node("ReduceSum", ["input"], ["content"], axes=[1],
                         keepdims=1),                              # [1,1,H,W]
        helper.make_node("ReduceMax", ["content"], ["col_any"], axes=[2],
                         keepdims=1),                              # [1,1,1,W]
        helper.make_node("ReduceSum", ["col_any"], ["Wreal"], axes=[3],
                         keepdims=1),                              # [1,1,1,1]
        # wall row: channel-5 row sum over width equals Wreal
        helper.make_node("ReduceSum", ["ch5"], ["ch5_rowsum"], axes=[3],
                         keepdims=1),                              # [1,1,H,1]
        helper.make_node("Sub", ["Wreal", "ch5_rowsum"], ["wdiff"]),
        # wall = (wdiff <= 0) i.e. rowsum >= Wreal  -> use Less(wdiff, 0.5)
        helper.make_node("Less", ["wdiff", "one_f"], ["wall_b"]),  # wdiff<1 => ==0
        helper.make_node("Cast", ["wall_b"], ["wall"], to=TensorProto.FLOAT),
        # column has a 1 anywhere
        helper.make_node("ReduceMax", ["ch1"], ["col_has1"], axes=[2],
                         keepdims=1),                              # [1,1,1,W]
        # place = wall (H) x col_has1 (W) outer product, broadcast
        helper.make_node("Mul", ["wall", "col_has1"], ["place"]),  # [1,1,H,W]
        # removed = ch1 cells that move away = ch1 AND not place
        helper.make_node("Sub", ["ch1", "place"], ["removed_raw"]),
        helper.make_node("Relu", ["removed_raw"], ["removed"]),
        # new channels
        helper.make_node("Add", ["ch0", "removed"], ["ch0_new"]),
        helper.make_node("Sub", ["ch5", "place"], ["ch5_new_raw"]),
        helper.make_node("Relu", ["ch5_new_raw"], ["ch5_new"]),
        # ch1_new = place
        # reassemble: [ch0_new][place(=ch1)][ch2..4][ch5_new][ch6..9]
        helper.make_node("Slice", ["input", "i2", "i5", "ax1", "st1"], ["ch234"]),
        helper.make_node("Slice", ["input", "i6", "i10", "ax1", "st1"], ["ch6789"]),
        helper.make_node("Concat",
                         ["ch0_new", "place", "ch234", "ch5_new", "ch6789"],
                         ["output"], axis=1),
    ]
    init += [i64("i2", [2]), i64("i5", [5]), i64("i6", [6]), i64("i10", [CHANNELS])]

    inputs = [helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, CHANNELS, HEIGHT, WIDTH])]
    outputs = [helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, CHANNELS, HEIGHT, WIDTH])]
    value_info = [
        helper.make_tensor_value_info(n, TensorProto.FLOAT, [1, 1, HEIGHT, WIDTH])
        for n in ["ch1", "ch5", "ch0", "content", "place", "removed_raw",
                  "removed", "ch0_new", "ch5_new_raw", "ch5_new"]
    ] + [
        helper.make_tensor_value_info("col_any", TensorProto.FLOAT, [1, 1, 1, WIDTH]),
        helper.make_tensor_value_info("Wreal", TensorProto.FLOAT, [1, 1, 1, 1]),
        helper.make_tensor_value_info("ch5_rowsum", TensorProto.FLOAT, [1, 1, HEIGHT, 1]),
        helper.make_tensor_value_info("wdiff", TensorProto.FLOAT, [1, 1, HEIGHT, 1]),
        helper.make_tensor_value_info("wall", TensorProto.FLOAT, [1, 1, HEIGHT, 1]),
        helper.make_tensor_value_info("col_has1", TensorProto.FLOAT, [1, 1, 1, WIDTH]),
        helper.make_tensor_value_info("ch234", TensorProto.FLOAT, [1, 3, HEIGHT, WIDTH]),
        helper.make_tensor_value_info("ch6789", TensorProto.FLOAT, [1, 4, HEIGHT, WIDTH]),
    ]
    graph = helper.make_graph(nodes, "drop_into_wall", inputs, outputs,
                              initializer=init, value_info=value_info)
    return helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
        ir_version=IR_VERSION)


def solve_drop_into_wall(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
