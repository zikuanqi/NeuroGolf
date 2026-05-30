"""Solver: fill each colour's bounding box solidly with that colour.

Every colour's scattered cells are replaced by a solid rectangle spanning that
colour's bounding box (task 132). Bounding boxes of different colours do not
overlap.

Per colour channel k the bounding box is the outer product of its row span and
its column span, each found with the prefix/suffix CumSum trick used by
`connect_dots`:

    row_has = any k in the row ;  rb = (cumsum(row_has) > 0) AND (reverse > 0)
    col_has = any k in the col ;  cb = (cumsum(col_has) > 0) AND (reverse > 0)
    box_k   = rb (outer) cb

Channels are processed together; the background channel is rebuilt inside the
grid so the output stays a valid one-hot.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from ..grids import CHANNELS, HEIGHT, WIDTH, all_examples

OPSET = 11
IR_VERSION = 8
NC = CHANNELS - 1


def _fill(grid):
    g = np.array(grid); H, W = g.shape
    out = np.zeros((H, W), int)
    for k in range(1, 10):
        m = (g == k).astype(int)
        if m.sum() == 0:
            continue
        rh = (m.sum(axis=1) > 0).astype(int)
        ch = (m.sum(axis=0) > 0).astype(int)
        rb = (np.cumsum(rh) > 0) & (np.cumsum(rh[::-1])[::-1] > 0)
        cb = (np.cumsum(ch) > 0) & (np.cumsum(ch[::-1])[::-1] > 0)
        out[np.outer(rb, cb)] = k
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
        if _fill(inp) != out:
            return False
        if inp != out:
            changed = True
    return changed


def _build() -> onnx.ModelProto:
    def i64(name, vals):
        return numpy_helper.from_array(np.array(vals, dtype=np.int64), name)

    def f32(name, arr):
        return numpy_helper.from_array(arr.astype(np.float32), name)

    rev_h = list(range(HEIGHT - 1, -1, -1))
    rev_w = list(range(WIDTH - 1, -1, -1))
    init = [
        i64("c1", [1]), i64("c10", [CHANNELS]), i64("ax1", [1]), i64("st1", [1]),
        i64("axis_h", [2]), i64("axis_w", [3]),
        i64("rev_h", rev_h), i64("rev_w", rev_w),
        f32("zero", np.array([0.0])), f32("one_f", np.array([1.0])),
    ]

    nodes = [
        helper.make_node("Slice", ["input", "c1", "c10", "ax1", "st1"],
                         ["colors"]),                          # [1,9,H,W]
        # row presence per colour: max over width -> [1,9,H,1]
        helper.make_node("ReduceMax", ["colors"], ["rowhas"], axes=[3],
                         keepdims=1),
        # col presence per colour: max over height -> [1,9,1,W]
        helper.make_node("ReduceMax", ["colors"], ["colhas"], axes=[2],
                         keepdims=1),
        # row between (first..last) via prefix/suffix cumsum along H
        helper.make_node("CumSum", ["rowhas", "axis_h"], ["rpre"]),
        helper.make_node("Gather", ["rowhas", "rev_h"], ["rrev"], axis=2),
        helper.make_node("CumSum", ["rrev", "axis_h"], ["rrev_cs"]),
        helper.make_node("Gather", ["rrev_cs", "rev_h"], ["rsuf"], axis=2),
        helper.make_node("Greater", ["rpre", "zero"], ["rpre_b"]),
        helper.make_node("Greater", ["rsuf", "zero"], ["rsuf_b"]),
        helper.make_node("And", ["rpre_b", "rsuf_b"], ["rb_b"]),
        helper.make_node("Cast", ["rb_b"], ["rb"], to=TensorProto.FLOAT),  # [1,9,H,1]
        # col between via prefix/suffix cumsum along W
        helper.make_node("CumSum", ["colhas", "axis_w"], ["cpre"]),
        helper.make_node("Gather", ["colhas", "rev_w"], ["crev"], axis=3),
        helper.make_node("CumSum", ["crev", "axis_w"], ["crev_cs"]),
        helper.make_node("Gather", ["crev_cs", "rev_w"], ["csuf"], axis=3),
        helper.make_node("Greater", ["cpre", "zero"], ["cpre_b"]),
        helper.make_node("Greater", ["csuf", "zero"], ["csuf_b"]),
        helper.make_node("And", ["cpre_b", "csuf_b"], ["cb_b"]),
        helper.make_node("Cast", ["cb_b"], ["cb"], to=TensorProto.FLOAT),  # [1,9,1,W]
        # box = rb * cb (broadcast outer product per channel) -> [1,9,H,W]
        helper.make_node("Mul", ["rb", "cb"], ["box"]),
        # rebuild background inside the grid
        helper.make_node("ReduceSum", ["input"], ["content"], axes=[1],
                         keepdims=1),
        helper.make_node("Greater", ["content", "zero"], ["in_grid_b"]),
        helper.make_node("Cast", ["in_grid_b"], ["in_grid"],
                         to=TensorProto.FLOAT),
        helper.make_node("ReduceSum", ["box"], ["bsum"], axes=[1], keepdims=1),
        helper.make_node("Sub", ["one_f", "bsum"], ["not_color"]),
        helper.make_node("Mul", ["in_grid", "not_color"], ["bg"]),
        helper.make_node("Concat", ["bg", "box"], ["output"], axis=1),
    ]

    inputs = [helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, CHANNELS, HEIGHT, WIDTH])]
    outputs = [helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, CHANNELS, HEIGHT, WIDTH])]
    value_info = [
        helper.make_tensor_value_info("colors", TensorProto.FLOAT, [1, NC, HEIGHT, WIDTH]),
        helper.make_tensor_value_info("rowhas", TensorProto.FLOAT, [1, NC, HEIGHT, 1]),
        helper.make_tensor_value_info("colhas", TensorProto.FLOAT, [1, NC, 1, WIDTH]),
        helper.make_tensor_value_info("rb", TensorProto.FLOAT, [1, NC, HEIGHT, 1]),
        helper.make_tensor_value_info("cb", TensorProto.FLOAT, [1, NC, 1, WIDTH]),
        helper.make_tensor_value_info("box", TensorProto.FLOAT, [1, NC, HEIGHT, WIDTH]),
        helper.make_tensor_value_info("bg", TensorProto.FLOAT, [1, 1, HEIGHT, WIDTH]),
    ]
    graph = helper.make_graph(nodes, "color_bbox_fill", inputs, outputs,
                              initializer=init, value_info=value_info)
    return helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
        ir_version=IR_VERSION)


def solve_color_bbox_fill(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
