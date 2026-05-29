"""Palindrome (mirror-concat) solvers.

`palindrome_h`: output is `[input | flip_h(input)]`. The right half is the
input mirrored about its right edge. Output shape is H x 2W.

`palindrome_v`: same idea along rows — output is `[input ; flip_v(input)]`,
2H x W.

`palindrome_2d`: output is the four-quadrant mirror of input —
`[[input,        flip_h(input)],
  [flip_v(input), rot180(input)]]`. Output shape is 2H x 2W. We implement it
by chaining the two single-axis palindromes through an intermediate tensor.

For a given axis the output position c maps to input column:

  c                  if c < W   (first tile)
  2W - 1 - c         if W <= c < 2W   (mirrored tile)
  anything           if c >= 2W   (masked to zero below)

We use a `Where` to select between the two index expressions, then a `Less`
mask zeros out cells past 2W.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from ..grids import CHANNELS, HEIGHT, WIDTH, all_examples

OPSET = 11
IR_VERSION = 8


def _is_palindrome_h(task: dict) -> bool:
    saw = False
    for ex in all_examples(task):
        i, o = ex["input"], ex["output"]
        if not i or not o:
            return False
        ih, iw = len(i), len(i[0])
        if len(o) != ih or len(o[0]) != 2 * iw:
            return False
        if 2 * iw > WIDTH:
            return False
        for r in range(ih):
            for c in range(iw):
                if o[r][c] != i[r][c] or o[r][iw + c] != i[r][iw - 1 - c]:
                    return False
        saw = True
    return saw


def _is_palindrome_v(task: dict) -> bool:
    saw = False
    for ex in all_examples(task):
        i, o = ex["input"], ex["output"]
        if not i or not o:
            return False
        ih, iw = len(i), len(i[0])
        if len(o) != 2 * ih or len(o[0]) != iw:
            return False
        if 2 * ih > HEIGHT:
            return False
        for r in range(ih):
            for c in range(iw):
                if o[r][c] != i[r][c] or o[ih + r][c] != i[ih - 1 - r][c]:
                    return False
        saw = True
    return saw


def _is_palindrome_2d(task: dict) -> bool:
    saw = False
    for ex in all_examples(task):
        i, o = ex["input"], ex["output"]
        if not i or not o:
            return False
        ih, iw = len(i), len(i[0])
        if len(o) != 2 * ih or len(o[0]) != 2 * iw:
            return False
        if 2 * ih > HEIGHT or 2 * iw > WIDTH:
            return False
        for r in range(ih):
            for c in range(iw):
                if (o[r][c] != i[r][c] or
                        o[r][iw + c] != i[r][iw - 1 - c] or
                        o[ih + r][c] != i[ih - 1 - r][c] or
                        o[ih + r][iw + c] != i[ih - 1 - r][iw - 1 - c]):
                    return False
        saw = True
    return saw


def _i64(name: str, vals) -> onnx.TensorProto:
    return numpy_helper.from_array(np.array(vals, dtype=np.int64), name)


def _f32(name: str, vals) -> onnx.TensorProto:
    return numpy_helper.from_array(np.array(vals, dtype=np.float32), name)


def _emit_palindrome(axis: int, suffix: str, in_name: str, out_name: str,
                     init: list, value_info: list) -> list:
    """Append the nodes / initializers / value_info that turn `in_name` into
    its single-axis palindrome along `axis` (2 = vertical, 3 = horizontal).

    Multiple invocations share globally named scalar constants (`one_f`,
    `two_f`, `one_i`) deduped by `_ensure_shared_inits`.
    """
    other = 3 if axis == 2 else 2
    side = HEIGHT if axis == 2 else WIDTH
    arange_shape4 = [1, 1, 1, 1]
    arange_shape4[axis] = side

    init += [
        _i64(f"arange_i_{suffix}", np.arange(side)),
        _f32(f"arange_f_{suffix}", np.arange(side)),
        _i64(f"shape4_{suffix}", arange_shape4),
        _i64(f"flat_{suffix}", [side]),
    ]

    proj_shape = [1, 1, HEIGHT, WIDTH]
    proj_shape[other] = 1
    arange_shape = arange_shape4

    value_info += [
        helper.make_tensor_value_info(
            f"chan_sum_{suffix}", TensorProto.FLOAT,
            [1, 1, HEIGHT, WIDTH]),
        helper.make_tensor_value_info(
            f"proj_{suffix}", TensorProto.FLOAT, proj_shape),
        helper.make_tensor_value_info(
            f"arange4f_{suffix}", TensorProto.FLOAT, arange_shape),
        helper.make_tensor_value_info(
            f"arange4i_{suffix}", TensorProto.INT64, arange_shape),
        helper.make_tensor_value_info(
            f"pos_{suffix}", TensorProto.FLOAT, proj_shape),
        helper.make_tensor_value_info(
            f"wm1f_{suffix}", TensorProto.FLOAT, [1, 1, 1, 1]),
        helper.make_tensor_value_info(
            f"wf_{suffix}", TensorProto.FLOAT, [1, 1, 1, 1]),
        helper.make_tensor_value_info(
            f"wi_{suffix}", TensorProto.INT64, [1, 1, 1, 1]),
        helper.make_tensor_value_info(
            f"twowf_{suffix}", TensorProto.FLOAT, [1, 1, 1, 1]),
        helper.make_tensor_value_info(
            f"twowi_pre_{suffix}", TensorProto.INT64, [1, 1, 1, 1]),
        helper.make_tensor_value_info(
            f"twowm1i_{suffix}", TensorProto.INT64, [1, 1, 1, 1]),
        helper.make_tensor_value_info(
            f"below_{suffix}", TensorProto.BOOL, arange_shape),
        helper.make_tensor_value_info(
            f"mirror_{suffix}", TensorProto.INT64, arange_shape),
        helper.make_tensor_value_info(
            f"idx_{suffix}", TensorProto.INT64, arange_shape),
        helper.make_tensor_value_info(
            f"idx1d_{suffix}", TensorProto.INT64, [side]),
        helper.make_tensor_value_info(
            f"gathered_{suffix}", TensorProto.FLOAT,
            [1, CHANNELS, HEIGHT, WIDTH]),
        helper.make_tensor_value_info(
            f"inrange_b_{suffix}", TensorProto.BOOL, arange_shape),
        helper.make_tensor_value_info(
            f"inrange_f_{suffix}", TensorProto.FLOAT, arange_shape),
    ]

    return [
        helper.make_node(
            "ReduceSum", [in_name], [f"chan_sum_{suffix}"],
            axes=[1], keepdims=1, name=f"chan_sum_{suffix}_op"),
        helper.make_node(
            "ReduceMax", [f"chan_sum_{suffix}"], [f"proj_{suffix}"],
            axes=[other], keepdims=1, name=f"proj_{suffix}_op"),
        helper.make_node(
            "Reshape", [f"arange_f_{suffix}", f"shape4_{suffix}"],
            [f"arange4f_{suffix}"], name=f"resh_arange_f_{suffix}"),
        helper.make_node(
            "Reshape", [f"arange_i_{suffix}", f"shape4_{suffix}"],
            [f"arange4i_{suffix}"], name=f"resh_arange_i_{suffix}"),
        helper.make_node(
            "Mul", [f"proj_{suffix}", f"arange4f_{suffix}"],
            [f"pos_{suffix}"], name=f"pos_{suffix}_op"),
        helper.make_node(
            "ReduceMax", [f"pos_{suffix}"], [f"wm1f_{suffix}"],
            axes=[axis], keepdims=1, name=f"wm1f_{suffix}_op"),
        helper.make_node(
            "Add", [f"wm1f_{suffix}", "one_f"], [f"wf_{suffix}"],
            name=f"wf_{suffix}_op"),
        helper.make_node(
            "Cast", [f"wf_{suffix}"], [f"wi_{suffix}"],
            to=TensorProto.INT64, name=f"wi_{suffix}_op"),
        helper.make_node(
            "Mul", [f"wf_{suffix}", "two_f"], [f"twowf_{suffix}"],
            name=f"twowf_{suffix}_op"),
        helper.make_node(
            "Add", [f"wi_{suffix}", f"wi_{suffix}"],
            [f"twowi_pre_{suffix}"], name=f"twowi_pre_{suffix}_op"),
        helper.make_node(
            "Sub", [f"twowi_pre_{suffix}", "one_i"], [f"twowm1i_{suffix}"],
            name=f"twowm1i_{suffix}_op"),
        helper.make_node(
            "Less", [f"arange4f_{suffix}", f"wf_{suffix}"],
            [f"below_{suffix}"], name=f"below_{suffix}_op"),
        helper.make_node(
            "Sub", [f"twowm1i_{suffix}", f"arange4i_{suffix}"],
            [f"mirror_{suffix}"], name=f"mirror_{suffix}_op"),
        helper.make_node(
            "Where",
            [f"below_{suffix}", f"arange4i_{suffix}", f"mirror_{suffix}"],
            [f"idx_{suffix}"], name=f"where_{suffix}_op"),
        helper.make_node(
            "Reshape", [f"idx_{suffix}", f"flat_{suffix}"],
            [f"idx1d_{suffix}"], name=f"flat_idx_{suffix}"),
        helper.make_node(
            "Gather", [in_name, f"idx1d_{suffix}"],
            [f"gathered_{suffix}"], axis=axis,
            name=f"gather_{suffix}_op"),
        helper.make_node(
            "Less", [f"arange4f_{suffix}", f"twowf_{suffix}"],
            [f"inrange_b_{suffix}"], name=f"inrange_b_{suffix}_op"),
        helper.make_node(
            "Cast", [f"inrange_b_{suffix}"], [f"inrange_f_{suffix}"],
            to=TensorProto.FLOAT, name=f"inrange_f_{suffix}_op"),
        helper.make_node(
            "Mul", [f"gathered_{suffix}", f"inrange_f_{suffix}"],
            [out_name], name=f"out_{suffix}_op"),
    ]


def _shared_inits() -> list:
    return [
        _f32("one_f", 1.0),
        _f32("two_f", 2.0),
        _i64("one_i", [1]),
    ]


def _build_single(axis: int) -> onnx.ModelProto:
    init = _shared_inits()
    value_info: list = []
    nodes = _emit_palindrome(axis, "h" if axis == 3 else "v",
                              "input", "output", init, value_info)
    inputs = [helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, CHANNELS, HEIGHT, WIDTH])]
    outputs = [helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, CHANNELS, HEIGHT, WIDTH])]
    graph = helper.make_graph(
        nodes, "palindrome", inputs, outputs,
        initializer=init, value_info=value_info)
    return helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
        ir_version=IR_VERSION)


def _build_2d() -> onnx.ModelProto:
    init = _shared_inits()
    value_info: list = []
    nodes = _emit_palindrome(3, "h", "input", "_mid", init, value_info)
    nodes += _emit_palindrome(2, "v", "_mid", "output", init, value_info)
    value_info.append(helper.make_tensor_value_info(
        "_mid", TensorProto.FLOAT, [1, CHANNELS, HEIGHT, WIDTH]))

    inputs = [helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, CHANNELS, HEIGHT, WIDTH])]
    outputs = [helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, CHANNELS, HEIGHT, WIDTH])]
    graph = helper.make_graph(
        nodes, "palindrome_2d", inputs, outputs,
        initializer=init, value_info=value_info)
    return helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
        ir_version=IR_VERSION)


def solve_palindrome_h(task: dict) -> Optional[onnx.ModelProto]:
    return _build_single(3) if _is_palindrome_h(task) else None


def solve_palindrome_v(task: dict) -> Optional[onnx.ModelProto]:
    return _build_single(2) if _is_palindrome_v(task) else None


def solve_palindrome_2d(task: dict) -> Optional[onnx.ModelProto]:
    return _build_2d() if _is_palindrome_2d(task) else None
