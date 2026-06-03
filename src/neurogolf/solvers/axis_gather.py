"""Static axis-gather solver.

Some tasks rearrange / duplicate whole rows (or whole columns) of the input by
a fixed permutation that is identical across every example — e.g. stacking the
input under its own vertical mirror (`[flip_v(input); input]`, task 116), or any
constant row/column shuffle that preserves the other axis.

When the input and output shapes are constant across all examples and each
output row equals some input row (width unchanged), the whole transform is a
single `Gather` along the height axis with a baked index vector. The column
analog gathers along the width axis. Both are essentially free (one op, a
30-element index initializer), so they score very high.

Output rows beyond the real output height point at a guaranteed-zero padding
row of the input, so the padded region stays all-zero.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from ..grids import CHANNELS, HEIGHT, WIDTH, all_examples

OPSET = 11
IR_VERSION = 8


def _examples(task: dict):
    out = []
    for ex in all_examples(task):
        i, o = ex["input"], ex["output"]
        if not i or not i[0] or not o or not o[0]:
            return None
        if len(i) > HEIGHT or len(i[0]) > WIDTH:
            return None
        out.append((np.array(i), np.array(o)))
    return out or None


def _find_map(exs, axis: int):
    """Return a fixed index map along `axis` (0=rows, 1=cols) such that every
    output line equals input line `map[k]`, consistent across all examples, or
    None. The orthogonal dimension must be preserved."""
    i0, o0 = exs[0]
    ih, iw = i0.shape
    oh, ow = o0.shape
    if axis == 0:
        if iw != ow or oh > HEIGHT or ih >= HEIGHT:
            return None
        in_len, out_len = ih, oh
        get_in = lambda g, k: g[k, :]
        get_out = lambda g, k: g[k, :]
    else:
        if ih != oh or ow > WIDTH or iw >= WIDTH:
            return None
        in_len, out_len = iw, ow
        get_in = lambda g, k: g[:, k]
        get_out = lambda g, k: g[:, k]

    idx = []
    for k in range(out_len):
        target = get_out(o0, k)
        found = None
        for s in range(in_len):
            if np.array_equal(target, get_in(i0, s)):
                found = s
                break
        if found is None:
            return None
        idx.append(found)
    if idx == list(range(out_len)):
        return None  # identity along this axis — leave to cheaper solvers

    for i, o in exs:
        if i.shape != i0.shape or o.shape != o0.shape:
            return None
        for k in range(out_len):
            if not np.array_equal(get_out(o, k), get_in(i, idx[k])):
                return None
    return idx, in_len, out_len


def _build(idx: list[int], in_len: int, out_len: int, axis: int) -> onnx.ModelProto:
    dim = HEIGHT if axis == 0 else WIDTH
    # rows/cols beyond the real output point at the first all-zero padding line
    full = list(idx) + [in_len] * (dim - out_len)
    onnx_axis = 2 if axis == 0 else 3
    init = [numpy_helper.from_array(np.array(full, dtype=np.int64), "gidx")]
    node = helper.make_node(
        "Gather", ["input", "gidx"], ["output"], axis=onnx_axis,
        name="axis_gather")
    inputs = [helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, CHANNELS, HEIGHT, WIDTH])]
    outputs = [helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, CHANNELS, HEIGHT, WIDTH])]
    graph = helper.make_graph([node], "axis_gather", inputs, outputs,
                              initializer=init)
    return helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", OPSET)],
        ir_version=IR_VERSION)


def solve_axis_gather(task: dict) -> Optional[onnx.ModelProto]:
    exs = _examples(task)
    if exs is None:
        return None
    for axis in (0, 1):
        res = _find_map(exs, axis)
        if res is not None:
            idx, in_len, out_len = res
            return _build(idx, in_len, out_len, axis)
    return None
