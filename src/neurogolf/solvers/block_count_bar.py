"""Solver: count 2x2 solid blocks of a colour -> fixed-size unary bar.

The grid contains several solid 2x2 squares of a marker colour C (amongst
other noise); the output is a constant-size 1xK row (task 38: 1x5) whose first
N cells are colour D and whose remaining K-N cells are background, where N is
the number of 2x2 C-blocks.

Unlike `count_bar`, the output keeps its trailing background cells (the bar is
a *fixed* width), so channel 0 must be set on the unfilled part of the bar.

Counting is a 2x2 all-ones `Conv` over channel C: a window summing to 4 is a
solid block. `ReduceSum` of the >3.5 mask gives N; a `Less(ramp, N)` mask
paints the bar.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from ..grids import CHANNELS, HEIGHT, WIDTH, all_examples

OPSET = 11
IR_VERSION = 8


def _count_blocks(grid, color) -> int:
    g = np.array(grid)
    if g.shape[0] < 2 or g.shape[1] < 2:
        return 0
    m = (g == color).astype(np.int32)
    s = m[:-1, :-1] + m[:-1, 1:] + m[1:, :-1] + m[1:, 1:]
    return int((s == 4).sum())


def _detect(task: dict):
    """Return (C, D, axis, K) or None. axis 0 = 1xK row, 1 = Kx1 column."""
    examples = list(all_examples(task))
    if not examples:
        return None
    out_shapes = {(len(ex["output"]), len(ex["output"][0]))
                  for ex in examples if ex["output"] and ex["output"][0]}
    if len(out_shapes) != 1:
        return None
    oh, ow = next(iter(out_shapes))
    if oh == 1:
        axis, K = 0, ow
    elif ow == 1:
        axis, K = 1, oh
    else:
        return None
    if K < 1 or K > max(HEIGHT, WIDTH):
        return None

    # output colour D (single non-bg colour across all bars)
    out_colors = set()
    for ex in examples:
        out_colors |= {v for row in ex["output"] for v in row if v != 0}
    if len(out_colors) != 1:
        return None
    D = out_colors.pop()

    for C in range(1, CHANNELS):
        counts = set()
        ok = True
        for ex in examples:
            inp, out = ex["input"], ex["output"]
            if len(inp) > HEIGHT or len(inp[0]) > WIDTH:
                ok = False
                break
            n = _count_blocks(inp, C)
            if n > K:
                ok = False
                break
            if axis == 0:
                exp = [[D] * n + [0] * (K - n)]
            else:
                exp = [[D]] * n + [[0]] * (K - n)
            if exp != out:
                ok = False
                break
            counts.add(n)
        if ok and len(counts) >= 2:
            return C, D, axis, K
    return None


def _build(C: int, D: int, axis: int, K: int) -> onnx.ModelProto:
    F = TensorProto.FLOAT
    B = TensorProto.BOOL
    n = helper.make_node

    # 2x2 all-ones conv kernel over the single channel C
    kernel = np.ones((1, 1, 2, 2), dtype=np.float32)
    side = WIDTH if axis == 0 else HEIGHT
    ramp = np.arange(side, dtype=np.float32)
    ramp = ramp.reshape(1, 1, 1, side) if axis == 0 else ramp.reshape(1, 1, side, 1)

    # bar region: the K cells of row 0 (axis 0) or column 0 (axis 1)
    region = np.zeros((1, 1, HEIGHT, WIDTH), dtype=np.float32)
    if axis == 0:
        region[0, 0, 0, :K] = 1.0
    else:
        region[0, 0, :K, 0] = 1.0

    init = [
        numpy_helper.from_array(np.array([0, C, 0, 0], dtype=np.int64), "c_s"),
        numpy_helper.from_array(np.array([1, C + 1, HEIGHT, WIDTH],
                                         dtype=np.int64), "c_e"),
        numpy_helper.from_array(np.array([0, 1, 2, 3], dtype=np.int64), "c_ax"),
        numpy_helper.from_array(kernel, "k2x2"),
        numpy_helper.from_array(np.array(3.5, dtype=np.float32), "thr"),
        numpy_helper.from_array(ramp, "ramp"),
        numpy_helper.from_array(region, "region"),
    ]

    nodes = [
        n("Slice", ["input", "c_s", "c_e", "c_ax"], ["chC"], name="slice_c"),
        n("Conv", ["chC", "k2x2"], ["bsum"], kernel_shape=[2, 2],
          pads=[0, 0, 0, 0], name="conv2x2"),
        n("Greater", ["bsum", "thr"], ["is_blk_b"], name="gt_blk"),
        n("Cast", ["is_blk_b"], ["is_blk"], to=F, name="cast_blk"),
        n("ReduceSum", ["is_blk"], ["N"], axes=[2, 3], keepdims=1, name="sumN"),
        n("Less", ["ramp", "N"], ["lt_b"], name="lt"),
        n("Cast", ["lt_b"], ["lt_f"], to=F, name="cast_lt"),
        n("Mul", ["region", "lt_f"], ["chD_plane"], name="bar_on"),
        n("Sub", ["region", "chD_plane"], ["ch0_plane"], name="bar_off"),
        n("Sub", ["region", "region"], ["zero_plane"], name="zero"),
    ]

    planes = []
    for k in range(CHANNELS):
        if k == 0:
            planes.append("ch0_plane")
        elif k == D:
            planes.append("chD_plane")
        else:
            planes.append("zero_plane")
    nodes.append(n("Concat", planes, ["output"], axis=1, name="assemble"))

    def vi(name, shape, dt=F):
        return helper.make_tensor_value_info(name, dt, shape)

    conv_h = HEIGHT - 1
    conv_w = WIDTH - 1
    s1 = [1, 1, HEIGHT, WIDTH]
    ramp_shape = [1, 1, 1, side] if axis == 0 else [1, 1, side, 1]
    value_info = [
        vi("chC", s1), vi("bsum", [1, 1, conv_h, conv_w]),
        vi("is_blk_b", [1, 1, conv_h, conv_w], B),
        vi("is_blk", [1, 1, conv_h, conv_w]),
        vi("N", [1, 1, 1, 1]),
        vi("lt_b", ramp_shape, B), vi("lt_f", ramp_shape),
        vi("chD_plane", s1), vi("ch0_plane", s1), vi("zero_plane", s1),
    ]
    inputs = [vi("input", [1, CHANNELS, HEIGHT, WIDTH])]
    outputs = [vi("output", [1, CHANNELS, HEIGHT, WIDTH])]
    graph = helper.make_graph(nodes, "block_count_bar", inputs, outputs,
                              initializer=init, value_info=value_info)
    return helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
        ir_version=IR_VERSION)


def solve_block_count_bar(task: dict) -> Optional[onnx.ModelProto]:
    params = _detect(task)
    if params is None:
        return None
    return _build(*params)
