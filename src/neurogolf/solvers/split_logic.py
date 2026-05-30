"""Solver: split the grid into two halves and combine them with a boolean op.

The input is two equal sub-grids - either touching (left|right or top|bottom)
or separated by a single divider row/column - and the output is one half-sized
grid whose on-cells come from a boolean combination of the two halves'
occupancy masks, painted in a single colour:

    out = OP(left != 0, right != 0)  ->  fill colour C where true, else 0

OP and the split direction are detected per task from {and, or, xor, nor,
nand} x {left-right, top-bottom, with/without a 1-cell divider}. This
generalises `split_and` (tasks 26, 72, 144, 227, 236, 318, 347, 386, 395).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from ..grids import CHANNELS, HEIGHT, WIDTH, all_examples

OPSET = 11
IR_VERSION = 8

# direction -> (axis, has_divider): axis 2 = vertical split (top/bottom),
# axis 3 = horizontal split (left/right)
_DIRS = {
    "LR": (3, False), "TB": (2, False),
    "LRsep": (3, True), "TBsep": (2, True),
}
_OPS = ("and", "or", "xor", "nor", "nand")


def _halves(g: np.ndarray, direction: str):
    h, w = g.shape
    if direction == "LR":
        return g[:, :w // 2], g[:, w - w // 2:]
    if direction == "TB":
        return g[:h // 2], g[h - h // 2:]
    if direction == "LRsep":
        return g[:, :w // 2], g[:, w // 2 + 1:]
    if direction == "TBsep":
        return g[:h // 2], g[h // 2 + 1:]
    raise ValueError(direction)


def _apply(am: np.ndarray, bm: np.ndarray, op: str) -> np.ndarray:
    if op == "and":
        return am & bm
    if op == "or":
        return am | bm
    if op == "xor":
        return am ^ bm
    if op == "nor":
        return ~(am | bm)
    if op == "nand":
        return ~(am & bm)
    raise ValueError(op)


def _detect(task: dict):
    """Return (direction, op, fill, oh, ow) or None."""
    examples = list(all_examples(task))
    if not examples:
        return None

    for direction in _DIRS:
        for op in _OPS:
            fill = None
            ok = True
            shape = None
            for ex in examples:
                g = np.array(ex["input"])
                o = np.array(ex["output"])
                h, w = g.shape
                # geometry must match the split
                if direction in ("LR", "LRsep") and h != o.shape[0]:
                    ok = False
                    break
                if direction in ("TB", "TBsep") and w != o.shape[1]:
                    ok = False
                    break
                if direction == "LR" and w % 2 != 0:
                    ok = False
                    break
                if direction == "TB" and h % 2 != 0:
                    ok = False
                    break
                if direction == "LRsep" and w % 2 != 1:
                    ok = False
                    break
                if direction == "TBsep" and h % 2 != 1:
                    ok = False
                    break
                a, b = _halves(g, direction)
                if a.shape != o.shape or b.shape != o.shape:
                    ok = False
                    break
                colours = sorted({int(v) for v in o[o != 0]})
                if len(colours) > 1:
                    ok = False
                    break
                cur_fill = colours[0] if colours else 0
                m = _apply(a != 0, b != 0, op)
                # nor/nand can fire on an empty output; require a real colour
                this_fill = cur_fill if cur_fill else (fill or 0)
                if not np.array_equal(np.where(m, this_fill, 0), o):
                    ok = False
                    break
                if cur_fill:
                    if fill is None:
                        fill = cur_fill
                    elif fill != cur_fill:
                        ok = False
                        break
                shape = o.shape
            if ok and fill:
                return direction, op, fill, shape[0], shape[1]
    return None


def _build(direction: str, op: str, fill: int, oh: int, ow: int):
    axis, has_div = _DIRS[direction]

    def i64(name, vals):
        return numpy_helper.from_array(np.array(vals, dtype=np.int64), name)

    def f32(name, arr):
        return numpy_helper.from_array(arr.astype(np.float32), name)

    # slice geometry for the two halves in the full 30x30 canvas
    if axis == 3:                       # horizontal split -> two W-halves
        a_lo, a_hi = 0, ow
        b_lo = ow + (1 if has_div else 0)
        b_hi = b_lo + ow
        a_starts, a_ends = [0, 0, 0, a_lo], [1, CHANNELS, oh, a_hi]
        b_starts, b_ends = [0, 0, 0, b_lo], [1, CHANNELS, oh, b_hi]
        pad_to = [0, 0, 0, 0, 0, 0, HEIGHT - oh, WIDTH - ow]
    else:                               # vertical split -> two H-halves
        a_lo, a_hi = 0, oh
        b_lo = oh + (1 if has_div else 0)
        b_hi = b_lo + oh
        a_starts, a_ends = [0, 0, a_lo, 0], [1, CHANNELS, a_hi, ow]
        b_starts, b_ends = [0, 0, b_lo, 0], [1, CHANNELS, b_hi, ow]
        pad_to = [0, 0, 0, 0, 0, 0, HEIGHT - oh, WIDTH - ow]

    init = [
        i64("a_starts", a_starts), i64("a_ends", a_ends),
        i64("b_starts", b_starts), i64("b_ends", b_ends),
        i64("axes_all", [0, 1, 2, 3]),
        f32("zero", np.array([0.0])), f32("one_f", np.array([1.0])),
        f32("two_f", np.array([2.0])), f32("fill_f", np.array([float(fill)])),
        i64("pads", pad_to),
    ]

    # occupancy = 1 - background-channel value (channel 0 is background)
    init += [i64("ch0_s", [0, 0, 0, 0]),
             i64("a_bg_e", ([1, 1, oh, ow])),
             i64("ch_ax", [0, 1, 2, 3])]
    nodes = [
        helper.make_node("Slice", ["input", "a_starts", "a_ends", "axes_all"],
                         ["a_oh"]),
        helper.make_node("Slice", ["input", "b_starts", "b_ends", "axes_all"],
                         ["b_oh"]),
        helper.make_node("Slice", ["a_oh", "ch0_s", "a_bg_e", "ch_ax"],
                         ["a_bg_ch"]),
        helper.make_node("Slice", ["b_oh", "ch0_s", "a_bg_e", "ch_ax"],
                         ["b_bg_ch"]),
        helper.make_node("Sub", ["one_f", "a_bg_ch"], ["am"]),   # [1,1,oh,ow]
        helper.make_node("Sub", ["one_f", "b_bg_ch"], ["bm"]),
    ]

    # boolean op in arithmetic form on 0/1 masks
    if op == "and":
        nodes.append(helper.make_node("Mul", ["am", "bm"], ["res"]))
    elif op == "or":
        nodes.append(helper.make_node("Max", ["am", "bm"], ["res"]))
    elif op == "nand":
        nodes += [helper.make_node("Mul", ["am", "bm"], ["a_and_b"]),
                  helper.make_node("Sub", ["one_f", "a_and_b"], ["res"])]
    elif op == "nor":
        nodes += [helper.make_node("Max", ["am", "bm"], ["a_or_b"]),
                  helper.make_node("Sub", ["one_f", "a_or_b"], ["res"])]
    elif op == "xor":
        # a + b - 2ab
        nodes += [helper.make_node("Add", ["am", "bm"], ["a_plus_b"]),
                  helper.make_node("Mul", ["am", "bm"], ["a_mul_b"]),
                  helper.make_node("Mul", ["a_mul_b", "two_f"], ["two_ab"]),
                  helper.make_node("Sub", ["a_plus_b", "two_ab"], ["res"])]

    # pad the boolean result to the full canvas; rebuild a valid one-hot where
    # the background channel is on inside the oh x ow region wherever res == 0.
    region = np.zeros((1, 1, HEIGHT, WIDTH), dtype=np.float32)
    region[:, :, :oh, :ow] = 1.0
    init.append(f32("region", region))
    nodes += [
        helper.make_node("Pad", ["res", "pads", "zero"], ["res_full"]),
        helper.make_node("Sub", ["region", "res_full"], ["bg_full"]),
    ]

    # colour channels: zeros except channel `fill` = res_full
    parts = ["bg_full"]
    if fill > 1:
        init.append(f32("zb", np.zeros((1, fill - 1, HEIGHT, WIDTH),
                                       dtype=np.float32)))
        parts.append("zb")
    parts.append("res_full")
    if fill < CHANNELS - 1:
        init.append(f32("za", np.zeros((1, CHANNELS - fill - 1, HEIGHT, WIDTH),
                                       dtype=np.float32)))
        parts.append("za")
    nodes.append(helper.make_node("Concat", parts, ["output"], axis=1))

    inputs = [helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, CHANNELS, HEIGHT, WIDTH])]
    outputs = [helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, CHANNELS, HEIGHT, WIDTH])]
    value_info = [
        helper.make_tensor_value_info("am", TensorProto.FLOAT, [1, 1, oh, ow]),
        helper.make_tensor_value_info("bm", TensorProto.FLOAT, [1, 1, oh, ow]),
        helper.make_tensor_value_info("res", TensorProto.FLOAT, [1, 1, oh, ow]),
        helper.make_tensor_value_info("res_full", TensorProto.FLOAT,
                                      [1, 1, HEIGHT, WIDTH]),
        helper.make_tensor_value_info("bg_full", TensorProto.FLOAT,
                                      [1, 1, HEIGHT, WIDTH]),
    ]
    graph = helper.make_graph(nodes, "split_logic", inputs, outputs,
                              initializer=init, value_info=value_info)
    return helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
        ir_version=IR_VERSION)


def solve_split_logic(task: dict) -> Optional[onnx.ModelProto]:
    spec = _detect(task)
    if spec is None:
        return None
    return _build(*spec)
