"""Solver: block-majority downscale (task 130).

The input is a (kh*oh) x (kw*ow) grid; the output is oh x ow, where each output
cell is the **majority colour** of its kh x kw block (ties broken by lowest
colour index). Task 130 is a constant 9x9 -> 3x3 / k=3.

`AveragePool` (kernel = stride = block size) turns each block into the
per-channel colour fraction; `ArgMax` over channels picks the majority; `OneHot`
re-expands to the one-hot frame, padded back to 30x30.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from ..grids import CHANNELS, HEIGHT, WIDTH, all_examples

OPSET = 11
IR_VERSION = 8
FULL = [1, CHANNELS, HEIGHT, WIDTH]


def _ref(g: np.ndarray, kh: int, kw: int, oh: int, ow: int) -> np.ndarray:
    out = np.zeros((oh, ow), dtype=int)
    for r in range(oh):
        for c in range(ow):
            blk = g[r * kh:(r + 1) * kh, c * kw:(c + 1) * kw]
            out[r, c] = int(np.argmax(np.bincount(blk.flatten(), minlength=CHANNELS)))
    return out


def _detect(task: dict):
    examples = list(all_examples(task))
    if not examples:
        return None
    kk = None
    oo = None
    saw_shrink = False
    for ex in examples:
        i, o = ex["input"], ex["output"]
        if not i or not i[0] or not o or not o[0]:
            return None
        H, W, oh, ow = len(i), len(i[0]), len(o), len(o[0])
        if H > HEIGHT or W > WIDTH:
            continue
        if oh == 0 or ow == 0 or H % oh or W % ow:
            return None
        kh, kw = H // oh, W // ow
        if kh < 1 or kw < 1 or (kh == 1 and kw == 1):
            return None
        if kh > 1 or kw > 1:
            saw_shrink = True
        if not np.array_equal(_ref(np.array(i), kh, kw, oh, ow), np.array(o)):
            return None
        if kk is None:
            kk, oo = (kh, kw), (oh, ow)
        elif kk != (kh, kw) or oo != (oh, ow):
            return None       # bake a single block / output size
    if kk is None or not saw_shrink:
        return None
    return kk[0], kk[1], oo[0], oo[1]


def _build(kh: int, kw: int, oh: int, ow: int) -> onnx.ModelProto:
    F = TensorProto.FLOAT
    n_ = helper.make_node
    pool_h = (HEIGHT - kh) // kh + 1
    pool_w = (WIDTH - kw) // kw + 1
    init = [
        numpy_helper.from_array(np.array([0, 0, 0, 0], np.int64), "sl_s"),
        numpy_helper.from_array(np.array([1, CHANNELS, oh, ow], np.int64), "sl_e"),
        numpy_helper.from_array(np.array([0, 1, 2, 3], np.int64), "sl_ax"),
        numpy_helper.from_array(np.array([1, oh, ow], np.int64), "rs"),
        numpy_helper.from_array(np.array(CHANNELS, np.int64), "depth"),
        numpy_helper.from_array(np.array([0.0, 1.0], np.float32), "onoff"),
        numpy_helper.from_array(
            np.array([0, 0, 0, 0, 0, 0, HEIGHT - oh, WIDTH - ow], np.int64), "pads"),
    ]
    nodes = [
        n_("AveragePool", ["input"], ["pooled"], kernel_shape=[kh, kw],
           strides=[kh, kw], name="pooled"),
        n_("Slice", ["pooled", "sl_s", "sl_e", "sl_ax"], ["blk"], name="blk"),
        n_("ArgMax", ["blk"], ["idx"], axis=1, keepdims=1, name="idx"),
        n_("Reshape", ["idx", "rs"], ["idx3"], name="idx3"),
        n_("OneHot", ["idx3", "depth", "onoff"], ["oneh"], axis=1, name="oneh"),
        n_("Pad", ["oneh", "pads"], ["output"], mode="constant", name="output"),
    ]
    value_info = [
        helper.make_tensor_value_info("pooled", F, [1, CHANNELS, pool_h, pool_w]),
        helper.make_tensor_value_info("blk", F, [1, CHANNELS, oh, ow]),
        helper.make_tensor_value_info("idx", TensorProto.INT64, [1, 1, oh, ow]),
        helper.make_tensor_value_info("idx3", TensorProto.INT64, [1, oh, ow]),
        helper.make_tensor_value_info("oneh", F, [1, CHANNELS, oh, ow]),
    ]
    graph = helper.make_graph(
        nodes, "downscale_majority",
        [helper.make_tensor_value_info("input", F, FULL)],
        [helper.make_tensor_value_info("output", F, FULL)],
        initializer=init, value_info=value_info)
    return helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
        ir_version=IR_VERSION)


def solve_downscale_majority(task: dict) -> Optional[onnx.ModelProto]:
    p = _detect(task)
    if p is None:
        return None
    return _build(*p)
