"""ConvKxK + bias solver with a non-padding mask.

Variant of `conv3x3` that includes a bias term. The bias breaks zero-input
cells (which decode to padding in the verifier), so we multiply the Conv
output by `(ReduceSum(input, axes=1) > 0)` to keep padding cells empty.

This catches tasks where the rule is "for each cell in the actual grid,
compute output color from a K×K neighborhood plus a per-channel constant
offset". We export two factory functions — one fixed at 3×3 and one that
takes a kernel size — so the registry can try several kernels in order.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from ..grids import CHANNELS, HEIGHT, WIDTH, to_onehot

OPSET = 11
IR_VERSION = 8


def _neighborhood_features(grid: np.ndarray, kernel: int) -> np.ndarray:
    _, h, w = grid.shape
    pad_amt = kernel // 2
    pad = np.pad(grid, ((0, 0), (pad_amt, pad_amt), (pad_amt, pad_amt)))
    feats = np.zeros((h * w, CHANNELS * kernel * kernel), dtype=np.float32)
    idx = 0
    for r in range(h):
        for c in range(w):
            patch = pad[:, r:r + kernel, c:c + kernel]
            feats[idx] = patch.reshape(-1)
            idx += 1
    return feats


def _build(weight: np.ndarray, bias: np.ndarray, kernel: int) -> onnx.ModelProto:
    w = weight.astype(np.float32)
    b = bias.astype(np.float32)
    w_init = helper.make_tensor("W", TensorProto.FLOAT, list(w.shape),
                                w.flatten())
    b_init = helper.make_tensor("B", TensorProto.FLOAT, list(b.shape),
                                b.flatten())

    pad_amt = kernel // 2
    nodes = [
        helper.make_node(
            "Conv", ["input", "W", "B"], ["conv_out"],
            kernel_shape=[kernel, kernel],
            pads=[pad_amt, pad_amt, pad_amt, pad_amt],
            name="conv_op"),
        # Non-padding mask: 1 where any input channel fires.
        helper.make_node(
            "ReduceSum", ["input"], ["pad_mask"],
            axes=[1], keepdims=1, name="pad_sum"),
        helper.make_node(
            "Mul", ["conv_out", "pad_mask"], ["output"],
            name="apply_pad_mask"),
    ]
    inputs = [helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, CHANNELS, HEIGHT, WIDTH])]
    outputs = [helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, CHANNELS, HEIGHT, WIDTH])]
    value_info = [
        helper.make_tensor_value_info(
            "conv_out", TensorProto.FLOAT,
            [1, CHANNELS, HEIGHT, WIDTH]),
        helper.make_tensor_value_info(
            "pad_mask", TensorProto.FLOAT, [1, 1, HEIGHT, WIDTH]),
    ]
    graph = helper.make_graph(
        nodes, f"conv{kernel}x{kernel}_masked", inputs, outputs,
        initializer=[w_init, b_init], value_info=value_info)
    return helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
        ir_version=IR_VERSION)


def _solve(task: dict, kernel: int) -> Optional[onnx.ModelProto]:
    pairs = []
    arc_gen_sample = task.get("arc-gen", [])[:30]
    for ex in task.get("train", []) + task.get("test", []) + arc_gen_sample:
        i, o = ex["input"], ex["output"]
        if not i or not o:
            return None
        if len(i) != len(o) or len(i[0]) != len(o[0]):
            return None
        inp = to_onehot(i)
        out = to_onehot(o)
        if inp is None or out is None:
            return None
        pairs.append((inp[0], out[0]))
    if not pairs:
        return None

    crops = []
    for inp, out in pairs:
        h = next((r for r in range(29, -1, -1)
                  if inp[:, r, :].any() or out[:, r, :].any()), -1) + 1
        w = next((c for c in range(29, -1, -1)
                  if inp[:, :, c].any() or out[:, :, c].any()), -1) + 1
        if h == 0 or w == 0:
            return None
        crops.append((inp[:, :h, :w], out[:, :h, :w]))

    feat_blocks = [_neighborhood_features(i, kernel) for i, _ in crops]
    targ_blocks = [o.reshape(CHANNELS, -1).T for _, o in crops]
    X = np.vstack(feat_blocks)
    Y = np.vstack(targ_blocks)
    X1 = np.hstack([X, np.ones((X.shape[0], 1), dtype=np.float32)])
    sol, _, _, _ = np.linalg.lstsq(X1, Y, rcond=None)
    W_flat = sol[:-1].T
    B = sol[-1]
    W = W_flat.reshape(CHANNELS, CHANNELS, kernel, kernel)

    pred = X @ W_flat.T + B
    if not np.all((pred > 0.0) == (Y > 0.5)):
        return None
    return _build(W, B, kernel)


def solve_conv1x1_masked(task: dict) -> Optional[onnx.ModelProto]:
    return _solve(task, 1)


def solve_conv3x3_masked(task: dict) -> Optional[onnx.ModelProto]:
    return _solve(task, 3)


def solve_conv5x5_masked(task: dict) -> Optional[onnx.ModelProto]:
    return _solve(task, 5)


def solve_conv7x7_masked(task: dict) -> Optional[onnx.ModelProto]:
    return _solve(task, 7)
