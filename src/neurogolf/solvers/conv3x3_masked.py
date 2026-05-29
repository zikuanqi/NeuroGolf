"""Conv3×3 + bias solver with a non-padding mask.

Variant of `conv3x3` that includes a bias term. The bias breaks zero-input
cells (which decode to padding in the verifier), so we multiply the Conv
output by `(ReduceSum(input, axes=1) > 0)` to keep padding cells empty.

This catches tasks where the rule is "for each cell in the actual grid,
compute output color from a 3×3 neighborhood plus a per-channel constant
offset" — e.g., "fill non-bg cells uniformly with color X, leave bg alone"
or other affine local rules. The mask makes the affine part safe.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from ..grids import CHANNELS, HEIGHT, WIDTH, to_onehot

KERNEL = 3
OPSET = 11
IR_VERSION = 8


def _neighborhood_features(grid: np.ndarray) -> np.ndarray:
    _, h, w = grid.shape
    pad = np.pad(grid, ((0, 0), (1, 1), (1, 1)))
    feats = np.zeros((h * w, CHANNELS * KERNEL * KERNEL), dtype=np.float32)
    idx = 0
    for r in range(h):
        for c in range(w):
            patch = pad[:, r:r + KERNEL, c:c + KERNEL]
            feats[idx] = patch.reshape(-1)
            idx += 1
    return feats


def _build(weight: np.ndarray, bias: np.ndarray) -> onnx.ModelProto:
    w = weight.astype(np.float32)
    b = bias.astype(np.float32)
    w_init = helper.make_tensor("W", TensorProto.FLOAT, list(w.shape),
                                w.flatten())
    b_init = helper.make_tensor("B", TensorProto.FLOAT, list(b.shape),
                                b.flatten())

    nodes = [
        helper.make_node(
            "Conv", ["input", "W", "B"], ["conv_out"],
            kernel_shape=[KERNEL, KERNEL], pads=[1, 1, 1, 1],
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
        nodes, "conv3x3_masked", inputs, outputs,
        initializer=[w_init, b_init], value_info=value_info)
    return helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
        ir_version=IR_VERSION)


def solve_conv3x3_masked(task: dict) -> Optional[onnx.ModelProto]:
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

    feat_blocks = [_neighborhood_features(i) for i, _ in crops]
    targ_blocks = [o.reshape(CHANNELS, -1).T for _, o in crops]
    X = np.vstack(feat_blocks)
    Y = np.vstack(targ_blocks)
    # Augment with bias column.
    X1 = np.hstack([X, np.ones((X.shape[0], 1), dtype=np.float32)])
    sol, _, _, _ = np.linalg.lstsq(X1, Y, rcond=None)
    W_flat = sol[:-1].T
    B = sol[-1]
    W = W_flat.reshape(CHANNELS, CHANNELS, KERNEL, KERNEL)

    # Quick sanity: on the cropped (non-padding) cells, after the mask the
    # raw conv output (W·feats + B) must agree with the one-hot target's
    # threshold-at-zero binarization.
    pred = X @ W_flat.T + B
    if not np.all((pred > 0.0) == (Y > 0.5)):
        return None
    return _build(W, B)
