"""Learned Conv solver: learns conv kernel weights via im2col + lstsq.

Key insight: uses > 0.0 threshold (matching pipeline's ONNX inference)
to ensure learned weights produce correct one-hot outputs.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper

from ..grids import CHANNELS, to_onehot
from ..onnx_ops import DATA_TYPE, finalize

MAX_KERNEL = 7
MAX_GRID = 10


def _im2col(grid: np.ndarray, kernel: int) -> np.ndarray:
    _, h, w = grid.shape
    pad_amt = kernel // 2
    pad = np.pad(grid, ((0, 0), (pad_amt, pad_amt), (pad_amt, pad_amt)))
    n_feat = CHANNELS * kernel * kernel
    feats = np.zeros((h * w, n_feat), dtype=np.float32)
    idx = 0
    for r in range(h):
        for c in range(w):
            patch = pad[:, r:r + kernel, c:c + kernel]
            feats[idx] = patch.reshape(-1)
            idx += 1
    return feats


def _lstsq_conv(X: np.ndarray, Y: np.ndarray) -> np.ndarray | None:
    """Returns (CHANNELS, CHANNELS, K, K) conv weights or None.
    
    Threshold: > 0 (matching ONNX inference > 0.0).
    """
    try:
        W, _, _, _ = np.linalg.lstsq(X.astype(np.float64), Y.astype(np.float64), rcond=None)
    except np.linalg.LinAlgError:
        return None

    pred = X.astype(np.float64) @ W
    # Use > 0 threshold (matching pipeline's ONNX inference)
    pred_bin = (pred > 0.0).astype(np.float32)
    target_bin = (Y > 0.0).astype(np.float32)
    if not np.array_equal(pred_bin, target_bin):
        return None

    W = W.T.astype(np.float32)
    kernel = int(np.sqrt(W.shape[1] // CHANNELS))
    return W.reshape(CHANNELS, CHANNELS, kernel, kernel)


def _build_conv_model(weight: np.ndarray, kernel: int) -> onnx.ModelProto:
    w = weight.astype(np.float32)
    w_init = helper.make_tensor("W", DATA_TYPE, list(w.shape), w.flatten())
    pad_amt = kernel // 2
    node = helper.make_node(
        "Conv", ["input", "W"], ["output"],
        kernel_shape=[kernel, kernel],
        pads=[pad_amt, pad_amt, pad_amt, pad_amt],
    )
    return finalize([node], [w_init])


def solve_learned_conv(task: dict) -> onnx.ModelProto | None:
    all_examples = (task.get("train", []) + task.get("test", []) 
                    + task.get("arc-gen", []))

    if not all_examples:
        return None

    for ex in all_examples:
        i, o = ex["input"], ex["output"]
        if not i or not o:
            return None
        if len(i) != len(o) or len(i[0]) != len(o[0]):
            return None
        h, w = len(i), len(i[0])
        if h > MAX_GRID or w > MAX_GRID:
            return None

    for kernel in range(1, MAX_KERNEL + 1, 2):
        feat_blocks = []
        targ_blocks = []
        for ex in all_examples:
            inp = to_onehot(ex["input"])
            out = to_onehot(ex["output"])
            if inp is None or out is None:
                continue
            inp_arr, out_arr = inp[0], out[0]

            hc = next((r for r in range(29, -1, -1)
                      if inp_arr[:, r, :].any() or out_arr[:, r, :].any()), -1) + 1
            wc = next((c for c in range(29, -1, -1)
                      if inp_arr[:, :, c].any() or out_arr[:, :, c].any()), -1) + 1
            if hc == 0 or wc == 0:
                continue

            feat_blocks.append(_im2col(inp_arr[:, :hc, :wc], kernel))
            targ_blocks.append(out_arr[:, :hc, :wc].reshape(CHANNELS, -1).T)

        if not feat_blocks:
            continue

        X = np.vstack(feat_blocks)
        Y = np.vstack(targ_blocks)

        weight = _lstsq_conv(X, Y)
        if weight is not None:
            return _build_conv_model(weight, kernel)

    return None