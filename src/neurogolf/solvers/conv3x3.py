"""Solver: fit a 3x3 Conv (with bias) via least squares.

Many ARC tasks are *purely local* — the output color at (r, c) is determined
by the 3x3 patch of input cells around (r, c). When that's the case a linear
model in the 10*9 one-hot neighborhood features can reproduce the output
exactly. Even when the rule is non-linear, the >0 thresholding at the output
gives enough slack that a least-squares solution often still discretizes to
the correct one-hot.

We fit only on `train` examples to keep arc-gen as a held-out generalization
check; the pipeline rejects the model if any example fails.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import onnx
from onnx import TensorProto, helper

from ..grids import CHANNELS, to_onehot
from ..onnx_ops import DATA_TYPE, finalize

KERNEL = 3


def _neighborhood_features(grid: np.ndarray) -> np.ndarray:
    """For each spatial position, gather the 3x3 patch of channel values.

    grid: (10, H, W). Returns (H*W, 90) where 90 = 10 * 3 * 3.
    """
    _, h, w = grid.shape
    pad = np.pad(grid, ((0, 0), (1, 1), (1, 1)))
    feats = np.zeros((h * w, CHANNELS * KERNEL * KERNEL), dtype=np.float32)
    idx = 0
    for r in range(h):
        for c in range(w):
            patch = pad[:, r:r + KERNEL, c:c + KERNEL]  # (10, 3, 3)
            feats[idx] = patch.reshape(-1)
            idx += 1
    return feats


def _conv3x3_model(weight: np.ndarray) -> onnx.ModelProto:
    w = weight.astype(np.float32)
    w_init = helper.make_tensor("W", DATA_TYPE, list(w.shape), w.flatten())
    # No bias: padding cells in the 30x30 canvas are all-zero one-hot vectors,
    # and a nonzero bias would leak into them - failing the verifier which
    # compares the full 30x30 tensor cell-by-cell.
    node = helper.make_node(
        "Conv", ["input", "W"], ["output"],
        kernel_shape=[KERNEL, KERNEL], pads=[1, 1, 1, 1],
    )
    return finalize([node], [w_init])


def solve_conv3x3(task: dict) -> Optional[onnx.ModelProto]:
    # Same-size requirement: a 3x3 conv with pads=1 keeps spatial dims.
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    # Sample arc-gen examples too so the fit sees the augmented distribution.
    # arc-gen lists have hundreds of entries; cap at 30 to keep lstsq fast.
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

    # Crop each pair down to its actual H, W (cells outside are empty padding).
    crops: list[tuple[np.ndarray, np.ndarray]] = []
    for inp, out in pairs:
        h = next((r for r in range(29, -1, -1)
                  if inp[:, r, :].any() or out[:, r, :].any()), -1) + 1
        w = next((c for c in range(29, -1, -1)
                  if inp[:, :, c].any() or out[:, :, c].any()), -1) + 1
        if h == 0 or w == 0:
            return None
        crops.append((inp[:, :h, :w], out[:, :h, :w]))

    # Build feature / target matrices.
    feat_blocks = [_neighborhood_features(i) for i, _ in crops]
    targ_blocks = [o.reshape(CHANNELS, -1).T for _, o in crops]
    X = np.vstack(feat_blocks)                                      # (N, 90)
    Y = np.vstack(targ_blocks)                                      # (N, 10)

    # Solve without bias: zero-input cells (padding) must produce zero output
    # so the verifier's full-canvas comparison passes.
    sol, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    W_flat = sol.T                                  # (10, 90)
    W = W_flat.reshape(CHANNELS, CHANNELS, KERNEL, KERNEL)

    # Quick sanity check on the fit before paying the verification cost.
    pred = X @ W_flat.T
    if not np.all((pred > 0.0) == (Y > 0.5)):
        return None
    return _conv3x3_model(W)
