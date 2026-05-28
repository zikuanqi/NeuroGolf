"""Color-remap solver: each input color maps to one fixed output color.

A 1x1 Conv with a permutation-like 10x10 weight matrix implements
`output[k, r, c] = sum_j W[k, j] * input[j, r, c]`. Since inputs are one-hot,
this is exactly a per-pixel color lookup.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import onnx

from ..grids import CHANNELS, all_examples
from ..onnx_ops import conv1x1_model


def _derive_mapping(task: dict) -> dict[int, int] | None:
    mapping: dict[int, int] = {}
    seen_input = False
    for ex in all_examples(task):
        inp, out = ex["input"], ex["output"]
        if not inp or not out:
            return None
        if len(inp) != len(out) or len(inp[0]) != len(out[0]):
            return None
        for r, row in enumerate(inp):
            for c, color in enumerate(row):
                target = out[r][c]
                if color in mapping and mapping[color] != target:
                    return None
                mapping[color] = target
                seen_input = True
    return mapping if seen_input else None


def solve_remap(task: dict) -> Optional[onnx.ModelProto]:
    mapping = _derive_mapping(task)
    if mapping is None:
        return None
    if all(k == v for k, v in mapping.items()):
        return None  # let the identity solver own this case
    w = np.zeros((CHANNELS, CHANNELS), dtype=np.float32)
    for src, dst in mapping.items():
        if not (0 <= src < CHANNELS) or not (0 <= dst < CHANNELS):
            return None
        w[dst, src] = 1.0
    return conv1x1_model(w)
