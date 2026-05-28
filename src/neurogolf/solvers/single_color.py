"""Solver: output is a uniform single color, filling the input shape.

Every filled input cell becomes color C in the output; empty (padding) cells
stay empty. A 1x1 Conv with weight row C set to all 1s and every other row 0
implements this exactly.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import onnx

from ..grids import CHANNELS, all_examples
from ..onnx_ops import conv1x1_model


def _derive_color(task: dict) -> int | None:
    color: int | None = None
    saw_any = False
    for ex in all_examples(task):
        inp, out = ex["input"], ex["output"]
        if not inp or not out:
            return None
        if len(inp) != len(out) or len(inp[0]) != len(out[0]):
            return None
        for r, row in enumerate(out):
            for c, cell in enumerate(row):
                if inp[r][c] == 0 and cell == 0:
                    # Color 0 could equally be "empty stays empty" — keep going.
                    pass
                if cell != 0 or any(0 != x for x in row):
                    if color is None:
                        color = cell
                    elif color != cell:
                        return None
                saw_any = True
    return color if saw_any else None


def solve_single_color(task: dict) -> Optional[onnx.ModelProto]:
    # Every output cell must be one constant color across all examples,
    # and the output shape must equal the input shape (so the "filled mask"
    # of the input recovers the output footprint).
    color: int | None = None
    for ex in all_examples(task):
        inp, out = ex["input"], ex["output"]
        if not inp or not out:
            return None
        if len(inp) != len(out) or len(inp[0]) != len(out[0]):
            return None
        for row in out:
            for cell in row:
                if color is None:
                    color = cell
                elif cell != color:
                    return None
    if color is None or not (0 <= color < CHANNELS):
        return None
    # Conv weight: row `color` reduces across all input channels.
    w = np.zeros((CHANNELS, CHANNELS), dtype=np.float32)
    w[color, :] = 1.0
    return conv1x1_model(w)
