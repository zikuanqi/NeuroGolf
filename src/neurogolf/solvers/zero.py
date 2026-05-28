"""Solver for tasks where the output is an empty (all-zeros) grid.

In the one-hot encoding "all zeros" means every cell is the special padding
marker, which converts back to an empty grid `[]`. Used by ARC tasks that ask
for "no answer" / "blank canvas" type outputs.
"""
from __future__ import annotations

from typing import Optional

import onnx

from ..grids import all_examples
from ..onnx_ops import zero_model


def _is_all_zeros(task: dict) -> bool:
    saw_any = False
    for ex in all_examples(task):
        saw_any = True
        for row in ex["output"]:
            for cell in row:
                if cell != 0:
                    # Color 0 has a dedicated one-hot channel; "all zeros" means
                    # the *output grid* is empty (zero rows). Stricter: only if
                    # the literal output list is empty.
                    pass
        if ex["output"]:  # non-empty grid → not the case we handle.
            return False
    return saw_any


def solve_zero(task: dict) -> Optional[onnx.ModelProto]:
    if _is_all_zeros(task):
        return zero_model()
    return None
