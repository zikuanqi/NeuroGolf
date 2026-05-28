"""Solver for tasks where output == input on every example."""
from __future__ import annotations

from typing import Optional

import onnx

from ..grids import is_identity
from ..onnx_ops import identity_model


def solve_identity(task: dict) -> Optional[onnx.ModelProto]:
    if is_identity(task):
        return identity_model()
    return None
