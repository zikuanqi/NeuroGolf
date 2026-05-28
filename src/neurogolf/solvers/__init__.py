"""Per-family solvers.

A solver is a callable `(task: dict) -> Optional[onnx.ModelProto]`. It returns
`None` if it can't handle this task, otherwise a candidate model that the
pipeline will verify.
"""
from __future__ import annotations

from typing import Callable, Optional

import onnx

from .identity import solve_identity
from .zero import solve_zero

Solver = Callable[[dict], Optional[onnx.ModelProto]]

ALL_SOLVERS: list[Solver] = [
    solve_identity,
    solve_zero,
]
