"""Per-family solvers.

A solver is a callable `(task: dict) -> Optional[onnx.ModelProto]`. It returns
`None` if it can't handle this task, otherwise a candidate model that the
pipeline will verify.

Order matters only for tie-breaking — the pipeline picks whichever passing
candidate has the most points, so cheaper solvers should come first to keep
build time down when many candidates would pass.
"""
from __future__ import annotations

from typing import Callable, Optional

import onnx

from .identity import solve_identity
from .remap import solve_remap
from .single_color import solve_single_color
from .zero import solve_zero

Solver = Callable[[dict], Optional[onnx.ModelProto]]

ALL_SOLVERS: list[Solver] = [
    solve_identity,
    solve_zero,
    solve_single_color,
    solve_remap,
]
