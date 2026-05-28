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

from .bbox_strip import solve_bbox_strip
from .conv3x3 import solve_conv3x3
from .identity import solve_identity
from .kron_scale import solve_kron_scale
from .marker_crop import solve_marker_crop
from .remap import solve_remap
from .resize_scale import solve_resize_scale
from .shape_aware_flip import (solve_flip_h_aware, solve_flip_v_aware,
                                solve_rot90_ccw_aware, solve_rot90_cw_aware,
                                solve_rot180_aware)
from .single_color import solve_single_color
from .spatial import solve_transpose
from .static_crop import solve_static_crop
from .zero import solve_zero

Solver = Callable[[dict], Optional[onnx.ModelProto]]

ALL_SOLVERS: list[Solver] = [
    solve_identity,
    solve_zero,
    solve_single_color,
    solve_remap,
    solve_transpose,
    solve_static_crop,
    solve_kron_scale,
    solve_resize_scale,
    solve_marker_crop,
    solve_flip_h_aware,
    solve_flip_v_aware,
    solve_rot180_aware,
    solve_rot90_ccw_aware,
    solve_rot90_cw_aware,
    solve_bbox_strip,
    solve_conv3x3,
]
