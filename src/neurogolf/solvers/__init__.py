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

from .bbox_color_extract import solve_bbox_color_extract
from .bbox_strip import solve_bbox_strip
from .conv3x3 import solve_conv3x3
from .conv3x3_masked import (solve_conv1x1_masked, solve_conv3x3_masked,
                              solve_conv5x5_masked)
from .scale_detector import solve_scale_detector
from .variable_shift import solve_variable_shift
from .gravity_right import solve_gravity_right
from .identity import solve_identity
from .kron_scale import solve_kron_scale
from .largest_bbox_fill import solve_largest_bbox_fill
from .majority_fill import solve_majority_fill
from .marker_crop import solve_marker_crop
from .remap import solve_remap
from .resize_scale import solve_resize_scale
from .row_uniform_indicator import solve_row_uniform_indicator
from .palindrome import (solve_palindrome_2d, solve_palindrome_h,
                          solve_palindrome_v)
from .shift import solve_shift
from .tile_h import solve_tile_h
from .shape_aware_flip import (solve_flip_h_aware, solve_flip_v_aware,
                                solve_rot90_ccw_aware, solve_rot90_cw_aware,
                                solve_rot180_aware)
from .single_color import solve_single_color
from .spatial import solve_transpose
from .split_and import solve_split_and
from .static_crop import solve_static_crop
from .variable_kron import solve_variable_kron
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
    solve_variable_kron,
    solve_marker_crop,
    solve_shift,
    solve_tile_h,
    solve_scale_detector,
    solve_variable_shift,
    solve_palindrome_h,
    solve_palindrome_v,
    solve_palindrome_2d,
    solve_majority_fill,
    solve_largest_bbox_fill,
    solve_row_uniform_indicator,
    solve_flip_h_aware,
    solve_flip_v_aware,
    solve_rot180_aware,
    solve_rot90_ccw_aware,
    solve_rot90_cw_aware,
    solve_bbox_strip,
    solve_bbox_color_extract,
    solve_split_and,
    solve_conv3x3,
    solve_conv1x1_masked,
    solve_conv3x3_masked,
    solve_conv5x5_masked,
]
