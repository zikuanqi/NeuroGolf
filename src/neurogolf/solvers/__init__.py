"""Per-family solvers.

A solver is a callable `(task: dict) -> Optional[onnx.ModelProto]`. It returns
`None` if it can't handle this task, otherwise a candidate model that the
pipeline will verify.

Order matters only for tie-breaking - the pipeline picks whichever passing
candidate has the most points, so cheaper solvers should come first to keep
build time down when many candidates would pass.
"""
from __future__ import annotations

from typing import Callable, Optional

import onnx

from .axis_gather import solve_axis_gather
from .bbox_color_extract import solve_bbox_color_extract
from .blob_recolor import solve_blob_recolor
from .block_count_bar import solve_block_count_bar
from .block_mask import solve_block_mask
from .cc_rank_recolor import solve_cc_rank_recolor
from .cc_size_recolor import solve_cc_size_recolor
from .color_bbox_fill import solve_color_bbox_fill
from .color_lines import solve_color_lines
from .stripe_seeds import solve_stripe_seeds
from .column_label import solve_column_label
from .connect_dots import solve_connect_dots
from .connect_fill import solve_connect_fill
from .count_bar import solve_count_bar
from .cross_laser import solve_cross_laser
from .denoise import solve_denoise
from .diag_ray import solve_diag_ray
from .diag_block_slide import solve_diag_block_slide
from .diag_connect import solve_diag_connect
from .stamp_top_row import solve_stamp_top_row
from .diag_tile import solve_diag_tile
from .dilate_ones import solve_dilate_ones
from .drop_into_wall import solve_drop_into_wall
from .endpoint_bridge import solve_endpoint_bridge
from .bbox_strip import solve_bbox_strip
from .bbox_strip_zero import solve_bbox_strip_zero
from .largest_comp_crop import solve_largest_comp_crop
from .conv3x3 import solve_conv3x3
from .conv3x3_masked import (solve_conv1x1_masked, solve_conv3x3_masked,
                              solve_conv5x5_masked)
from .filled_rect import solve_filled_rect
from .framed_regions import solve_framed_regions
from .plus_panels import solve_plus_panels
from .rot180_repair import solve_rot180_repair
from .lattice_count import solve_lattice_count
from .quadrant_crop import solve_quadrant_crop
from .connect_box_markers import solve_connect_box_markers
from .recolor_in_block import solve_recolor_in_block
from .corner_rays import solve_corner_rays
from .divider_fold import solve_divider_fold
from .band_sort import solve_band_sort
from .interior_recolor import solve_interior_recolor
from .interior_recolor_aware import solve_interior_recolor_aware
from .float_up import solve_float_up
from .diag_x import solve_diag_x
from .staircase import solve_staircase
from .box_stretch import solve_box_stretch
from .gap_fill import solve_gap_fill
from .merge_pair import solve_merge_pair
from .cross_move import solve_cross_move
from .row_checker import solve_row_checker
from .five_isolate import solve_five_isolate
from .color_sort_column import solve_color_sort_column
from .rect_interior_rank import solve_rect_interior_rank
from .ring_recolor import solve_ring_recolor
from .line_cross_swap import solve_line_cross_swap
from .explode_corners import solve_explode_corners
from .l_connect import solve_l_connect
from .block_quadrant import solve_block_quadrant
from .move_toward import solve_move_toward
from .cut_diagonals import solve_cut_diagonals
from .odd_panel_shape import solve_odd_panel_shape
from .band_majority import solve_band_majority
from .connect_pairs import solve_connect_pairs
from .panel_summary import solve_panel_summary
from .column_template import solve_column_template
from .fractal_blocks import solve_fractal_blocks
from .diagonal_markers import solve_diagonal_markers
from .odd_col_recolor import solve_odd_col_recolor
from .triangle_diag import solve_triangle_diag
from .pocket_drop import solve_pocket_drop
from .square_complete import solve_square_complete
from .midpoint_plus import solve_midpoint_plus
from .elbow_connect import solve_elbow_connect
from .mirror_quad import solve_mirror_quad
from .arrow_ray import solve_arrow_ray
from .gravity_down import solve_gravity_down
from .slide_to_wall import solve_slide_to_wall
from .slide_to_line import solve_slide_to_line
from .project_to_block import solve_project_to_block
from .gravity_up import solve_gravity_up
from .gravity_right import solve_gravity_right
from .gravity_right_diag import solve_gravity_right_diag
from .halo import solve_halo
from .hspan_fill import solve_hspan_fill
from .identity import solve_identity
from .isolate_recolor import solve_isolate_recolor
from .keep_majority import solve_keep_majority
from .kron_scale import solve_kron_scale
from .largest_bbox_fill import solve_largest_bbox_fill
from .majority_fill import solve_majority_fill
from .marker_crop import solve_marker_crop
from .mirror_complete import solve_mirror_complete
from .nearest_wall import solve_nearest_wall
from .odd_panel import solve_odd_panel
from .odd_panel_aware import solve_odd_panel_aware
from .outline import solve_outline
from .palindrome import (solve_palindrome_2d, solve_palindrome_h,
                          solve_palindrome_v)
from .ray_down import solve_ray_down
from .recolor_fives import solve_recolor_fives
from .remap import solve_remap
from .repeat_top_rows import solve_repeat_top_rows
from .resize_scale import solve_resize_scale
from .row_uniform_indicator import solve_row_uniform_indicator
from .scale_detector import solve_scale_detector
from .downscale_majority import solve_downscale_majority
from .self_fractal import solve_self_fractal
from .rot_tile import solve_rot_tile
from .rot_tile_aware import solve_rot_tile_aware
from .period_extend_h import solve_period_extend_h
from .untile_half import solve_untile_half
from .periodic_fill import solve_periodic_fill
from .shape_aware_flip import (solve_flip_h_aware, solve_flip_v_aware,
                                solve_rot90_ccw_aware, solve_rot90_cw_aware,
                                solve_rot180_aware)
from .shift import solve_shift
from .single_color import solve_single_color
from .spatial import solve_transpose
from .split_and import solve_split_and
from .split_logic import solve_split_logic
from .stamp import solve_stamp
from .static_crop import solve_static_crop
from .tile_h import solve_tile_h
from .variable_kron import solve_variable_kron
from .zero import solve_zero
from .zero_color import solve_zero_color
from .flood_fill_enclosure import solve_flood_fill_enclosure
from .symmetry_classify import solve_symmetry_classify
from .shape_classify import solve_shape_classify
from .count_pattern import solve_count_pattern
from .colorcount_pattern import solve_colorcount_pattern
from .position_color import solve_position_color
from .spatial_classify import solve_spatial_classify
from .scattered_color import solve_scattered_color
from .shift_down_recolor import solve_shift_down_recolor
from .learned_conv import solve_learned_conv

Solver = Callable[[dict], Optional[onnx.ModelProto]]

ALL_SOLVERS: list[Solver] = [
    solve_identity,
    solve_zero,
    solve_axis_gather,
    solve_single_color,
    solve_remap,
    solve_repeat_top_rows,
    solve_transpose,
    solve_block_mask,
    solve_static_crop,
    solve_kron_scale,
    solve_resize_scale,
    solve_variable_kron,
    solve_marker_crop,
    solve_shift,
    solve_tile_h,
    solve_period_extend_h,
    solve_untile_half,
    solve_scale_detector,
    solve_downscale_majority,
    solve_self_fractal,
    solve_rot_tile,
    solve_rot_tile_aware,
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
    solve_bbox_strip_zero,
    solve_largest_comp_crop,
    solve_bbox_color_extract,
    solve_split_and,
    solve_split_logic,
    solve_odd_panel,
    solve_odd_panel_aware,
    solve_connect_dots,
    solve_connect_fill,
    solve_color_bbox_fill,
    solve_mirror_complete,
    solve_denoise,
    solve_drop_into_wall,
    solve_nearest_wall,
    solve_cross_laser,
    solve_halo,
    solve_color_lines,
    solve_stripe_seeds,
    solve_endpoint_bridge,
    solve_keep_majority,
    solve_blob_recolor,
    solve_recolor_fives,
    solve_dilate_ones,
    solve_count_bar,
    solve_block_count_bar,
    solve_hspan_fill,
    solve_stamp,
    solve_outline,
    solve_ray_down,
    solve_isolate_recolor,
    solve_cc_size_recolor,
    solve_cc_rank_recolor,
    solve_zero_color,
    solve_column_label,
    solve_diag_ray,
    solve_diag_block_slide,
    solve_diag_connect,
    solve_stamp_top_row,
    solve_diag_tile,
    solve_gravity_down,
    solve_slide_to_wall,
    solve_slide_to_line,
    solve_project_to_block,
    solve_gravity_up,
    solve_gravity_right,
    solve_gravity_right_diag,
    solve_filled_rect,
    solve_framed_regions,
    solve_plus_panels,
    solve_rot180_repair,
    solve_lattice_count,
    solve_quadrant_crop,
    solve_connect_box_markers,
    solve_recolor_in_block,
    solve_corner_rays,
    solve_divider_fold,
    solve_band_sort,
    solve_interior_recolor,
    solve_interior_recolor_aware,
    solve_float_up,
    solve_diag_x,
    solve_staircase,
    solve_box_stretch,
    solve_gap_fill,
    solve_merge_pair,
    solve_cross_move,
    solve_row_checker,
    solve_five_isolate,
    solve_color_sort_column,
    solve_rect_interior_rank,
    solve_ring_recolor,
    solve_line_cross_swap,
    solve_explode_corners,
    solve_l_connect,
    solve_block_quadrant,
    solve_move_toward,
    solve_cut_diagonals,
    solve_odd_panel_shape,
    solve_band_majority,
    solve_connect_pairs,
    solve_panel_summary,
    solve_column_template,
    solve_fractal_blocks,
    solve_diagonal_markers,
    solve_odd_col_recolor,
    solve_triangle_diag,
    solve_pocket_drop,
    solve_square_complete,
    solve_midpoint_plus,
    solve_elbow_connect,
    solve_mirror_quad,
    solve_arrow_ray,
    solve_periodic_fill,
    solve_conv3x3,
    solve_conv1x1_masked,
    solve_conv3x3_masked,
    solve_conv5x5_masked,
    solve_position_color,
    solve_colorcount_pattern,
    solve_count_pattern,
    solve_shape_classify,
    solve_scattered_color,
    solve_flood_fill_enclosure,
    solve_symmetry_classify,
    solve_spatial_classify,
    solve_learned_conv,
]
