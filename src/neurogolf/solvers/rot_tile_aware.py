"""Shape-aware rotational four-quadrant tiling (task 106).

Same target as `solve_rot_tile` — a square N x N input maps to a 2N x 2N output
of four rotated copies::

    +-------+-------+
    |   I   | rot270|
    +-------+-------+
    | rot90 | rot180|
    +-------+-------+

— but here N is **not** constant across the examples (task 106 mixes 2x2 and
3x3 grids), so the cheap baked-index `solve_rot_tile` declines. This version
detects the content extent at runtime:

  * the three rotations reuse the shape-aware `_flip_along` machinery
    (transpose + content-aware row/column flip), each landing top-left;
  * `_shift_along` slides a top-left block down / right by its own content
    extent N, so the four quadrants drop into place via three adds.

It costs far more memory than the static version (~20 full intermediates), so
it only wins when the static solver can't.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from ..grids import CHANNELS, HEIGHT, WIDTH, all_examples
from .shape_aware_flip import _flip_along
from .rot_tile import _rot_tile

OPSET = 11
IR_VERSION = 8

FULL = [1, CHANNELS, HEIGHT, WIDTH]


def _detect(task: dict) -> bool:
    """rot_tile rule holds on every (square) example. N may vary."""
    examples = list(all_examples(task))
    if not examples:
        return False
    saw_multi = False
    sizes = set()
    for ex in examples:
        inp = ex["input"]
        if not inp or len(inp) != len(inp[0]):
            return False
        n = len(inp)
        if n < 1 or 2 * n > min(HEIGHT, WIDTH):
            return False
        if _rot_tile(inp) != ex["output"]:
            return False
        sizes.add(n)
        if any(v for row in inp for v in row):
            saw_multi = True
    # only meaningful when there is real content; constant-N is left to the
    # cheaper static solver (it will simply outscore this one).
    return saw_multi and len(sizes) >= 1


def _shift_along(axis: int, suffix: str, in_name: str, out_name: str,
                 init: list, vi: list) -> list:
    """Slide `in_name` (a top-left block) along `axis` (2=down, 3=right) by its
    own content extent N, so its content moves into positions [N, 2N)."""
    other = 3 if axis == 2 else 2
    side = HEIGHT if axis == 2 else WIDTH

    arange_shape = [1, 1, 1, 1]
    arange_shape[axis] = side
    proj_shape = [1, 1, HEIGHT, WIDTH]
    proj_shape[other] = 1
    scalar = [1, 1, 1, 1]

    init += [
        numpy_helper.from_array(np.arange(side, dtype=np.float32),
                                f"sa_arange_{suffix}"),
        numpy_helper.from_array(np.array(arange_shape, dtype=np.int64),
                                f"sa_areshape_{suffix}"),
        numpy_helper.from_array(np.array([side], dtype=np.int64),
                                f"sa_flat_{suffix}"),
        numpy_helper.from_array(np.array(1.0, dtype=np.float32),
                                f"sa_one_{suffix}"),
        numpy_helper.from_array(np.array(0.0, dtype=np.float32),
                                f"sa_zero_{suffix}"),
        numpy_helper.from_array(np.array(float(side - 1), dtype=np.float32),
                                f"sa_max_{suffix}"),
    ]
    vi += [
        helper.make_tensor_value_info(f"sa_mask_{suffix}", TensorProto.FLOAT,
                                      [1, 1, HEIGHT, WIDTH]),
        helper.make_tensor_value_info(f"sa_proj_{suffix}", TensorProto.FLOAT,
                                      proj_shape),
        helper.make_tensor_value_info(f"sa_a4_{suffix}", TensorProto.FLOAT,
                                      arange_shape),
        helper.make_tensor_value_info(f"sa_pos_{suffix}", TensorProto.FLOAT,
                                      proj_shape),
        helper.make_tensor_value_info(f"sa_maxidx_{suffix}", TensorProto.FLOAT,
                                      scalar),
        helper.make_tensor_value_info(f"sa_shift_{suffix}", TensorProto.FLOAT,
                                      scalar),
        helper.make_tensor_value_info(f"sa_idxf_{suffix}", TensorProto.FLOAT,
                                      arange_shape),
        helper.make_tensor_value_info(f"sa_idxc_{suffix}", TensorProto.FLOAT,
                                      arange_shape),
        helper.make_tensor_value_info(f"sa_idxi_{suffix}", TensorProto.INT64,
                                      arange_shape),
        helper.make_tensor_value_info(f"sa_idx1d_{suffix}", TensorProto.INT64,
                                      [side]),
        helper.make_tensor_value_info(f"sa_gathered_{suffix}",
                                      TensorProto.FLOAT, FULL),
        helper.make_tensor_value_info(f"sa_keepb_{suffix}", TensorProto.BOOL,
                                      arange_shape),
        helper.make_tensor_value_info(f"sa_keepf_{suffix}", TensorProto.FLOAT,
                                      arange_shape),
    ]
    return [
        helper.make_node("ReduceSum", [in_name], [f"sa_mask_{suffix}"],
                         axes=[1], keepdims=1, name=f"sa_sum_{suffix}"),
        helper.make_node("ReduceMax", [f"sa_mask_{suffix}"],
                         [f"sa_proj_{suffix}"], axes=[other], keepdims=1,
                         name=f"sa_projop_{suffix}"),
        helper.make_node("Reshape",
                         [f"sa_arange_{suffix}", f"sa_areshape_{suffix}"],
                         [f"sa_a4_{suffix}"], name=f"sa_resh_{suffix}"),
        helper.make_node("Mul", [f"sa_proj_{suffix}", f"sa_a4_{suffix}"],
                         [f"sa_pos_{suffix}"], name=f"sa_mul_{suffix}"),
        helper.make_node("ReduceMax", [f"sa_pos_{suffix}"],
                         [f"sa_maxidx_{suffix}"], axes=[axis], keepdims=1,
                         name=f"sa_maxop_{suffix}"),
        helper.make_node("Add", [f"sa_maxidx_{suffix}", f"sa_one_{suffix}"],
                         [f"sa_shift_{suffix}"], name=f"sa_addop_{suffix}"),
        helper.make_node("Sub", [f"sa_a4_{suffix}", f"sa_shift_{suffix}"],
                         [f"sa_idxf_{suffix}"], name=f"sa_subop_{suffix}"),
        helper.make_node("Clip",
                         [f"sa_idxf_{suffix}", f"sa_zero_{suffix}",
                          f"sa_max_{suffix}"],
                         [f"sa_idxc_{suffix}"], name=f"sa_clip_{suffix}"),
        helper.make_node("Cast", [f"sa_idxc_{suffix}"], [f"sa_idxi_{suffix}"],
                         to=TensorProto.INT64, name=f"sa_cast_{suffix}"),
        helper.make_node("Reshape",
                         [f"sa_idxi_{suffix}", f"sa_flat_{suffix}"],
                         [f"sa_idx1d_{suffix}"], name=f"sa_flatop_{suffix}"),
        helper.make_node("Gather", [in_name, f"sa_idx1d_{suffix}"],
                         [f"sa_gathered_{suffix}"], axis=axis,
                         name=f"sa_gather_{suffix}"),
        helper.make_node("Greater", [f"sa_a4_{suffix}", f"sa_maxidx_{suffix}"],
                         [f"sa_keepb_{suffix}"], name=f"sa_gt_{suffix}"),
        helper.make_node("Cast", [f"sa_keepb_{suffix}"], [f"sa_keepf_{suffix}"],
                         to=TensorProto.FLOAT, name=f"sa_keepcast_{suffix}"),
        helper.make_node("Mul", [f"sa_gathered_{suffix}", f"sa_keepf_{suffix}"],
                         [out_name], name=f"sa_keepmul_{suffix}"),
    ]


def _build() -> onnx.ModelProto:
    init: list = []
    vi: list = []
    nodes: list = []

    # --- rotations (all land top-left) ---
    # rot270 (cw90) = transpose(flip_v(input))
    nodes += _flip_along(2, "cw", "input", "cw_v", init, vi)
    nodes.append(helper.make_node("Transpose", ["cw_v"], ["rot270"],
                                  perm=[0, 1, 3, 2], name="cw_t"))
    # rot90 (ccw90) = flip_v(transpose(input))
    nodes.append(helper.make_node("Transpose", ["input"], ["ccw_t"],
                                  perm=[0, 1, 3, 2], name="ccw_pre"))
    nodes += _flip_along(2, "ccw", "ccw_t", "rot90", init, vi)
    # rot180 = flip_v(flip_h(input))
    nodes += _flip_along(3, "r180h", "input", "r180_h", init, vi)
    nodes += _flip_along(2, "r180v", "r180_h", "rot180", init, vi)

    vi += [
        helper.make_tensor_value_info("cw_v", TensorProto.FLOAT, FULL),
        helper.make_tensor_value_info("rot270", TensorProto.FLOAT, FULL),
        helper.make_tensor_value_info("ccw_t", TensorProto.FLOAT, FULL),
        helper.make_tensor_value_info("rot90", TensorProto.FLOAT, FULL),
        helper.make_tensor_value_info("r180_h", TensorProto.FLOAT, FULL),
        helper.make_tensor_value_info("rot180", TensorProto.FLOAT, FULL),
        helper.make_tensor_value_info("rot90_lo", TensorProto.FLOAT, FULL),
        helper.make_tensor_value_info("rot180_lo", TensorProto.FLOAT, FULL),
        helper.make_tensor_value_info("left", TensorProto.FLOAT, FULL),
        helper.make_tensor_value_info("right", TensorProto.FLOAT, FULL),
        helper.make_tensor_value_info("right_sh", TensorProto.FLOAT, FULL),
    ]

    # --- assemble quadrants via dynamic shifts ---
    nodes += _shift_along(2, "rot90lo", "rot90", "rot90_lo", init, vi)
    nodes += _shift_along(2, "rot180lo", "rot180", "rot180_lo", init, vi)
    nodes.append(helper.make_node("Add", ["input", "rot90_lo"], ["left"],
                                  name="add_left"))
    nodes.append(helper.make_node("Add", ["rot270", "rot180_lo"], ["right"],
                                  name="add_right"))
    nodes += _shift_along(3, "rightsh", "right", "right_sh", init, vi)
    nodes.append(helper.make_node("Add", ["left", "right_sh"], ["output"],
                                  name="add_out"))

    inputs = [helper.make_tensor_value_info("input", TensorProto.FLOAT, FULL)]
    outputs = [helper.make_tensor_value_info("output", TensorProto.FLOAT, FULL)]
    graph = helper.make_graph(nodes, "rot_tile_aware", inputs, outputs,
                              initializer=init, value_info=vi)
    return helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
        ir_version=IR_VERSION)


def solve_rot_tile_aware(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
