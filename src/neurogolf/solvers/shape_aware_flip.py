"""Shape-aware horizontal / vertical / 180-degree flips.

The simple "Gather with indices [29, 28, ..., 0]" trick fails here because ARC
grids are top-left-aligned inside the 30x30 canvas. Reversing all 30 positions
shifts the content to the opposite edge instead of flipping it in place.

This solver finds the content extent at runtime:

  1. mask  = ReduceSum(input, axis=channel)        — 1 where any color, 0 else.
  2. proj  = ReduceMax(mask, axis=spatial_other)   — per-axis presence vector.
  3. W-1   = ReduceMax(proj * arange)              — largest occupied index.
  4. idx   = Clip(W-1 - arange, 0, 29)             — gather indices for flip.
  5. out   = Gather(input, idx) * proj             — flipped & masked back.

Step 5's mask multiplication zeros out cells that the clipped gather would
otherwise duplicate from column 0.

Three solvers share the same machinery, differing only in which axis is
flipped (`solve_flip_h_aware` / `solve_flip_v_aware` / `solve_rot180_aware`).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from ..grids import CHANNELS, HEIGHT, WIDTH, all_examples

OPSET = 11
IR_VERSION = 8


# ---------- pattern detectors ----------

def _check_same_shape(task: dict) -> bool:
    saw = False
    for ex in all_examples(task):
        i, o = ex["input"], ex["output"]
        if not i or not o:
            return False
        if len(i) != len(o) or len(i[0]) != len(o[0]):
            return False
        saw = True
    return saw


def _is_flip_h(task: dict) -> bool:
    if not _check_same_shape(task):
        return False
    for ex in all_examples(task):
        i, o = ex["input"], ex["output"]
        for r, row in enumerate(i):
            w = len(row)
            for c, v in enumerate(row):
                if o[r][w - 1 - c] != v:
                    return False
    return True


def _is_flip_v(task: dict) -> bool:
    if not _check_same_shape(task):
        return False
    for ex in all_examples(task):
        i, o = ex["input"], ex["output"]
        h = len(i)
        for r, row in enumerate(i):
            for c, v in enumerate(row):
                if o[h - 1 - r][c] != v:
                    return False
    return True


def _is_rot180(task: dict) -> bool:
    if not _check_same_shape(task):
        return False
    for ex in all_examples(task):
        i, o = ex["input"], ex["output"]
        h, w = len(i), len(i[0])
        for r, row in enumerate(i):
            for c, v in enumerate(row):
                if o[h - 1 - r][w - 1 - c] != v:
                    return False
    return True


# ---------- graph building blocks ----------

def _make_arange(size: int, name: str) -> onnx.TensorProto:
    return numpy_helper.from_array(
        np.arange(size, dtype=np.float32), name)


def _flip_along(axis: int, suffix: str, input_name: str, output_name: str,
                init_carry: list, vi_carry: list) -> list:
    """Emit the node sequence that flips `input_name` along `axis` (2 or 3).

    All new initializers and value-info entries are appended to `init_carry`
    and `vi_carry` respectively. Returns the list of nodes.
    """
    other = 3 if axis == 2 else 2
    side = HEIGHT if axis == 2 else WIDTH

    arange = _make_arange(side, f"arange_{suffix}")
    clip_min = numpy_helper.from_array(
        np.array(0.0, dtype=np.float32), f"clip_min_{suffix}")
    clip_max = numpy_helper.from_array(
        np.array(float(side - 1), dtype=np.float32), f"clip_max_{suffix}")
    reshape_to_1d = numpy_helper.from_array(
        np.array([side], dtype=np.int64), f"reshape_1d_{suffix}")
    init_carry += [arange, clip_min, clip_max, reshape_to_1d]

    mask = f"mask_{suffix}"
    proj = f"proj_{suffix}"
    proj_pos = f"projpos_{suffix}"
    max_idx_f = f"maxf_{suffix}"
    flip_idx_f = f"flipf_{suffix}"
    flip_idx_clipped = f"flipclip_{suffix}"
    flip_idx_i64 = f"flipi64_{suffix}"
    flip_idx_1d = f"flip1d_{suffix}"
    flipped = f"flipped_{suffix}"

    # Shape bookkeeping for value_info.
    mask_shape = [1, 1, HEIGHT, WIDTH]
    proj_shape = list(mask_shape)
    proj_shape[other] = 1
    proj_shape[1] = 1  # already 1
    pos_shape = proj_shape
    max_shape = [1] * 4
    max_shape[other] = 1
    # ReduceMax keepdims=1 over `axis`: that dim becomes 1; the `other` dim
    # already is 1 from the earlier reduction, so all dims except `1` channel
    # are 1.
    flip_shape = proj_shape  # same as proj since arange broadcasts along axis

    vi_carry += [
        helper.make_tensor_value_info(mask, TensorProto.FLOAT, mask_shape),
        helper.make_tensor_value_info(proj, TensorProto.FLOAT, proj_shape),
        helper.make_tensor_value_info(proj_pos, TensorProto.FLOAT, pos_shape),
        helper.make_tensor_value_info(max_idx_f, TensorProto.FLOAT,
                                      [1, 1, 1, 1]),
        helper.make_tensor_value_info(flip_idx_f, TensorProto.FLOAT,
                                      flip_shape),
        helper.make_tensor_value_info(flip_idx_clipped, TensorProto.FLOAT,
                                      flip_shape),
        helper.make_tensor_value_info(flip_idx_i64, TensorProto.INT64,
                                      flip_shape),
        helper.make_tensor_value_info(flip_idx_1d, TensorProto.INT64,
                                      [side]),
        helper.make_tensor_value_info(flipped, TensorProto.FLOAT,
                                      [1, CHANNELS, HEIGHT, WIDTH]),
    ]

    # Reshape arange to broadcast against proj along `axis`.
    arange_shape = [1, 1, 1, 1]
    arange_shape[axis] = side
    arange_4d = f"arange4d_{suffix}"
    arange_4d_reshape_init = numpy_helper.from_array(
        np.array(arange_shape, dtype=np.int64), f"arange_resh_{suffix}")
    init_carry += [arange_4d_reshape_init]
    vi_carry.append(helper.make_tensor_value_info(
        arange_4d, TensorProto.FLOAT, arange_shape))

    nodes = [
        # Sum across channels → per-cell "any color present" (still float).
        helper.make_node(
            "ReduceSum", [input_name], [mask],
            axes=[1], keepdims=1, name=f"sum_ch_{suffix}",
        ),
        # Max across the OTHER spatial axis → per-axis presence vector.
        helper.make_node(
            "ReduceMax", [mask], [proj],
            axes=[other], keepdims=1, name=f"proj_{suffix}",
        ),
        helper.make_node(
            "Reshape", [f"arange_{suffix}", f"arange_resh_{suffix}"],
            [arange_4d], name=f"reshape_arange_{suffix}",
        ),
        # Position = presence * index along the flip axis.
        helper.make_node(
            "Mul", [proj, arange_4d], [proj_pos],
            name=f"mulpos_{suffix}",
        ),
        # Largest occupied index along the flip axis.
        helper.make_node(
            "ReduceMax", [proj_pos], [max_idx_f],
            axes=[axis], keepdims=1, name=f"argmaxf_{suffix}",
        ),
        # idx[c] = (W-1) - c (still float; clipped next).
        helper.make_node(
            "Sub", [max_idx_f, arange_4d], [flip_idx_f],
            name=f"sub_{suffix}",
        ),
        helper.make_node(
            "Clip",
            [flip_idx_f, f"clip_min_{suffix}", f"clip_max_{suffix}"],
            [flip_idx_clipped], name=f"clip_{suffix}",
        ),
        helper.make_node(
            "Cast", [flip_idx_clipped], [flip_idx_i64],
            to=TensorProto.INT64, name=f"cast_{suffix}",
        ),
        # Squeeze to 1D for Gather.
        helper.make_node(
            "Reshape", [flip_idx_i64, f"reshape_1d_{suffix}"], [flip_idx_1d],
            name=f"squeeze_{suffix}",
        ),
        helper.make_node(
            "Gather", [input_name, flip_idx_1d], [flipped],
            axis=axis, name=f"gather_{suffix}",
        ),
        # Mask out cells past the content extent.
        helper.make_node(
            "Mul", [flipped, proj], [output_name],
            name=f"maskout_{suffix}",
        ),
    ]
    return nodes


def _build_flip(axis: int) -> onnx.ModelProto:
    """A network that flips the input along a single spatial axis."""
    initializers: list = []
    value_info: list = []
    nodes = _flip_along(axis, "f", "input", "output",
                        initializers, value_info)

    inputs = [helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, CHANNELS, HEIGHT, WIDTH])]
    outputs = [helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, CHANNELS, HEIGHT, WIDTH])]
    graph = helper.make_graph(
        nodes, f"flip_axis_{axis}", inputs, outputs,
        initializer=initializers, value_info=value_info,
    )
    return helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
        ir_version=IR_VERSION,
    )


def _build_rot180() -> onnx.ModelProto:
    """Compose horizontal then vertical flip."""
    initializers: list = []
    value_info: list = []
    nodes = _flip_along(3, "h", "input", "_horiz",
                        initializers, value_info)
    nodes += _flip_along(2, "v", "_horiz", "output",
                         initializers, value_info)
    value_info.append(helper.make_tensor_value_info(
        "_horiz", TensorProto.FLOAT, [1, CHANNELS, HEIGHT, WIDTH]))

    inputs = [helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, CHANNELS, HEIGHT, WIDTH])]
    outputs = [helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, CHANNELS, HEIGHT, WIDTH])]
    graph = helper.make_graph(
        nodes, "rot180", inputs, outputs,
        initializer=initializers, value_info=value_info,
    )
    return helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
        ir_version=IR_VERSION,
    )


# ---------- 90° rotations (transpose + shape-aware flip) ----------

def _is_rot90_ccw(task: dict) -> bool:
    """output[r][c] = input[c][W-1-r]; output shape = (W, H)."""
    saw = False
    for ex in all_examples(task):
        i, o = ex["input"], ex["output"]
        if not i or not o:
            return False
        H, W = len(i), len(i[0])
        if len(o) != W or len(o[0]) != H:
            return False
        for r in range(W):
            for c in range(H):
                if o[r][c] != i[c][W - 1 - r]:
                    return False
        saw = True
    return saw


def _is_rot90_cw(task: dict) -> bool:
    """output[r][c] = input[H-1-c][r]; output shape = (W, H)."""
    saw = False
    for ex in all_examples(task):
        i, o = ex["input"], ex["output"]
        if not i or not o:
            return False
        H, W = len(i), len(i[0])
        if len(o) != W or len(o[0]) != H:
            return False
        for r in range(W):
            for c in range(H):
                if o[r][c] != i[H - 1 - c][r]:
                    return False
        saw = True
    return saw


def _build_rot90(direction: str) -> onnx.ModelProto:
    """Transpose composed with a shape-aware flip.

    rot90 CCW (output[r,c] = input[c, W-1-r]):
        transpose first, then flip rows of the result based on the new
        content height (which equals the original content width W).

    rot90 CW (output[r,c] = input[H-1-c, r]):
        flip rows of the input based on content height H, then transpose.
    """
    initializers: list = []
    value_info: list = []
    if direction == "ccw":
        nodes = [helper.make_node(
            "Transpose", ["input"], ["_t"], perm=[0, 1, 3, 2],
            name="pre_transpose")]
        value_info.append(helper.make_tensor_value_info(
            "_t", TensorProto.FLOAT, [1, CHANNELS, HEIGHT, WIDTH]))
        nodes += _flip_along(2, "v", "_t", "output",
                             initializers, value_info)
        graph_name = "rot90_ccw"
    else:
        nodes = _flip_along(2, "v", "input", "_v",
                            initializers, value_info)
        value_info.append(helper.make_tensor_value_info(
            "_v", TensorProto.FLOAT, [1, CHANNELS, HEIGHT, WIDTH]))
        nodes.append(helper.make_node(
            "Transpose", ["_v"], ["output"], perm=[0, 1, 3, 2],
            name="post_transpose"))
        graph_name = "rot90_cw"

    inputs = [helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, CHANNELS, HEIGHT, WIDTH])]
    outputs = [helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, CHANNELS, HEIGHT, WIDTH])]
    graph = helper.make_graph(
        nodes, graph_name, inputs, outputs,
        initializer=initializers, value_info=value_info,
    )
    return helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
        ir_version=IR_VERSION,
    )


def solve_rot90_ccw_aware(task: dict) -> Optional[onnx.ModelProto]:
    return _build_rot90("ccw") if _is_rot90_ccw(task) else None


def solve_rot90_cw_aware(task: dict) -> Optional[onnx.ModelProto]:
    return _build_rot90("cw") if _is_rot90_cw(task) else None


# ---------- public solvers ----------

def solve_flip_h_aware(task: dict) -> Optional[onnx.ModelProto]:
    return _build_flip(3) if _is_flip_h(task) else None


def solve_flip_v_aware(task: dict) -> Optional[onnx.ModelProto]:
    return _build_flip(2) if _is_flip_v(task) else None


def solve_rot180_aware(task: dict) -> Optional[onnx.ModelProto]:
    if _is_rot180(task):
        return _build_rot180()
    return None
