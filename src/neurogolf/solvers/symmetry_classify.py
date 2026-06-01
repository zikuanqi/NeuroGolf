"""Symmetry classification solver for 1x1 output tasks.

Detects whether the input grid has both horizontal and vertical (h&v) symmetry,
and outputs a single pixel at (0,0): color 1 if symmetric, color 7 otherwise.

Works by slicing the 30x30 canvas to extract the compact content region (3x3
for task 103), then performing symmetry detection via Gather-based flips +
Sub+Abs+ReduceMax+Clip on that smaller region. This avoids the content-shift
problem that plagues full-canvas Gather flips.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import onnx
from onnx import TensorProto, helper

from ..grids import CHANNELS, HEIGHT, WIDTH, all_examples
from ..onnx_ops import IR_VERSION, OPSET_IMPORTS, DATA_TYPE, GRID_SHAPE


def _vt(name: str, shape: list[int]) -> onnx.ValueInfoProto:
    """Shorthand for building a value_info with the standard data type."""
    return helper.make_tensor_value_info(name, DATA_TYPE, shape)


def _check_hv_symmetry_mapping(task: dict) -> bool:
    """Verify the task follows: h&v symmetric -> color 1, else -> color 7.

    Returns True only if *every* example (train + test + arc-gen) obeys this
    mapping and at least one symmetric and one non-symmetric example exist.
    """
    saw_sym = False
    saw_asym = False
    for ex in all_examples(task):
        inp, out = ex["input"], ex["output"]
        if not inp or not out:
            return False
        if len(out) != 1 or len(out[0]) != 1:
            return False
        h = len(inp)
        w = len(inp[0]) if h > 0 else 0
        h_sym = True
        v_sym = True
        for r in range(h):
            for c in range(w):
                if inp[r][c] != inp[r][w - 1 - c]:
                    h_sym = False
                if inp[r][c] != inp[h - 1 - r][c]:
                    v_sym = False
        is_sym = h_sym and v_sym
        expected = 1 if is_sym else 7
        if out[0][0] != expected:
            return False
        if is_sym:
            saw_sym = True
        else:
            saw_asym = True
    return saw_sym and saw_asym


def _derive_grid_size(task: dict) -> tuple[int, int]:
    """Derive the content grid size (H, W) from task examples."""
    for ex in all_examples(task):
        inp = ex["input"]
        if inp:
            return len(inp), len(inp[0])
    return 0, 0


def solve_symmetry_classify(task: dict) -> Optional[onnx.ModelProto]:
    """Build an ONNX model for h&v symmetry classification (task 103)."""
    if not _check_hv_symmetry_mapping(task):
        return None

    gh, gw = _derive_grid_size(task)
    if gh == 0 or gw == 0 or gh > HEIGHT or gw > WIDTH:
        return None

    # --- Initializers ---

    # Gather indices for flipping within the content region
    rev_h = np.arange(gh - 1, -1, -1, dtype=np.int64)
    rev_w = np.arange(gw - 1, -1, -1, dtype=np.int64)

    rev_h_init = helper.make_tensor(
        "rev_h", TensorProto.INT64, [gh], rev_h.tolist(),
    )
    rev_w_init = helper.make_tensor(
        "rev_w", TensorProto.INT64, [gw], rev_w.tolist(),
    )

    # Slice parameters as initializers (opset 10: starts/ends/axes are inputs)
    slice_starts = helper.make_tensor(
        "slice_starts", TensorProto.INT64, [4], [0, 0, 0, 0],
    )
    slice_ends = helper.make_tensor(
        "slice_ends", TensorProto.INT64, [4], [1, CHANNELS, gh, gw],
    )
    slice_axes = helper.make_tensor(
        "slice_axes", TensorProto.INT64, [4], [0, 1, 2, 3],
    )

    # Scalar 1.0 for computing (1 - s)
    one = helper.make_tensor("one", TensorProto.FLOAT, [1], [1.0])

    # color1: one-hot tensor with color 1 at spatial position (0,0)
    color1_arr = np.zeros((1, CHANNELS, HEIGHT, WIDTH), dtype=np.float32)
    color1_arr[0, 1, 0, 0] = 1.0
    color1_init = helper.make_tensor(
        "color1", TensorProto.FLOAT, list(color1_arr.shape), color1_arr.flatten(),
    )

    # color7: one-hot tensor with color 7 at spatial position (0,0)
    color7_arr = np.zeros((1, CHANNELS, HEIGHT, WIDTH), dtype=np.float32)
    color7_arr[0, 7, 0, 0] = 1.0
    color7_init = helper.make_tensor(
        "color7", TensorProto.FLOAT, list(color7_arr.shape), color7_arr.flatten(),
    )

    # Shape of the sliced content region
    SLICE_SHAPE = [1, CHANNELS, gh, gw]

    # --- Nodes ---

    nodes = [
        # ---- Slice to extract content region ----
        helper.make_node(
            "Slice", ["input", "slice_starts", "slice_ends", "slice_axes"],
            ["small"],
        ),

        # ---- Horizontal symmetry check ----
        helper.make_node("Gather", ["small", "rev_w"], ["h_flip"], axis=3),
        helper.make_node("Sub", ["small", "h_flip"], ["h_diff"]),
        helper.make_node("Abs", ["h_diff"], ["h_abs"]),
        helper.make_node("ReduceMax", ["h_abs"], ["h_max"],
                         axes=[1, 2, 3], keepdims=1),
        helper.make_node("Clip", ["h_max"], ["h_asym"], min=0.0, max=1.0),
        helper.make_node("Sub", ["one", "h_asym"], ["h_sym"]),

        # ---- Vertical symmetry check ----
        helper.make_node("Gather", ["small", "rev_h"], ["v_flip"], axis=2),
        helper.make_node("Sub", ["small", "v_flip"], ["v_diff"]),
        helper.make_node("Abs", ["v_diff"], ["v_abs"]),
        helper.make_node("ReduceMax", ["v_abs"], ["v_max"],
                         axes=[1, 2, 3], keepdims=1),
        helper.make_node("Clip", ["v_max"], ["v_asym"], min=0.0, max=1.0),
        helper.make_node("Sub", ["one", "v_asym"], ["v_sym"]),

        # ---- Combine: s = h_sym AND v_sym ----
        helper.make_node("Mul", ["h_sym", "v_sym"], ["s"]),

        # ---- not_s = 1 - s ----
        helper.make_node("Sub", ["one", "s"], ["not_s"]),

        # ---- output = s*color1 + (1-s)*color7 ----
        helper.make_node("Mul", ["s", "color1"], ["out1"]),
        helper.make_node("Mul", ["not_s", "color7"], ["out7"]),
        helper.make_node("Add", ["out1", "out7"], ["output"]),
    ]

    # --- Value info for intermediate tensors ---
    value_info = [
        _vt("small", SLICE_SHAPE),
        _vt("h_flip", SLICE_SHAPE),
        _vt("h_diff", SLICE_SHAPE),
        _vt("h_abs", SLICE_SHAPE),
        _vt("h_max", [1, 1, 1, 1]),
        _vt("h_asym", [1, 1, 1, 1]),
        _vt("h_sym", [1, 1, 1, 1]),
        _vt("v_flip", SLICE_SHAPE),
        _vt("v_diff", SLICE_SHAPE),
        _vt("v_abs", SLICE_SHAPE),
        _vt("v_max", [1, 1, 1, 1]),
        _vt("v_asym", [1, 1, 1, 1]),
        _vt("v_sym", [1, 1, 1, 1]),
        _vt("s", [1, 1, 1, 1]),
        _vt("not_s", [1, 1, 1, 1]),
        _vt("out1", GRID_SHAPE),
        _vt("out7", GRID_SHAPE),
    ]

    # --- Assemble graph ---
    x = helper.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = helper.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    graph = helper.make_graph(
        nodes, "symmetry_classify", [x], [y],
        [
            rev_h_init, rev_w_init,
            slice_starts, slice_ends, slice_axes,
            one, color1_init, color7_init,
        ],
        value_info=value_info,
    )
    return helper.make_model(
        graph, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS,
    )