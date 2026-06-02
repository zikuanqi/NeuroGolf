"""Task 7 solver: Diagonal cyclic tiling.

The output tiles the entire grid with a cyclic pattern based on anti-diagonals
(i+j). Each anti-diagonal residue class (i+j) % n gets a unique color,
determined by which color appears on that residue in the input.

Pattern: output[i][j] = color_order[(i+j) % n]
where color_order[r] = the non-zero color that appears at residue r in the input.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from ..grids import CHANNELS, HEIGHT, WIDTH, all_examples, to_onehot

OPSET = 11
IR_VERSION = 8


# ----- Reference NumPy implementation (also used for detection) -------------

def _solve_array(inp: list) -> list:
    """Reference implementation of Task 7."""
    h, w = len(inp), len(inp[0])
    
    # Find unique non-zero colors
    colors = set()
    for row in inp:
        for v in row:
            if v != 0:
                colors.add(v)
    
    if not colors:
        return [[0] * w for _ in range(h)]
    
    colors = sorted(colors)
    n = len(colors)
    
    # For each residue r, find which color appears there
    residue_color = {}  # r -> color
    for r in range(n):
        for i in range(h):
            for j in range(w):
                if (i + j) % n == r and inp[i][j] != 0:
                    residue_color[r] = inp[i][j]
                    break
            if r in residue_color:
                break
    
    # Build color_order by residue index
    color_order = [residue_color[r] for r in range(n)]
    
    # Generate output
    out = [[0] * w for _ in range(h)]
    for i in range(h):
        for j in range(w):
            r = (i + j) % n
            out[i][j] = color_order[r]
    
    return out


def _detect(task: dict) -> bool:
    """Check if this task matches the diagonal tiling pattern."""
    examples = list(all_examples(task))
    if not examples:
        return False
    
    for ex in examples:
        inp, out = ex["input"], ex["output"]
        if len(inp) != len(out) or len(inp[0]) != len(out[0]):
            return False
        if _solve_array(inp) != out:
            return False
    
    # Must have non-zero colors (not just identity)
    has_nonzero = any(
        any(v != 0 for row in ex["input"] for v in row)
        for ex in examples
    )
    return has_nonzero


# ----- ONNX graph -----------------------------------------------------------

def _build() -> onnx.ModelProto:
    """Build the ONNX graph for diagonal tiling.

    The graph:
    1. Creates 3 pre-computed diagonal residue masks (mod 3 of i+j)
    2. For each residue, finds which color appears there in the input
    3. Creates the output by assigning each residue's color to its mask positions
    """
    def i64(name, data):
        return numpy_helper.from_array(np.array(data, dtype=np.int64), name)

    def f32(name, data):
        return numpy_helper.from_array(np.array(data, dtype=np.float32), name)

    init = []
    nodes = []
    value_info = []

    # ---- Pre-compute residue masks ----
    # mask_r0[i][j] = 1.0 if (i+j)%3 == 0 else 0.0
    # mask_r1[i][j] = 1.0 if (i+j)%3 == 1 else 0.0
    # mask_r2[i][j] = 1.0 if (i+j)%3 == 2 else 0.0
    for r in range(3):
        mask = np.zeros((1, 1, HEIGHT, WIDTH), dtype=np.float32)
        for i in range(HEIGHT):
            for j in range(WIDTH):
                if (i + j) % 3 == r:
                    mask[0, 0, i, j] = 1.0
        init.append(f32(f"mask_r{r}", mask))
        value_info.append(
            helper.make_tensor_value_info(
                f"mask_r{r}", TensorProto.FLOAT, [1, 1, HEIGHT, WIDTH]))

    # ---- Slice input to exclude channel 0 (background) ----
    # Input: [1, 10, 30, 30] → slice channels [1:10] → [1, 9, 30, 30]
    init.extend([
        i64("ch1_start", [0, 1, 0, 0]),
        i64("ch1_end", [1, CHANNELS, HEIGHT, WIDTH]),
        i64("ch_axes", [0, 1, 2, 3]),
        i64("ch_step", [1, 1, 1, 1]),
    ])
    nodes.append(helper.make_node(
        "Slice", ["input", "ch1_start", "ch1_end", "ch_axes", "ch_step"],
        ["ch19"], name="slice_ch19"))

    # Shared zero scalar for Greater comparison
    init.append(f32("zero_scalar", [0.0]))

    # ---- For each residue r: find colors that appear there ----
    N_CH = CHANNELS - 1  # 9 channels (1-9)
    for r in range(3):
        # Step 1: Multiply ch19 with residue mask to isolate residue-r cells
        # ch19: [1, 9, 30, 30] * mask_r: [1, 1, 30, 30] → [1, 9, 30, 30]
        nodes.append(helper.make_node(
            "Mul", ["ch19", f"mask_r{r}"], [f"res{r}_cells"],
            name=f"mul_res{r}"))

        # Step 2: ReduceSum over spatial dims → [1, 9, 1, 1]
        nodes.append(helper.make_node(
            "ReduceSum", [f"res{r}_cells"], [f"res{r}_count"],
            axes=[2, 3], keepdims=1, name=f"sum_res{r}"))

        # Step 3: Greater-than-zero to get color indicator
        nodes.append(helper.make_node(
            "Greater", [f"res{r}_count", "zero_scalar"], [f"res{r}_bool"],
            name=f"gt_res{r}"))

        # Step 4: Cast to float
        nodes.append(helper.make_node(
            "Cast", [f"res{r}_bool"], [f"res{r}_indicator"],
            to=TensorProto.FLOAT, name=f"cast_res{r}"))

        # Step 5: Expand to full grid by multiplying with residue mask
        # res_r_indicator: [1, 9, 1, 1] * mask_r: [1, 1, 30, 30] → [1, 9, 30, 30]
        nodes.append(helper.make_node(
            "Mul", [f"res{r}_indicator", f"mask_r{r}"], [f"res{r}_output"],
            name=f"expand_res{r}"))

    # ---- Combine all residue outputs → [1, 9, 30, 30] ----
    nodes.append(helper.make_node(
        "Add", ["res0_output", "res1_output"], ["out01"],
        name="add_01"))
    nodes.append(helper.make_node(
        "Add", ["out01", "res2_output"], ["out19"],
        name="add_012"))

    # ---- Create zero channel 0 and concat → [1, 10, 30, 30] ----
    init.append(f32("ch0_zero", np.zeros((1, 1, HEIGHT, WIDTH), dtype=np.float32)))
    nodes.append(helper.make_node(
        "Concat", ["ch0_zero", "out19"], ["full_output"],
        axis=1, name="concat_ch0"))

    # ---- Compute bounding box mask from input ----
    # Step B1: sum all channels → [1, 1, 30, 30] presence map
    nodes.append(helper.make_node(
        "ReduceSum", ["input"], ["presence_map"],
        axes=[1], keepdims=1, name="sum_all_ch"))

    # Step B2: threshold to binary → [1, 1, 30, 30] (bool)
    nodes.append(helper.make_node(
        "Greater", ["presence_map", "zero_scalar"], ["nonzero_mask_bool"],
        name="gt_presence"))

    # Cast to float for ReduceMax compatibility
    nodes.append(helper.make_node(
        "Cast", ["nonzero_mask_bool"], ["nonzero_mask"],
        to=TensorProto.FLOAT, name="cast_nzmask"))

    # Step B3: Find rows with any non-zero → [1, 1, 30, 1]
    nodes.append(helper.make_node(
        "ReduceMax", ["nonzero_mask"], ["row_presence"],
        axes=[3], keepdims=1, name="rmax_cols"))

    # Step B4: Find cols with any non-zero → [1, 1, 1, 30]
    nodes.append(helper.make_node(
        "ReduceMax", ["nonzero_mask"], ["col_presence"],
        axes=[2], keepdims=1, name="rmax_rows"))

    # Step B5: Create bounding box mask via Min
    # row_presence: [1, 1, 30, 1], col_presence: [1, 1, 1, 30]
    # Min with broadcast → [1, 1, 30, 30]
    nodes.append(helper.make_node(
        "Min", ["row_presence", "col_presence"], ["bbox_mask"],
        name="min_bbox"))

    # Step B6: Multiply output with bounding box mask
    # bbox_mask: [1, 1, 30, 30], full_output: [1, 10, 30, 30] → broadcast
    nodes.append(helper.make_node(
        "Mul", ["full_output", "bbox_mask"], ["output"],
        name="apply_bbox"))

    inputs = [helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, CHANNELS, HEIGHT, WIDTH])]
    outputs = [helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, CHANNELS, HEIGHT, WIDTH])]
    
    graph = helper.make_graph(
        nodes, "diag_tile", inputs, outputs,
        initializer=init, value_info=value_info)
    return helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", OPSET)],
        ir_version=IR_VERSION)


def solve_diag_tile(task: dict) -> onnx.ModelProto | None:
    if not _detect(task):
        return None
    return _build()