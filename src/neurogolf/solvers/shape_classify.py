"""Shape classification solver for 3x3 / NxN 1x1 output tasks.

Detects spatial patterns in small grids and maps them to output colors
using a feature hash that is invariant to the specific input colors.

Approach:
- Slice content region, skip channel 0 (background)
- ReduceSum over channels -> ink map (1 where any non-bg color exists)
- Flatten + weighted sum -> spatial hash
- Map hash -> output color via Equal+Cast+Mul+Add chain
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from ..grids import CHANNELS, HEIGHT, WIDTH, all_examples

OPSET = 10
IR_VERSION_V = 8
DATA_TYPE = TensorProto.FLOAT
GRID_SHAPE = [1, CHANNELS, HEIGHT, WIDTH]

HASH_WEIGHTS = [1, 2, 4, 8, 16, 32, 64, 128, 256]


def _check_task(task: dict) -> bool:
    """Check if task has 3x3 inputs with single-color non-bg pixels and 1x1 outputs."""
    for ex in all_examples(task):
        inp, out = ex["input"], ex["output"]
        if len(inp) != 3 or any(len(row) != 3 for row in inp):
            return False
        if len(out) != 1 or len(out[0]) != 1:
            return False
        colors = set()
        nz = 0
        for row in inp:
            for v in row:
                if v != 0:
                    colors.add(v)
                    nz += 1
        if len(colors) != 1:
            return False
    return True


def _build_pattern_map(task: dict) -> dict[int, int]:
    """Map hash values to output colors from train examples."""
    pmap: dict[int, int] = {}
    for ex in task["train"]:
        inp, out = ex["input"], ex["output"]
        h = sum(HASH_WEIGHTS[r * 3 + c]
                for r, row in enumerate(inp)
                for c, v in enumerate(row) if v != 0)
        pmap[h] = out[0][0]
    return pmap


def _verify_rule(task: dict) -> bool:
    """Verify that the hash->color mapping works for all examples."""
    pmap = _build_pattern_map(task)
    for ex in all_examples(task):
        inp, out = ex["input"], ex["output"]
        h = sum(HASH_WEIGHTS[r * 3 + c]
                for r, row in enumerate(inp)
                for c, v in enumerate(row) if v != 0)
        if pmap.get(h) != out[0][0]:
            return False
    return True


def solve_shape_classify(task: dict) -> Optional[onnx.ModelProto]:
    if not _check_task(task):
        return None
    if not _verify_rule(task):
        return None

    pmap = _build_pattern_map(task)
    candidates = sorted(pmap.items())

    def _i64(name, vals):
        return numpy_helper.from_array(np.array(vals, dtype=np.int64), name)

    def _f32(name, vals):
        return numpy_helper.from_array(np.array(vals, dtype=np.float32), name)

    nodes = []
    initializers = []

    # Step 1: Slice to 3x3 content, skip channel 0 (background).
    # Channel 0 is used by to_onehot for empty cells; we want only channels 1-9.
    initializers.extend([
        _i64("slice_starts", [0, 1, 0, 0]),
        _i64("slice_ends", [1, CHANNELS, 3, 3]),
        _i64("slice_axes", [0, 1, 2, 3]),
        _i64("slice_steps", [1, 1, 1, 1]),
    ])
    nodes.append(helper.make_node(
        "Slice", ["input", "slice_starts", "slice_ends", "slice_axes", "slice_steps"],
        ["content"]))

    # Step 2: Sum over channels -> ink map (1,1,3,3), then flatten to (1,9)
    nodes.append(helper.make_node(
        "ReduceSum", ["content"], ["ink"], axes=[1], keepdims=0))
    initializers.append(_i64("shape_1x9", [1, 9]))
    nodes.append(helper.make_node(
        "Reshape", ["ink", "shape_1x9"], ["ink_flat"]))

    # Step 3: MatMul with hash weights (9,1) -> (1,1)
    hw = np.array(HASH_WEIGHTS, dtype=np.float32).reshape(9, 1)
    initializers.append(_f32("hash_weights", hw))
    nodes.append(helper.make_node(
        "MatMul", ["ink_flat", "hash_weights"], ["hash_2d"]))

    # Reshape to scalar
    initializers.append(_i64("scalar_shape", [1]))
    nodes.append(helper.make_node(
        "Reshape", ["hash_2d", "scalar_shape"], ["hash_f"]))

    # Step 4: Cast hash to int64 (onnxruntime in this env rejects Equal with float)
    nodes.append(helper.make_node(
        "Cast", ["hash_f"], ["hash_i64"], to=int(TensorProto.INT64)))

    # Step 5: For each (hash_val, color) pair, add contribution at (0,0)
    base = np.zeros((1, CHANNELS, HEIGHT, WIDTH), dtype=np.float32)
    initializers.append(_f32("base_grid", base))
    prev_acc = "base_grid"

    for i, (hval, color) in enumerate(candidates):
        # One-hot canvas for this color at (0,0)
        canvas = np.zeros((1, CHANNELS, HEIGHT, WIDTH), dtype=np.float32)
        canvas[0, color, 0, 0] = 1.0
        initializers.append(_f32(f"canvas_{i}", canvas))
        initializers.append(_i64(f"target_{i}", [hval]))

        nodes.append(helper.make_node(
            "Equal", ["hash_i64", f"target_{i}"], [f"eq_{i}"]))
        nodes.append(helper.make_node(
            "Cast", [f"eq_{i}"], [f"match_{i}"], to=int(TensorProto.FLOAT)))
        nodes.append(helper.make_node(
            "Mul", [f"match_{i}", f"canvas_{i}"], [f"scaled_{i}"]))
        acc_name = "base_grid" if i == 0 else f"acc_{i-1}"
        nodes.append(helper.make_node(
            "Add", [acc_name, f"scaled_{i}"], [f"acc_{i}"]))
        prev_acc = f"acc_{i}"

    nodes.append(helper.make_node("Identity", [prev_acc], ["output"]))

    x = helper.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = helper.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    graph = helper.make_graph(nodes, "shape_classify", [x], [y], initializers)
    return helper.make_model(graph, ir_version=IR_VERSION_V,
                             opset_imports=[helper.make_opsetid("", OPSET)])