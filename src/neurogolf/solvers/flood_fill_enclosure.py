"""Solver: flood-fill enclosure.

Fill color-0 cells that are completely enclosed by a single source color,
recoloring them to a target color.

Task 002: color-0 cells NOT reachable from the grid border without crossing
color-3 walls are recolored to 4.

ONNX has no loops, so we unroll an N-step border flood-fill. Starting from the
30x30 border, repeatedly dilate the "reachable from outside" mask by one cell
(Pad + plus-shaped Conv + Greater) and intersect with the non-wall mask. After
enough iterations the reachable set is stable; color-0 cells that remain
unreached are the enclosed interior. We clear their channel-0 bit and set their
target channel.
"""
from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from ..grids import CHANNELS, HEIGHT, WIDTH

OPSET = 11
IR_VERSION = 8
# Max BFS reach distance on a 30x30 grid is bounded by ~56; this many one-step
# dilations guarantees the reachable mask has converged.
NUM_ITERS = 58


def all_examples(task: dict):
    for ex in task.get("train", []):
        yield ex
    for ex in task.get("test", []):
        yield ex


def _flood_fill(grid, source, target):
    """Reference CPU implementation used to validate the pattern at detect time."""
    H, W = len(grid), len(grid[0])
    visited = [[False] * W for _ in range(H)]
    q: deque = deque()
    for r in range(H):
        for c in (0, W - 1):
            if grid[r][c] != source and not visited[r][c]:
                visited[r][c] = True
                q.append((r, c))
    for c in range(W):
        for r in (0, H - 1):
            if grid[r][c] != source and not visited[r][c]:
                visited[r][c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and not visited[nr][nc] \
                    and grid[nr][nc] != source:
                visited[nr][nc] = True
                q.append((nr, nc))
    result = [row[:] for row in grid]
    for r in range(H):
        for c in range(W):
            if grid[r][c] == 0 and not visited[r][c]:
                result[r][c] = target
    return result


def _detect(task: dict) -> Optional[dict]:
    examples = list(all_examples(task))
    if not examples:
        return None

    first = examples[0]
    inp0, out0 = first["input"], first["output"]
    h0, w0 = len(inp0), len(inp0[0])
    if len(out0) != h0 or len(out0[0]) != w0:
        return None

    # Only color-0 cells may change, and they all change to one target color.
    target_val = None
    for r in range(h0):
        for c in range(w0):
            if inp0[r][c] != out0[r][c]:
                if inp0[r][c] != 0:
                    return None
                if target_val is None:
                    target_val = out0[r][c]
                elif out0[r][c] != target_val:
                    return None
    if target_val is None or target_val == 0:
        return None

    # Exactly one non-zero, non-target color acts as the wall/source.
    sources = set()
    for r in range(h0):
        for c in range(w0):
            v = inp0[r][c]
            if v != 0 and v != target_val:
                sources.add(v)
    if len(sources) != 1:
        return None
    source_val = next(iter(sources))

    for ex in examples:
        inp, out = ex["input"], ex["output"]
        if len(inp) != len(out) or (inp and len(inp[0]) != len(out[0])):
            return None
        if _flood_fill(inp, source_val, target_val) != out:
            return None

    return {"source_val": int(source_val), "target_val": int(target_val)}


def _build(config: dict) -> onnx.ModelProto:
    source_val = config["source_val"]
    target_idx = config["target_val"]

    def f32(name, vals):
        return numpy_helper.from_array(np.array(vals, dtype=np.float32), name)

    def i64(name, vals):
        return numpy_helper.from_array(np.array(vals, dtype=np.int64), name)

    kernel = np.array([[0., 1., 0.], [1., 0., 1.], [0., 1., 0.]],
                      dtype=np.float32).reshape(1, 1, 3, 3)
    edge = np.zeros((1, 1, HEIGHT, WIDTH), dtype=np.float32)
    edge[0, 0, 0, :] = 1.0
    edge[0, 0, :, 0] = 1.0
    edge[0, 0, HEIGHT - 1, :] = 1.0
    edge[0, 0, :, WIDTH - 1] = 1.0
    one_4d = np.ones((1, 1, HEIGHT, WIDTH), dtype=np.float32)

    init = [
        f32("kernel_W", kernel),
        f32("edge_mask", edge),
        f32("zero", [0.0]),
        f32("one_f", [1.0]),
        f32("one_4d", one_4d),
        f32("source_c", [float(source_val)]),
        i64("pads", [0, 0, 1, 1, 0, 0, 1, 1]),  # 4D Pad => 2*rank = 8 values
        i64("idx0", [0]),
        i64("idxT", [target_idx]),
        i64("s_mid", [1]),
        i64("e_mid", [target_idx]),
        i64("s_aft", [target_idx + 1]),
        i64("e_aft", [CHANNELS]),
        i64("axes_c", [1]),
        i64("step_1", [1]),
    ]

    nodes = [
        # one-hot -> color index; in-grid cells map to their color, out-of-grid
        # padding is all-zero so ArgMax gives 0 (never equals a non-zero source).
        helper.make_node("ArgMax", ["input"], ["argmax"], axis=1, keepdims=0),
        helper.make_node("Cast", ["argmax"], ["argmax_f"], to=TensorProto.FLOAT),
        helper.make_node("Equal", ["argmax_f", "source_c"], ["wall_b"]),
        helper.make_node("Cast", ["wall_b"], ["wall"], to=TensorProto.FLOAT),
        helper.make_node("Sub", ["one_f", "wall"], ["non_wall"]),
        helper.make_node("Unsqueeze", ["non_wall"], ["non_wall_4d"], axes=[1]),
        helper.make_node("Mul", ["edge_mask", "non_wall_4d"], ["outside_0"]),
    ]

    prev = "outside_0"
    for i in range(NUM_ITERS):
        nodes += [
            helper.make_node("Pad", [prev, "pads", "zero"], [f"pad_{i}"],
                             mode="constant"),
            helper.make_node("Conv", [f"pad_{i}", "kernel_W"], [f"conv_{i}"],
                             kernel_shape=[3, 3]),
            helper.make_node("Greater", [f"conv_{i}", "zero"], [f"dil_{i}"]),
            helper.make_node("Cast", [f"dil_{i}"], [f"dilf_{i}"],
                             to=TensorProto.FLOAT),
            helper.make_node("Mul", [f"dilf_{i}", "non_wall_4d"], [f"reach_{i}"]),
            helper.make_node("Max", [prev, f"reach_{i}"], [f"outside_{i + 1}"]),
        ]
        prev = f"outside_{i + 1}"

    nodes += [
        helper.make_node("Gather", ["input", "idx0"], ["ch0"], axis=1),
        helper.make_node("Sub", ["one_4d", prev], ["not_outside"]),
        helper.make_node("Mul", ["ch0", "not_outside"], ["inside"]),
        helper.make_node("Sub", ["ch0", "inside"], ["ch0_new"]),
        helper.make_node("Gather", ["input", "idxT"], ["tc"], axis=1),
        helper.make_node("Add", ["tc", "inside"], ["tc_new"]),
    ]

    # Reassemble 10 channels: [ch0_new] [1..T-1] [tc_new] [T+1..9].
    parts = ["ch0_new"]
    if target_idx > 1:
        nodes.append(helper.make_node(
            "Slice", ["input", "s_mid", "e_mid", "axes_c", "step_1"], ["mid"]))
        parts.append("mid")
    parts.append("tc_new")
    if target_idx < CHANNELS - 1:
        nodes.append(helper.make_node(
            "Slice", ["input", "s_aft", "e_aft", "axes_c", "step_1"], ["aft"]))
        parts.append("aft")
    nodes.append(helper.make_node("Concat", parts, ["output"], axis=1))

    inputs = [helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, CHANNELS, HEIGHT, WIDTH])]
    outputs = [helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, CHANNELS, HEIGHT, WIDTH])]
    graph = helper.make_graph(nodes, "flood_fill", inputs, outputs,
                              initializer=init)
    return helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
        ir_version=IR_VERSION)


def solve_flood_fill_enclosure(task: dict) -> Optional[onnx.ModelProto]:
    config = _detect(task)
    if config is None:
        return None
    return _build(config)
