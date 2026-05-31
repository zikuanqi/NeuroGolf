"""Solver: keep every marker and ring its background neighbours with colour 1.

Each non-background cell is preserved; every background cell that touches a
marker in its 3x3 neighbourhood is repainted colour 1, forming a one-pixel halo
around each shape. Background cells with no marker nearby stay background
(task 95).

One-hot channel arithmetic on the (1, 10, 30, 30) tensor:
  nonbg  = content - channel-0                 (1 on marker cells)
  cnt    = 3x3 box convolution of nonbg        (markers in the neighbourhood)
  dilate = cnt > 0                             (cells within one step of a marker)
  border = real background cell AND dilated    -> colour 1
  output = markers (own colour) + border (1) + remaining background (0)
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from ..grids import CHANNELS, HEIGHT, WIDTH, all_examples

OPSET = 11
IR_VERSION = 8
HALO_COLOR = 1


def _transform(grid):
    g = np.array(grid)
    H, W = g.shape
    out = [[0] * W for _ in range(H)]
    for r in range(H):
        for c in range(W):
            if g[r, c] != 0:
                out[r][c] = int(g[r, c])
                continue
            near = False
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < H and 0 <= cc < W and g[rr, cc] != 0:
                        near = True
            if near:
                out[r][c] = HALO_COLOR
    return out


def _detect(task: dict) -> bool:
    examples = list(all_examples(task))
    if not examples:
        return False
    changed = False
    for ex in examples:
        inp, out = ex["input"], ex["output"]
        if len(inp) > 30 or len(inp[0]) > 30:
            continue
        if len(inp) != len(out) or len(inp[0]) != len(out[0]):
            return False
        if _transform(inp) != out:
            return False
        if inp != out:
            changed = True
    return changed


def _build() -> onnx.ModelProto:
    F = TensorProto.FLOAT
    B = TensorProto.BOOL
    I = TensorProto.INT64

    def f32(name, arr):
        return numpy_helper.from_array(arr.astype(np.float32), name)

    e0 = np.zeros((1, CHANNELS, 1, 1), dtype=np.float32)
    e0[0, 0, 0, 0] = 1.0
    e1 = np.zeros((1, CHANNELS, 1, 1), dtype=np.float32)
    e1[0, HALO_COLOR, 0, 0] = 1.0
    ones_k = np.ones((1, 1, 3, 3), dtype=np.float32)

    init = [
        f32("e0", e0), f32("e1", e1), f32("ones_k", ones_k),
        f32("one", np.array([[[[1.0]]]])),
        f32("half", np.array([[[[0.5]]]])),
        numpy_helper.from_array(np.array([0], dtype=np.int64), "idx0"),
    ]

    n = helper.make_node
    nodes = [
        n("Gather", ["input", "idx0"], ["ch0"], axis=1),
        n("ReduceSum", ["input"], ["content"], axes=[1], keepdims=1),
        n("Sub", ["content", "ch0"], ["nonbg"]),
        n("Conv", ["nonbg", "ones_k"], ["cnt"],
          kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        n("Greater", ["cnt", "half"], ["dil_b"]),
        n("Cast", ["dil_b"], ["dilated"], to=F),
        n("Sub", ["one", "nonbg"], ["notnonbg"]),
        n("Mul", ["content", "notnonbg"], ["realbg"]),
        n("Mul", ["realbg", "dilated"], ["border_mask"]),
        n("Sub", ["one", "dilated"], ["notdil"]),
        n("Mul", ["realbg", "notdil"], ["farbg_mask"]),
        n("Mul", ["input", "nonbg"], ["out_markers"]),
        n("Mul", ["e1", "border_mask"], ["out_border"]),
        n("Mul", ["e0", "farbg_mask"], ["out_farbg"]),
        n("Add", ["out_markers", "out_border"], ["tmp_add"]),
        n("Add", ["tmp_add", "out_farbg"], ["output"]),
    ]

    def vi(name, shape, dt=F):
        return helper.make_tensor_value_info(name, dt, shape)

    g4 = [1, CHANNELS, HEIGHT, WIDTH]
    s1 = [1, 1, HEIGHT, WIDTH]
    value_info = [
        vi("ch0", s1), vi("content", s1), vi("nonbg", s1), vi("cnt", s1),
        vi("dil_b", s1, B), vi("dilated", s1), vi("notnonbg", s1),
        vi("realbg", s1), vi("border_mask", s1), vi("notdil", s1),
        vi("farbg_mask", s1),
        vi("out_markers", g4), vi("out_border", g4), vi("out_farbg", g4),
        vi("tmp_add", g4),
    ]

    inputs = [vi("input", g4)]
    outputs = [vi("output", g4)]
    graph = helper.make_graph(nodes, "halo", inputs, outputs,
                              initializer=init, value_info=value_info)
    return helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
        ir_version=IR_VERSION)


def solve_halo(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
