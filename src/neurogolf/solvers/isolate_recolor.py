"""Solver: recolour cells by whether they are isolated or touching same colour.

Every coloured cell is classified as *isolated* (no orthogonal neighbour of its
own colour) or *connected*, and recoloured through a task-specific map keyed by
(colour, isolated?). Background is untouched (tasks 147, 272). This is the
size-1-vs-larger special case of connected-component recolouring, so it needs
only a depthwise same-colour-neighbour count, not full labelling.

One-hot channel arithmetic on the (1, 10, 30, 30) tensor:
  nbr   = per-channel count of same-colour 4-neighbours (depthwise 3x3 cross)
  iso   = cells with nbr == 0 ; conn = cells with nbr > 0
  output = 1x1 remap of iso (isolated map) + 1x1 remap of conn (connected map)
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from ..grids import CHANNELS, HEIGHT, WIDTH, all_examples


OPSET = 11
IR_VERSION = 8


def _isolated(g, r, c):
    H, W = len(g), len(g[0])
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < H and 0 <= cc < W and g[rr][cc] == g[r][c]:
            return False
    return True


def _transform(grid, m):
    H, W = len(grid), len(grid[0])
    out = [[0] * W for _ in range(H)]
    for r in range(H):
        for c in range(W):
            v = grid[r][c]
            if v == 0:
                continue
            key = (v, _isolated(grid, r, c))
            if key not in m:
                return None
            out[r][c] = m[key]
    return out


def _params(task: dict):
    m = {}
    examples = list(all_examples(task))
    for ex in examples:
        i, o = ex["input"], ex["output"]
        if len(i) > 30 or len(i[0]) > 30:
            continue
        if len(i) != len(o) or len(i[0]) != len(o[0]):
            return None
        for r in range(len(i)):
            for c in range(len(i[0])):
                v = i[r][c]
                if v == 0:
                    if o[r][c] != 0:
                        return None
                    continue
                key = (v, _isolated(i, r, c))
                if m.get(key, o[r][c]) != o[r][c]:
                    return None
                m[key] = o[r][c]
    if not m:
        return None
    changed = any(o != i for i, o in ((ex["input"], ex["output"])
                  for ex in examples)
                  if len(ex["input"]) <= 30 and len(ex["input"][0]) <= 30)
    return m if changed else None


def _build(m: dict) -> onnx.ModelProto:
    F = TensorProto.FLOAT
    B = TensorProto.BOOL

    def f32(name, arr):
        return numpy_helper.from_array(arr.astype(np.float32), name)

    cross = np.zeros((CHANNELS, 1, 3, 3), dtype=np.float32)
    for (r, c) in ((0, 1), (1, 0), (1, 2), (2, 1)):
        cross[:, 0, r, c] = 1.0
    m_iso = np.zeros((CHANNELS, CHANNELS, 1, 1), dtype=np.float32)
    m_conn = np.zeros((CHANNELS, CHANNELS, 1, 1), dtype=np.float32)
    m_iso[0, 0, 0, 0] = 1.0           # background stays background
    m_conn[0, 0, 0, 0] = 1.0
    for (colour, iso), out_colour in m.items():
        mat = m_iso if iso else m_conn
        mat[out_colour, colour, 0, 0] = 1.0

    init = [
        f32("cross", cross), f32("m_iso", m_iso), f32("m_conn", m_conn),
        f32("half", np.array([[[[0.5]]]])),
    ]

    n = helper.make_node
    nodes = [
        n("Conv", ["input", "cross"], ["nbr"],
          kernel_shape=[3, 3], pads=[1, 1, 1, 1], group=CHANNELS),
        n("Less", ["nbr", "half"], ["iso_b"]),
        n("Cast", ["iso_b"], ["iso_f"], to=F),
        n("Mul", ["input", "iso_f"], ["iso"]),
        n("Greater", ["nbr", "half"], ["conn_b"]),
        n("Cast", ["conn_b"], ["conn_f"], to=F),
        n("Mul", ["input", "conn_f"], ["conn"]),
        n("Conv", ["iso", "m_iso"], ["out_iso"], kernel_shape=[1, 1]),
        n("Conv", ["conn", "m_conn"], ["out_conn"], kernel_shape=[1, 1]),
        n("Add", ["out_iso", "out_conn"], ["output"]),
    ]

    def vi(name, dt=F):
        return helper.make_tensor_value_info(name, dt, [1, CHANNELS, HEIGHT, WIDTH])

    value_info = [vi("nbr"), vi("iso_b", B), vi("iso_f"), vi("iso"),
                  vi("conn_b", B), vi("conn_f"), vi("conn"),
                  vi("out_iso"), vi("out_conn")]

    graph = helper.make_graph(nodes, "isolate_recolor", [vi("input")],
                              [vi("output")], initializer=init,
                              value_info=value_info)
    return helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
        ir_version=IR_VERSION)


def solve_isolate_recolor(task: dict) -> Optional[onnx.ModelProto]:
    params = _params(task)
    if params is None:
        return None
    # full check against the python reference
    for ex in all_examples(task):
        i, o = ex["input"], ex["output"]
        if len(i) > 30 or len(i[0]) > 30:
            continue
        if _transform(i, params) != o:
            return None
    return _build(params)
