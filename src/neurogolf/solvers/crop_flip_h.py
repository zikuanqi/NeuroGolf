"""Solver: crop the shape's bounding box and mirror it horizontally (task 177).

A rectangle (with noise cells inside) sits somewhere on the grid; the output
is its bounding-box crop flipped left-right::

    8 8 8 2          2 8 8 8
    8 2 8 8    ->    8 8 2 8

The crop-to-top-left reuses the bbox Gather pipeline; the flip is a runtime
reflection matrix ``S[j,c] = 1 iff j + c == w - 1`` applied with ``MatMul``,
where ``w`` is the bbox width.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from ..grids import CHANNELS, HEIGHT, WIDTH, all_examples

OPSET = 11
IR_VERSION = 8
F = TensorProto.FLOAT
I64 = TensorProto.INT64


def _ref(g: np.ndarray) -> Optional[np.ndarray]:
    ys, xs = np.where(g != 0)
    if len(ys) == 0:
        return None
    crop = g[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return crop[:, ::-1]


def _detect(task: dict) -> bool:
    saw = False
    changed = False
    for ex in all_examples(task):
        i, o = ex["input"], ex["output"]
        if not i or not i[0] or len(i) > HEIGHT or len(i[0]) > WIDTH:
            continue
        r = _ref(np.array(i))
        if r is None or r.shape != np.array(o).shape or not np.array_equal(r, np.array(o)):
            return False
        saw = True
        if not np.array_equal(r, r[:, ::-1]):
            changed = True
    return saw and changed


def _build() -> onnx.ModelProto:
    n = helper.make_node
    idx = np.arange(WIDTH)
    D2sum = (idx[:, None] + idx[None, :]).astype(np.float32).reshape(1, 1, WIDTH, WIDTH)
    init = [
        numpy_helper.from_array(D2sum, "D2sum"),
        numpy_helper.from_array(np.arange(HEIGHT, dtype=np.float32).reshape(1, 1, HEIGHT, 1), "ah"),
        numpy_helper.from_array(np.arange(WIDTH, dtype=np.float32).reshape(1, 1, 1, WIDTH), "aw"),
        numpy_helper.from_array(np.array(0.5, np.float32), "half"),
        numpy_helper.from_array(np.array(1.0, np.float32), "one"),
        numpy_helper.from_array(np.array(0.0, np.float32), "zero"),
        numpy_helper.from_array(np.array(float(HEIGHT - 1), np.float32), "max29"),
        numpy_helper.from_array(np.array([0], np.int64), "c0s"),
        numpy_helper.from_array(np.array([1], np.int64), "c1e"),
        numpy_helper.from_array(np.array([1], np.int64), "ax1"),
        numpy_helper.from_array(np.array([HEIGHT], np.int64), "hsh"),
        numpy_helper.from_array(np.array([WIDTH], np.int64), "wsh"),
    ]
    nodes = [
        n("ReduceSum", ["input"], ["occ"], axes=[1], keepdims=1),
        n("Slice", ["input", "c0s", "c1e", "ax1"], ["is0"]),
        n("Sub", ["occ", "is0"], ["nbg"]),
        n("ReduceMax", ["nbg"], ["rowH"], axes=[3], keepdims=1),
        n("ReduceMax", ["nbg"], ["colH"], axes=[2], keepdims=1),
        n("ArgMax", ["rowH"], ["r0i"], axis=2, keepdims=1), n("Cast", ["r0i"], ["r0"], to=F),
        n("ArgMax", ["colH"], ["c0i"], axis=3, keepdims=1), n("Cast", ["c0i"], ["c0"], to=F),
        n("Mul", ["rowH", "ah"], ["rp"]), n("ReduceMax", ["rp"], ["r1"], axes=[2], keepdims=1),
        n("Mul", ["colH", "aw"], ["cpos"]), n("ReduceMax", ["cpos"], ["c1m"], axes=[3], keepdims=1),
        # crop to top-left via clamped gathers
        n("Add", ["ah", "r0"], ["sr"]),
        n("Clip", ["sr", "zero", "max29"], ["src"]),
        n("Cast", ["src"], ["sri"], to=I64), n("Reshape", ["sri", "hsh"], ["sr1"]),
        n("Add", ["aw", "c0"], ["sc"]),
        n("Clip", ["sc", "zero", "max29"], ["scc"]),
        n("Cast", ["scc"], ["sci"], to=I64), n("Reshape", ["sci", "wsh"], ["sc1"]),
        n("Gather", ["input", "sr1"], ["gr"], axis=2),
        n("Gather", ["gr", "sc1"], ["shifted"], axis=3),
        # bbox mask
        n("Sub", ["r1", "r0"], ["hm1"]), n("Add", ["hm1", "one"], ["hv"]),
        n("Sub", ["c1m", "c0"], ["wm1"]),
        n("Less", ["ah", "hv"], ["hin_b"]), n("Cast", ["hin_b"], ["hin"], to=F),
        n("Add", ["wm1", "one"], ["wv"]),
        n("Less", ["aw", "wv"], ["win_b"]), n("Cast", ["win_b"], ["winf"], to=F),
        n("Mul", ["hin", "winf"], ["bm"]),
        n("Mul", ["shifted", "bm"], ["cropped"]),
        # horizontal flip within width w: S[j,c]=1 iff j+c == w-1
        n("Sub", ["D2sum", "wm1"], ["dws"]), n("Abs", ["dws"], ["dwa"]),
        n("Less", ["dwa", "half"], ["Sf_b"]), n("Cast", ["Sf_b"], ["Sf"], to=F),
        n("MatMul", ["cropped", "Sf"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "crop_flip_h",
                              [helper.make_tensor_value_info("input", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              [helper.make_tensor_value_info("output", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_crop_flip_h(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
