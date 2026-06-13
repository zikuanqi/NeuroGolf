"""Solver: recolour the inner 8-pattern by quadrant corner keys (task 183).

A frame of colour-1 lines (two full rows, two full columns) encloses an inner
region holding an 8-pattern; the four grid corners carry colour keys.  The
output is the inner region with every 8 replaced by the key of its quadrant::

    2 1 . . 1 3        inner  8 8     ->   2 3
    1 1 1 1 1 1               8 .          4 .
    . 1 8 8 1 .
    . 1 8 . 1 .       (TL / TR / BL / BR keys = the four grid corners)
    1 1 1 1 1 1
    4 1 . . 1 6

The two divider rows/cols are found by per-line colour-1 counts equal to the
grid width/height; the inner window is gathered to the top-left and quadrant
membership is an ``arange < size/2`` test.
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
    H, W = g.shape
    frows = [r for r in range(H) if np.all(g[r] == 1)]
    fcols = [c for c in range(W) if np.all(g[:, c] == 1)]
    if len(frows) != 2 or len(fcols) != 2:
        return None
    r0, r1 = frows[0] + 1, frows[1] - 1
    c0, c1 = fcols[0] + 1, fcols[1] - 1
    if r1 < r0 or c1 < c0:
        return None
    sub = g[r0:r1 + 1, c0:c1 + 1]
    h, w = sub.shape
    keys = (g[0, 0], g[0, W - 1], g[H - 1, 0], g[H - 1, W - 1])
    out = np.zeros((h, w), int)
    for r in range(h):
        for c in range(w):
            if sub[r, c] == 8:
                top = r < h / 2
                left = c < w / 2
                out[r, c] = keys[0] if (top and left) else keys[1] if top \
                    else keys[2] if left else keys[3]
    return out


def _detect(task: dict) -> bool:
    saw = False
    for ex in all_examples(task):
        i, o = ex["input"], ex["output"]
        if not i or not i[0] or len(i) > HEIGHT or len(i[0]) > WIDTH:
            continue
        r = _ref(np.array(i))
        if r is None or r.shape != np.array(o).shape or not np.array_equal(r, np.array(o)):
            return False
        saw = True
    return saw


def _build() -> onnx.ModelProto:
    n = helper.make_node
    e0 = np.zeros((1, CHANNELS, 1, 1), np.float32); e0[0, 0] = 1.0
    init = [
        numpy_helper.from_array(e0, "e0"),
        numpy_helper.from_array(np.arange(HEIGHT, dtype=np.float32).reshape(1, 1, HEIGHT, 1), "ah"),
        numpy_helper.from_array(np.arange(WIDTH, dtype=np.float32).reshape(1, 1, 1, WIDTH), "aw"),
        numpy_helper.from_array(np.array(0.5, np.float32), "half"),
        numpy_helper.from_array(np.array(1.0, np.float32), "one"),
        numpy_helper.from_array(np.array(2.0, np.float32), "two"),
        numpy_helper.from_array(np.array(0.0, np.float32), "zero"),
        numpy_helper.from_array(np.array(29.0, np.float32), "max29"),
        numpy_helper.from_array(np.array([0], np.int64), "z0"),
        numpy_helper.from_array(np.array([1], np.int64), "z1"),
        numpy_helper.from_array(np.array([8], np.int64), "c8s"),
        numpy_helper.from_array(np.array([9], np.int64), "c9e"),
        numpy_helper.from_array(np.array([1], np.int64), "ax1"),
        numpy_helper.from_array(np.array([2], np.int64), "ax2v"),
        numpy_helper.from_array(np.array([3], np.int64), "ax3v"),
        numpy_helper.from_array(np.array([1], np.int64), "sh1"),
        numpy_helper.from_array(np.array([HEIGHT], np.int64), "shH"),
        numpy_helper.from_array(np.array([WIDTH], np.int64), "shW"),
    ]
    nodes = [
        n("ReduceSum", ["input"], ["occ"], axes=[1], keepdims=1),
        n("Slice", ["input", "z1", "c8s", "ax1"], ["mid"]),     # unused span guard (1..7)
        n("ReduceMax", ["occ"], ["rowD"], axes=[3], keepdims=1),
        n("ReduceMax", ["occ"], ["colD"], axes=[2], keepdims=1),
        n("ReduceSum", ["colD"], ["W"], axes=[3], keepdims=1),
        n("ReduceSum", ["rowD"], ["Hh"], axes=[2], keepdims=1),
        # full-1 rows/cols
        n("Slice", ["input", "z1", "c8s", "ax1"], ["span17"]),
    ]
    # is1 channel
    nodes = [
        n("ReduceSum", ["input"], ["occ"], axes=[1], keepdims=1),
        n("ReduceMax", ["occ"], ["rowD"], axes=[3], keepdims=1),
        n("ReduceMax", ["occ"], ["colD"], axes=[2], keepdims=1),
        n("ReduceSum", ["colD"], ["W"], axes=[3], keepdims=1),
        n("ReduceSum", ["rowD"], ["Hh"], axes=[2], keepdims=1),
        n("Slice", ["input", "z1", "ax1", "ax1"], ["_dummy"]),
    ]
    # rebuild cleanly
    init.append(numpy_helper.from_array(np.array([2], np.int64), "c2e"))
    nodes = [
        n("ReduceSum", ["input"], ["occ"], axes=[1], keepdims=1),
        n("ReduceMax", ["occ"], ["rowD"], axes=[3], keepdims=1),
        n("ReduceMax", ["occ"], ["colD"], axes=[2], keepdims=1),
        n("ReduceSum", ["colD"], ["W"], axes=[3], keepdims=1),
        n("ReduceSum", ["rowD"], ["Hh"], axes=[2], keepdims=1),
        n("Slice", ["input", "z1", "c2e", "ax1"], ["is1"]),
        n("Slice", ["input", "c8s", "c9e", "ax1"], ["is8"]),
        n("ReduceSum", ["is1"], ["r1cnt"], axes=[3], keepdims=1),
        n("Sub", ["W", "half"], ["Wm"]),
        n("Greater", ["r1cnt", "Wm"], ["fr_b"]), n("Cast", ["fr_b"], ["frow"], to=F),
        n("ReduceSum", ["is1"], ["c1cnt"], axes=[2], keepdims=1),
        n("Sub", ["Hh", "half"], ["Hm"]),
        n("Greater", ["c1cnt", "Hm"], ["fc_b"]), n("Cast", ["fc_b"], ["fcol"], to=F),
        # divider positions: first (ArgMax) and last (max of pos)
        n("ArgMax", ["frow"], ["rai"], axis=2, keepdims=1), n("Cast", ["rai"], ["ra"], to=F),
        n("Mul", ["frow", "ah"], ["frp"]), n("ReduceMax", ["frp"], ["rb"], axes=[2], keepdims=1),
        n("ArgMax", ["fcol"], ["cai"], axis=3, keepdims=1), n("Cast", ["cai"], ["ca"], to=F),
        n("Mul", ["fcol", "aw"], ["fcp"]), n("ReduceMax", ["fcp"], ["cb"], axes=[3], keepdims=1),
        # inner window origin and size
        n("Add", ["ra", "one"], ["r0"]),
        n("Add", ["ca", "one"], ["c0"]),
        n("Sub", ["rb", "r0"], ["hi"]),                      # h = rb-1 - r0 + 1 = rb - r0
        n("Sub", ["cb", "c0"], ["wi"]),
        # gather inner to top-left
        n("Add", ["ah", "r0"], ["gr_f"]),
        n("Clip", ["gr_f", "zero", "max29"], ["gr_c"]),
        n("Cast", ["gr_c"], ["gr_i"], to=I64), n("Reshape", ["gr_i", "shH"], ["gr1"]),
        n("Add", ["aw", "c0"], ["gc_f"]),
        n("Clip", ["gc_f", "zero", "max29"], ["gc_c"]),
        n("Cast", ["gc_c"], ["gc_i"], to=I64), n("Reshape", ["gc_i", "shW"], ["gc1"]),
        n("Gather", ["is8", "gr1"], ["g8r"], axis=2),
        n("Gather", ["g8r", "gc1"], ["m8"], axis=3),         # shifted 8-mask
        # window + quadrants
        n("Less", ["ah", "hi"], ["ri_b"]), n("Cast", ["ri_b"], ["ri"], to=F),
        n("Less", ["aw", "wi"], ["ci_b"]), n("Cast", ["ci_b"], ["ci"], to=F),
        n("Mul", ["ri", "ci"], ["win"]),
        n("Mul", ["m8", "win"], ["m8w"]),
        n("Div", ["hi", "two"], ["h2"]),
        n("Div", ["wi", "two"], ["w2"]),
        n("Less", ["ah", "h2"], ["tp_b"]), n("Cast", ["tp_b"], ["top"], to=F),
        n("Less", ["aw", "w2"], ["lf_b"]), n("Cast", ["lf_b"], ["left"], to=F),
        n("Sub", ["one", "top"], ["bot"]),
        n("Sub", ["one", "left"], ["right"]),
        # corner key vectors (1,10,1,1)
        n("Sub", ["Hh", "one"], ["Hm1"]),
        n("Cast", ["Hm1"], ["Hm1i0"], to=I64), n("Reshape", ["Hm1i0", "sh1"], ["hi1"]),
        n("Sub", ["W", "one"], ["Wm1"]),
        n("Cast", ["Wm1"], ["Wm1i0"], to=I64), n("Reshape", ["Wm1i0", "sh1"], ["wi1"]),
        n("Slice", ["input", "z0", "z1", "ax2v"], ["rowTop"]),
        n("Gather", ["input", "hi1"], ["rowBot"], axis=2),
        n("Slice", ["rowTop", "z0", "z1", "ax3v"], ["kTL"]),
        n("Gather", ["rowTop", "wi1"], ["kTR"], axis=3),
        n("Slice", ["rowBot", "z0", "z1", "ax3v"], ["kBL"]),
        n("Gather", ["rowBot", "wi1"], ["kBR"], axis=3),
        # paint: each quadrant of m8w gets its key
        n("Mul", ["top", "left"], ["qTL"]), n("Mul", ["qTL", "m8w"], ["mTL"]), n("Mul", ["mTL", "kTL"], ["pTL"]),
        n("Mul", ["top", "right"], ["qTR"]), n("Mul", ["qTR", "m8w"], ["mTR"]), n("Mul", ["mTR", "kTR"], ["pTR"]),
        n("Mul", ["bot", "left"], ["qBL"]), n("Mul", ["qBL", "m8w"], ["mBL"]), n("Mul", ["mBL", "kBL"], ["pBL"]),
        n("Mul", ["bot", "right"], ["qBR"]), n("Mul", ["qBR", "m8w"], ["mBR"]), n("Mul", ["mBR", "kBR"], ["pBR"]),
        n("Add", ["pTL", "pTR"], ["p1"]),
        n("Add", ["pBL", "pBR"], ["p2"]),
        n("Add", ["p1", "p2"], ["paint"]),
        # background within the window
        n("Sub", ["win", "m8w"], ["bgm"]),
        n("Mul", ["bgm", "e0"], ["bgp"]),
        n("Add", ["paint", "bgp"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "quadrant_corner_map",
                              [helper.make_tensor_value_info("input", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              [helper.make_tensor_value_info("output", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_quadrant_corner_map(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
