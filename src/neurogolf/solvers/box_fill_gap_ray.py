"""Solver: fill a 5-box interior with 8 and ray out through its gap (task 336).

A rectangular ``5`` frame has one missing border cell.  The interior is filled
``8`` and an ``8`` ray escapes through the gap, travelling outward (perpendicular
to that border) to the grid edge.

Bounding box from ``5`` projections; the gap is the frame cell that is
background; its border side picks the ray direction.
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
    ys, xs = np.where(g == 5)
    if len(ys) == 0:
        return None
    r0, r1, c0, c1 = ys.min(), ys.max(), xs.min(), xs.max()
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None
    out = g.copy()
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            if g[r, c] == 0:
                out[r, c] = 8
    gap = None
    for c in range(c0, c1 + 1):
        if g[r0, c] == 0:
            gap = (r0, c, "U")
        if g[r1, c] == 0:
            gap = (r1, c, "D")
    for r in range(r0, r1 + 1):
        if g[r, c0] == 0:
            gap = (r, c0, "L")
        if g[r, c1] == 0:
            gap = (r, c1, "R")
    if gap is None:
        return None
    gr, gc, d = gap
    out[gr, gc] = 8
    if d == "U":
        out[:gr, gc] = 8
    elif d == "D":
        out[gr + 1:, gc] = 8
    elif d == "L":
        out[gr, :gc] = 8
    else:
        out[gr, gc + 1:] = 8
    return out


def _detect(task: dict) -> bool:
    saw = False
    for ex in all_examples(task):
        i, o = ex["input"], ex["output"]
        if not i or not i[0] or len(i) > HEIGHT or len(i[0]) > WIDTH:
            continue
        r = _ref(np.array(i))
        if r is None or not np.array_equal(r, np.array(o)):
            return False
        saw = True
    return saw


def _build() -> onnx.ModelProto:
    n = helper.make_node
    e8me0 = np.zeros((1, CHANNELS, 1, 1), np.float32); e8me0[0, 8] = 1.0; e8me0[0, 0] = -1.0
    rowidx = np.arange(HEIGHT, dtype=np.float32).reshape(1, 1, HEIGHT, 1)
    colidx = np.arange(WIDTH, dtype=np.float32).reshape(1, 1, 1, WIDTH)
    init = [
        numpy_helper.from_array(e8me0, "e8me0"),
        numpy_helper.from_array(rowidx, "rowidx"),
        numpy_helper.from_array(colidx, "colidx"),
        numpy_helper.from_array(np.array(0.5, np.float32), "half"),
        numpy_helper.from_array(np.array([0], np.int64), "c0s"),
        numpy_helper.from_array(np.array([1], np.int64), "c1e"),
        numpy_helper.from_array(np.array([5], np.int64), "c5s"),
        numpy_helper.from_array(np.array([6], np.int64), "c6e"),
        numpy_helper.from_array(np.array([1], np.int64), "ax1"),
    ]

    def ge(x, v, out):  # x > v-0.5
        return [n("Sub", [v, "half"], [out + "_t"]), n("Greater", [x, out + "_t"], [out + "_b"]),
                n("Cast", [out + "_b"], [out], to=F)]

    def le(x, v, out):  # x < v+0.5
        return [n("Add", [v, "half"], [out + "_t"]), n("Less", [x, out + "_t"], [out + "_b"]),
                n("Cast", [out + "_b"], [out], to=F)]

    def lt(x, v, out):  # x < v-0.5
        return [n("Sub", [v, "half"], [out + "_t"]), n("Less", [x, out + "_t"], [out + "_b"]),
                n("Cast", [out + "_b"], [out], to=F)]

    def gt(x, v, out):  # x > v+0.5
        return [n("Add", [v, "half"], [out + "_t"]), n("Greater", [x, out + "_t"], [out + "_b"]),
                n("Cast", [out + "_b"], [out], to=F)]

    def eqd(a, b, out):  # |a-b| < 0.5
        return [n("Sub", [a, b], [out + "_d"]), n("Abs", [out + "_d"], [out + "_a"]),
                n("Less", [out + "_a", "half"], [out + "_b"]), n("Cast", [out + "_b"], [out], to=F)]

    nodes = [
        n("Slice", ["input", "c0s", "c1e", "ax1"], ["is0"]),
        n("Slice", ["input", "c5s", "c6e", "ax1"], ["is5"]),
        n("ReduceMax", ["is5"], ["rowP"], axes=[3], keepdims=1),
        n("ReduceMax", ["is5"], ["colP"], axes=[2], keepdims=1),
        n("ArgMax", ["rowP"], ["r0i"], axis=2, keepdims=1), n("Cast", ["r0i"], ["r0"], to=F),
        n("ArgMax", ["colP"], ["c0i"], axis=3, keepdims=1), n("Cast", ["c0i"], ["c0"], to=F),
        n("Mul", ["rowP", "rowidx"], ["rP"]), n("ReduceMax", ["rP"], ["r1"], axes=[2], keepdims=1),
        n("Mul", ["colP", "colidx"], ["cP"]), n("ReduceMax", ["cP"], ["c1"], axes=[3], keepdims=1),
    ]
    nodes += ge("rowidx", "r0", "geR0") + le("rowidx", "r1", "leR1")
    nodes += ge("colidx", "c0", "geC0") + le("colidx", "c1", "leC1")
    nodes += [n("Mul", ["geR0", "leR1"], ["inBoxRow"]), n("Mul", ["geC0", "leC1"], ["inBoxCol"]),
              n("Mul", ["inBoxRow", "inBoxCol"], ["boxMask"])]
    nodes += gt("rowidx", "r0", "gtR0") + lt("rowidx", "r1", "ltR1")
    nodes += gt("colidx", "c0", "gtC0") + lt("colidx", "c1", "ltC1")
    nodes += [n("Mul", ["gtR0", "ltR1"], ["inIntRow"]), n("Mul", ["gtC0", "ltC1"], ["inIntCol"]),
              n("Mul", ["inIntRow", "inIntCol"], ["interiorMask"]),
              n("Sub", ["boxMask", "interiorMask"], ["frameMask"]),
              n("Mul", ["frameMask", "is0"], ["gapMask"]),
              n("Mul", ["gapMask", "rowidx"], ["grm"]), n("ReduceSum", ["grm"], ["gr"], axes=[2, 3], keepdims=1),
              n("Mul", ["gapMask", "colidx"], ["gcm"]), n("ReduceSum", ["gcm"], ["gc"], axes=[2, 3], keepdims=1),
              n("ReduceMax", ["gapMask"], ["colGap"], axes=[2], keepdims=1),
              n("ReduceMax", ["gapMask"], ["rowGap"], axes=[3], keepdims=1)]
    nodes += eqd("gr", "r0", "onTop") + eqd("gr", "r1", "onBot") + eqd("gc", "c0", "onLeft") + eqd("gc", "c1", "onRight")
    nodes += lt("rowidx", "r0", "aboveBox") + gt("rowidx", "r1", "belowBox")
    nodes += lt("colidx", "c0", "leftBox") + gt("colidx", "c1", "rightBox")
    nodes += [
        n("Mul", ["colGap", "aboveBox"], ["upA"]), n("Mul", ["upA", "onTop"], ["upRay"]),
        n("Mul", ["colGap", "belowBox"], ["dnA"]), n("Mul", ["dnA", "onBot"], ["downRay"]),
        n("Mul", ["rowGap", "leftBox"], ["lfA"]), n("Mul", ["lfA", "onLeft"], ["leftRay"]),
        n("Mul", ["rowGap", "rightBox"], ["rtA"]), n("Mul", ["rtA", "onRight"], ["rightRay"]),
        n("Add", ["upRay", "downRay"], ["ray1"]), n("Add", ["leftRay", "rightRay"], ["ray2"]),
        n("Add", ["ray1", "ray2"], ["ray"]),
        n("Max", ["interiorMask", "ray"], ["f1"]), n("Max", ["f1", "gapMask"], ["fillAll"]),
        n("Mul", ["fillAll", "is0"], ["fill8"]),
        n("Mul", ["fill8", "e8me0"], ["delta"]),
        n("Add", ["input", "delta"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "box_fill_gap_ray",
                              [helper.make_tensor_value_info("input", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              [helper.make_tensor_value_info("output", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_box_fill_gap_ray(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
