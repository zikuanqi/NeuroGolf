"""Solver: crop the interior of a 4-corner marker box, recoloured (task 88).

Two colours appear: four markers at the corners of a rectangle, and a shape
inside it.  The output is the box *interior* (the rectangle shrunk by one on
every side), with every cell of the inner shape repainted in the marker
colour::

    6 . . . 6                          (interior, shape 8 -> marker 6)
    . . 8 . .          . 6 .
    . 8 8 8 .   ->     6 6 6
    . . 8 . .          . 6 .
    6 . . . 6

The marker box is just the bounding box of all non-background cells (the
markers are the extreme corners); the marker colour is the cell at that box's
top-left corner.  The interior is gathered to the top-left and every non-bg
cell there is set to the marker colour.
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
    r0, r1, c0, c1 = ys.min(), ys.max(), xs.min(), xs.max()
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None
    mk = g[r0, c0]
    crop = g[r0 + 1:r1, c0 + 1:c1].copy()
    out = np.where(crop != 0, mk, 0)
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
        numpy_helper.from_array(np.array(1.0, np.float32), "one"),
        numpy_helper.from_array(np.array(0.0, np.float32), "cmin"),
        numpy_helper.from_array(np.array(float(HEIGHT - 1), np.float32), "cmax"),
        numpy_helper.from_array(np.array([0], np.int64), "c0idx"),
        numpy_helper.from_array(np.array([1], np.int64), "c1idx"),
        numpy_helper.from_array(np.array([1], np.int64), "ax1"),
        numpy_helper.from_array(np.array([1], np.int64), "one_shape"),
        numpy_helper.from_array(np.array([HEIGHT], np.int64), "h_shape"),
        numpy_helper.from_array(np.array([WIDTH], np.int64), "w_shape"),
    ]
    nodes = [
        n("ReduceSum", ["input"], ["occ"], axes=[1], keepdims=1),
        n("Slice", ["input", "c0idx", "c1idx", "ax1"], ["is0"]),
        n("Sub", ["occ", "is0"], ["nonbg"]),
        n("ReduceMax", ["nonbg"], ["rowHas"], axes=[3], keepdims=1),
        n("ReduceMax", ["nonbg"], ["colHas"], axes=[2], keepdims=1),
        n("ArgMax", ["rowHas"], ["r0i"], axis=2, keepdims=1),
        n("ArgMax", ["colHas"], ["c0i"], axis=3, keepdims=1),
        n("Cast", ["r0i"], ["r0f"], to=F),
        n("Cast", ["c0i"], ["c0f"], to=F),
        n("Mul", ["rowHas", "ah"], ["rpos"]),
        n("ReduceMax", ["rpos"], ["r1f"], axes=[2], keepdims=1),
        n("Mul", ["colHas", "aw"], ["cpos"]),
        n("ReduceMax", ["cpos"], ["c1f"], axes=[3], keepdims=1),
        # interior size = (r1 - r0 - 1, c1 - c0 - 1)
        n("Sub", ["r1f", "r0f"], ["dh"]), n("Sub", ["dh", "one"], ["intH"]),
        n("Sub", ["c1f", "c0f"], ["dw"]), n("Sub", ["dw", "one"], ["intW"]),
        # marker colour = input[r0, c0]
        n("Reshape", ["r0i", "one_shape"], ["r0_1d"]),
        n("Reshape", ["c0i", "one_shape"], ["c0_1d"]),
        n("Gather", ["input", "r0_1d"], ["grow"], axis=2),
        n("Gather", ["grow", "c0_1d"], ["mkvec"], axis=3),
        # shift the interior to the top-left: gather rows arange+r0+1, cols arange+c0+1
        n("Add", ["r0f", "one"], ["r0p1"]),
        n("Add", ["ah", "r0p1"], ["srf"]),
        n("Clip", ["srf", "cmin", "cmax"], ["src"]),
        n("Cast", ["src"], ["sri"], to=I64),
        n("Reshape", ["sri", "h_shape"], ["sr1d"]),
        n("Add", ["c0f", "one"], ["c0p1"]),
        n("Add", ["aw", "c0p1"], ["scf"]),
        n("Clip", ["scf", "cmin", "cmax"], ["scc"]),
        n("Cast", ["scc"], ["sci"], to=I64),
        n("Reshape", ["sci", "w_shape"], ["sc1d"]),
        n("Gather", ["input", "sr1d"], ["shr"], axis=2),
        n("Gather", ["shr", "sc1d"], ["shifted"], axis=3),
        n("ReduceSum", ["shifted"], ["socc"], axes=[1], keepdims=1),
        n("Slice", ["shifted", "c0idx", "c1idx", "ax1"], ["sis0"]),
        n("Sub", ["socc", "sis0"], ["snonbg"]),
        # interior mask = (arange_h < intH) & (arange_w < intW)
        n("Less", ["ah", "intH"], ["rin"]), n("Cast", ["rin"], ["rinf"], to=F),
        n("Less", ["aw", "intW"], ["cin"]), n("Cast", ["cin"], ["cinf"], to=F),
        n("Mul", ["rinf", "cinf"], ["imask"]),
        # output: shape cells -> marker colour ; interior bg -> background
        n("Mul", ["imask", "snonbg"], ["shapeC"]),
        n("Sub", ["imask", "shapeC"], ["bgC"]),
        n("Mul", ["shapeC", "mkvec"], ["a"]),
        n("Mul", ["bgC", "e0"], ["b"]),
        n("Add", ["a", "b"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "marker_box_interior",
                              [helper.make_tensor_value_info("input", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              [helper.make_tensor_value_info("output", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_marker_box_interior(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
