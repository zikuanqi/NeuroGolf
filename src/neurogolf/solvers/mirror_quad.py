"""Solver: reflect a 2-shape into four quadrants around a 3-block (task 112).

A shape of colour ``2`` and a small block of colour ``3`` share the grid.  The
2-shape is mirrored across the horizontal and vertical axes that pass through the
centre of the 3-block, producing four copies (original + h-mirror + v-mirror +
both).  The 3-block is left untouched.

Build: the 3-centroid gives the reflection axes ``(cr, cc)``; column indices
``round(2*cc - c)`` and row indices ``round(2*cr - r)`` drive two ``Gather``s
(and their composition) to build the reflected 2-masks, each masked by an
in-bounds flag; the union is painted ``e_2`` over the real grid.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from ..grids import CHANNELS, HEIGHT, WIDTH, all_examples

OPSET = 11
IR_VERSION = 8
FULL = [1, CHANNELS, HEIGHT, WIDTH]


def _ref(g: np.ndarray) -> Optional[np.ndarray]:
    H, W = g.shape
    y2, x2 = np.where(g == 2); y3, x3 = np.where(g == 3)
    if len(y2) == 0 or len(y3) == 0:
        return None
    cr = 2 * y3.mean(); cc = 2 * x3.mean()
    out = g.copy()
    pts = set()
    for r, c in zip(y2, x2):
        for nr in (r, cr - r):
            for nc in (c, cc - c):
                R = int(round(nr)); C = int(round(nc))
                if 0 <= R < H and 0 <= C < W:
                    pts.add((R, C))
    for R, C in pts:
        out[R, C] = 2
    return out


def _detect(task: dict) -> bool:
    saw = changed = False
    for ex in all_examples(task):
        i, o = ex["input"], ex["output"]
        if not i or not i[0] or len(i) > HEIGHT or len(i[0]) > WIDTH:
            continue
        gi = np.array(i)
        r = _ref(gi)
        if r is None or not np.array_equal(r, np.array(o)):
            return False
        if not np.array_equal(r, gi):
            changed = True
        saw = True
    return saw and changed


def _build() -> onnx.ModelProto:
    F = TensorProto.FLOAT
    I = TensorProto.INT64
    n = helper.make_node

    row_idx = np.arange(HEIGHT, dtype=np.float32).reshape(1, 1, HEIGHT, 1)
    col_idx = np.arange(WIDTH, dtype=np.float32).reshape(1, 1, 1, WIDTH)
    col1d = np.arange(WIDTH, dtype=np.float32)
    row1d = np.arange(HEIGHT, dtype=np.float32)
    e2 = np.zeros((1, CHANNELS, 1, 1), np.float32); e2[0, 2] = 1.0
    init = [
        numpy_helper.from_array(row_idx, "row_idx"),
        numpy_helper.from_array(col_idx, "col_idx"),
        numpy_helper.from_array(col1d, "col1d"),
        numpy_helper.from_array(row1d, "row1d"),
        numpy_helper.from_array(e2, "e2"),
        numpy_helper.from_array(np.array(0.5, np.float32), "half"),
        numpy_helper.from_array(np.array(-0.5, np.float32), "neghalf"),
        numpy_helper.from_array(np.array(1.0, np.float32), "one"),
        numpy_helper.from_array(np.array(0.0, np.float32), "zero"),
        numpy_helper.from_array(np.array(float(WIDTH - 1), np.float32), "wmax"),
        numpy_helper.from_array(np.array(float(HEIGHT - 1), np.float32), "hmax"),
        numpy_helper.from_array(np.array(WIDTH - 0.5, np.float32), "wlim"),
        numpy_helper.from_array(np.array(HEIGHT - 0.5, np.float32), "hlim"),
        numpy_helper.from_array(np.array([2], np.int64), "two_s"),
        numpy_helper.from_array(np.array([3], np.int64), "three_s"),
        numpy_helper.from_array(np.array([4], np.int64), "four_s"),
        numpy_helper.from_array(np.array([1], np.int64), "ax1"),
        numpy_helper.from_array(np.array([1, 1, 1, WIDTH], np.int64), "shp_c"),
        numpy_helper.from_array(np.array([1, 1, HEIGHT, 1], np.int64), "shp_r"),
    ]
    nodes = [
        n("Slice", ["input", "two_s", "three_s", "ax1"], ["is2"]),
        n("Slice", ["input", "three_s", "four_s", "ax1"], ["is3"]),
        n("ReduceSum", ["is3"], ["cnt3"], keepdims=0),
        n("Mul", ["is3", "col_idx"], ["is3c"]), n("ReduceSum", ["is3c"], ["s3c"], keepdims=0),
        n("Div", ["s3c", "cnt3"], ["cc"]), n("Add", ["cc", "cc"], ["twocc"]),
        n("Mul", ["is3", "row_idx"], ["is3r"]), n("ReduceSum", ["is3r"], ["s3r"], keepdims=0),
        n("Div", ["s3r", "cnt3"], ["cr"]), n("Add", ["cr", "cr"], ["twocr"]),
        # column reflect indices + validity
        n("Sub", ["twocc", "col1d"], ["idxc_raw"]),
        n("Add", ["idxc_raw", "half"], ["idxc_h"]),
        n("Clip", ["idxc_h", "zero", "wmax"], ["idxc_clip"]),
        n("Cast", ["idxc_clip"], ["idxc_int"], to=I),
        n("Greater", ["idxc_raw", "neghalf"], ["vc_lo_b"]), n("Cast", ["vc_lo_b"], ["vc_lo"], to=F),
        n("Less", ["idxc_raw", "wlim"], ["vc_hi_b"]), n("Cast", ["vc_hi_b"], ["vc_hi"], to=F),
        n("Mul", ["vc_lo", "vc_hi"], ["validc1"]),
        n("Reshape", ["validc1", "shp_c"], ["validc"]),
        # row reflect indices + validity
        n("Sub", ["twocr", "row1d"], ["idxr_raw"]),
        n("Add", ["idxr_raw", "half"], ["idxr_h"]),
        n("Clip", ["idxr_h", "zero", "hmax"], ["idxr_clip"]),
        n("Cast", ["idxr_clip"], ["idxr_int"], to=I),
        n("Greater", ["idxr_raw", "neghalf"], ["vr_lo_b"]), n("Cast", ["vr_lo_b"], ["vr_lo"], to=F),
        n("Less", ["idxr_raw", "hlim"], ["vr_hi_b"]), n("Cast", ["vr_hi_b"], ["vr_hi"], to=F),
        n("Mul", ["vr_lo", "vr_hi"], ["validr1"]),
        n("Reshape", ["validr1", "shp_r"], ["validr"]),
        # gathers (reflections)
        n("Gather", ["is2", "idxc_int"], ["is2_h"], axis=3),
        n("Mul", ["is2_h", "validc"], ["C01"]),
        n("Gather", ["is2", "idxr_int"], ["is2_v"], axis=2),
        n("Mul", ["is2_v", "validr"], ["C10"]),
        n("Gather", ["is2_h", "idxr_int"], ["is2_hv"], axis=2),
        n("Mul", ["is2_hv", "validc"], ["C11a"]), n("Mul", ["C11a", "validr"], ["C11"]),
        # union + paint
        n("Max", ["is2", "C01", "C10", "C11"], ["twoflag"]),
        n("ReduceSum", ["input"], ["content"], axes=[1], keepdims=1),
        n("Mul", ["twoflag", "content"], ["paint"]),
        n("Sub", ["one", "paint"], ["inv"]), n("Mul", ["input", "inv"], ["kept"]),
        n("Mul", ["e2", "paint"], ["addc"]), n("Add", ["kept", "addc"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "mirror_quad",
                              [helper.make_tensor_value_info("input", F, FULL)],
                              [helper.make_tensor_value_info("output", F, FULL)],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_mirror_quad(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
