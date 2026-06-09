"""Solver: reflect a shape across a 2-marker fold line; background -> 3 (task 62).

A single coloured shape sits next to a ``2`` marker (a short line or a single
cell).  The shape is mirror-folded across the line through the marker (the fold
axis sits half a cell inside the marker), the union of shape + reflection is
painted the shape colour, and every other cell becomes ``3``::

    . 4 . .            . 4 4 .       (2-line is the fold axis;
    4 4 . .    ->      4 4 4 4 ...    the 4-shape reflects across it,
    . 4 2 .            . 4 4 .        the rest of the grid -> 3)

Build: shape and marker centroids decide the fold orientation (vertical vs
horizontal); the reflection is a ``Gather`` on ``round(2A + off - i)`` indices
(bounds-masked), the V/H results selected by the orientation flag; the union is
painted the runtime shape colour, everything else ``e_3``.
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
F = TensorProto.FLOAT
I = TensorProto.INT64


def _ref(g: np.ndarray) -> Optional[np.ndarray]:
    H, W = g.shape
    y2, x2 = np.where(g == 2)
    if len(y2) == 0:
        return None
    cols = [c for c in np.unique(g) if c not in (0, 2)]
    if len(cols) != 1:
        return None
    sc = cols[0]; sy, sx = np.where(g == sc)
    if len(sy) == 0:
        return None
    c2r, c2c = y2.mean(), x2.mean(); scr, scc = sy.mean(), sx.mean()
    out = np.full((H, W), 3, dtype=g.dtype)
    if abs(c2c - scc) >= abs(c2r - scr):
        A = int(round(x2.mean())); off = -1 if scc < c2c else 1
        for r, c in zip(sy, sx):
            out[r, c] = sc; mc = 2 * A + off - c
            if 0 <= mc < W:
                out[r, mc] = sc
    else:
        A = int(round(y2.mean())); off = -1 if scr < c2r else 1
        for r, c in zip(sy, sx):
            out[r, c] = sc; mr = 2 * A + off - r
            if 0 <= mr < H:
                out[mr, c] = sc
    return out if not np.array_equal(out, g) else None


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
    R = np.broadcast_to(np.arange(HEIGHT, dtype=np.float32).reshape(1, 1, HEIGHT, 1),
                        (1, 1, HEIGHT, WIDTH)).copy()
    C = np.broadcast_to(np.arange(WIDTH, dtype=np.float32).reshape(1, 1, 1, WIDTH),
                        (1, 1, HEIGHT, WIDTH)).copy()
    e3 = np.zeros((1, CHANNELS, 1, 1), np.float32); e3[0, 3] = 1.0
    init = [
        numpy_helper.from_array(R, "R"), numpy_helper.from_array(C, "C"),
        numpy_helper.from_array(np.arange(WIDTH, dtype=np.float32), "col1d"),
        numpy_helper.from_array(np.arange(HEIGHT, dtype=np.float32), "row1d"),
        numpy_helper.from_array(e3, "e3"),
        numpy_helper.from_array(np.array(0.5, np.float32), "half"),
        numpy_helper.from_array(np.array(1.0, np.float32), "one"),
        numpy_helper.from_array(np.array(2.0, np.float32), "two"),
        numpy_helper.from_array(np.array(0.0, np.float32), "zero"),
        numpy_helper.from_array(np.array(float(WIDTH - 1), np.float32), "wmax"),
        numpy_helper.from_array(np.array(float(HEIGHT - 1), np.float32), "hmax"),
        numpy_helper.from_array(np.array(-0.5, np.float32), "neghalf"),
        numpy_helper.from_array(np.array(WIDTH - 0.5, np.float32), "wlim"),
        numpy_helper.from_array(np.array(HEIGHT - 0.5, np.float32), "hlim"),
        numpy_helper.from_array(np.array([2], np.int64), "c2s"),
        numpy_helper.from_array(np.array([3], np.int64), "c3e"),
        numpy_helper.from_array(np.array([1], np.int64), "ax1"),
        numpy_helper.from_array(np.array([1, 1, 1, WIDTH], np.int64), "shp_c"),
        numpy_helper.from_array(np.array([1, 1, HEIGHT, 1], np.int64), "shp_r"),
    ]
    nodes = [
        n("ReduceSum", ["input"], ["content"], axes=[1], keepdims=1),
        # most-common = background
        n("ReduceSum", ["input"], ["hist"], axes=[2, 3], keepdims=1),
        n("ReduceMax", ["hist"], ["bgcount"], axes=[1], keepdims=1),
        n("Sub", ["bgcount", "half"], ["bghalf"]),
        n("Greater", ["hist", "bghalf"], ["bgsel_b"]), n("Cast", ["bgsel_b"], ["bgsel"], to=F),
        n("Mul", ["input", "bgsel"], ["ibg"]),
        n("ReduceSum", ["ibg"], ["is_bg"], axes=[1], keepdims=1),
        n("Slice", ["input", "c2s", "c3e", "ax1"], ["is2"]),
        n("Sub", ["content", "is_bg"], ["nb"]),
        n("Sub", ["nb", "is2"], ["is_sc"]),
        # shape colour one-hot
        n("Mul", ["input", "is_sc"], ["isc_ch"]),
        n("ReduceSum", ["isc_ch"], ["schist"], axes=[2, 3], keepdims=1),
        n("Greater", ["schist", "half"], ["esc_b"]), n("Cast", ["esc_b"], ["e_sc"], to=F),
        # centroids
        n("ReduceSum", ["is2"], ["cnt2"], keepdims=0),
        n("Mul", ["is2", "R"], ["i2r"]), n("ReduceSum", ["i2r"], ["s2r"], keepdims=0), n("Div", ["s2r", "cnt2"], ["c2r"]),
        n("Mul", ["is2", "C"], ["i2c"]), n("ReduceSum", ["i2c"], ["s2c"], keepdims=0), n("Div", ["s2c", "cnt2"], ["c2c"]),
        n("ReduceSum", ["is_sc"], ["cntsc"], keepdims=0),
        n("Mul", ["is_sc", "R"], ["isr"]), n("ReduceSum", ["isr"], ["ssr"], keepdims=0), n("Div", ["ssr", "cntsc"], ["scr"]),
        n("Mul", ["is_sc", "C"], ["isc2"]), n("ReduceSum", ["isc2"], ["ssc"], keepdims=0), n("Div", ["ssc", "cntsc"], ["scc"]),
        # orientation: vert if |c2c-scc| >= |c2r-scr|
        n("Sub", ["c2c", "scc"], ["ddc"]), n("Abs", ["ddc"], ["addc"]),
        n("Sub", ["c2r", "scr"], ["ddr"]), n("Abs", ["ddr"], ["addr"]),
        n("Less", ["addr", "addc"], ["isv_b"]), n("Cast", ["isv_b"], ["isvert"], to=F),
        # vertical axis A_v=round(c2c), off_v
        n("Add", ["c2c", "half"], ["avh"]), n("Floor", ["avh"], ["Av"]),
        n("Less", ["scc", "c2c"], ["sl_b"]), n("Cast", ["sl_b"], ["sl"], to=F),
        n("Mul", ["sl", "two"], ["sl2"]), n("Sub", ["one", "sl2"], ["offv"]),     # 1-2*left
        n("Add", ["Av", "Av"], ["Av2"]), n("Add", ["Av2", "offv"], ["basev"]),
        n("Sub", ["basev", "col1d"], ["idxc_raw"]),
        n("Add", ["idxc_raw", "half"], ["idxc_h"]), n("Clip", ["idxc_h", "zero", "wmax"], ["idxc_cl"]),
        n("Cast", ["idxc_cl"], ["idxc"], to=I),
        n("Greater", ["idxc_raw", "neghalf"], ["vcl_b"]), n("Cast", ["vcl_b"], ["vcl"], to=F),
        n("Less", ["idxc_raw", "wlim"], ["vch_b"]), n("Cast", ["vch_b"], ["vch"], to=F),
        n("Mul", ["vcl", "vch"], ["validc1"]), n("Reshape", ["validc1", "shp_c"], ["validc"]),
        n("Gather", ["is_sc", "idxc"], ["refV0"], axis=3), n("Mul", ["refV0", "validc"], ["refV"]),
        # horizontal axis A_h=round(c2r), off_h
        n("Add", ["c2r", "half"], ["ahh"]), n("Floor", ["ahh"], ["Ah"]),
        n("Less", ["scr", "c2r"], ["su_b"]), n("Cast", ["su_b"], ["su"], to=F),
        n("Mul", ["su", "two"], ["su2"]), n("Sub", ["one", "su2"], ["offh"]),
        n("Add", ["Ah", "Ah"], ["Ah2"]), n("Add", ["Ah2", "offh"], ["baseh"]),
        n("Sub", ["baseh", "row1d"], ["idxr_raw"]),
        n("Add", ["idxr_raw", "half"], ["idxr_h"]), n("Clip", ["idxr_h", "zero", "hmax"], ["idxr_cl"]),
        n("Cast", ["idxr_cl"], ["idxr"], to=I),
        n("Greater", ["idxr_raw", "neghalf"], ["vrl_b"]), n("Cast", ["vrl_b"], ["vrl"], to=F),
        n("Less", ["idxr_raw", "hlim"], ["vrh_b"]), n("Cast", ["vrh_b"], ["vrh"], to=F),
        n("Mul", ["vrl", "vrh"], ["validr1"]), n("Reshape", ["validr1", "shp_r"], ["validr"]),
        n("Gather", ["is_sc", "idxr"], ["refH0"], axis=2), n("Mul", ["refH0", "validr"], ["refH"]),
        # select reflection by orientation
        n("Mul", ["isvert", "refV"], ["selV"]),
        n("Sub", ["one", "isvert"], ["ishor"]), n("Mul", ["ishor", "refH"], ["selH"]),
        n("Add", ["selV", "selH"], ["reflection"]),
        n("Max", ["is_sc", "reflection"], ["shape_out"]),
        # output: shape_out -> sc, other real cells -> 3
        n("Sub", ["content", "shape_out"], ["paintbg"]),
        n("Mul", ["e3", "paintbg"], ["a3"]),
        n("Mul", ["e_sc", "shape_out"], ["asc"]),
        n("Add", ["a3", "asc"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "fold_mirror",
                              [helper.make_tensor_value_info("input", F, FULL)],
                              [helper.make_tensor_value_info("output", F, FULL)],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_fold_mirror(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
