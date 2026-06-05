"""Solver: read stacked colour bands into an ordered colour strip (task 115).

The grid is a stack of solid (if jagged) colour bands along one axis.  The
output lists the distinct band colours in spatial order: vertical bands give a
1xN row, horizontal bands give an Nx1 column.

Construction (all at runtime):
  * per-colour count and centroid (row/col) via `ReduceSum` of index ramps;
  * orientation = whichever axis the colour centroids spread along more;
  * rank(c) = how many present colours have a smaller centroid on that axis,
    computed as a 10x10 pairwise `Less` reduced over one axis;
  * a placement matrix P[c,k] = present(c) & rank(c)==k is reshaped into a row
    or column strip and padded back onto the 30x30 canvas.
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
    cols = [int(c) for c in np.unique(g)]
    if len(cols) < 2:
        return None
    cen = {}
    for c in cols:
        ys, xs = np.where(g == c)
        cen[c] = (ys.mean(), xs.mean())
    rspread = max(cen[c][0] for c in cols) - min(cen[c][0] for c in cols)
    cspread = max(cen[c][1] for c in cols) - min(cen[c][1] for c in cols)
    if cspread >= rspread:
        order = sorted(cols, key=lambda c: cen[c][1])
        return np.array(order).reshape(1, -1)
    order = sorted(cols, key=lambda c: cen[c][0])
    return np.array(order).reshape(-1, 1)


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
    F = TensorProto.FLOAT
    n = helper.make_node
    row_idx = np.arange(HEIGHT, dtype=np.float32).reshape(1, 1, HEIGHT, 1)
    col_idx = np.arange(WIDTH, dtype=np.float32).reshape(1, 1, 1, WIDTH)
    init = [
        numpy_helper.from_array(row_idx, "row_idx"),
        numpy_helper.from_array(col_idx, "col_idx"),
        numpy_helper.from_array(np.arange(CHANNELS, dtype=np.float32).reshape(1, CHANNELS), "ar10"),
        numpy_helper.from_array(np.array(0.5, np.float32), "half"),
        numpy_helper.from_array(np.array(1.0, np.float32), "one"),
        numpy_helper.from_array(np.array(1000.0, np.float32), "big"),
        numpy_helper.from_array(np.array(0.0, np.float32), "zero"),
        numpy_helper.from_array(np.array([CHANNELS, 1], np.int64), "shp_c1"),
        numpy_helper.from_array(np.array([1, CHANNELS], np.int64), "shp_1c"),
        numpy_helper.from_array(np.array([1, CHANNELS, 1, CHANNELS], np.int64), "shp_prow"),
        numpy_helper.from_array(np.array([1, CHANNELS, CHANNELS, 1], np.int64), "shp_pcol"),
        numpy_helper.from_array(
            np.array([0, 0, 0, 0, 0, 0, HEIGHT - 1, WIDTH - CHANNELS], np.int64), "pads_row"),
        numpy_helper.from_array(
            np.array([0, 0, 0, 0, 0, 0, HEIGHT - CHANNELS, WIDTH - 1], np.int64), "pads_col"),
    ]
    nodes = [
        n("ReduceSum", ["input"], ["tot"], axes=[2, 3], keepdims=1),       # (1,10,1,1)
        n("Mul", ["input", "row_idx"], ["rinp"]),
        n("ReduceSum", ["rinp"], ["rsum"], axes=[2, 3], keepdims=1),
        n("Mul", ["input", "col_idx"], ["cinp"]),
        n("ReduceSum", ["cinp"], ["csum"], axes=[2, 3], keepdims=1),
        n("Greater", ["tot", "half"], ["pres_b"]), n("Cast", ["pres_b"], ["present"], to=F),
        n("Sub", ["one", "present"], ["absent"]),
        n("Add", ["tot", "absent"], ["cnt_safe"]),
        n("Div", ["rsum", "cnt_safe"], ["row_cen"]),
        n("Div", ["csum", "cnt_safe"], ["col_cen"]),
        # orientation: spread of centroids per axis (present colours only)
        n("Mul", ["absent", "big"], ["absbig"]),
        n("Mul", ["col_cen", "present"], ["ccp"]),
        n("Sub", ["ccp", "absbig"], ["cc_hi"]), n("Add", ["ccp", "absbig"], ["cc_lo"]),
        n("ReduceMax", ["cc_hi"], ["cc_max"], axes=[1], keepdims=1),
        n("ReduceMin", ["cc_lo"], ["cc_min"], axes=[1], keepdims=1),
        n("Sub", ["cc_max", "cc_min"], ["cspread"]),
        n("Mul", ["row_cen", "present"], ["rcp"]),
        n("Sub", ["rcp", "absbig"], ["rc_hi"]), n("Add", ["rcp", "absbig"], ["rc_lo"]),
        n("ReduceMax", ["rc_hi"], ["rc_max"], axes=[1], keepdims=1),
        n("ReduceMin", ["rc_lo"], ["rc_min"], axes=[1], keepdims=1),
        n("Sub", ["rc_max", "rc_min"], ["rspread"]),
        n("Less", ["cspread", "rspread"], ["horiz_b"]), n("Cast", ["horiz_b"], ["horiz"], to=F),
        n("Sub", ["one", "horiz"], ["vert"]),                              # 1 if vertical bands
        # axis centroid = vertical ? col_cen : row_cen
        n("Mul", ["vert", "col_cen"], ["acv"]),
        n("Mul", ["horiz", "row_cen"], ["ach"]),
        n("Add", ["acv", "ach"], ["axis_cen"]),                           # (1,10,1,1)
        # pairwise rank
        n("Reshape", ["axis_cen", "shp_c1"], ["aci"]),                    # (10,1)
        n("Reshape", ["axis_cen", "shp_1c"], ["acj"]),                    # (1,10)
        n("Less", ["acj", "aci"], ["less_b"]), n("Cast", ["less_b"], ["less"], to=F),  # (10,10)
        n("Reshape", ["present", "shp_1c"], ["prj"]),                     # (1,10)
        n("Mul", ["less", "prj"], ["lessp"]),
        n("ReduceSum", ["lessp"], ["rank"], axes=[1], keepdims=1),        # (10,1)
        n("Sub", ["rank", "ar10"], ["rkd"]), n("Abs", ["rkd"], ["rka"]),
        n("Less", ["rka", "half"], ["eq_b"]), n("Cast", ["eq_b"], ["eq"], to=F),  # (10,10)
        n("Reshape", ["present", "shp_c1"], ["pri"]),                     # (10,1)
        n("Mul", ["eq", "pri"], ["P"]),                                   # (10,10) P[c,k]
        # placement
        n("Reshape", ["P", "shp_prow"], ["P_row"]),                       # (1,10,1,10)
        n("Pad", ["P_row", "pads_row", "zero"], ["row_placed"]),          # (1,10,30,30)
        n("Reshape", ["P", "shp_pcol"], ["P_col"]),                       # (1,10,10,1)
        n("Pad", ["P_col", "pads_col", "zero"], ["col_placed"]),
        n("Mul", ["vert", "row_placed"], ["o_row"]),
        n("Mul", ["horiz", "col_placed"], ["o_col"]),
        n("Add", ["o_row", "o_col"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "band_sort",
                              [helper.make_tensor_value_info("input", F, FULL)],
                              [helper.make_tensor_value_info("output", F, FULL)],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_band_sort(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
