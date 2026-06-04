"""Solver: fill the central plus of an 8-line panel grid (task 55).

Two full rows and two full columns of colour 8 split the grid into a 3x3
arrangement of panels.  The four corner panels stay background; the central
plus is painted with fixed colours:

    top-centre = 2   bottom-centre = 1   mid-left = 4   mid-right = 3
    centre = 6

Build: a divider row is a row whose every *real* cell is an 8 (compared
against the content mask so the 30x30 padding is ignored).  `CumSum` over the
divider indicators gives a band index per row/column; the three bands are
isolated with `Less`/`Greater` thresholds, combined by outer product into the
five plus regions, and painted only on real background cells (channel 0).
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
EIGHT = 8
TOP, BOT, LEFT, RIGHT, CEN = 2, 1, 4, 3, 6


def _ref(g: np.ndarray) -> Optional[np.ndarray]:
    H, W = g.shape
    ch8 = (g == EIGHT).astype(np.int64)
    rowdiv = (ch8.sum(axis=1) == W).astype(np.int64)
    coldiv = (ch8.sum(axis=0) == H).astype(np.int64)
    if rowdiv.sum() != 2 or coldiv.sum() != 2:
        return None
    cr = np.cumsum(rowdiv)
    cc = np.cumsum(coldiv)
    top_r = cr == 0
    mid_r = (cr == 1) & (rowdiv == 0)
    bot_r = (cr == 2) & (rowdiv == 0)
    left_c = cc == 0
    mid_c = (cc == 1) & (coldiv == 0)
    right_c = (cc == 2) & (coldiv == 0)
    out = g.copy()

    def paint(rmask, cmask, col):
        m = np.outer(rmask, cmask) & (g == 0)
        out[m] = col

    paint(top_r, mid_c, TOP)
    paint(bot_r, mid_c, BOT)
    paint(mid_r, left_c, LEFT)
    paint(mid_r, right_c, RIGHT)
    paint(mid_r, mid_c, CEN)
    return out


def _detect(task: dict) -> bool:
    saw = False
    for ex in all_examples(task):
        i, o = ex["input"], ex["output"]
        if not i or not i[0]:
            continue
        if len(i) > HEIGHT or len(i[0]) > WIDTH:
            continue  # scorer skips examples that don't fit the 30x30 canvas
        ref = _ref(np.array(i))
        if ref is None or not np.array_equal(ref, np.array(o)):
            return False
        saw = True
    return saw


def _ev(col: int) -> np.ndarray:
    v = np.zeros((1, CHANNELS, 1, 1), np.float32)
    v[0, col] = 1.0
    v[0, 0] = -1.0  # background (ch0) -> col
    return v


def _build() -> onnx.ModelProto:
    F = TensorProto.FLOAT
    B = TensorProto.BOOL
    n = helper.make_node
    init = [
        numpy_helper.from_array(np.array(0.5, np.float32), "half"),
        numpy_helper.from_array(np.array(1.5, np.float32), "oneh"),
        numpy_helper.from_array(np.array(2, np.int64), "ax2"),
        numpy_helper.from_array(np.array(3, np.int64), "ax3"),
        numpy_helper.from_array(np.array([0, EIGHT, 0, 0], np.int64), "e8_s"),
        numpy_helper.from_array(np.array([1, EIGHT + 1, HEIGHT, WIDTH], np.int64), "e8_e"),
        numpy_helper.from_array(np.array([0, 0, 0, 0], np.int64), "c0_s"),
        numpy_helper.from_array(np.array([1, 1, HEIGHT, WIDTH], np.int64), "c0_e"),
        numpy_helper.from_array(np.array([0, 1, 2, 3], np.int64), "ax4"),
        numpy_helper.from_array(_ev(TOP), "evTOP"),
        numpy_helper.from_array(_ev(BOT), "evBOT"),
        numpy_helper.from_array(_ev(LEFT), "evLEFT"),
        numpy_helper.from_array(_ev(RIGHT), "evRIGHT"),
        numpy_helper.from_array(_ev(CEN), "evCEN"),
    ]
    rc = [1, 1, HEIGHT, 1]; cc_s = [1, 1, 1, WIDTH]; s1 = [1, 1, HEIGHT, WIDTH]

    def gt(a, name, thr="half"):
        return [n("Greater", [a, thr], [name + "_b"]), n("Cast", [name + "_b"], [name], to=F)]

    def lt(a, name, thr="half"):
        return [n("Less", [a, thr], [name + "_b"]), n("Cast", [name + "_b"], [name], to=F)]

    nodes = [
        n("Slice", ["input", "e8_s", "e8_e", "ax4"], ["ch8"]),          # (1,1,H,W)
        n("Slice", ["input", "c0_s", "c0_e", "ax4"], ["ch0"]),          # background
        n("ReduceSum", ["input"], ["content"], axes=[1], keepdims=1),   # (1,1,H,W)
        # ---- row dividers ----
        n("ReduceSum", ["content"], ["row_content"], axes=[3], keepdims=1),
        n("ReduceSum", ["ch8"], ["row_eight"], axes=[3], keepdims=1),
        n("Sub", ["row_content", "row_eight"], ["row_diff"]),
        *lt("row_diff", "row_full"),
        *gt("row_content", "row_pos"),
        n("Mul", ["row_full", "row_pos"], ["rowdiv"]),                  # (1,1,H,1)
        n("CumSum", ["rowdiv", "ax2"], ["cr"]),
        # ---- col dividers ----
        n("ReduceSum", ["content"], ["col_content"], axes=[2], keepdims=1),
        n("ReduceSum", ["ch8"], ["col_eight"], axes=[2], keepdims=1),
        n("Sub", ["col_content", "col_eight"], ["col_diff"]),
        *lt("col_diff", "col_full"),
        *gt("col_content", "col_pos"),
        n("Mul", ["col_full", "col_pos"], ["coldiv"]),                  # (1,1,1,W)
        n("CumSum", ["coldiv", "ax3"], ["cc"]),
        # ---- row bands ----
        *lt("cr", "top_r"),                                             # cr<0.5
        *gt("cr", "cr_g0"), *lt("cr", "cr_l1", "oneh"),
        n("Mul", ["cr_g0", "cr_l1"], ["mid_r0"]),
        n("Sub", ["row_pos", "rowdiv"], ["not_rowdiv"]),               # 1-rowdiv
        n("Mul", ["mid_r0", "not_rowdiv"], ["mid_r"]),
        *gt("cr", "cr_g1", "oneh"),
        n("Mul", ["cr_g1", "not_rowdiv"], ["bot_r"]),
        # ---- col bands ----
        *lt("cc", "left_c"),
        *gt("cc", "cc_g0"), *lt("cc", "cc_l1", "oneh"),
        n("Mul", ["cc_g0", "cc_l1"], ["mid_c0"]),
        n("Sub", ["col_pos", "coldiv"], ["not_coldiv"]),
        n("Mul", ["mid_c0", "not_coldiv"], ["mid_c"]),
        *gt("cc", "cc_g1", "oneh"),
        n("Mul", ["cc_g1", "not_coldiv"], ["right_c"]),
        # ---- regions (outer products) restricted to real background ----
        n("Mul", ["top_r", "mid_c"], ["m_top0"]), n("Mul", ["m_top0", "ch0"], ["p_top"]),
        n("Mul", ["bot_r", "mid_c"], ["m_bot0"]), n("Mul", ["m_bot0", "ch0"], ["p_bot"]),
        n("Mul", ["mid_r", "left_c"], ["m_left0"]), n("Mul", ["m_left0", "ch0"], ["p_left"]),
        n("Mul", ["mid_r", "right_c"], ["m_right0"]), n("Mul", ["m_right0", "ch0"], ["p_right"]),
        n("Mul", ["mid_r", "mid_c"], ["m_cen0"]), n("Mul", ["m_cen0", "ch0"], ["p_cen"]),
        # ---- paint ----
        n("Mul", ["evTOP", "p_top"], ["d_top"]),
        n("Mul", ["evBOT", "p_bot"], ["d_bot"]),
        n("Mul", ["evLEFT", "p_left"], ["d_left"]),
        n("Mul", ["evRIGHT", "p_right"], ["d_right"]),
        n("Mul", ["evCEN", "p_cen"], ["d_cen"]),
        n("Add", ["d_top", "d_bot"], ["d1"]),
        n("Add", ["d1", "d_left"], ["d2"]),
        n("Add", ["d2", "d_right"], ["d3"]),
        n("Add", ["d3", "d_cen"], ["delta"]),
        n("Add", ["input", "delta"], ["output"]),
    ]

    def vif(name, shp):
        return helper.make_tensor_value_info(name, F, shp)

    def vib(name, shp):
        return helper.make_tensor_value_info(name, B, shp)

    vi = [
        vif("ch8", s1), vif("ch0", s1), vif("content", s1),
        vif("row_content", rc), vif("row_eight", rc), vif("row_diff", rc),
        vib("row_full_b", rc), vif("row_full", rc), vib("row_pos_b", rc), vif("row_pos", rc),
        vif("rowdiv", rc), vif("cr", rc), vif("zero_r", rc),
        vif("col_content", cc_s), vif("col_eight", cc_s), vif("col_diff", cc_s),
        vib("col_full_b", cc_s), vif("col_full", cc_s), vib("col_pos_b", cc_s), vif("col_pos", cc_s),
        vif("coldiv", cc_s), vif("cc", cc_s),
        vib("top_r_b", rc), vif("top_r", rc), vib("cr_g0_b", rc), vif("cr_g0", rc),
        vib("cr_l1_b", rc), vif("cr_l1", rc), vif("mid_r0", rc), vif("not_rowdiv", rc),
        vif("mid_r", rc), vib("cr_g1_b", rc), vif("cr_g1", rc), vif("bot_r", rc),
        vib("left_c_b", cc_s), vif("left_c", cc_s), vib("cc_g0_b", cc_s), vif("cc_g0", cc_s),
        vib("cc_l1_b", cc_s), vif("cc_l1", cc_s), vif("mid_c0", cc_s), vif("not_coldiv", cc_s),
        vif("mid_c", cc_s), vib("cc_g1_b", cc_s), vif("cc_g1", cc_s), vif("right_c", cc_s),
        vif("m_top0", s1), vif("p_top", s1), vif("m_bot0", s1), vif("p_bot", s1),
        vif("m_left0", s1), vif("p_left", s1), vif("m_right0", s1), vif("p_right", s1),
        vif("m_cen0", s1), vif("p_cen", s1),
        vif("d_top", FULL), vif("d_bot", FULL), vif("d_left", FULL),
        vif("d_right", FULL), vif("d_cen", FULL),
        vif("d1", FULL), vif("d2", FULL), vif("d3", FULL), vif("delta", FULL),
    ]
    graph = helper.make_graph(nodes, "plus_panels",
                              [vif("input", FULL)], [vif("output", FULL)],
                              initializer=init, value_info=vi)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_plus_panels(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
