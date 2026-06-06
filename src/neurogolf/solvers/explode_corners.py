"""Solver: explode a 2x2 block to four point-reflected 2x2 stamps (task 225).

The only content is a 2x2 colour block.  Each of its four cells is copied, as a
solid 2x2 stamp, to the diagonally opposite side: the top-left cell lands two
cells down-right, the top-right cell down-left, etc. (a point reflection through
the block centre).  The original block stays.

Build (all shifts are constant offsets, so no position detection is needed):
each cell is identified by which orthogonal neighbours are also foreground
(top-left = has-below & has-right, ...).  The four classified cells are
translated by their fixed offsets, merged, and dilated down-right into 2x2
stamps; the result is OR-ed with the input.
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
    ys, xs = np.where(g != 0)
    if len(ys) == 0:
        return None
    r0, c0, r1, c1 = ys.min(), xs.min(), ys.max(), xs.max()
    if r1 - r0 != 1 or c1 - c0 != 1 or (g[r0:r0 + 2, c0:c0 + 2] == 0).any():
        return None
    out = g.copy()
    targets = {(0, 0): (r0 + 2, c0 + 2), (0, 1): (r0 + 2, c0 - 2),
               (1, 0): (r0 - 2, c0 + 2), (1, 1): (r0 - 2, c0 - 2)}
    for (i, j), (tr, tc) in targets.items():
        col = g[r0 + i, c0 + j]
        for di in (0, 1):
            for dj in (0, 1):
                rr, cc = tr + di, tc + dj
                if 0 <= rr < H and 0 <= cc < W:
                    out[rr, cc] = col
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
    F = TensorProto.FLOAT
    n = helper.make_node
    note0 = np.ones((1, CHANNELS, 1, 1), np.float32); note0[0, 0] = 0.0
    init = [numpy_helper.from_array(note0, "note0"),
            numpy_helper.from_array(np.array([0.0], np.float32), "zero"),
            numpy_helper.from_array(np.array(1.0, np.float32), "one")]
    nodes = []
    seen = {}

    def shift(src, dr, dc, tag):
        """Append Pad+Slice so out[r,c] = src[r-dr, c-dc] (0 outside)."""
        hb, he = max(dr, 0), max(-dr, 0)
        wb, we = max(dc, 0), max(-dc, 0)
        pname, sname, ename = f"pad_{tag}", f"sst_{tag}", f"sen_{tag}"
        init.append(numpy_helper.from_array(
            np.array([0, 0, hb, wb, 0, 0, he, we], np.int64), pname))
        init.append(numpy_helper.from_array(np.array([he, we], np.int64), sname))
        init.append(numpy_helper.from_array(np.array([he + HEIGHT, we + WIDTH], np.int64), ename))
        if "ax23" not in seen:
            init.append(numpy_helper.from_array(np.array([2, 3], np.int64), "ax23"))
            seen["ax23"] = True
        nodes.append(n("Pad", [src, pname, "zero"], [f"p_{tag}"]))
        nodes.append(n("Slice", [f"p_{tag}", sname, ename, "ax23"], [f"o_{tag}"]))
        return f"o_{tag}"

    nodes += [
        n("Mul", ["input", "note0"], ["nb_c"]),
        n("ReduceSum", ["nb_c"], ["nonbg"], axes=[1], keepdims=1),       # (1,1,H,W)
    ]
    below = shift("nonbg", -1, 0, "below")    # nonbg[r+1]
    right = shift("nonbg", 0, -1, "right")    # nonbg[c+1]
    above = shift("nonbg", 1, 0, "above")
    left = shift("nonbg", 0, 1, "left")
    nodes += [
        n("Mul", ["nonbg", below], ["nb_b"]), n("Mul", ["nb_b", right], ["tl_m"]),
        n("Mul", ["nb_b", left], ["tr_m"]),
        n("Mul", ["nonbg", above], ["nb_a"]), n("Mul", ["nb_a", right], ["bl_m"]),
        n("Mul", ["nb_a", left], ["br_m"]),
        n("Mul", ["input", "tl_m"], ["tl_cell"]),
        n("Mul", ["input", "tr_m"], ["tr_cell"]),
        n("Mul", ["input", "bl_m"], ["bl_cell"]),
        n("Mul", ["input", "br_m"], ["br_cell"]),
    ]
    def dilate_translate(cell, dr, dc, tag):
        # dilate the cell down-right into a 2x2 (on-grid), then translate by (dr,dc);
        # dilating first lets parts that fold back onto the grid survive a negative shift.
        cr = shift(cell, 0, 1, tag + "_r")
        nodes.append(n("Max", [cell, cr], [tag + "_h"]))
        cd = shift(tag + "_h", 1, 0, tag + "_d")
        nodes.append(n("Max", [tag + "_h", cd], [tag + "_2"]))
        return shift(tag + "_2", dr, dc, tag + "_t")

    tl_s = dilate_translate("tl_cell", 2, 2, "tl")
    tr_s = dilate_translate("tr_cell", 2, -3, "tr")
    bl_s = dilate_translate("bl_cell", -3, 2, "bl")
    br_s = dilate_translate("br_cell", -3, -3, "br")
    nodes += [
        n("Max", [tl_s, tr_s], ["mtop"]),
        n("Max", [bl_s, br_s], ["mbot"]),
        n("Max", ["mtop", "mbot"], ["stamps0"]),
        n("ReduceSum", ["input"], ["grid"], axes=[1], keepdims=1),  # 1 over the actual grid
        n("Mul", ["stamps0", "grid"], ["stamps"]),                  # clip stamps to the grid
        n("ReduceSum", ["stamps"], ["stamp_mask"], axes=[1], keepdims=1),
        n("Sub", ["one", "stamp_mask"], ["keep"]),                  # clear input (incl. ch0) at stamps
        n("Mul", ["input", "keep"], ["kept"]),
        n("Add", ["kept", "stamps"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "explode_corners",
                              [helper.make_tensor_value_info("input", F, FULL)],
                              [helper.make_tensor_value_info("output", F, FULL)],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_explode_corners(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
