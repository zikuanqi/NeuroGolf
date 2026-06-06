"""Solver: fill the gap between two rectangles with colour 8 (task 341).

Two solid rectangles are separated by a gap (vertically or horizontally).  The
output fills that gap with colour 8, spanning the intersection of the two
rectangles' interiors on the perpendicular axis (their shared range shrunk by
one cell on each side).

Build (all at runtime, colours never baked):
  * a gap row/column is an empty line enclosed by rectangle content above and
    below (left and right) - found with prefix/suffix `Max` floods;
  * an interior column is one where a colour is present at c-1, c and c+1 (a
    1-D erosion); the fill columns are the product over colours of those
    erosions (absent colours contribute 1), giving the interior intersection;
  * the gap axis is whichever of rows/columns actually has a gap, and colour 8
    is painted on the gap x interior rectangle.
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
TARGET = 8


def _bbox(g, c):
    ys, xs = np.where(g == c)
    return ys.min(), ys.max(), xs.min(), xs.max()


def _ref(g: np.ndarray) -> Optional[np.ndarray]:
    cols = [int(c) for c in np.unique(g) if c != 0]
    if len(cols) != 2:
        return None
    A, B = cols
    ar0, ar1, ac0, ac1 = _bbox(g, A)
    br0, br1, bc0, bc1 = _bbox(g, B)
    out = g.copy()
    if ar1 < br0 or br1 < ar0:
        g0, g1 = (ar1 + 1, br0 - 1) if ar1 < br0 else (br1 + 1, ar0 - 1)
        lo, hi = max(ac0 + 1, bc0 + 1), min(ac1 - 1, bc1 - 1)
        if lo <= hi and g0 <= g1:
            out[g0:g1 + 1, lo:hi + 1] = TARGET
        return out
    if ac1 < bc0 or bc1 < ac0:
        g0, g1 = (ac1 + 1, bc0 - 1) if ac1 < bc0 else (bc1 + 1, ac0 - 1)
        lo, hi = max(ar0 + 1, br0 + 1), min(ar1 - 1, br1 - 1)
        if lo <= hi and g0 <= g1:
            out[lo:hi + 1, g0:g1 + 1] = TARGET
        return out
    return None


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
    e_delta = np.zeros((1, CHANNELS, 1, 1), np.float32)
    e_delta[0, TARGET] = 1.0; e_delta[0, 0] = -1.0
    init = [
        numpy_helper.from_array(np.array(0.5, np.float32), "half"),
        numpy_helper.from_array(np.array(1.0, np.float32), "one"),
        numpy_helper.from_array(np.array([1], np.int64), "st1"),
        numpy_helper.from_array(np.array([0.0], np.float32), "zero"),
        numpy_helper.from_array(np.array([0, 1, 0, 0], np.int64), "nb_s"),
        numpy_helper.from_array(np.array([1, CHANNELS, HEIGHT, WIDTH], np.int64), "nb_e"),
        numpy_helper.from_array(np.array([0, 1, 2, 3], np.int64), "ax4"),
        numpy_helper.from_array(e_delta, "e_delta"),
    ]
    nodes = []

    def shift(src, axis, d, lo_pad, tag):
        """Shift `src` by d along `axis`. lo_pad=True pads the low side
        (brings index-d), else pads the high side (brings index+d)."""
        pad = [0, 0, 0, 0, 0, 0, 0, 0]
        if lo_pad:
            pad[axis] = d; slo, shi = 0, (HEIGHT if axis == 2 else WIDTH)
        else:
            pad[4 + axis] = d; slo, shi = d, (HEIGHT if axis == 2 else WIDTH) + d
        init.append(numpy_helper.from_array(np.array(pad, np.int64), tag + "p"))
        init.append(numpy_helper.from_array(np.array([slo], np.int64), tag + "lo"))
        init.append(numpy_helper.from_array(np.array([shi], np.int64), tag + "hi"))
        init.append(numpy_helper.from_array(np.array([axis], np.int64), tag + "ax"))
        nodes.append(n("Pad", [src, tag + "p", "zero"], [tag + "pp"]))
        nodes.append(n("Slice", [tag + "pp", tag + "lo", tag + "hi", tag + "ax", "st1"], [tag]))
        return tag

    def flood(src, axis, lo_pad, tag):
        """Prefix (lo_pad=True) or suffix max along axis via log-doubling."""
        D = HEIGHT if axis == 2 else WIDTH
        cur = src; d = 1; s = 0
        while d < D:
            sh = shift(cur, axis, d, lo_pad, f"{tag}{s}")
            nodes.append(n("Max", [cur, sh], [f"{tag}m{s}"]))
            cur = f"{tag}m{s}"; d *= 2; s += 1
        return cur

    # non-background content
    nodes += [
        n("Slice", ["input", "nb_s", "nb_e", "ax4"], ["nb"]),
        n("ReduceSum", ["nb"], ["nbc"], axes=[1], keepdims=1),          # (1,1,H,W)
        n("ReduceMax", ["nbc"], ["row_has"], axes=[3], keepdims=1),     # (1,1,H,1)
        n("ReduceMax", ["nbc"], ["col_has"], axes=[2], keepdims=1),     # (1,1,1,W)
    ]
    # gap rows / cols = empty line enclosed by content both sides
    rpre = flood("row_has", 2, True, "rp"); rsuf = flood("row_has", 2, False, "rs")
    cpre = flood("col_has", 3, True, "cp"); csuf = flood("col_has", 3, False, "cs")
    nodes += [
        n("Sub", ["one", "row_has"], ["norow"]),
        n("Mul", ["norow", rpre], ["gr0"]), n("Mul", ["gr0", rsuf], ["gap_rows"]),
        n("Sub", ["one", "col_has"], ["nocol"]),
        n("Mul", ["nocol", cpre], ["gc0"]), n("Mul", ["gc0", csuf], ["gap_cols"]),
        # per-channel profiles
        n("ReduceMax", ["input"], ["colprof"], axes=[2], keepdims=1),    # (1,10,1,W)
        n("ReduceMax", ["input"], ["rowprof"], axes=[3], keepdims=1),    # (1,10,H,1)
        n("ReduceMax", ["input"], ["present"], axes=[2, 3], keepdims=1),  # (1,10,1,1)
        n("Sub", ["one", "present"], ["absent"]),
    ]
    # erode column profile: colprof & shift_left & shift_right
    cl = shift("colprof", 3, 1, False, "cl")   # colprof[c+1]
    cr = shift("colprof", 3, 1, True, "cr")    # colprof[c-1]
    nodes += [
        n("Mul", ["colprof", cl], ["ce0"]), n("Mul", ["ce0", cr], ["erode_col"]),
        n("Mul", ["erode_col", "present"], ["eec0"]), n("Add", ["eec0", "absent"], ["erode_col_eff"]),
        n("Slice", ["erode_col_eff", "nb_s", "nb_e", "ax4"], ["ece_nb"]),
        n("ReduceProd", ["ece_nb"], ["fill_cols"], axes=[1], keepdims=1),  # (1,1,1,W)
    ]
    rl = shift("rowprof", 2, 1, False, "rl")
    rr = shift("rowprof", 2, 1, True, "rr")
    nodes += [
        n("Mul", ["rowprof", rl], ["re0"]), n("Mul", ["re0", rr], ["erode_row"]),
        n("Mul", ["erode_row", "present"], ["eer0"]), n("Add", ["eer0", "absent"], ["erode_row_eff"]),
        n("Slice", ["erode_row_eff", "nb_s", "nb_e", "ax4"], ["ere_nb"]),
        n("ReduceProd", ["ere_nb"], ["fill_rows"], axes=[1], keepdims=1),  # (1,1,H,1)
        # combine: vertical gap uses gap_rows x fill_cols, else horizontal
        n("Mul", ["gap_rows", "fill_cols"], ["vfill"]),
        n("Mul", ["fill_rows", "gap_cols"], ["hfill"]),
        n("ReduceMax", ["gap_rows"], ["anyrow"]),
        n("Greater", ["anyrow", "half"], ["isv_b"]), n("Cast", ["isv_b"], ["isv"], to=F),
        n("Sub", ["one", "isv"], ["ish"]),
        n("Mul", ["isv", "vfill"], ["fv"]), n("Mul", ["ish", "hfill"], ["fh"]),
        n("Add", ["fv", "fh"], ["fillmask"]),                            # (1,1,H,W)
        n("Mul", ["e_delta", "fillmask"], ["delta"]),
        n("Add", ["input", "delta"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "gap_fill",
                              [helper.make_tensor_value_info("input", F, FULL)],
                              [helper.make_tensor_value_info("output", F, FULL)],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_gap_fill(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
