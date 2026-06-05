"""Solver: connect aligned markers to a filled box with coloured lines (task 64).

A single solid rectangle (the most common non-background colour) sits in the
grid.  Scattered marker cells of other colours that line up with the box -
either in a row the box spans or a column it spans - are joined to the box by
a straight line painted in the marker's own colour, filling the background gap
between the box edge and the marker.  Markers that line up with nothing are
left untouched.

Construction (everything at runtime):
  * background = global arg-max channel, box = arg-max of the rest;
  * `marker_field` = the input with the background and box channels removed;
  * for each of the four directions the marker colours are flooded toward the
    box with a log-doubling shift-`Max`, and the box mask is flooded the
    opposite way; a gap cell is one that is background, has the box on one side
    and a marker on the other - it is painted with the flooded marker colour.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from ..grids import CHANNELS, HEIGHT, WIDTH, all_examples

OPSET = 11
IR_VERSION = 8


# ----- reference (also used for detection) ----------------------------------

def _box(g: np.ndarray, bg: int):
    """Return (boxcolour, r0, r1, c0, c1) for the unique solid non-bg rectangle."""
    for c in np.unique(g):
        if c == bg:
            continue
        ys, xs = np.where(g == c)
        if len(ys) < 2:
            continue
        r0, r1, c0, c1 = ys.min(), ys.max(), xs.min(), xs.max()
        if (g[r0:r1 + 1, c0:c1 + 1] == c).all() and \
           (r1 - r0 + 1) * (c1 - c0 + 1) == len(ys):
            return int(c), int(r0), int(r1), int(c0), int(c1)
    return None


def _ref(g: np.ndarray) -> Optional[np.ndarray]:
    from collections import Counter
    bg = Counter(g.flatten().tolist()).most_common(1)[0][0]
    box = _box(g, bg)
    if box is None:
        return None
    bc, r0, r1, c0, c1 = box
    out = g.copy()
    for c in np.unique(g):
        if c == bg or c == bc:
            continue
        for (y, x) in zip(*np.where(g == c)):
            if r0 <= y <= r1:
                if x > c1:
                    out[y, c1 + 1:x] = c
                elif x < c0:
                    out[y, x + 1:c0] = c
            elif c0 <= x <= c1:
                if y > r1:
                    out[r1 + 1:y, x] = c
                elif y < r0:
                    out[y + 1:r0, x] = c
    return out


def _detect(task: dict) -> bool:
    saw = False
    for ex in all_examples(task):
        i, o = ex["input"], ex["output"]
        if not i or not i[0]:
            continue
        if len(i) > HEIGHT or len(i[0]) > WIDTH:
            continue
        r = _ref(np.array(i))
        if r is None or not np.array_equal(r, np.array(o)):
            return False
        saw = True
    return saw


# ----- ONNX graph -----------------------------------------------------------

def _build() -> onnx.ModelProto:
    def i64(name, data):
        return numpy_helper.from_array(np.array(data, np.int64), name)

    def f32(name, data):
        return numpy_helper.from_array(np.array(data, np.float32), name)

    init = [
        i64("st1", [1]), f32("zero", [0.0]), f32("halff", [0.5]),
        i64("ch_lo", [0, 1, 0, 0]), i64("ch_hi", [1, CHANNELS, HEIGHT, WIDTH]),
        i64("ax1", [1]),
    ]
    nodes = []

    # ---- background & box one-hot channel selectors (1,10,1,1) ----
    nodes += [
        helper.make_node("ReduceSum", ["input"], ["tot"], axes=[2, 3], keepdims=1),
        helper.make_node("ReduceMax", ["tot"], ["mx1"], axes=[1], keepdims=1),
        helper.make_node("Sub", ["mx1", "halff"], ["mx1m"]),
        helper.make_node("Greater", ["tot", "mx1m"], ["bg_b"]),
        helper.make_node("Cast", ["bg_b"], ["bg_ch"], to=TensorProto.FLOAT),  # (1,10,1,1)
        # box = the channel whose cells fill their bbox solidly: count == bbox
        # area where bbox area = (rmax-rmin+1)*(cmax-cmin+1).  Exact for a solid
        # rectangle, strictly larger for scattered cells, so it isolates the box
        # and excludes background (which has holes) and markers.
        helper.make_node("ReduceMax", ["input"], ["row_pres"], axes=[3], keepdims=1),  # (1,10,H,1)
        helper.make_node("ReduceMax", ["input"], ["col_pres"], axes=[2], keepdims=1),  # (1,10,1,W)
        helper.make_node("Mul", ["row_pres", "row_idx"], ["r_on"]),
        helper.make_node("ReduceMax", ["r_on"], ["rmax"], axes=[2], keepdims=1),
        helper.make_node("Sub", ["onef", "row_pres"], ["row_off"]),
        helper.make_node("Mul", ["row_off", "big"], ["row_offb"]),
        helper.make_node("Add", ["r_on", "row_offb"], ["r_min_src"]),
        helper.make_node("ReduceMin", ["r_min_src"], ["rmin"], axes=[2], keepdims=1),
        helper.make_node("Sub", ["rmax", "rmin"], ["rdiff"]),
        helper.make_node("Add", ["rdiff", "onef"], ["bh"]),            # bbox height
        helper.make_node("Mul", ["col_pres", "col_idx"], ["c_on"]),
        helper.make_node("ReduceMax", ["c_on"], ["cmax"], axes=[3], keepdims=1),
        helper.make_node("Sub", ["onef", "col_pres"], ["col_off"]),
        helper.make_node("Mul", ["col_off", "big"], ["col_offb"]),
        helper.make_node("Add", ["c_on", "col_offb"], ["c_min_src"]),
        helper.make_node("ReduceMin", ["c_min_src"], ["cmin"], axes=[3], keepdims=1),
        helper.make_node("Sub", ["cmax", "cmin"], ["cdiff"]),
        helper.make_node("Add", ["cdiff", "onef"], ["bw"]),            # bbox width
        helper.make_node("Mul", ["bh", "bw"], ["area"]),               # (1,10,1,1)
        helper.make_node("Sub", ["area", "tot"], ["area_gap"]),
        helper.make_node("Less", ["area_gap", "halff"], ["solid_b"]),  # area == count
        helper.make_node("Cast", ["solid_b"], ["solid_f"], to=TensorProto.FLOAT),
        helper.make_node("Sub", ["tot", "onef"], ["cnt_m1"]),
        helper.make_node("Greater", ["cnt_m1", "halff"], ["ge2_b"]),   # count >= 2
        helper.make_node("Cast", ["ge2_b"], ["ge2_f"], to=TensorProto.FLOAT),
        helper.make_node("Mul", ["solid_f", "ge2_f"], ["box_ch"]),
    ]
    init.append(f32("onef10", np.ones((1, CHANNELS, 1, 1), np.float32)))
    init.append(f32("onef", [1.0]))
    init.append(f32("big", [1000.0]))
    init.append(numpy_helper.from_array(
        np.arange(HEIGHT, dtype=np.float32).reshape(1, 1, HEIGHT, 1), "row_idx"))
    init.append(numpy_helper.from_array(
        np.arange(WIDTH, dtype=np.float32).reshape(1, 1, 1, WIDTH), "col_idx"))

    # ---- per-cell masks ----
    nodes += [
        helper.make_node("Mul", ["input", "bg_ch"], ["bg_cells_c"]),
        helper.make_node("ReduceSum", ["bg_cells_c"], ["is_bg"], axes=[1], keepdims=1),  # (1,1,H,W)
        helper.make_node("Mul", ["input", "box_ch"], ["box_cells_c"]),
        helper.make_node("ReduceSum", ["box_cells_c"], ["box_mask"], axes=[1], keepdims=1),
        # marker field = input with bg and box channels zeroed
        helper.make_node("Add", ["bg_ch", "box_ch"], ["bgbox_ch"]),
        helper.make_node("Sub", ["onef10", "bgbox_ch"], ["mk_keep"]),
        helper.make_node("Mul", ["input", "mk_keep"], ["marker"]),  # (1,10,H,W)
    ]

    dim = {2: HEIGHT, 3: WIDTH}

    def flood(src, axis, hi, tag):
        """Log-doubling shift-Max flood of `src` along `axis`.
        hi=True  -> each cell takes max over higher indices (info flows to low);
        hi=False -> max over lower indices.
        """
        D = dim[axis]
        cur = src
        d = 1
        s = 0
        while d < D:
            if hi:
                pad = [0, 0, 0, 0, 0, 0, 0, 0]; pad[4 + axis] = d
                slo, shi = d, D + d
            else:
                pad = [0, 0, 0, 0, 0, 0, 0, 0]; pad[axis] = d
                slo, shi = 0, D
            pn = f"{tag}_p{s}"; sn = f"{tag}_s{s}"; mn = f"{tag}_m{s}"
            init.append(i64(f"{tag}_pad{s}", pad))
            init.append(i64(f"{tag}_lo{s}", [slo]))
            init.append(i64(f"{tag}_hi{s}", [shi]))
            init.append(i64(f"{tag}_ax{s}", [axis]))
            nodes.append(helper.make_node("Pad", [cur, f"{tag}_pad{s}", "zero"], [pn]))
            nodes.append(helper.make_node(
                "Slice", [pn, f"{tag}_lo{s}", f"{tag}_hi{s}", f"{tag}_ax{s}", "st1"], [sn]))
            nodes.append(helper.make_node("Max", [cur, sn], [mn]))
            cur = mn
            d *= 2
            s += 1
        return cur

    # marker colour flooded toward the box from each side
    col_R = flood("marker", 3, True, "cR")    # marker to the right -> fills leftward
    col_L = flood("marker", 3, False, "cL")
    col_D = flood("marker", 2, True, "cD")
    col_U = flood("marker", 2, False, "cU")
    # box presence flooded the opposite way (box on the far side of the gap)
    boxL = flood("box_mask", 3, False, "bL")  # box to the left
    boxR = flood("box_mask", 3, True, "bR")
    boxU = flood("box_mask", 2, False, "bU")  # box above
    boxD = flood("box_mask", 2, True, "bD")

    def present(colt, tag):
        nodes.append(helper.make_node("ReduceSum", [colt], [tag + "_pr"], axes=[1], keepdims=1))
        return tag + "_pr"

    prR = present(col_R, "cR"); prL = present(col_L, "cL")
    prD = present(col_D, "cD"); prU = present(col_U, "cU")

    def gap(boxside, markpres, color, tag):
        # mask = boxside * markpres * is_bg   (1,1,H,W); fill = color * mask
        nodes.append(helper.make_node("Mul", [boxside, markpres], [tag + "_a"]))
        nodes.append(helper.make_node("Mul", [tag + "_a", "is_bg"], [tag + "_msk"]))
        nodes.append(helper.make_node("Mul", [color, tag + "_msk"], [tag + "_fill"]))
        return tag + "_fill"

    fR = gap(boxL, prR, col_R, "gR")   # box on left, marker on right
    fL = gap(boxR, prL, col_L, "gL")
    fD = gap(boxU, prD, col_D, "gD")   # box above, marker below
    fU = gap(boxD, prU, col_U, "gU")

    nodes += [
        helper.make_node("Add", [fR, fL], ["f1"]),
        helper.make_node("Add", ["f1", fD], ["f2"]),
        helper.make_node("Add", ["f2", fU], ["fill_all"]),       # (1,10,H,W)
        helper.make_node("ReduceSum", ["fill_all"], ["fill_any"], axes=[1], keepdims=1),
        helper.make_node("Mul", ["bg_ch", "fill_any"], ["bg_remove"]),  # bg channel at fill cells
        helper.make_node("Sub", ["fill_all", "bg_remove"], ["delta"]),
        helper.make_node("Add", ["input", "delta"], ["output"]),
    ]

    inputs = [helper.make_tensor_value_info("input", TensorProto.FLOAT,
                                            [1, CHANNELS, HEIGHT, WIDTH])]
    outputs = [helper.make_tensor_value_info("output", TensorProto.FLOAT,
                                             [1, CHANNELS, HEIGHT, WIDTH])]
    graph = helper.make_graph(nodes, "connect_box_markers", inputs, outputs,
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_connect_box_markers(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
