"""Solver: recover the patch hidden under a 3-block from grid symmetry (task 351).

The grid is symmetric (horizontal, vertical, or 180-degree); a solid ``3`` block
occludes a rectangular region.  The output is that region's true content,
recovered from whichever symmetry is valid (matches the grid outside the block)
and whose mirror source is itself unoccluded.

Each candidate patch is a two-axis ``Gather`` of the input at the mirrored
coordinates of the 3-bbox; validity is checked by comparing the input to its full
flip outside the 3-cells; a clean (3-free) valid patch is selected by priority.
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


def _ref(g: np.ndarray) -> Optional[np.ndarray]:
    H, W = g.shape
    ys, xs = np.where(g == 3)
    if len(ys) == 0:
        return None
    R0, R1, C0, C1 = ys.min(), ys.max(), xs.min(), xs.max()
    flips = {"h": g[:, ::-1], "v": g[::-1, :], "p": g[::-1, ::-1]}
    for key, G in (("h", flips["h"]), ("v", flips["v"]), ("p", flips["p"])):
        sub = G[R0:R1 + 1, C0:C1 + 1]
        if (sub == 3).any():
            continue
        m = (g != 3) & (G != 3)
        if np.array_equal(g[m], G[m]):
            return sub
    return None


def _detect(task: dict) -> bool:
    saw = False
    for ex in all_examples(task):
        i, o = ex["input"], ex["output"]
        if not i or not i[0] or len(i) > HEIGHT or len(i[0]) > WIDTH:
            continue
        r = _ref(np.array(i))
        if r is None or r.tolist() != o:
            return False
        saw = True
    return saw


def _build() -> onnx.ModelProto:
    n = helper.make_node
    rowidx = np.arange(HEIGHT, dtype=np.float32).reshape(1, 1, HEIGHT, 1)
    colidx = np.arange(WIDTH, dtype=np.float32).reshape(1, 1, 1, WIDTH)
    init = [
        numpy_helper.from_array(rowidx, "rowidx"),
        numpy_helper.from_array(colidx, "colidx"),
        numpy_helper.from_array(np.array(1.0, np.float32), "one"),
        numpy_helper.from_array(np.array(0.5, np.float32), "half"),
        numpy_helper.from_array(np.array(1e4, np.float32), "big"),
        numpy_helper.from_array(np.array(0.0, np.float32), "clipmin"),
        numpy_helper.from_array(np.array(float(HEIGHT - 1), np.float32), "clipmax"),
        numpy_helper.from_array(np.array([3], np.int64), "c3s"),
        numpy_helper.from_array(np.array([4], np.int64), "c4e"),
        numpy_helper.from_array(np.array([1], np.int64), "ax1"),
        numpy_helper.from_array(np.array([HEIGHT], np.int64), "to30"),
    ]

    def clampcast(src, out):
        return [
            n("Clip", [src, "clipmin", "clipmax"], [out + "_c"]),
            n("Cast", [out + "_c"], [out + "_i"], to=TensorProto.INT64),
            n("Reshape", [out + "_i", "to30"], [out]),
        ]

    nodes = [
        n("ReduceSum", ["input"], ["occ"], axes=[1], keepdims=1),
        n("ReduceMax", ["occ"], ["rowHas"], axes=[3], keepdims=1),
        n("ReduceMax", ["occ"], ["colHas"], axes=[2], keepdims=1),
        n("ReduceSum", ["rowHas"], ["Hn"], axes=[2], keepdims=1),     # (1,1,1,1)
        n("ReduceSum", ["colHas"], ["Wn"], axes=[3], keepdims=1),
        n("Sub", ["Hn", "one"], ["Hm1"]), n("Sub", ["Wn", "one"], ["Wm1"]),
        n("Slice", ["input", "c3s", "c4e", "ax1"], ["is3"]),          # (1,1,30,30)
        n("ReduceMax", ["is3"], ["r3"], axes=[3], keepdims=1),        # (1,1,30,1)
        n("ReduceMax", ["is3"], ["c3"], axes=[2], keepdims=1),        # (1,1,1,30)
        # R0,R1,C0,C1
        n("Mul", ["r3", "rowidx"], ["r3i"]), n("ReduceMax", ["r3i"], ["R1"], axes=[2], keepdims=1),
        n("Sub", ["one", "r3"], ["r3inv"]), n("Mul", ["r3inv", "big"], ["r3b"]),
        n("Add", ["r3i", "r3b"], ["r3min"]), n("ReduceMin", ["r3min"], ["R0"], axes=[2], keepdims=1),
        n("Mul", ["c3", "colidx"], ["c3i"]), n("ReduceMax", ["c3i"], ["C1"], axes=[3], keepdims=1),
        n("Sub", ["one", "c3"], ["c3inv"]), n("Mul", ["c3inv", "big"], ["c3b"]),
        n("Add", ["c3i", "c3b"], ["c3min"]), n("ReduceMin", ["c3min"], ["C0"], axes=[3], keepdims=1),
        n("Sub", ["R1", "R0"], ["rspan"]), n("Add", ["rspan", "one"], ["bboxH"]),
        n("Sub", ["C1", "C0"], ["cspan"]), n("Add", ["cspan", "one"], ["bboxW"]),
        # valid mask
        n("Less", ["rowidx", "bboxH"], ["vmr_b"]), n("Cast", ["vmr_b"], ["vmr"], to=F),
        n("Less", ["colidx", "bboxW"], ["vmc_b"]), n("Cast", ["vmc_b"], ["vmc"], to=F),
        n("Mul", ["vmr", "vmc"], ["vmask"]),
        # index sequences
        n("Add", ["rowidx", "R0"], ["rowA0"]),
        n("Sub", ["Hm1", "R0"], ["rdBase"]), n("Sub", ["rdBase", "rowidx"], ["rowD0"]),
        n("Add", ["colidx", "C0"], ["colA0"]),
        n("Sub", ["Wm1", "C0"], ["cdBase"]), n("Sub", ["cdBase", "colidx"], ["colD0"]),
        n("Sub", ["Wm1", "colidx"], ["hrev0"]),
        n("Sub", ["Hm1", "rowidx"], ["vrev0"]),
    ]
    for src, out in (("rowA0", "rowA"), ("rowD0", "rowD"), ("colA0", "colA"),
                     ("colD0", "colD"), ("hrev0", "hrev"), ("vrev0", "vrev")):
        nodes += clampcast(src, out)

    nodes += [
        # candidate patches
        n("Gather", ["input", "rowA"], ["pH_r"], axis=2), n("Gather", ["pH_r", "colD"], ["pH0"], axis=3),
        n("Mul", ["pH0", "vmask"], ["patchH"]),
        n("Gather", ["input", "rowD"], ["pV_r"], axis=2), n("Gather", ["pV_r", "colA"], ["pV0"], axis=3),
        n("Mul", ["pV0", "vmask"], ["patchV"]),
        n("Gather", ["input", "rowD"], ["pP_r"], axis=2), n("Gather", ["pP_r", "colD"], ["pP0"], axis=3),
        n("Mul", ["pP0", "vmask"], ["patchP"]),
        # full flips for validity
        n("Gather", ["input", "hrev"], ["hflip"], axis=3),
        n("Gather", ["input", "vrev"], ["vflip"], axis=2),
        n("Gather", ["vflip", "hrev"], ["pflip"], axis=3),
    ]

    def is3ch(src, out):
        return [n("Slice", [src, "c3s", "c4e", "ax1"], [out])]

    def validity(flip, tag):
        return [
            n("Mul", ["input", flip], ["prod_" + tag]),
            n("ReduceSum", ["prod_" + tag], ["same_" + tag], axes=[1], keepdims=1),
            *is3ch(flip, "f3_" + tag),
            n("Sub", ["one", "is3"], ["ni_" + tag]),
            n("Sub", ["one", "f3_" + tag], ["nf_" + tag]),
            n("Mul", ["ni_" + tag, "nf_" + tag], ["bn_" + tag]),
            n("Mul", ["bn_" + tag, "occ"], ["bng_" + tag]),
            n("Sub", ["one", "same_" + tag], ["nsame_" + tag]),
            n("Mul", ["bng_" + tag, "nsame_" + tag], ["mis_" + tag]),
            n("ReduceSum", ["mis_" + tag], ["misc_" + tag], axes=[2, 3], keepdims=1),
            n("Less", ["misc_" + tag, "half"], ["vld_b_" + tag]), n("Cast", ["vld_b_" + tag], ["valid_" + tag], to=F),
        ]
    nodes += validity("hflip", "h") + validity("vflip", "v") + validity("pflip", "p")

    def clean(patch, tag):
        return [
            n("Slice", [patch, "c3s", "c4e", "ax1"], ["p3_" + tag]),
            n("ReduceSum", ["p3_" + tag], ["p3c_" + tag], axes=[2, 3], keepdims=1),
            n("Less", ["p3c_" + tag, "half"], ["cl_b_" + tag]), n("Cast", ["cl_b_" + tag], ["clean_" + tag], to=F),
        ]
    nodes += clean("patchH", "h") + clean("patchV", "v") + clean("patchP", "p")

    nodes += [
        n("Mul", ["valid_h", "clean_h"], ["gh"]),
        n("Mul", ["valid_v", "clean_v"], ["gv"]),
        n("Mul", ["valid_p", "clean_p"], ["gp"]),
        n("Sub", ["one", "gh"], ["ngh"]),
        n("Mul", ["gv", "ngh"], ["useV"]),
        n("Mul", ["gp", "ngh"], ["gp1"]), n("Sub", ["one", "gv"], ["ngv"]), n("Mul", ["gp1", "ngv"], ["useP"]),
        n("Mul", ["patchH", "gh"], ["oH"]),
        n("Mul", ["patchV", "useV"], ["oV"]),
        n("Mul", ["patchP", "useP"], ["oP"]),
        n("Add", ["oH", "oV"], ["o1"]), n("Add", ["o1", "oP"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "symmetry_patch",
                              [helper.make_tensor_value_info("input", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              [helper.make_tensor_value_info("output", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_symmetry_patch(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
