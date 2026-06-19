"""Solver: restore a mirror-symmetric figure occluded by a solid rectangle (task 71).

A symmetric figure is partly hidden under a solid rectangle of one colour.  The
occluded cells are restored from the figure's vertical mirror image; the rest is
left untouched.

The occluder is found input-only as the colour whose cells fill their bounding
box (a solid rectangle).  The mirror axis is unknown (the occluder skews any
bbox guess), so all ``2W-1`` candidate vertical axes are scored on the integer
grid via a single 5-D ``Gather``: the winning axis is the one whose reflection
agrees on every non-occluder pair with zero disagreement, chosen by ``ArgMax``.
The one-hot is then reflected through that axis to paint the hidden cells.
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
A = 2 * WIDTH - 1


def _occluder(i: np.ndarray) -> Optional[int]:
    for col in [c for c in np.unique(i) if c != 0]:
        ys, xs = np.where(i == col)
        area = (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)
        if len(ys) == area and area >= 2 and np.all(
                i[ys.min():ys.max() + 1, xs.min():xs.max() + 1] == col):
            return int(col)
    return None


def _vaxis(i: np.ndarray, occ: int) -> Optional[int]:
    H, W = i.shape
    non = (i != occ)
    best = None
    for k in range(0, 2 * W - 1):
        a = d = 0
        for r in range(H):
            for c in range(W):
                mc = k - c
                if 0 <= mc < W and non[r, c] and non[r, mc]:
                    if i[r, c] == i[r, mc]:
                        a += 1
                    else:
                        d += 1
        if d == 0 and a > 0 and (best is None or a > best[1]):
            best = (k, a)
    return best[0] if best else None


def _ref(g: np.ndarray) -> Optional[np.ndarray]:
    occ = _occluder(g)
    if occ is None:
        return None
    kv = _vaxis(g, occ)
    if kv is None:
        return None
    H, W = g.shape
    out = g.copy()
    for r in range(H):
        for c in range(W):
            if g[r, c] == occ:
                mc = kv - c
                out[r, c] = g[r, mc] if (0 <= mc < W and g[r, mc] != occ) else 0
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
    chvec = np.arange(CHANNELS, dtype=np.float32).reshape(1, CHANNELS, 1, 1)
    notbg = np.ones((1, CHANNELS, 1, 1), np.float32); notbg[0, 0] = 0.0
    e0 = np.zeros((1, CHANNELS, 1, 1), np.float32); e0[0, 0] = 1.0
    idxV = np.zeros((A, WIDTH), np.int64)
    validV = np.zeros((A, WIDTH), np.float32)
    for k in range(A):
        for c in range(WIDTH):
            mc = k - c
            idxV[k, c] = min(max(mc, 0), WIDTH - 1)
            validV[k, c] = 1.0 if 0 <= mc <= WIDTH - 1 else 0.0
    init = [
        numpy_helper.from_array(chvec, "chvec"),
        numpy_helper.from_array(notbg, "notbg"),
        numpy_helper.from_array(e0, "e0"),
        numpy_helper.from_array(np.arange(HEIGHT, dtype=np.float32).reshape(1, 1, HEIGHT, 1), "rowidx"),
        numpy_helper.from_array(np.arange(WIDTH, dtype=np.float32).reshape(1, 1, 1, WIDTH), "colidx"),
        numpy_helper.from_array(np.array(1.0, np.float32), "one"),
        numpy_helper.from_array(np.array(1e4, np.float32), "big"),
        numpy_helper.from_array(np.array(0.5, np.float32), "half"),
        numpy_helper.from_array(np.array(0.25, np.float32), "q25"),
        numpy_helper.from_array(np.array(1.5, np.float32), "c15"),
        numpy_helper.from_array(idxV, "idxV"),
        numpy_helper.from_array(validV, "validV2"),
        numpy_helper.from_array(np.array([1, 1, 1, A, WIDTH], np.int64), "shp5d"),
        numpy_helper.from_array(np.array([A], np.int64), "shpA"),
        numpy_helper.from_array(np.array([WIDTH], np.int64), "shpW"),
        numpy_helper.from_array(np.array([1, 1, 1, WIDTH], np.int64), "shp114W"),
    ]
    nodes = [
        # integer-valued grid
        n("Mul", ["input", "chvec"], ["wc"]),
        n("ReduceSum", ["wc"], ["gridInt"], axes=[1], keepdims=1),
        # occluder = colour filling its bbox (count == area, area >= 2)
        n("ReduceSum", ["input"], ["cntc"], axes=[2, 3], keepdims=1),
        n("ReduceMax", ["input"], ["rowHas"], axes=[3], keepdims=1),
        n("ReduceMax", ["input"], ["colHas"], axes=[2], keepdims=1),
        n("Mul", ["rowHas", "rowidx"], ["rh"]),
        n("ReduceMax", ["rh"], ["rmax"], axes=[2], keepdims=1),
        n("Sub", ["one", "rowHas"], ["rhi"]), n("Mul", ["rhi", "big"], ["rhb"]),
        n("Add", ["rh", "rhb"], ["rhm"]), n("ReduceMin", ["rhm"], ["rmin"], axes=[2], keepdims=1),
        n("Mul", ["colHas", "colidx"], ["ch_"]),
        n("ReduceMax", ["ch_"], ["cmax"], axes=[3], keepdims=1),
        n("Sub", ["one", "colHas"], ["chi"]), n("Mul", ["chi", "big"], ["chb"]),
        n("Add", ["ch_", "chb"], ["chm"]), n("ReduceMin", ["chm"], ["cmin"], axes=[3], keepdims=1),
        n("Sub", ["rmax", "rmin"], ["hh0"]), n("Add", ["hh0", "one"], ["hh"]),
        n("Sub", ["cmax", "cmin"], ["ww0"]), n("Add", ["ww0", "one"], ["ww"]),
        n("Mul", ["hh", "ww"], ["area"]),
        n("Sub", ["cntc", "area"], ["cad"]), n("Mul", ["cad", "cad"], ["cad2"]),
        n("Less", ["cad2", "q25"], ["eqca_b"]), n("Cast", ["eqca_b"], ["eqca"], to=F),
        n("Greater", ["area", "c15"], ["a2_b"]), n("Cast", ["a2_b"], ["a2"], to=F),
        n("Greater", ["cntc", "half"], ["hc_b"]), n("Cast", ["hc_b"], ["hc"], to=F),
        n("Mul", ["eqca", "a2"], ["s1"]), n("Mul", ["s1", "hc"], ["s2"]),
        n("Mul", ["s2", "notbg"], ["occOH"]),                       # (1,10,1,1)
        n("Mul", ["input", "occOH"], ["occSel"]),
        n("ReduceSum", ["occSel"], ["occMask"], axes=[1], keepdims=1),   # (1,1,H,W)
        # non = real (occupancy==1) AND not occluder; excludes padding beyond the grid
        n("ReduceSum", ["input"], ["realMask"], axes=[1], keepdims=1),
        n("Sub", ["one", "occMask"], ["nocc"]),
        n("Mul", ["realMask", "nocc"], ["non"]),
        # 5-D reflection of integer grid over all vertical axes
        n("Gather", ["gridInt", "idxV"], ["gReflV"], axis=3),       # (1,1,H,A,W)
        n("Gather", ["non", "idxV"], ["nonReflV"], axis=3),
        n("Unsqueeze", ["gridInt"], ["gIntE"], axes=[3]),           # (1,1,H,1,W)
        n("Unsqueeze", ["non"], ["nonE"], axes=[3]),
        n("Reshape", ["validV2", "shp5d"], ["validV"]),             # (1,1,1,A,W)
        n("Sub", ["gIntE", "gReflV"], ["dV"]), n("Mul", ["dV", "dV"], ["dV2"]),
        n("Less", ["dV2", "q25"], ["eqV_b"]), n("Cast", ["eqV_b"], ["eqV"], to=F),
        n("Sub", ["one", "eqV"], ["neqV"]),
        n("Mul", ["nonE", "nonReflV"], ["bn0"]), n("Mul", ["bn0", "validV"], ["both"]),
        n("Mul", ["eqV", "both"], ["agp"]), n("ReduceSum", ["agp"], ["agree"], axes=[2, 4], keepdims=1),
        n("Mul", ["neqV", "both"], ["dsp"]), n("ReduceSum", ["dsp"], ["dis"], axes=[2, 4], keepdims=1),
        n("Greater", ["dis", "half"], ["dp_b"]), n("Cast", ["dp_b"], ["dp"], to=F),
        n("Mul", ["dp", "big"], ["dpb"]), n("Sub", ["agree", "dpb"], ["scoreV"]),
        n("Reshape", ["scoreV", "shpA"], ["scoreVr"]),
        n("ArgMax", ["scoreVr"], ["kv"], axis=0, keepdims=1),       # (1,)
        # reflect one-hot and occMask through the chosen axis
        n("Gather", ["idxV", "kv"], ["idxKv0"], axis=0),           # (1,W)
        n("Reshape", ["idxKv0", "shpW"], ["idxKv"]),               # (W,)
        n("Gather", ["validV2", "kv"], ["valKv0"], axis=0),
        n("Reshape", ["valKv0", "shp114W"], ["valKv"]),            # (1,1,1,W)
        n("Gather", ["input", "idxKv"], ["ohRefl"], axis=3),       # (1,10,H,W)
        n("Gather", ["non", "idxKv"], ["nonReflKv"], axis=3),      # (1,1,H,W): mirror is real & non-occ
        n("Mul", ["valKv", "nonReflKv"], ["srcOk"]),
        n("Mul", ["ohRefl", "srcOk"], ["fillC"]),
        n("Sub", ["one", "srcOk"], ["noSrc"]), n("Mul", ["noSrc", "e0"], ["bgAdd"]),
        n("Add", ["fillC", "bgAdd"], ["fillRaw"]), n("Mul", ["fillRaw", "occMask"], ["filled"]),
        n("Sub", ["one", "occMask"], ["keepM"]), n("Mul", ["input", "keepM"], ["keep"]),
        n("Add", ["keep", "filled"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "mirror_repair",
                              [helper.make_tensor_value_info("input", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              [helper.make_tensor_value_info("output", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_mirror_repair(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
