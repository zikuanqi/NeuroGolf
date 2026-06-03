"""Solver: periodic stripes from two colour seeds (task 13).

The grid holds exactly two single-cell seeds of two colours. They sit on a pair
of opposite edges:

* both on the top/bottom rows  -> full-height **vertical** stripes, spaced by
  the column gap, alternating the two seed colours from the left seed;
* both on the left/right cols  -> full-width **horizontal** stripes, spaced by
  the row gap, alternating from the top seed.

`out[r][c]` repeats with period p = the relevant gap. Detection of the
orientation is `minr,maxr in {0,H-1}` (seeds span the height -> vertical).

The graph reads the two seeds via a colour-value projection (no per-channel
loop), computes the stripe pattern with a dynamic `Mod`, builds both
orientations, selects with the edge flag, and finally `OneHot`s the colour grid
back to the (1,10,30,30) one-hot frame, masked to the real H x W region.
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


def _ref(grid) -> Optional[list]:
    g = np.array(grid)
    H, W = g.shape
    sd = [(r, c, int(g[r, c])) for r in range(H) for c in range(W) if g[r, c] != 0]
    if len(sd) != 2:
        return None
    (r1, c1, a), (r2, c2, b) = sd
    minr, maxr = min(r1, r2), max(r1, r2)
    minc, maxc = min(c1, c2), max(c1, c2)
    out = np.zeros((H, W), dtype=int)
    flagV = (minr in (0, H - 1)) and (maxr in (0, H - 1))
    if flagV:
        p = maxc - minc
        if p == 0:
            return None
        bcol, ocol = (a, b) if c1 < c2 else (b, a)
        for c in range(W):
            if c >= minc and (c - minc) % p == 0:
                out[:, c] = bcol if ((c - minc) // p) % 2 == 0 else ocol
    else:
        p = maxr - minr
        if p == 0:
            return None
        bcol, ocol = (a, b) if r1 < r2 else (b, a)
        for r in range(H):
            if r >= minr and (r - minr) % p == 0:
                out[r, :] = bcol if ((r - minr) // p) % 2 == 0 else ocol
    return out.tolist()


def _detect(task: dict) -> bool:
    saw = False
    for ex in all_examples(task):
        i, o = ex["input"], ex["output"]
        if not i or not i[0]:
            return False
        if len(i) > HEIGHT or len(i[0]) > WIDTH:
            continue
        ref = _ref(i)
        if ref is None or ref != o:
            return False
        saw = True
    return saw


def _build() -> onnx.ModelProto:
    F = TensorProto.FLOAT
    I = TensorProto.INT64
    n_ = helper.make_node

    kvec = np.arange(CHANNELS, dtype=np.float32).reshape(1, CHANNELS, 1, 1)
    row_ar = np.arange(HEIGHT, dtype=np.float32).reshape(1, 1, HEIGHT, 1)
    col_ar = np.arange(WIDTH, dtype=np.float32).reshape(1, 1, 1, WIDTH)
    init = [
        numpy_helper.from_array(kvec, "kvec"),
        numpy_helper.from_array(row_ar, "row_ar"),
        numpy_helper.from_array(col_ar, "col_ar"),
        numpy_helper.from_array(np.array(0.0, np.float32), "zero"),
        numpy_helper.from_array(np.array(1.0, np.float32), "one"),
        numpy_helper.from_array(np.array(2.0, np.float32), "two"),
        numpy_helper.from_array(np.array(0.5, np.float32), "half"),
        numpy_helper.from_array(np.array(-0.5, np.float32), "neghalf"),
        numpy_helper.from_array(np.array(1000.0, np.float32), "big"),
        numpy_helper.from_array(np.array(10, np.int64), "depth"),
        numpy_helper.from_array(np.array([0.0, 1.0], np.float32), "ohvals"),
        numpy_helper.from_array(np.array([1, HEIGHT, WIDTH], np.int64), "rsh3"),
    ]

    nodes = [
        # colour-value grid (0 on background/padding, seed colour at seeds)
        n_("Mul", ["input", "kvec"], ["ck"]),
        n_("ReduceSum", ["ck"], ["cgrid"], axes=[1], keepdims=1),
        # seed presence + extents
        n_("Greater", ["cgrid", "zero"], ["seed_b"]),
        n_("Cast", ["seed_b"], ["seed"], to=F),
        n_("ReduceMax", ["seed"], ["rowhas"], axes=[3], keepdims=1),
        n_("ReduceMax", ["seed"], ["colhas"], axes=[2], keepdims=1),
        n_("Mul", ["rowhas", "row_ar"], ["rpos"]),
        n_("Mul", ["colhas", "col_ar"], ["cpos"]),
        n_("ReduceMax", ["rpos"], ["maxr"], axes=[2], keepdims=1),
        n_("ReduceMax", ["cpos"], ["maxc"], axes=[3], keepdims=1),
        n_("Sub", ["one", "rowhas"], ["rinv"]),
        n_("Sub", ["one", "colhas"], ["cinv"]),
        n_("Mul", ["rinv", "big"], ["rbig"]),
        n_("Mul", ["cinv", "big"], ["cbig"]),
        n_("Add", ["rpos", "rbig"], ["rposb"]),
        n_("Add", ["cpos", "cbig"], ["cposb"]),
        n_("ReduceMin", ["rposb"], ["minr"], axes=[2], keepdims=1),
        n_("ReduceMin", ["cposb"], ["minc"], axes=[3], keepdims=1),
        # per-row / per-col seed colour
        n_("ReduceMax", ["cgrid"], ["rowcolor"], axes=[3], keepdims=1),
        n_("ReduceMax", ["cgrid"], ["colcolor"], axes=[2], keepdims=1),
        # grid extent (includes background) -> masks
        n_("ReduceSum", ["input"], ["presIn"], axes=[1], keepdims=1),
        n_("ReduceMax", ["presIn"], ["rmaskH"], axes=[3], keepdims=1),
        n_("ReduceMax", ["presIn"], ["cmaskW"], axes=[2], keepdims=1),
        n_("Mul", ["rmaskH", "row_ar"], ["rallpos"]),
        n_("Mul", ["cmaskW", "col_ar"], ["callpos"]),
        n_("ReduceMax", ["rallpos"], ["Hm1"], axes=[2], keepdims=1),
        n_("ReduceMax", ["callpos"], ["Wm1"], axes=[3], keepdims=1),
    ]

    def eqmask(out, ar, scalar, s):
        # |ar - scalar| < 0.5  -> float mask
        return [
            n_("Sub", [ar, scalar], [f"d_{s}"]),
            n_("Abs", [f"d_{s}"], [f"ad_{s}"]),
            n_("Less", [f"ad_{s}", "half"], [f"b_{s}"]),
            n_("Cast", [f"b_{s}"], [out], to=F),
        ]

    # base/other colours
    nodes += eqmask("eq_minc", "col_ar", "minc", "minc")
    nodes += eqmask("eq_maxc", "col_ar", "maxc", "maxc")
    nodes += eqmask("eq_minr", "row_ar", "minr", "minr")
    nodes += eqmask("eq_maxr", "row_ar", "maxr", "maxr")
    nodes += [
        n_("Mul", ["colcolor", "eq_minc"], ["avc"]),
        n_("ReduceSum", ["avc"], ["a_v"], axes=[3], keepdims=1),
        n_("Mul", ["colcolor", "eq_maxc"], ["bvc"]),
        n_("ReduceSum", ["bvc"], ["b_v"], axes=[3], keepdims=1),
        n_("Mul", ["rowcolor", "eq_minr"], ["avr"]),
        n_("ReduceSum", ["avr"], ["a_h"], axes=[2], keepdims=1),
        n_("Mul", ["rowcolor", "eq_maxr"], ["bvr"]),
        n_("ReduceSum", ["bvr"], ["b_h"], axes=[2], keepdims=1),
    ]

    def stripe(axis, ar, base, hi, a_val, b_val, s):
        # period p = hi - base ; pattern along `ar` (col_ar for V, row_ar for H)
        out = []
        out += [
            n_("Sub", [hi, base], [f"p_{s}"]),
            n_("Mul", [f"p_{s}", "two"], [f"twop_{s}"]),
            n_("Max", [f"twop_{s}", "one"], [f"twops_{s}"]),
            n_("Sub", [ar, base], [f"off_{s}"]),
            n_("Greater", [f"off_{s}", "neghalf"], [f"ge_b_{s}"]),
            n_("Cast", [f"ge_b_{s}"], [f"ge_{s}"], to=F),
            n_("Mul", [f"off_{s}", f"ge_{s}"], [f"aoff_{s}"]),
            n_("Cast", [f"aoff_{s}"], [f"aoffi_{s}"], to=I),
            n_("Cast", [f"twops_{s}"], [f"twopi_{s}"], to=I),
            n_("Mod", [f"aoffi_{s}", f"twopi_{s}"], [f"modi_{s}"]),
            n_("Cast", [f"modi_{s}"], [f"mod_{s}"], to=F),
        ]
        out += eqmask(f"aeq_{s}", f"mod_{s}", "zero", f"a0_{s}")
        out += eqmask(f"beq_{s}", f"mod_{s}", f"p_{s}", f"bp_{s}")
        out += [
            n_("Mul", [f"aeq_{s}", f"ge_{s}"], [f"aon_{s}"]),
            n_("Mul", [f"beq_{s}", f"ge_{s}"], [f"bon_{s}"]),
            n_("Mul", [a_val, f"aon_{s}"], [f"at_{s}"]),
            n_("Mul", [b_val, f"bon_{s}"], [f"bt_{s}"]),
            n_("Add", [f"at_{s}", f"bt_{s}"], [f"pat_{s}"]),
        ]
        return out

    nodes += stripe(3, "col_ar", "minc", "maxc", "a_v", "b_v", "v")
    nodes += stripe(2, "row_ar", "minr", "maxr", "a_h", "b_h", "h")

    nodes += [
        # broadcast each pattern over the grid, mask to H x W
        n_("Mul", ["pat_v", "rmaskH"], ["vg0"]),
        n_("Mul", ["vg0", "cmaskW"], ["vgrid"]),
        n_("Mul", ["pat_h", "cmaskW"], ["hg0"]),
        n_("Mul", ["hg0", "rmaskH"], ["hgrid"]),
        # flagV = minr,maxr both in {0, H-1}
    ]
    nodes += eqmask("minr_0", "minr", "zero", "minr0")
    nodes += eqmask("minr_H", "minr", "Hm1", "minrH")
    nodes += eqmask("maxr_0", "maxr", "zero", "maxr0")
    nodes += eqmask("maxr_H", "maxr", "Hm1", "maxrH")
    nodes += [
        n_("Add", ["minr_0", "minr_H"], ["minr_e"]),
        n_("Add", ["maxr_0", "maxr_H"], ["maxr_e"]),
        n_("Mul", ["minr_e", "maxr_e"], ["flagV"]),     # 1 iff both edges
        n_("Sub", ["one", "flagV"], ["flagH"]),
        n_("Mul", ["vgrid", "flagV"], ["vsel"]),
        n_("Mul", ["hgrid", "flagH"], ["hsel"]),
        n_("Add", ["vsel", "hsel"], ["ocolor"]),
        # one-hot back to (1,10,30,30)
        n_("Reshape", ["ocolor", "rsh3"], ["ocolor3"]),
        n_("Cast", ["ocolor3"], ["oidx"], to=I),
        n_("OneHot", ["oidx", "depth", "ohvals"], ["oh"], axis=1),
        n_("Mul", ["rmaskH", "cmaskW"], ["extent"]),
        n_("Mul", ["oh", "extent"], ["output"]),
    ]
    for nd in nodes:
        nd.name = nd.output[0]

    # ---- value_info ----
    def vi(name, shape, dt=F):
        return helper.make_tensor_value_info(name, dt, shape)

    s1 = [1, 1, HEIGHT, WIDTH]
    rc = [1, 1, HEIGHT, 1]
    cc = [1, 1, 1, WIDTH]
    sc = [1, 1, 1, 1]
    B = TensorProto.BOOL
    value_info = [
        vi("ck", FULL), vi("cgrid", s1), vi("seed_b", s1, B), vi("seed", s1),
        vi("rowhas", rc), vi("colhas", cc), vi("rpos", rc), vi("cpos", cc),
        vi("maxr", sc), vi("maxc", sc), vi("rinv", rc), vi("cinv", cc),
        vi("rbig", rc), vi("cbig", cc), vi("rposb", rc), vi("cposb", cc),
        vi("minr", sc), vi("minc", sc), vi("rowcolor", rc), vi("colcolor", cc),
        vi("presIn", s1), vi("rmaskH", rc), vi("cmaskW", cc),
        vi("rallpos", rc), vi("callpos", cc), vi("Hm1", sc), vi("Wm1", sc),
        vi("avc", cc), vi("a_v", sc), vi("bvc", cc), vi("b_v", sc),
        vi("avr", rc), vi("a_h", sc), vi("bvr", rc), vi("b_h", sc),
        vi("vg0", s1), vi("vgrid", s1), vi("hg0", s1), vi("hgrid", s1),
        vi("minr_e", sc), vi("maxr_e", sc), vi("flagV", sc), vi("flagH", sc),
        vi("vsel", s1), vi("hsel", s1), vi("ocolor", s1),
        vi("ocolor3", [1, HEIGHT, WIDTH]), vi("oidx", [1, HEIGHT, WIDTH], I),
        vi("oh", FULL), vi("extent", s1),
    ]
    for nm, sh in [("minc", cc), ("maxc", cc), ("minr", rc), ("maxr", rc)]:
        s = nm
        value_info += [vi(f"d_{s}", sh), vi(f"ad_{s}", sh),
                       vi(f"b_{s}", sh, B), vi(f"eq_{s}", sh)]
    # eqmask temporaries for colour pick use suffixes minc/maxc/minr/maxr above;
    # outputs eq_minc/eq_maxc/eq_minr/eq_maxr already declared via eq_{s}
    for s, sh in [("minr0", sc), ("minrH", sc), ("maxr0", sc), ("maxrH", sc)]:
        value_info += [vi(f"d_{s}", sh), vi(f"ad_{s}", sh),
                       vi(f"b_{s}", sh, B)]
    value_info += [vi("minr_0", sc), vi("minr_H", sc), vi("maxr_0", sc),
                   vi("maxr_H", sc)]
    for s, ar_sh in [("v", cc), ("h", rc)]:
        value_info += [
            vi(f"p_{s}", sc), vi(f"twop_{s}", sc), vi(f"twops_{s}", sc),
            vi(f"off_{s}", ar_sh), vi(f"ge_b_{s}", ar_sh, B), vi(f"ge_{s}", ar_sh),
            vi(f"aoff_{s}", ar_sh), vi(f"aoffi_{s}", ar_sh, I),
            vi(f"twopi_{s}", sc, I), vi(f"modi_{s}", ar_sh, I), vi(f"mod_{s}", ar_sh),
            vi(f"aon_{s}", ar_sh), vi(f"bon_{s}", ar_sh),
            vi(f"at_{s}", ar_sh), vi(f"bt_{s}", ar_sh), vi(f"pat_{s}", ar_sh),
            vi(f"aeq_{s}", ar_sh), vi(f"beq_{s}", ar_sh),
            vi(f"d_a0_{s}", ar_sh), vi(f"ad_a0_{s}", ar_sh), vi(f"b_a0_{s}", ar_sh, B),
            vi(f"d_bp_{s}", ar_sh), vi(f"ad_bp_{s}", ar_sh), vi(f"b_bp_{s}", ar_sh, B),
        ]

    graph = helper.make_graph(nodes, "stripe_seeds", [vi("input", FULL)],
                              [vi("output", FULL)], initializer=init,
                              value_info=value_info)
    return helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
        ir_version=IR_VERSION)


def solve_stripe_seeds(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
