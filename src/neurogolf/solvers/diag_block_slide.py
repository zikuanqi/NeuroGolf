"""Solver: a 2x2 block slides diagonally, leaving a trail (task 34).

The grid holds one 2x2 block: some cells of a colour C and 1-3 cells of colour
2. Each colour-2 cell marks a diagonal corner; the block slides in that diagonal
direction (up to 3 directions at once), painting a 2-wide trail of colour C
(the markers become C too).

Build: the block mask propagates in each of the four diagonals by log-doubling
shift-and-max; each diagonal is gated by whether the block's corresponding bbox
corner holds a colour-2 marker; the union is painted C with a background fill.
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
DIRS = {"UL": (-1, -1), "UR": (-1, 1), "DL": (1, -1), "DR": (1, 1)}


def _ref(g: np.ndarray) -> Optional[np.ndarray]:
    H, W = g.shape
    nz = np.argwhere(g != 0)
    if len(nz) == 0:
        return None
    r0, r1 = nz[:, 0].min(), nz[:, 0].max()
    c0, c1 = nz[:, 1].min(), nz[:, 1].max()
    if r1 - r0 != 1 or c1 - c0 != 1:
        return None
    blk = g[r0:r0 + 2, c0:c0 + 2]
    cs = set(int(v) for v in blk.flatten())
    if 2 not in cs:
        return None
    cc = [v for v in cs if v != 2]
    if len(cc) != 1:
        return None
    C = cc[0]
    corners = {"UL": (r0, c0), "UR": (r0, c1), "DL": (r1, c0), "DR": (r1, c1)}
    out = np.zeros_like(g)
    out[r0:r0 + 2, c0:c0 + 2] = C
    for nm, (dr, dc) in DIRS.items():
        cr, ccol = corners[nm]
        if g[cr, ccol] != 2:
            continue
        k = 1
        while True:
            placed = False
            for i in range(2):
                for j in range(2):
                    nr, ncl = r0 + i + dr * k, c0 + j + dc * k
                    if 0 <= nr < H and 0 <= ncl < W:
                        out[nr, ncl] = C
                        placed = True
            if not placed:
                break
            k += 1
    return out


def _detect(task: dict) -> bool:
    saw = False
    for ex in all_examples(task):
        i, o = ex["input"], ex["output"]
        if not i or not i[0] or len(i) > HEIGHT or len(i[0]) > WIDTH:
            return False
        ref = _ref(np.array(i))
        if ref is None or not np.array_equal(ref, np.array(o)):
            return False
        saw = True
    return saw


def _i64(name, v): return numpy_helper.from_array(np.array(v, np.int64), name)
def _f32(name, v): return numpy_helper.from_array(np.asarray(v, np.float32), name)


def _build() -> onnx.ModelProto:
    F = TensorProto.FLOAT
    B = TensorProto.BOOL
    n = helper.make_node
    row_ar = np.arange(HEIGHT, dtype=np.float32).reshape(1, 1, HEIGHT, 1)
    col_ar = np.arange(WIDTH, dtype=np.float32).reshape(1, 1, 1, WIDTH)
    mask_c = np.ones((1, CHANNELS, 1, 1), np.float32); mask_c[0, 0] = 0; mask_c[0, 2] = 0
    e0 = np.zeros((1, CHANNELS, 1, 1), np.float32); e0[0, 0] = 1
    init = [
        _f32("row_ar", row_ar), _f32("col_ar", col_ar), _f32("mask_c", mask_c),
        _f32("e0", e0), _f32("half", 0.5), _f32("one", 1.0), _f32("big", 1e4),
        _i64("ch2_s", [0, 2, 0, 0]), _i64("ch2_e", [1, 3, HEIGHT, WIDTH]),
        _i64("ch0_s", [0, 0, 0, 0]), _i64("ch0_e", [1, 1, HEIGHT, WIDTH]),
        _i64("ax4", [0, 1, 2, 3]),
    ]
    vi = []

    def V(name, shape, dt=F): vi.append(helper.make_tensor_value_info(name, dt, shape))

    s1 = [1, 1, HEIGHT, WIDTH]; rc = [1, 1, HEIGHT, 1]; cc = [1, 1, 1, WIDTH]; sc = [1, 1, 1, 1]
    nodes = [
        n("ReduceSum", ["input"], ["content"], axes=[1], keepdims=1),
        n("Slice", ["input", "ch0_s", "ch0_e", "ax4"], ["ch0"]),
        n("Sub", ["content", "ch0"], ["block"]),
        n("Slice", ["input", "ch2_s", "ch2_e", "ax4"], ["ch2"]),
        # block bbox r0,r1,c0,c1
        n("ReduceMax", ["block"], ["rpres"], axes=[3], keepdims=1),
        n("ReduceMax", ["block"], ["cpres"], axes=[2], keepdims=1),
        n("Mul", ["rpres", "row_ar"], ["rpos"]),
        n("Mul", ["cpres", "col_ar"], ["cpos"]),
        n("ReduceMax", ["rpos"], ["r1"], axes=[2], keepdims=1),
        n("ReduceMax", ["cpos"], ["c1"], axes=[3], keepdims=1),
        n("Sub", ["one", "rpres"], ["rinv"]), n("Mul", ["rinv", "big"], ["rbig"]),
        n("Add", ["rpos", "rbig"], ["rpb"]), n("ReduceMin", ["rpb"], ["r0"], axes=[2], keepdims=1),
        n("Sub", ["one", "cpres"], ["cinv"]), n("Mul", ["cinv", "big"], ["cbig"]),
        n("Add", ["cpos", "cbig"], ["cpb"]), n("ReduceMin", ["cpb"], ["c0"], axes=[3], keepdims=1),
    ]
    for nm in ("content", "ch0", "block", "ch2"): V(nm, s1)
    for nm in ("rpres", "rpos", "rinv", "rbig", "rpb"): V(nm, rc)
    for nm in ("cpres", "cpos", "cinv", "cbig", "cpb"): V(nm, cc)
    for nm in ("r0", "r1", "c0", "c1"): V(nm, sc)

    # row/col equality masks for r0,r1,c0,c1
    for nm, ar in (("r0", "row_ar"), ("r1", "row_ar"), ("c0", "col_ar"), ("c1", "col_ar")):
        shp = rc if ar == "row_ar" else cc
        nodes += [
            n("Sub", [ar, nm], [f"d_{nm}"]), n("Abs", [f"d_{nm}"], [f"ad_{nm}"]),
            n("Less", [f"ad_{nm}", "half"], [f"mb_{nm}"]), n("Cast", [f"mb_{nm}"], [f"m_{nm}"], to=F),
        ]
        V(f"d_{nm}", shp); V(f"ad_{nm}", shp); V(f"mb_{nm}", shp, B); V(f"m_{nm}", shp)

    # corner -> (rowmask, colmask); flag = sum(ch2 * rowmask * colmask)
    corners = {"UL": ("m_r0", "m_c0"), "UR": ("m_r0", "m_c1"),
               "DL": ("m_r1", "m_c0"), "DR": ("m_r1", "m_c1")}
    for nm, (rm, cm) in corners.items():
        nodes += [
            n("Mul", ["ch2", rm], [f"cn_{nm}_a"]), n("Mul", [f"cn_{nm}_a", cm], [f"cn_{nm}"]),
            n("ReduceSum", [f"cn_{nm}"], [f"flag_{nm}"], axes=[0, 1, 2, 3], keepdims=1),
        ]
        V(f"cn_{nm}_a", s1); V(f"cn_{nm}", s1); V(f"flag_{nm}", sc)

    # diagonal propagation per direction (log-doubling shift+max), gated by flag
    bands = []
    for nm, (dr, dc) in DIRS.items():
        cur = "block"
        for i in range(5):
            sr, scl = dr * (2 ** i), dc * (2 ** i)
            ptop, pbot = max(sr, 0), max(-sr, 0)
            plef, prig = max(scl, 0), max(-scl, 0)
            pname = f"pad_{nm}_{i}"; ss = f"ss_{nm}_{i}"; se = f"se_{nm}_{i}"
            init += [
                _i64(pname, [0, 0, ptop, plef, 0, 0, pbot, prig]),
                _i64(ss, [0, 0, pbot, prig]),
                _i64(se, [1, 1, pbot + HEIGHT, prig + WIDTH]),
            ]
            pad_o = f"p_{nm}_{i}"; sh_o = f"sh_{nm}_{i}"; mx_o = f"cur_{nm}_{i}"
            nodes += [
                n("Pad", [cur, pname], [pad_o], mode="constant"),
                n("Slice", [pad_o, ss, se, "ax4"], [sh_o]),
                n("Max", [cur, sh_o], [mx_o]),
            ]
            V(pad_o, [1, 1, HEIGHT + pbot + ptop, WIDTH + plef + prig])
            V(sh_o, s1); V(mx_o, s1)
            cur = mx_o
        nodes.append(n("Mul", [cur, f"flag_{nm}"], [f"band_{nm}"])); V(f"band_{nm}", s1)
        bands.append(f"band_{nm}")

    # union of block + gated bands, masked to the real grid (drop padding leak)
    nodes.append(n("Max", ["block"] + bands, ["union0"])); V("union0", s1)
    nodes.append(n("Mul", ["union0", "content"], ["union"])); V("union", s1)
    # colour C = the non-0 non-2 channel present in the block
    nodes += [
        n("ReduceSum", ["input"], ["counts"], axes=[2, 3], keepdims=1),
        n("Mul", ["counts", "mask_c"], ["counts_c"]),
        n("Greater", ["counts_c", "half"], ["conehot_b"]),
        n("Cast", ["conehot_b"], ["conehot"], to=F),
        n("Mul", ["conehot", "union"], ["band_c"]),
        # background fill on channel 0 within the grid (content) where not union
        n("Sub", ["content", "union"], ["bg"]),
        n("Mul", ["e0", "bg"], ["bg_c"]),
        n("Add", ["band_c", "bg_c"], ["output"]),
    ]
    c11 = [1, CHANNELS, 1, 1]
    V("counts", c11); V("counts_c", c11); V("conehot_b", c11, B); V("conehot", c11)
    V("band_c", FULL); V("bg", s1); V("bg_c", FULL)

    graph = helper.make_graph(nodes, "diag_block_slide",
                              [helper.make_tensor_value_info("input", F, FULL)],
                              [helper.make_tensor_value_info("output", F, FULL)],
                              initializer=init, value_info=vi)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_diag_block_slide(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
