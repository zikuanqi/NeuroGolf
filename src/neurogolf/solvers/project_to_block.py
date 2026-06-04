"""Solver: border markers project into a block (task 35).

The grid holds a solid colour-8 rectangle plus single coloured markers lined up
with the block's rows/columns. Each marker fires a ray toward the block and
recolours the first block cell it hits (the marker stays put); the block
interior is unchanged.

Per direction the markers are projected onto the matching block edge with a
`ReduceMax`, placed on that edge row/column, and merged into a colour-value
grid that is `OneHot`-expanded back to the one-hot frame.
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
BLOCK = 8


def _ref(g: np.ndarray) -> Optional[np.ndarray]:
    m = (g == BLOCK)
    nz = np.argwhere(m)
    if len(nz) == 0:
        return None
    r0, r1 = nz[:, 0].min(), nz[:, 0].max()
    c0, c1 = nz[:, 1].min(), nz[:, 1].max()
    if not m[r0:r1 + 1, c0:c1 + 1].all() or (r1 == r0 and c1 == c0):
        return None
    out = g.copy()
    H, W = g.shape
    for r in range(H):
        for c in range(W):
            v = int(g[r, c])
            if v == 0 or v == BLOCK:
                continue
            if c0 <= c <= c1 and r < r0:
                out[r0, c] = v
            elif c0 <= c <= c1 and r > r1:
                out[r1, c] = v
            elif r0 <= r <= r1 and c < c0:
                out[r, c0] = v
            elif r0 <= r <= r1 and c > c1:
                out[r, c1] = v
            else:
                return None
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
    kvec = np.arange(CHANNELS, dtype=np.float32).reshape(1, CHANNELS, 1, 1)
    init = [
        _f32("row_ar", row_ar), _f32("col_ar", col_ar), _f32("kvec", kvec),
        _f32("half", 0.5), _f32("one", 1.0), _f32("big", 1e4), _f32("eight", 8.0),
        _i64("b_s", [0, BLOCK, 0, 0]), _i64("b_e", [1, BLOCK + 1, HEIGHT, WIDTH]),
        _i64("ax4", [0, 1, 2, 3]), _i64("depth", CHANNELS),
        _f32("onoff", [0.0, 1.0]), _i64("rs3", [1, HEIGHT, WIDTH]),
    ]
    vi = []
    def V(nm, sh, dt=F): vi.append(helper.make_tensor_value_info(nm, dt, sh))
    s1 = [1, 1, HEIGHT, WIDTH]; rc = [1, 1, HEIGHT, 1]; cc = [1, 1, 1, WIDTH]; sc = [1, 1, 1, 1]

    nodes = [
        n("Slice", ["input", "b_s", "b_e", "ax4"], ["block"]),
        n("Mul", ["input", "kvec"], ["kgrid"]),
        n("ReduceSum", ["kgrid"], ["colorgrid"], axes=[1], keepdims=1),
        n("Mul", ["block", "eight"], ["block8"]),
        n("Sub", ["colorgrid", "block8"], ["mcol"]),  # marker colours (0 at block/bg)
        n("ReduceSum", ["input"], ["content"], axes=[1], keepdims=1),
        # block bbox
        n("ReduceMax", ["block"], ["brow"], axes=[3], keepdims=1),
        n("ReduceMax", ["block"], ["bcol"], axes=[2], keepdims=1),
        n("Mul", ["brow", "row_ar"], ["brp"]), n("Mul", ["bcol", "col_ar"], ["bcp"]),
        n("ReduceMax", ["brp"], ["r1"], axes=[2], keepdims=1),
        n("ReduceMax", ["bcp"], ["c1"], axes=[3], keepdims=1),
        n("Sub", ["one", "brow"], ["rinv"]), n("Mul", ["rinv", "big"], ["rbig"]),
        n("Add", ["brp", "rbig"], ["brpb"]), n("ReduceMin", ["brpb"], ["r0"], axes=[2], keepdims=1),
        n("Sub", ["one", "bcol"], ["cinv"]), n("Mul", ["cinv", "big"], ["cbig"]),
        n("Add", ["bcp", "cbig"], ["bcpb"]), n("ReduceMin", ["bcpb"], ["c0"], axes=[3], keepdims=1),
    ]
    for nm in ("block", "block8", "mcol", "content", "colorgrid"): V(nm, s1)
    V("kgrid", FULL)
    for nm in ("brow", "brp", "rinv", "rbig", "brpb"): V(nm, rc)
    for nm in ("bcol", "bcp", "cinv", "cbig", "bcpb"): V(nm, cc)
    for nm in ("r0", "r1", "c0", "c1"): V(nm, sc)

    # edge masks and span masks
    def cmp(out, op, a, b, shp):
        nodes.append(n(op, [a, b], [out + "_b"])); V(out + "_b", shp, B)
        nodes.append(n("Cast", [out + "_b"], [out], to=F)); V(out, shp)
    cmp("aboveR", "Less", "row_ar", "r0", rc)
    cmp("belowR", "Greater", "row_ar", "r1", rc)
    cmp("leftR", "Less", "col_ar", "c0", cc)
    cmp("rightR", "Greater", "col_ar", "c1", cc)
    # span bounds: c0-0.5, c1+0.5, r0-0.5, r1+0.5
    nodes += [n("Sub", ["c0", "half"], ["c0m"]), n("Add", ["c1", "half"], ["c1p"]),
              n("Sub", ["r0", "half"], ["r0m"]), n("Add", ["r1", "half"], ["r1p"])]
    for nm in ("c0m", "c1p", "r0m", "r1p"): V(nm, sc)
    cmp("cge", "Greater", "col_ar", "c0m", cc)
    cmp("cle", "Less", "col_ar", "c1p", cc)
    nodes.append(n("Mul", ["cge", "cle"], ["cspan"])); V("cspan", cc)
    cmp("rge", "Greater", "row_ar", "r0m", rc)
    cmp("rle", "Less", "row_ar", "r1p", rc)
    nodes.append(n("Mul", ["rge", "rle"], ["rspan"])); V("rspan", rc)
    # edge placement masks: row==r0/r1, col==c0/c1
    for nm, base, shp in (("er0", "r0", rc), ("er1", "r1", rc)):
        nodes += [n("Sub", ["row_ar", base], ["d" + nm]), n("Abs", ["d" + nm], ["ad" + nm]),
                  n("Less", ["ad" + nm, "half"], [nm + "_b"]), n("Cast", [nm + "_b"], [nm], to=F)]
        V("d" + nm, shp); V("ad" + nm, shp); V(nm + "_b", shp, B); V(nm, shp)
    for nm, base, shp in (("ec0", "c0", cc), ("ec1", "c1", cc)):
        nodes += [n("Sub", ["col_ar", base], ["d" + nm]), n("Abs", ["d" + nm], ["ad" + nm]),
                  n("Less", ["ad" + nm, "half"], [nm + "_b"]), n("Cast", [nm + "_b"], [nm], to=F)]
        V("d" + nm, shp); V("ad" + nm, shp); V(nm + "_b", shp, B); V(nm, shp)

    # projections
    def proj(name, region, span_or_none, redu_axis, place_mask):
        # marker colours in region (and span) projected along redu_axis, placed
        src = f"{name}_src"
        nodes.append(n("Mul", ["mcol", region], [src + "0"])); V(src + "0", s1)
        if span_or_none:
            nodes.append(n("Mul", [src + "0", span_or_none], [src])); V(src, s1)
        else:
            src = src + "0"
        pr = f"{name}_pr"
        nodes.append(n("ReduceMax", [src], [pr], axes=[redu_axis], keepdims=1))
        V(pr, cc if redu_axis == 2 else rc)
        gname = f"{name}_g"
        nodes.append(n("Mul", [pr, place_mask], [gname])); V(gname, s1)
        return gname

    g_above = proj("above", "aboveR", "cspan", 2, "er0")
    g_below = proj("below", "belowR", "cspan", 2, "er1")
    g_left = proj("left", "leftR", "rspan", 3, "ec0")
    g_right = proj("right", "rightR", "rspan", 3, "ec1")
    nodes.append(n("Max", [g_above, g_below, g_left, g_right], ["projected"])); V("projected", s1)

    # out_color = where(projected>0, projected, colorgrid); OneHot; mask to grid
    nodes += [
        n("Greater", ["projected", "half"], ["pmask_b"]), n("Cast", ["pmask_b"], ["pmask"], to=F),
        n("Sub", ["one", "pmask"], ["npmask"]),
        n("Mul", ["colorgrid", "npmask"], ["keepc"]),
        n("Add", ["projected", "keepc"], ["outc"]),
        n("Reshape", ["outc", "rs3"], ["outc3"]),
        n("Cast", ["outc3"], ["outc_i"], to=TensorProto.INT64),
        n("OneHot", ["outc_i", "depth", "onoff"], ["oneh"], axis=1),
        n("Mul", ["oneh", "content"], ["output"]),
    ]
    V("pmask_b", s1, B); V("pmask", s1); V("npmask", s1); V("keepc", s1); V("outc", s1)
    V("outc3", [1, HEIGHT, WIDTH]); V("outc_i", [1, HEIGHT, WIDTH], TensorProto.INT64)
    V("oneh", FULL)

    graph = helper.make_graph(nodes, "project_to_block",
                              [helper.make_tensor_value_info("input", F, FULL)],
                              [helper.make_tensor_value_info("output", F, FULL)],
                              initializer=init, value_info=vi)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_project_to_block(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
