"""Solver: crop to the largest connected component (task 36).

A noisy grid holds many scattered single cells plus one solid blob; the output
is that blob cropped to its bounding box. The blob is the largest 4-connected
same-colour component.

Pipeline = `cc_size_recolor`'s labelling + sizing, then `bbox_strip`'s crop:
  1. label each cell with the max per-cell id in its 4-connected same-colour
     component (iterated max-propagation);
  2. component size per cell via an all-pairs label match;
  3. keep only the cells whose size == the global max (the largest component);
  4. crop that masked grid to its bounding box, top-left aligned.
"""
from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from ..grids import CHANNELS, HEIGHT, WIDTH, all_examples

OPSET = 11
IR_VERSION = 8
N_ITERS = 20
NCELL = HEIGHT * WIDTH


def _ref(g: np.ndarray) -> Optional[np.ndarray]:
    H, W = g.shape
    seen = np.zeros_like(g, bool)
    comps = []
    for r in range(H):
        for c in range(W):
            if g[r, c] and not seen[r, c]:
                col = g[r, c]; q = deque([(r, c)]); seen[r, c] = True; cells = [(r, c)]
                while q:
                    y, x = q.popleft()
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < H and 0 <= nx < W and not seen[ny, nx] and g[ny, nx] == col:
                            seen[ny, nx] = True; q.append((ny, nx)); cells.append((ny, nx))
                comps.append(cells)
    if not comps:
        return None
    szs = sorted((len(c) for c in comps), reverse=True)
    if len(szs) > 1 and szs[0] == szs[1]:
        return None  # ambiguous largest
    big = max(comps, key=len)
    ys = [y for y, x in big]; xs = [x for y, x in big]
    return g[min(ys):max(ys) + 1, min(xs):max(xs) + 1]


def _detect(task: dict) -> bool:
    saw = False
    for ex in all_examples(task):
        i, o = ex["input"], ex["output"]
        if not i or not i[0] or not o or not o[0]:
            return False
        if len(i) > HEIGHT or len(i[0]) > WIDTH:
            continue
        ref = _ref(np.array(i))
        if ref is None or not np.array_equal(ref, np.array(o)):
            return False
        saw = True
    return saw


def _build() -> onnx.ModelProto:
    F = TensorProto.FLOAT
    B = TensorProto.BOOL

    def f32(name, arr): return numpy_helper.from_array(np.asarray(arr, np.float32), name)
    def i64(name, vals): return numpy_helper.from_array(np.array(vals, np.int64), name)

    cellid = (np.arange(NCELL).reshape(1, 1, HEIGHT, WIDTH) + 1).astype(np.float32)
    e_nonbg = np.ones((1, CHANNELS, 1, 1), dtype=np.float32); e_nonbg[0, 0, 0, 0] = 0.0
    e0 = np.zeros((1, CHANNELS, 1, 1), dtype=np.float32); e0[0, 0, 0, 0] = 1.0
    init = [
        f32("cellid", cellid), f32("e_nonbg", e_nonbg), f32("e0", e0), i64("idx0", [0]),
        i64("pad_T", [0, 0, 1, 0, 0, 0, 0, 0]), i64("pad_B", [0, 0, 0, 0, 0, 0, 1, 0]),
        i64("pad_L", [0, 0, 0, 1, 0, 0, 0, 0]), i64("pad_R", [0, 0, 0, 0, 0, 0, 0, 1]),
        i64("r0", [0]), i64("r30", [HEIGHT]), i64("r1", [1]), i64("r31", [HEIGHT + 1]),
        i64("c30", [WIDTH]), i64("c31", [WIDTH + 1]), i64("ax2", [2]), i64("ax3", [3]),
        i64("to_col", [1, 1, NCELL, 1]), i64("to_row", [1, 1, 1, NCELL]),
        i64("to_grid", [1, 1, HEIGHT, WIDTH]),
        f32("half", 0.5),
        # bbox-crop constants
        f32("arange_h", np.arange(HEIGHT)), f32("arange_w", np.arange(WIDTH)),
        i64("ah_shape", [1, 1, HEIGHT, 1]), i64("aw_shape", [1, 1, 1, WIDTH]),
        i64("hidx_shape", [HEIGHT]), i64("widx_shape", [WIDTH]),
        f32("clipmin", 0.0), f32("clipmax", float(HEIGHT - 1)), f32("one_f", 1.0),
    ]
    n = helper.make_node
    nodes = []; vinfo = []
    g4 = [1, CHANNELS, HEIGHT, WIDTH]; s1 = [1, 1, HEIGHT, WIDTH]

    def vi(name, shape, dt=F): vinfo.append(helper.make_tensor_value_info(name, dt, shape))

    src_ch = {}

    def shift(src, d, name):
        if d == 'T': pad, st, en, ax = "pad_T", "r0", "r30", "ax2"
        elif d == 'B': pad, st, en, ax = "pad_B", "r1", "r31", "ax2"
        elif d == 'L': pad, st, en, ax = "pad_L", "r0", "c30", "ax3"
        else: pad, st, en, ax = "pad_R", "r1", "c31", "ax3"
        padded = name + "_p"
        if ax == "ax2": vi(padded, [1, src_ch[src], HEIGHT + 1, WIDTH])
        else: vi(padded, [1, src_ch[src], HEIGHT, WIDTH + 1])
        vi(name, [1, src_ch[src], HEIGHT, WIDTH])
        nodes.append(n("Pad", [src, pad], [padded]))
        nodes.append(n("Slice", [padded, st, en, ax], [name]))

    # nonbg + initial labels
    nodes.append(n("ReduceSum", ["input"], ["content"], axes=[1], keepdims=1)); vi("content", s1)
    nodes.append(n("Gather", ["input", "idx0"], ["ch0"], axis=1)); vi("ch0", s1)
    nodes.append(n("Sub", ["content", "ch0"], ["nonbg"])); vi("nonbg", s1)
    nodes.append(n("Mul", ["cellid", "nonbg"], ["label0"])); vi("label0", s1)

    # same-colour neighbour indicators
    src_ch["input"] = CHANNELS
    for d in ("T", "B", "L", "R"):
        sname = f"sin_{d}"; src_ch[sname] = CHANNELS
        shift("input", d, sname)
        nodes.append(n("Mul", ["input", sname], [f"sprod_{d}"])); vi(f"sprod_{d}", g4); src_ch[f"sprod_{d}"] = CHANNELS
        nodes.append(n("Mul", [f"sprod_{d}", "e_nonbg"], [f"sprodn_{d}"])); vi(f"sprodn_{d}", g4); src_ch[f"sprodn_{d}"] = CHANNELS
        nodes.append(n("ReduceSum", [f"sprodn_{d}"], [f"sc_{d}"], axes=[1], keepdims=1)); vi(f"sc_{d}", s1); src_ch[f"sc_{d}"] = 1

    cur = "label0"; src_ch["label0"] = 1
    for it in range(N_ITERS):
        contribs = [cur]
        for d in ("T", "B", "L", "R"):
            sl = f"slab_{it}_{d}"; src_ch[sl] = 1
            shift(cur, d, sl)
            cn = f"contrib_{it}_{d}"
            nodes.append(n("Mul", [sl, f"sc_{d}"], [cn])); vi(cn, s1); src_ch[cn] = 1
            contribs.append(cn)
        nxt = f"label_{it+1}"
        nodes.append(n("Max", contribs, [nxt])); vi(nxt, s1); src_ch[nxt] = 1
        cur = nxt
    label = cur

    # component size via all-pairs label match
    nodes.append(n("Reshape", [label, "to_col"], ["lcol"])); vi("lcol", [1, 1, NCELL, 1])
    nodes.append(n("Reshape", [label, "to_row"], ["lrow"])); vi("lrow", [1, 1, 1, NCELL])
    nodes.append(n("Equal", ["lcol", "lrow"], ["eqm"])); vi("eqm", [1, 1, NCELL, NCELL], B)
    nodes.append(n("Cast", ["eqm"], ["eqmf"], to=F)); vi("eqmf", [1, 1, NCELL, NCELL])
    nodes.append(n("ReduceSum", ["eqmf"], ["szcol"], axes=[3], keepdims=1)); vi("szcol", [1, 1, NCELL, 1])
    nodes.append(n("Reshape", ["szcol", "to_grid"], ["size"])); vi("size", s1)
    # background cells get size = NCELL (all bg share label 0); exclude them
    nodes.append(n("Mul", ["size", "nonbg"], ["size_nb"])); vi("size_nb", s1)
    nodes.append(n("ReduceMax", ["size_nb"], ["maxsz"], axes=[0, 1, 2, 3], keepdims=1)); vi("maxsz", [1, 1, 1, 1])
    nodes.append(n("Sub", ["maxsz", "half"], ["maxsz_t"])); vi("maxsz_t", [1, 1, 1, 1])
    nodes.append(n("Greater", ["size_nb", "maxsz_t"], ["lmask_b"])); vi("lmask_b", s1, B)
    nodes.append(n("Cast", ["lmask_b"], ["lmask"], to=F)); vi("lmask", s1)
    nodes.append(n("Mul", ["input", "lmask"], ["keep"])); vi("keep", g4)

    # ---- bbox crop of `keep` (bg = 0) ----
    nodes.append(n("ReduceSum", ["keep"], ["ksum"], axes=[1], keepdims=1)); vi("ksum", s1)
    nodes.append(n("ReduceMax", ["ksum"], ["rproj"], axes=[3], keepdims=1)); vi("rproj", [1, 1, HEIGHT, 1])
    nodes.append(n("ReduceMax", ["ksum"], ["cproj"], axes=[2], keepdims=1)); vi("cproj", [1, 1, 1, WIDTH])
    nodes.append(n("Reshape", ["arange_h", "ah_shape"], ["ah4"])); vi("ah4", [1, 1, HEIGHT, 1])
    nodes.append(n("Reshape", ["arange_w", "aw_shape"], ["aw4"])); vi("aw4", [1, 1, 1, WIDTH])
    nodes.append(n("Mul", ["rproj", "ah4"], ["rpos"])); vi("rpos", [1, 1, HEIGHT, 1])
    nodes.append(n("Mul", ["cproj", "aw4"], ["cpos"])); vi("cpos", [1, 1, 1, WIDTH])
    nodes.append(n("ReduceMax", ["rpos"], ["rmaxf"], axes=[2], keepdims=1)); vi("rmaxf", [1, 1, 1, 1])
    nodes.append(n("ReduceMax", ["cpos"], ["cmaxf"], axes=[3], keepdims=1)); vi("cmaxf", [1, 1, 1, 1])
    nodes.append(n("ArgMax", ["rproj"], ["rmini"], axis=2, keepdims=1)); vi("rmini", [1, 1, 1, 1], TensorProto.INT64)
    nodes.append(n("ArgMax", ["cproj"], ["cmini"], axis=3, keepdims=1)); vi("cmini", [1, 1, 1, 1], TensorProto.INT64)
    nodes.append(n("Cast", ["rmini"], ["rminf"], to=F)); vi("rminf", [1, 1, 1, 1])
    nodes.append(n("Cast", ["cmini"], ["cminf"], to=F)); vi("cminf", [1, 1, 1, 1])
    nodes.append(n("Sub", ["rmaxf", "rminf"], ["hm1"])); vi("hm1", [1, 1, 1, 1])
    nodes.append(n("Sub", ["cmaxf", "cminf"], ["wm1"])); vi("wm1", [1, 1, 1, 1])
    nodes.append(n("Add", ["ah4", "rminf"], ["srf"])); vi("srf", [1, 1, HEIGHT, 1])
    nodes.append(n("Clip", ["srf", "clipmin", "clipmax"], ["src"])); vi("src", [1, 1, HEIGHT, 1])
    nodes.append(n("Cast", ["src"], ["sri"], to=TensorProto.INT64)); vi("sri", [1, 1, HEIGHT, 1], TensorProto.INT64)
    nodes.append(n("Reshape", ["sri", "hidx_shape"], ["sr1d"])); vi("sr1d", [HEIGHT], TensorProto.INT64)
    nodes.append(n("Add", ["aw4", "cminf"], ["scf"])); vi("scf", [1, 1, 1, WIDTH])
    nodes.append(n("Clip", ["scf", "clipmin", "clipmax"], ["scc"])); vi("scc", [1, 1, 1, WIDTH])
    nodes.append(n("Cast", ["scc"], ["sci"], to=TensorProto.INT64)); vi("sci", [1, 1, 1, WIDTH], TensorProto.INT64)
    nodes.append(n("Reshape", ["sci", "widx_shape"], ["sc1d"])); vi("sc1d", [WIDTH], TensorProto.INT64)
    nodes.append(n("Gather", ["keep", "sr1d"], ["shr"], axis=2)); vi("shr", g4)
    nodes.append(n("Gather", ["shr", "sc1d"], ["shifted"], axis=3)); vi("shifted", g4)
    nodes.append(n("Add", ["hm1", "one_f"], ["hf"])); vi("hf", [1, 1, 1, 1])
    nodes.append(n("Add", ["wm1", "one_f"], ["wf"])); vi("wf", [1, 1, 1, 1])
    nodes.append(n("Less", ["ah4", "hf"], ["hle"])); vi("hle", [1, 1, HEIGHT, 1], B)
    nodes.append(n("Less", ["aw4", "wf"], ["wle"])); vi("wle", [1, 1, 1, WIDTH], B)
    nodes.append(n("Cast", ["hle"], ["hlef"], to=F)); vi("hlef", [1, 1, HEIGHT, 1])
    nodes.append(n("Cast", ["wle"], ["wlef"], to=F)); vi("wlef", [1, 1, 1, WIDTH])
    nodes.append(n("Mul", ["hlef", "wlef"], ["bmask"])); vi("bmask", s1)
    nodes.append(n("Mul", ["shifted", "bmask"], ["masked"])); vi("masked", g4)
    # fill the bbox's non-object cells with background (channel 0)
    nodes.append(n("ReduceSum", ["masked"], ["hascol2"], axes=[1], keepdims=1)); vi("hascol2", s1)
    nodes.append(n("Sub", ["bmask", "hascol2"], ["bgcell"])); vi("bgcell", s1)
    nodes.append(n("Mul", ["e0", "bgcell"], ["bgch0"])); vi("bgch0", g4)
    nodes.append(n("Add", ["masked", "bgch0"], ["output"]))

    inp = helper.make_tensor_value_info("input", F, g4)
    outp = helper.make_tensor_value_info("output", F, g4)
    graph = helper.make_graph(nodes, "largest_comp_crop", [inp], [outp],
                              initializer=init, value_info=vinfo)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_largest_comp_crop(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
