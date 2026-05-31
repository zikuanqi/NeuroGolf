"""Solver: recolour each same-colour connected component by its cell count.

Every 4-connected same-colour component is repainted a single colour determined
by its size, through a per-task size->colour map (tasks 169, 196, 330).

The graph: (1) label components by iterated same-colour max-propagation of a
unique per-cell id; (2) count each component by an all-pairs label match
(reshape to 900, Equal, ReduceSum); (3) map size -> colour.
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
N_ITERS = 26
NCELL = HEIGHT * WIDTH


def _components(g):
    H, W = len(g), len(g[0])
    seen = [[False] * W for _ in range(H)]
    out = []
    for r in range(H):
        for c in range(W):
            if g[r][c] == 0 or seen[r][c]:
                continue
            cells = []
            dq = deque([(r, c)]); seen[r][c] = True
            while dq:
                y, x = dq.popleft(); cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < H and 0 <= xx < W and not seen[yy][xx] \
                            and g[yy][xx] == g[y][x]:
                        seen[yy][xx] = True; dq.append((yy, xx))
            out.append(cells)
    return out


def _transform(grid, size_map):
    H, W = len(grid), len(grid[0])
    out = [[0] * W for _ in range(H)]
    for comp in _components(grid):
        s = len(comp)
        if s not in size_map:
            return None
        for (y, x) in comp:
            out[y][x] = size_map[s]
    return out


def _params(task: dict):
    size_map = {}
    for ex in all_examples(task):
        i, o = ex["input"], ex["output"]
        if len(i) > 30 or len(i[0]) > 30:
            continue
        if len(i) != len(o) or len(i[0]) != len(o[0]):
            return None
        for comp in _components(i):
            cols = {o[y][x] for (y, x) in comp}
            if len(cols) != 1:
                return None
            oc = cols.pop()
            s = len(comp)
            if size_map.get(s, oc) != oc:
                return None
            size_map[s] = oc
    return size_map or None


def _build(size_map: dict) -> onnx.ModelProto:
    F = TensorProto.FLOAT
    B = TensorProto.BOOL

    def f32(name, arr):
        return numpy_helper.from_array(arr.astype(np.float32), name)

    def i64(name, vals):
        return numpy_helper.from_array(np.array(vals, dtype=np.int64), name)

    cellid = (np.arange(NCELL).reshape(1, 1, HEIGHT, WIDTH) + 1).astype(np.float32)
    e_nonbg = np.ones((1, CHANNELS, 1, 1), dtype=np.float32)
    e_nonbg[0, 0, 0, 0] = 0.0
    e0 = np.zeros((1, CHANNELS, 1, 1), dtype=np.float32)
    e0[0, 0, 0, 0] = 1.0

    init = [
        f32("cellid", cellid), f32("e_nonbg", e_nonbg), f32("e0", e0),
        i64("idx0", [0]),
        i64("pad_T", [0, 0, 1, 0, 0, 0, 0, 0]),   # pad top -> neighbour above
        i64("pad_B", [0, 0, 0, 0, 0, 0, 1, 0]),
        i64("pad_L", [0, 0, 0, 1, 0, 0, 0, 0]),
        i64("pad_R", [0, 0, 0, 0, 0, 0, 0, 1]),
        i64("r0", [0]), i64("r30", [HEIGHT]), i64("r1", [1]), i64("r31", [HEIGHT + 1]),
        i64("c30", [WIDTH]), i64("c31", [WIDTH + 1]),
        i64("ax2", [2]), i64("ax3", [3]),
        i64("to_col", [1, 1, NCELL, 1]), i64("to_row", [1, 1, 1, NCELL]),
        i64("to_grid", [1, 1, HEIGHT, WIDTH]),
    ]
    color_consts = sorted({c for c in size_map.values()})
    for c in color_consts:
        a = np.zeros((1, CHANNELS, 1, 1), dtype=np.float32); a[0, c, 0, 0] = 1.0
        init.append(f32(f"ecol_{c}", a))
    for s in sorted(size_map):
        init.append(f32(f"sz_{s}", np.array([[[[float(s)]]]])))

    n = helper.make_node
    nodes = []
    vinfo = []
    g4 = [1, CHANNELS, HEIGHT, WIDTH]
    s1 = [1, 1, HEIGHT, WIDTH]

    def vi(name, shape, dt=F):
        vinfo.append(helper.make_tensor_value_info(name, dt, shape))

    # neighbour-shift helper: produce <name> = value of the neighbour in `d`
    def shift(src, d, name):
        # returns node list appended; declares value_info
        if d == 'T':      # neighbour above (pad top, take first H rows)
            pad, st, en, ax, h = "pad_T", "r0", "r30", "ax2", HEIGHT + 1
        elif d == 'B':
            pad, st, en, ax, h = "pad_B", "r1", "r31", "ax2", HEIGHT + 1
        elif d == 'L':
            pad, st, en, ax, h = "pad_L", "r0", "c30", "ax3", WIDTH + 1
        else:             # 'R'
            pad, st, en, ax, h = "pad_R", "r1", "c31", "ax3", WIDTH + 1
        padded = name + "_p"
        if ax == "ax2":
            vi(padded, [1, src_ch[src], HEIGHT + 1, WIDTH])
        else:
            vi(padded, [1, src_ch[src], HEIGHT, WIDTH + 1])
        vi(name, [1, src_ch[src], HEIGHT, WIDTH])
        nodes.append(n("Pad", [src, pad], [padded]))
        nodes.append(n("Slice", [padded, st, en, ax], [name]))

    src_ch = {}  # track channel count per tensor for value_info of pads

    # content / nonbg
    nodes.append(n("ReduceSum", ["input"], ["content"], axes=[1], keepdims=1)); vi("content", s1)
    nodes.append(n("Gather", ["input", "idx0"], ["ch0"], axis=1)); vi("ch0", s1)
    nodes.append(n("Sub", ["content", "ch0"], ["nonbg"])); vi("nonbg", s1)
    nodes.append(n("Mul", ["cellid", "nonbg"], ["label0"])); vi("label0", s1)

    # same-colour neighbour indicators sc_T/B/L/R  (input has CHANNELS channels)
    src_ch["input"] = CHANNELS
    for d in ("T", "B", "L", "R"):
        sname = f"sin_{d}"; src_ch[sname] = CHANNELS
        shift("input", d, sname)
        nodes.append(n("Mul", ["input", sname], [f"sprod_{d}"])); vi(f"sprod_{d}", g4); src_ch[f"sprod_{d}"] = CHANNELS
        nodes.append(n("Mul", [f"sprod_{d}", "e_nonbg"], [f"sprodn_{d}"])); vi(f"sprodn_{d}", g4); src_ch[f"sprodn_{d}"] = CHANNELS
        nodes.append(n("ReduceSum", [f"sprodn_{d}"], [f"sc_{d}"], axes=[1], keepdims=1)); vi(f"sc_{d}", s1); src_ch[f"sc_{d}"] = 1

    # iterative label propagation
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

    # per-component size via all-pairs label match
    nodes.append(n("Reshape", [label, "to_col"], ["lcol"])); vi("lcol", [1, 1, NCELL, 1])
    nodes.append(n("Reshape", [label, "to_row"], ["lrow"])); vi("lrow", [1, 1, 1, NCELL])
    nodes.append(n("Equal", ["lcol", "lrow"], ["eqm"])); vi("eqm", [1, 1, NCELL, NCELL], B)
    nodes.append(n("Cast", ["eqm"], ["eqmf"], to=F)); vi("eqmf", [1, 1, NCELL, NCELL])
    nodes.append(n("ReduceSum", ["eqmf"], ["szcol"], axes=[3], keepdims=1)); vi("szcol", [1, 1, NCELL, 1])
    nodes.append(n("Reshape", ["szcol", "to_grid"], ["size"])); vi("size", s1)

    # map size -> colour
    terms = []
    for s in sorted(size_map):
        col = size_map[s]
        nodes.append(n("Equal", ["size", f"sz_{s}"], [f"eqs_{s}"])); vi(f"eqs_{s}", s1, B)
        nodes.append(n("Cast", [f"eqs_{s}"], [f"eqsf_{s}"], to=F)); vi(f"eqsf_{s}", s1)
        nodes.append(n("Mul", [f"eqsf_{s}", "nonbg"], [f"sel_{s}"])); vi(f"sel_{s}", s1)
        nodes.append(n("Mul", [f"ecol_{col}", f"sel_{s}"], [f"term_{s}"])); vi(f"term_{s}", g4)
        terms.append(f"term_{s}")
    nodes.append(n("Mul", ["e0", "ch0"], ["out_bg"])); vi("out_bg", g4)
    acc = "out_bg"
    for j, tname in enumerate(terms):
        out = "output" if j == len(terms) - 1 else f"acc_{j}"
        nodes.append(n("Add", [acc, tname], [out]))
        if out != "output":
            vi(out, g4)
        acc = out

    inp = helper.make_tensor_value_info("input", F, g4)
    outp = helper.make_tensor_value_info("output", F, g4)
    graph = helper.make_graph(nodes, "cc_size_recolor", [inp], [outp],
                              initializer=init, value_info=vinfo)
    return helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
        ir_version=IR_VERSION)


def solve_cc_size_recolor(task: dict) -> Optional[onnx.ModelProto]:
    size_map = _params(task)
    if size_map is None:
        return None
    for ex in all_examples(task):
        i, o = ex["input"], ex["output"]
        if len(i) > 30 or len(i[0]) > 30:
            continue
        if _transform(i, size_map) != o:
            return None
    return _build(size_map)
