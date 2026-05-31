"""Solver: recolour each same-colour component by its size *rank*.

Components are ranked by cell count (largest = rank 0); each is repainted a
colour from a per-task rank->colour map (task 374). Builds on the cc-size
machinery: label by max-propagation, count by all-pairs label match, then a
second all-pairs gives each cell's rank = number of component representatives
strictly larger than it.
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
            cells = []; dq = deque([(r, c)]); seen[r][c] = True
            while dq:
                y, x = dq.popleft(); cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < H and 0 <= xx < W and not seen[yy][xx] \
                            and g[yy][xx] == g[y][x]:
                        seen[yy][xx] = True; dq.append((yy, xx))
            out.append(cells)
    return out


def _ranks(g):
    cs = _components(g)
    sizes = [len(c) for c in cs]
    return cs, [sum(1 for s in sizes if s > len(c)) for c in cs]


def _transform(grid, rank_map):
    H, W = len(grid), len(grid[0])
    out = [[0] * W for _ in range(H)]
    cs, ranks = _ranks(grid)
    for comp, rk in zip(cs, ranks):
        if rk not in rank_map:
            return None
        for (y, x) in comp:
            out[y][x] = rank_map[rk]
    return out


def _params(task: dict):
    rank_map = {}
    for ex in all_examples(task):
        i, o = ex["input"], ex["output"]
        if len(i) > 30 or len(i[0]) > 30:
            continue
        if len(i) != len(o) or len(i[0]) != len(o[0]):
            return None
        cs, ranks = _ranks(i)
        sizes = [len(c) for c in cs]
        if len(set(sizes)) != len(sizes):
            return None                      # ties: rank ambiguous for this graph
        for comp, rk in zip(cs, ranks):
            cols = {o[y][x] for (y, x) in comp}
            if len(cols) != 1:
                return None
            oc = cols.pop()
            if rank_map.get(rk, oc) != oc:
                return None
            rank_map[rk] = oc
    return rank_map or None


def _build(rank_map: dict) -> onnx.ModelProto:
    F = TensorProto.FLOAT
    B = TensorProto.BOOL

    def f32(name, arr):
        return numpy_helper.from_array(arr.astype(np.float32), name)

    def i64(name, vals):
        return numpy_helper.from_array(np.array(vals, dtype=np.int64), name)

    cellid = (np.arange(NCELL).reshape(1, 1, HEIGHT, WIDTH) + 1).astype(np.float32)
    e_nonbg = np.ones((1, CHANNELS, 1, 1), dtype=np.float32); e_nonbg[0, 0, 0, 0] = 0.0
    e0 = np.zeros((1, CHANNELS, 1, 1), dtype=np.float32); e0[0, 0, 0, 0] = 1.0

    init = [
        f32("cellid", cellid), f32("e_nonbg", e_nonbg), f32("e0", e0),
        i64("idx0", [0]),
        i64("pad_T", [0, 0, 1, 0, 0, 0, 0, 0]), i64("pad_B", [0, 0, 0, 0, 0, 0, 1, 0]),
        i64("pad_L", [0, 0, 0, 1, 0, 0, 0, 0]), i64("pad_R", [0, 0, 0, 0, 0, 0, 0, 1]),
        i64("r0", [0]), i64("r30", [HEIGHT]), i64("r1", [1]), i64("r31", [HEIGHT + 1]),
        i64("c30", [WIDTH]), i64("c31", [WIDTH + 1]), i64("ax2", [2]), i64("ax3", [3]),
        i64("to_col", [1, 1, NCELL, 1]), i64("to_row", [1, 1, 1, NCELL]),
        i64("to_grid", [1, 1, HEIGHT, WIDTH]),
    ]
    for rk in sorted(rank_map):
        col = rank_map[rk]
        a = np.zeros((1, CHANNELS, 1, 1), dtype=np.float32); a[0, col, 0, 0] = 1.0
        init.append(f32(f"ecol_{rk}", a))
        init.append(f32(f"rk_{rk}", np.array([[[[float(rk)]]]])))

    n = helper.make_node
    nodes, vinfo = [], []
    g4 = [1, CHANNELS, HEIGHT, WIDTH]; s1 = [1, 1, HEIGHT, WIDTH]
    src_ch = {}

    def vi(name, shape, dt=F):
        vinfo.append(helper.make_tensor_value_info(name, dt, shape))

    def shift(src, d, name):
        if d == 'T':
            pad, st, en, ax = "pad_T", "r0", "r30", "ax2"
        elif d == 'B':
            pad, st, en, ax = "pad_B", "r1", "r31", "ax2"
        elif d == 'L':
            pad, st, en, ax = "pad_L", "r0", "c30", "ax3"
        else:
            pad, st, en, ax = "pad_R", "r1", "c31", "ax3"
        ch = src_ch[src]
        if ax == "ax2":
            vi(name + "_p", [1, ch, HEIGHT + 1, WIDTH])
        else:
            vi(name + "_p", [1, ch, HEIGHT, WIDTH + 1])
        vi(name, [1, ch, HEIGHT, WIDTH]); src_ch[name] = ch
        nodes.append(n("Pad", [src, pad], [name + "_p"]))
        nodes.append(n("Slice", [name + "_p", st, en, ax], [name]))

    nodes.append(n("ReduceSum", ["input"], ["content"], axes=[1], keepdims=1)); vi("content", s1)
    nodes.append(n("Gather", ["input", "idx0"], ["ch0"], axis=1)); vi("ch0", s1)
    nodes.append(n("Sub", ["content", "ch0"], ["nonbg"])); vi("nonbg", s1)
    nodes.append(n("Mul", ["cellid", "nonbg"], ["label0"])); vi("label0", s1)
    src_ch["input"] = CHANNELS; src_ch["label0"] = 1

    for d in ("T", "B", "L", "R"):
        shift("input", d, f"sin_{d}")
        nodes.append(n("Mul", ["input", f"sin_{d}"], [f"sp_{d}"])); vi(f"sp_{d}", g4); src_ch[f"sp_{d}"] = CHANNELS
        nodes.append(n("Mul", [f"sp_{d}", "e_nonbg"], [f"spn_{d}"])); vi(f"spn_{d}", g4); src_ch[f"spn_{d}"] = CHANNELS
        nodes.append(n("ReduceSum", [f"spn_{d}"], [f"sc_{d}"], axes=[1], keepdims=1)); vi(f"sc_{d}", s1); src_ch[f"sc_{d}"] = 1

    cur = "label0"
    for it in range(N_ITERS):
        contribs = [cur]
        for d in ("T", "B", "L", "R"):
            shift(cur, d, f"sl_{it}_{d}")
            nodes.append(n("Mul", [f"sl_{it}_{d}", f"sc_{d}"], [f"cb_{it}_{d}"])); vi(f"cb_{it}_{d}", s1); src_ch[f"cb_{it}_{d}"] = 1
            contribs.append(f"cb_{it}_{d}")
        nodes.append(n("Max", contribs, [f"label_{it+1}"])); vi(f"label_{it+1}", s1); src_ch[f"label_{it+1}"] = 1
        cur = f"label_{it+1}"
    label = cur

    # size via all-pairs label match
    nodes.append(n("Reshape", [label, "to_col"], ["lcol"])); vi("lcol", [1, 1, NCELL, 1])
    nodes.append(n("Reshape", [label, "to_row"], ["lrow"])); vi("lrow", [1, 1, 1, NCELL])
    nodes.append(n("Equal", ["lcol", "lrow"], ["eqm"])); vi("eqm", [1, 1, NCELL, NCELL], B)
    nodes.append(n("Cast", ["eqm"], ["eqmf"], to=F)); vi("eqmf", [1, 1, NCELL, NCELL])
    nodes.append(n("ReduceSum", ["eqmf"], ["szcol"], axes=[3], keepdims=1)); vi("szcol", [1, 1, NCELL, 1])
    nodes.append(n("Reshape", ["szcol", "to_grid"], ["size"])); vi("size", s1)

    # rank via second all-pairs: count of representatives strictly larger
    nodes.append(n("Equal", ["cellid", label], ["rep_b"])); vi("rep_b", s1, B)
    nodes.append(n("Cast", ["rep_b"], ["rep"], to=F)); vi("rep", s1)
    nodes.append(n("Reshape", ["size", "to_col"], ["sz_col"])); vi("sz_col", [1, 1, NCELL, 1])
    nodes.append(n("Reshape", ["size", "to_row"], ["sz_row"])); vi("sz_row", [1, 1, 1, NCELL])
    nodes.append(n("Reshape", ["rep", "to_row"], ["rep_row"])); vi("rep_row", [1, 1, 1, NCELL])
    nodes.append(n("Greater", ["sz_row", "sz_col"], ["gt_b"])); vi("gt_b", [1, 1, NCELL, NCELL], B)
    nodes.append(n("Cast", ["gt_b"], ["gtf"], to=F)); vi("gtf", [1, 1, NCELL, NCELL])
    nodes.append(n("Mul", ["gtf", "rep_row"], ["gtrep"])); vi("gtrep", [1, 1, NCELL, NCELL])
    nodes.append(n("ReduceSum", ["gtrep"], ["rkcol"], axes=[3], keepdims=1)); vi("rkcol", [1, 1, NCELL, 1])
    nodes.append(n("Reshape", ["rkcol", "to_grid"], ["rank"])); vi("rank", s1)

    # rank -> colour
    terms = []
    for rk in sorted(rank_map):
        nodes.append(n("Equal", ["rank", f"rk_{rk}"], [f"eqr_{rk}"])); vi(f"eqr_{rk}", s1, B)
        nodes.append(n("Cast", [f"eqr_{rk}"], [f"eqrf_{rk}"], to=F)); vi(f"eqrf_{rk}", s1)
        nodes.append(n("Mul", [f"eqrf_{rk}", "nonbg"], [f"sel_{rk}"])); vi(f"sel_{rk}", s1)
        nodes.append(n("Mul", [f"ecol_{rk}", f"sel_{rk}"], [f"term_{rk}"])); vi(f"term_{rk}", g4)
        terms.append(f"term_{rk}")
    nodes.append(n("Mul", ["e0", "ch0"], ["out_bg"])); vi("out_bg", g4)
    acc = "out_bg"
    for j, tname in enumerate(terms):
        out = "output" if j == len(terms) - 1 else f"acc_{j}"
        nodes.append(n("Add", [acc, tname], [out]))
        if out != "output":
            vi(out, g4)
        acc = out

    graph = helper.make_graph(
        nodes, "cc_rank_recolor",
        [helper.make_tensor_value_info("input", F, g4)],
        [helper.make_tensor_value_info("output", F, g4)],
        initializer=init, value_info=vinfo)
    return helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
        ir_version=IR_VERSION)


def solve_cc_rank_recolor(task: dict) -> Optional[onnx.ModelProto]:
    rank_map = _params(task)
    if rank_map is None:
        return None
    for ex in all_examples(task):
        i, o = ex["input"], ex["output"]
        if len(i) > 30 or len(i[0]) > 30:
            continue
        if _transform(i, rank_map) != o:
            return None
    return _build(rank_map)
