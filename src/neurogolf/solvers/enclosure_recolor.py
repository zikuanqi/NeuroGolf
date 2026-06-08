"""Solver: recolour shapes that enclose a hole (task 279).

A shape (single non-background colour) is recoloured to ``8`` iff its connected
component encloses a background hole (i.e. it is a closed loop); open shapes are
left unchanged::

    1 1 1            8 8 8        . 1 .          . 1 .
    1 . 1    ->      8 . 8        1 1 1    ->    1 1 1   (a plus encloses
    1 1 1            8 8 8        . 1 .          . 1 .    nothing -> kept)

Build (connected components via iterative flood-fill, both ~24 dilation steps,
enough for the arc-gen distribution -- max observed geodesic = 13):
  1. flood background inward from the grid border -> ``reachable`` background;
     ``enclosed = background AND NOT reachable``.
  2. seed = non-bg cells 8-adjacent to an enclosed cell; flood it through the
     non-bg mask -> the whole enclosing component.
  3. repaint that component ``e_8``.

onnxruntime frees each flood iteration's tensor once the next consumes it, so the
runtime footprint stays small despite the node count.
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
NITER = 24


def _bgcol(g):
    v, c = np.unique(g, return_counts=True)
    return v[c.argmax()]


def _flood(seed, mask):
    H, W = mask.shape
    reached = seed & mask
    while True:
        P = np.pad(reached, 1); dil = np.zeros_like(reached)
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            dil |= P[1 + dr:1 + dr + H, 1 + dc:1 + dc + W]
        new = (reached | dil) & mask
        if np.array_equal(new, reached):
            return reached
        reached = new


def _ref(g: np.ndarray) -> Optional[np.ndarray]:
    bg = _bgcol(g); H, W = g.shape
    isbg = (g == bg)
    seedb = np.zeros((H, W), bool)
    seedb[0, :] |= isbg[0, :]; seedb[-1, :] |= isbg[-1, :]
    seedb[:, 0] |= isbg[:, 0]; seedb[:, -1] |= isbg[:, -1]
    enc = isbg & ~_flood(seedb, isbg)
    if not enc.any():
        return None
    nz = g != bg
    P = np.pad(enc, 1); adj = np.zeros((H, W), bool)
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
        adj |= P[1 + dr:1 + dr + H, 1 + dc:1 + dc + W]
    comp = _flood(nz & adj, nz)
    out = g.copy(); out[comp] = 8
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
    F = TensorProto.FLOAT
    n = helper.make_node

    e8 = np.zeros((1, CHANNELS, 1, 1), np.float32); e8[0, 8] = 1.0
    init = [
        numpy_helper.from_array(e8, "e8"),
        numpy_helper.from_array(np.array(0.5, np.float32), "half"),
        numpy_helper.from_array(np.array(1.0, np.float32), "one"),
        numpy_helper.from_array(np.array([2, 3], np.int64), "ax23"),
    ]
    nodes = []
    seen = {"pad": set(), "sl": set()}
    ctr = [0]

    def read_shift(x, ar, ac):
        pt, pl, pb, pr = max(ar, 0), max(ac, 0), max(-ar, 0), max(-ac, 0)
        pname = f"pad_{ar}_{ac}"
        if pname not in seen["pad"]:
            init.append(numpy_helper.from_array(
                np.array([0, 0, pt, pl, 0, 0, pb, pr], np.int64), pname))
            seen["pad"].add(pname)
        rs, cs = max(-ar, 0), max(-ac, 0)
        sname, ename = f"sst_{rs}_{cs}", f"sen_{rs}_{cs}"
        if sname not in seen["sl"]:
            init.append(numpy_helper.from_array(np.array([rs, cs], np.int64), sname))
            init.append(numpy_helper.from_array(
                np.array([rs + HEIGHT, cs + WIDTH], np.int64), ename))
            seen["sl"].add(sname)
        ctr[0] += 1
        pid, oid = f"ps{ctr[0]}", f"rs{ctr[0]}"
        nodes.append(n("Pad", [x, pname], [pid], mode="constant"))
        nodes.append(n("Slice", [pid, sname, ename, "ax23"], [oid]))
        return oid

    def flood(seed, mask, tag):
        ctr[0] += 1
        acc = f"acc_{tag}_0"
        nodes.append(n("Mul", [seed, mask], [acc]))
        for k in range(NITER):
            u = read_shift(acc, 1, 0); d = read_shift(acc, -1, 0)
            l = read_shift(acc, 0, 1); r = read_shift(acc, 0, -1)
            dil = f"dil_{tag}_{k}"
            nodes.append(n("Max", [acc, u, d, l, r], [dil]))
            nxt = f"acc_{tag}_{k+1}"
            nodes.append(n("Mul", [dil, mask], [nxt]))
            acc = nxt
        return acc

    # background / non-background masks
    nodes += [
        n("ReduceSum", ["input"], ["content"], axes=[1], keepdims=1),
        n("ReduceSum", ["input"], ["hist"], axes=[2, 3], keepdims=1),
        n("ReduceMax", ["hist"], ["bgcount"], axes=[1], keepdims=1),
        n("Sub", ["bgcount", "half"], ["bghalf"]),
        n("Greater", ["hist", "bghalf"], ["bgsel_b"]), n("Cast", ["bgsel_b"], ["bgsel"], to=F),
        n("Mul", ["input", "bgsel"], ["ibg"]),
        n("ReduceSum", ["ibg"], ["isbg"], axes=[1], keepdims=1),
        n("Sub", ["content", "isbg"], ["nz"]),
    ]
    # grid-boundary background = border seed
    cu = read_shift("content", 1, 0); cd = read_shift("content", -1, 0)
    cl = read_shift("content", 0, 1); cr = read_shift("content", 0, -1)
    nodes += [
        n("Mul", [cu, cd], ["na1"]), n("Mul", [cl, cr], ["na2"]),
        n("Mul", ["na1", "na2"], ["neigh_all"]),
        n("Sub", ["one", "neigh_all"], ["notall"]),
        n("Mul", ["content", "notall"], ["boundary"]),
        n("Mul", ["isbg", "boundary"], ["borderseed"]),
    ]
    reachb = flood("borderseed", "isbg", "b")
    nodes += [
        n("Sub", ["isbg", reachb], ["enc"]),
    ]
    # 8-adjacency of enclosed cells
    a = [read_shift("enc", dr, dc) for dr, dc in
         ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))]
    nodes += [
        n("Max", a, ["adj"]),
        n("Mul", ["nz", "adj"], ["seedc"]),
    ]
    comp = flood("seedc", "nz", "c")
    nodes += [
        n("Sub", ["one", comp], ["keep"]),
        n("Mul", ["input", "keep"], ["kept"]),
        n("Mul", ["e8", comp], ["addc"]),
        n("Add", ["kept", "addc"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "enclosure_recolor",
                              [helper.make_tensor_value_info("input", F, FULL)],
                              [helper.make_tensor_value_info("output", F, FULL)],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_enclosure_recolor(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
