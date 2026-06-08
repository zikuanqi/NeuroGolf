"""Solver: fill each enclosed square hole with colour 5 + side (task 302).

Boxes of colour 5 enclose square holes; each hole is filled by a colour that
encodes its side length -- 1x1 -> 6, 2x2 -> 7, 3x3 -> 8 (i.e. ``5 + side``)::

    5 5 5 5 5          5 5 5 5 5
    5 . . . 5    ->    5 8 8 8 5     (3x3 hole -> 8)
    5 . . . 5          5 8 8 8 5
    5 5 5 5 5          5 5 5 5 5

No component labelling is needed: for a *square* hole the horizontal run-length
of enclosed cells is uniform and equals the side, so ``fill = 5 + run``.

Build:
  * background flood from the grid border -> ``enclosed`` cells (24 dilation
    steps);
  * masked left/right cumulative counts (24 unrolled steps) give the run-length
    ``L + R - 1`` per enclosed cell;
  * convert ``5 + run`` to a one-hot and paint it on the holes.
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
NRUN = 24


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


def _enc(g, bg):
    H, W = g.shape; isbg = (g == bg)
    seed = np.zeros((H, W), bool)
    seed[0, :] |= isbg[0, :]; seed[-1, :] |= isbg[-1, :]
    seed[:, 0] |= isbg[:, 0]; seed[:, -1] |= isbg[:, -1]
    return isbg & ~_flood(seed, isbg)


def _runlen(enc):
    H, W = enc.shape; L = np.zeros((H, W), int); R = np.zeros((H, W), int)
    for r in range(H):
        for c in range(W):
            L[r, c] = (L[r, c - 1] + 1) if (enc[r, c] and c > 0) else (1 if enc[r, c] else 0)
        for c in range(W - 1, -1, -1):
            R[r, c] = (R[r, c + 1] + 1) if (enc[r, c] and c < W - 1) else (1 if enc[r, c] else 0)
    return L + R - 1


def _fillcolor(run):
    return 5 + run


def _ref(g: np.ndarray) -> Optional[np.ndarray]:
    bg = _bgcol(g)
    enc = _enc(g, bg)
    if not enc.any():
        return None
    run = _runlen(enc)
    out = g.copy()
    ys, xs = np.where(enc)
    for r, c in zip(ys, xs):
        out[r, c] = _fillcolor(run[r, c])
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


# shared graph fragments -------------------------------------------------------

def _emit_common(n, init, nodes, seen, ctr):
    """Emit nodes computing `enc` and `run`; return their tensor names."""
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

    nodes += [
        n("ReduceSum", ["input"], ["content"], axes=[1], keepdims=1),
        n("ReduceSum", ["input"], ["hist"], axes=[2, 3], keepdims=1),
        n("ReduceMax", ["hist"], ["bgcount"], axes=[1], keepdims=1),
        n("Sub", ["bgcount", "half"], ["bghalf"]),
        n("Greater", ["hist", "bghalf"], ["bgsel_b"]), n("Cast", ["bgsel_b"], ["bgsel"], to=F),
        n("Mul", ["input", "bgsel"], ["ibg"]),
        n("ReduceSum", ["ibg"], ["isbg"], axes=[1], keepdims=1),
    ]
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
    nodes.append(n("Sub", ["isbg", reachb], ["enc"]))
    # run-length: left then right masked cumulative count
    Lp = "enc"
    for k in range(NRUN):
        sh = read_shift(Lp, 0, 1)
        s1 = f"L1_{k}"; s2 = f"L2_{k}"
        nodes.append(n("Add", ["one", sh], [s1]))
        nodes.append(n("Mul", ["enc", s1], [s2]))
        Lp = s2
    Rp = "enc"
    for k in range(NRUN):
        sh = read_shift(Rp, 0, -1)
        s1 = f"R1_{k}"; s2 = f"R2_{k}"
        nodes.append(n("Add", ["one", sh], [s1]))
        nodes.append(n("Mul", ["enc", s1], [s2]))
        Rp = s2
    nodes += [
        n("Add", [Lp, Rp], ["lr"]),
        n("Sub", ["lr", "one"], ["run"]),
    ]
    return read_shift  # not used further; enc/run names fixed


F = TensorProto.FLOAT


def _build() -> onnx.ModelProto:
    n = helper.make_node
    chramp = np.arange(CHANNELS, dtype=np.float32).reshape(1, CHANNELS, 1, 1)
    init = [
        numpy_helper.from_array(chramp, "chramp"),
        numpy_helper.from_array(np.array(0.5, np.float32), "half"),
        numpy_helper.from_array(np.array(1.0, np.float32), "one"),
        numpy_helper.from_array(np.array(5.0, np.float32), "five"),
        numpy_helper.from_array(np.array([2, 3], np.int64), "ax23"),
    ]
    nodes = []
    seen = {"pad": set(), "sl": set()}
    ctr = [0]
    _emit_common(n, init, nodes, seen, ctr)
    nodes += [
        n("Add", ["run", "five"], ["fillval"]),                  # 5 + side
        n("Sub", ["fillval", "chramp"], ["fdiff"]),
        n("Abs", ["fdiff"], ["fad"]),
        n("Less", ["fad", "half"], ["oh_b"]), n("Cast", ["oh_b"], ["oh"], to=F),
        n("Mul", ["oh", "enc"], ["paint"]),
        n("Sub", ["one", "enc"], ["keep"]),
        n("Mul", ["input", "keep"], ["kept"]),
        n("Add", ["kept", "paint"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "hole_size_fill",
                              [helper.make_tensor_value_info("input", F, FULL)],
                              [helper.make_tensor_value_info("output", F, FULL)],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_hole_size_fill(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
