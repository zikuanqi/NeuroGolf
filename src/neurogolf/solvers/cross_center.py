"""Solver: draw a full-grid cross through each box's centre (task 94).

Each hollow rectangle (single non-background colour) gets a grid-spanning cross
of colour 6 through its bounding-box centre, painted over background only (the
box border is kept)::

    8 8 8 8 8            8 8 6 8 8
    8 1 1 1 8     ->     6 1 1 1 6     (cross through the box centre,
    8 1 8 1 8            8 1 6 1 8      over the 8-background)
    8 1 1 1 8            6 1 1 1 6
    8 8 8 8 8            8 8 6 8 8

Each shape cell learns its component's ``(minr,maxr,minc,maxc)`` via min/max
floods *through the shape mask*; the centre row/col are ``(min+max)/2``.  A
hollow box's centre row holds its left/right border cells, so ``ReduceMax`` over
the shape cells that lie on their own centre row/col yields per-row / per-col
cross flags, broadcast to a full cross and painted ``e_6`` over the background.
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
BIG = 1.0e6
F = TensorProto.FLOAT


def _bgcol(g):
    v, c = np.unique(g, return_counts=True)
    return v[c.argmax()]


def _shift(a, dr, dc, fill):
    H, W = a.shape; r = np.full_like(a, fill)
    ys = slice(max(0, dr), H + min(0, dr)); yd = slice(max(0, -dr), H + min(0, -dr))
    xs = slice(max(0, dc), W + min(0, dc)); xd = slice(max(0, -dc), W + min(0, -dc))
    r[yd, xd] = a[ys, xs]; return r


def _bounds(shape):
    H, W = shape.shape
    R = np.broadcast_to(np.arange(H).reshape(-1, 1), (H, W)).astype(float)
    C = np.broadcast_to(np.arange(W).reshape(1, -1), (H, W)).astype(float)
    minr = np.where(shape, R, BIG); maxr = np.where(shape, R, -BIG)
    minc = np.where(shape, C, BIG); maxc = np.where(shape, C, -BIG)
    for _ in range(80):
        a = minr.copy(); b = maxr.copy(); c = minc.copy(); d = maxc.copy()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a = np.minimum(a, _shift(minr, dr, dc, BIG))
            b = np.maximum(b, _shift(maxr, dr, dc, -BIG))
            c = np.minimum(c, _shift(minc, dr, dc, BIG))
            d = np.maximum(d, _shift(maxc, dr, dc, -BIG))
        a = np.where(shape, a, BIG); b = np.where(shape, b, -BIG)
        c = np.where(shape, c, BIG); d = np.where(shape, d, -BIG)
        if np.array_equal(a, minr) and np.array_equal(b, maxr) and \
           np.array_equal(c, minc) and np.array_equal(d, maxc):
            break
        minr, maxr, minc, maxc = a, b, c, d
    return minr, maxr, minc, maxc


def _ref(g: np.ndarray) -> Optional[np.ndarray]:
    bg = _bgcol(g); H, W = g.shape
    shape = (g != bg)
    if not shape.any():
        return None
    minr, maxr, minc, maxc = _bounds(shape)
    cr = (minr + maxr) / 2.0; cc = (minc + maxc) / 2.0
    R = np.broadcast_to(np.arange(H).reshape(-1, 1), (H, W)).astype(float)
    C = np.broadcast_to(np.arange(W).reshape(1, -1), (H, W)).astype(float)
    crowflag = (shape & (np.abs(R - cr) < 0.5)).any(axis=1, keepdims=True)
    ccolflag = (shape & (np.abs(C - cc) < 0.5)).any(axis=0, keepdims=True)
    cross = crowflag | ccolflag
    out = g.copy()
    out[(g == bg) & cross] = 6
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
    Rarr = np.broadcast_to(np.arange(HEIGHT, dtype=np.float32).reshape(1, 1, HEIGHT, 1),
                           (1, 1, HEIGHT, WIDTH)).copy()
    Carr = np.broadcast_to(np.arange(WIDTH, dtype=np.float32).reshape(1, 1, 1, WIDTH),
                           (1, 1, HEIGHT, WIDTH)).copy()
    e6 = np.zeros((1, CHANNELS, 1, 1), np.float32); e6[0, 6] = 1.0
    init = [
        numpy_helper.from_array(Rarr, "R"),
        numpy_helper.from_array(Carr, "C"),
        numpy_helper.from_array(e6, "e6"),
        numpy_helper.from_array(np.array(0.5, np.float32), "half"),
        numpy_helper.from_array(np.array(1.0, np.float32), "one"),
        numpy_helper.from_array(np.array(BIG, np.float32), "big"),
        numpy_helper.from_array(np.array(-BIG, np.float32), "negbig"),
        numpy_helper.from_array(np.array([2, 3], np.int64), "ax23"),
    ]
    nodes = []
    seen = set(); ctr = [0]

    def read_shift(x, ar, ac, padname):
        pt, pl, pb, pr = max(ar, 0), max(ac, 0), max(-ar, 0), max(-ac, 0)
        pname = f"pad_{ar}_{ac}"
        if pname not in seen:
            init.append(numpy_helper.from_array(
                np.array([0, 0, pt, pl, 0, 0, pb, pr], np.int64), pname)); seen.add(pname)
        rs, cs = max(-ar, 0), max(-ac, 0)
        sname, ename = f"sst_{rs}_{cs}", f"sen_{rs}_{cs}"
        if sname not in seen:
            init.append(numpy_helper.from_array(np.array([rs, cs], np.int64), sname))
            init.append(numpy_helper.from_array(np.array([rs + HEIGHT, cs + WIDTH], np.int64), ename))
            seen.add(sname)
        ctr[0] += 1
        pid, oid = f"ps{ctr[0]}", f"rs{ctr[0]}"
        nodes.append(n("Pad", [x, pname, padname], [pid], mode="constant"))
        nodes.append(n("Slice", [pid, sname, ename, "ax23"], [oid]))
        return oid

    DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))
    nodes += [
        n("ReduceSum", ["input"], ["content"], axes=[1], keepdims=1),
        n("ReduceSum", ["input"], ["hist"], axes=[2, 3], keepdims=1),
        n("ReduceMax", ["hist"], ["bgcount"], axes=[1], keepdims=1),
        n("Sub", ["bgcount", "half"], ["bghalf"]),
        n("Greater", ["hist", "bghalf"], ["bgsel_b"]), n("Cast", ["bgsel_b"], ["bgsel"], to=F),
        n("Mul", ["input", "bgsel"], ["ibg"]),
        n("ReduceSum", ["ibg"], ["is_bg"], axes=[1], keepdims=1),
        n("Sub", ["content", "is_bg"], ["shape"]),
        n("Sub", ["one", "shape"], ["notshape"]),
        n("Mul", ["R", "shape"], ["Rs"]), n("Mul", ["C", "shape"], ["Cs"]),
        n("Mul", ["big", "notshape"], ["bigout"]), n("Mul", ["negbig", "notshape"], ["negout"]),
        n("Add", ["Rs", "bigout"], ["minr0"]), n("Add", ["Rs", "negout"], ["maxr0"]),
        n("Add", ["Cs", "bigout"], ["minc0"]), n("Add", ["Cs", "negout"], ["maxc0"]),
    ]
    minr, maxr, minc, maxc = "minr0", "maxr0", "minc0", "maxc0"
    for k in range(NITER):
        ir = [minr]; xr = [maxr]; ic = [minc]; xc = [maxc]
        for (dr, dc) in DIRS:
            ir.append(read_shift(minr, dr, dc, "big"))
            xr.append(read_shift(maxr, dr, dc, "negbig"))
            ic.append(read_shift(minc, dr, dc, "big"))
            xc.append(read_shift(maxc, dr, dc, "negbig"))
        amnr = f"amnr{k}"; amxr = f"amxr{k}"; amnc = f"amnc{k}"; amxc = f"amxc{k}"
        nodes += [n("Min", ir, [amnr]), n("Max", xr, [amxr]), n("Min", ic, [amnc]), n("Max", xc, [amxc])]
        # re-mask to shape (keep BIG/-BIG outside) -- reuse precomputed bigout/negout
        nmnr = f"minr{k+1}"; nmxr = f"maxr{k+1}"; nmnc = f"minc{k+1}"; nmxc = f"maxc{k+1}"
        nodes += [
            n("Mul", [amnr, "shape"], [nmnr + "s"]), n("Add", [nmnr + "s", "bigout"], [nmnr]),
            n("Mul", [amxr, "shape"], [nmxr + "s"]), n("Add", [nmxr + "s", "negout"], [nmxr]),
            n("Mul", [amnc, "shape"], [nmnc + "s"]), n("Add", [nmnc + "s", "bigout"], [nmnc]),
            n("Mul", [amxc, "shape"], [nmxc + "s"]), n("Add", [nmxc + "s", "negout"], [nmxc]),
        ]
        minr, maxr, minc, maxc = nmnr, nmxr, nmnc, nmxc
    nodes += [
        n("Add", [minr, maxr], ["sumr"]), n("Mul", ["sumr", "half"], ["cr"]),
        n("Add", [minc, maxc], ["sumc"]), n("Mul", ["sumc", "half"], ["ccol"]),
        n("Sub", ["R", "cr"], ["dr"]), n("Abs", ["dr"], ["adr"]),
        n("Less", ["adr", "half"], ["oncr_b"]), n("Cast", ["oncr_b"], ["oncr_f"], to=F),
        n("Mul", ["shape", "oncr_f"], ["oncr"]),
        n("ReduceMax", ["oncr"], ["crowflag"], axes=[3], keepdims=1),
        n("Sub", ["C", "ccol"], ["dc"]), n("Abs", ["dc"], ["adc"]),
        n("Less", ["adc", "half"], ["oncc_b"]), n("Cast", ["oncc_b"], ["oncc_f"], to=F),
        n("Mul", ["shape", "oncc_f"], ["oncc"]),
        n("ReduceMax", ["oncc"], ["ccolflag"], axes=[2], keepdims=1),
        n("Max", ["crowflag", "ccolflag"], ["cross"]),
        n("Mul", ["cross", "is_bg"], ["paint"]),
        n("Sub", ["one", "paint"], ["keep"]),
        n("Mul", ["input", "keep"], ["kept"]),
        n("Mul", ["e6", "paint"], ["addc"]),
        n("Add", ["kept", "addc"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "cross_center",
                              [helper.make_tensor_value_info("input", F, FULL)],
                              [helper.make_tensor_value_info("output", F, FULL)],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_cross_center(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
