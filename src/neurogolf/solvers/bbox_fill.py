"""Solver: fill each colour-2 component's bounding box (task 77).

Every connected blob of colour 2 has a bounding box; the structure-colour cells
(the single non-background, non-2 colour) that fall inside any such box are
recoloured to 4::

    2 1 1 1            2 4 4 4
    1 2 1 2     ->     4 2 4 2     (1-cells inside the 2-blob's bbox -> 4)
    2 1 2 2            2 4 2 2

Per-component bounding boxes are rasterised with a **co-evolving bound-flood**: a
growing "inbox" carries per-cell ``(minr,maxr,minc,maxc)`` (seeded on the 2-cells
with their own coordinates).  Each step a candidate cell adjacent to the inbox
takes the min/max bounds of its inbox neighbours (BIG/-BIG padded shifts pick out
inbox neighbours automatically) and joins the inbox iff it lies within them.
After convergence the inbox is exactly the union of the component bounding boxes.
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
NITER = 14
BIG = 1.0e6
F = TensorProto.FLOAT


def _shift(a, dr, dc, fill):
    H, W = a.shape; r = np.full_like(a, fill)
    ys = slice(max(0, dr), H + min(0, dr)); yd = slice(max(0, -dr), H + min(0, -dr))
    xs = slice(max(0, dc), W + min(0, dc)); xd = slice(max(0, -dc), W + min(0, -dc))
    r[yd, xd] = a[ys, xs]; return r


def _raster(is2):
    H, W = is2.shape
    R = np.broadcast_to(np.arange(H).reshape(-1, 1), (H, W)).astype(float)
    C = np.broadcast_to(np.arange(W).reshape(1, -1), (H, W)).astype(float)
    inbox = is2.copy()
    minr = np.where(inbox, R, BIG); maxr = np.where(inbox, R, -BIG)
    minc = np.where(inbox, C, BIG); maxc = np.where(inbox, C, -BIG)
    for _ in range(80):
        cminr = np.full((H, W), BIG); cmaxr = np.full((H, W), -BIG)
        cminc = np.full((H, W), BIG); cmaxc = np.full((H, W), -BIG)
        adj = np.zeros((H, W), bool)
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            adj |= _shift(inbox, dr, dc, False)
            cminr = np.minimum(cminr, _shift(minr, dr, dc, BIG))
            cmaxr = np.maximum(cmaxr, _shift(maxr, dr, dc, -BIG))
            cminc = np.minimum(cminc, _shift(minc, dr, dc, BIG))
            cmaxc = np.maximum(cmaxc, _shift(maxc, dr, dc, -BIG))
        within = (R >= cminr) & (R <= cmaxr) & (C >= cminc) & (C <= cmaxc)
        newcell = adj & within & (~inbox)
        if not newcell.any():
            break
        inbox = inbox | newcell
        minr = np.where(newcell, cminr, minr); maxr = np.where(newcell, cmaxr, maxr)
        minc = np.where(newcell, cminc, minc); maxc = np.where(newcell, cmaxc, maxc)
    return inbox


def _ref(g: np.ndarray) -> Optional[np.ndarray]:
    is2 = (g == 2)
    if not is2.any():
        return None
    cols = [c for c in np.unique(g) if c not in (0, 2)]
    if len(cols) != 1:
        return None
    sc = cols[0]
    box = _raster(is2)
    out = g.copy()
    out[(g == sc) & box] = 4
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
    e4 = np.zeros((1, CHANNELS, 1, 1), np.float32); e4[0, 4] = 1.0
    init = [
        numpy_helper.from_array(Rarr, "R"),
        numpy_helper.from_array(Carr, "C"),
        numpy_helper.from_array(e4, "e4"),
        numpy_helper.from_array(np.array(0.5, np.float32), "half"),
        numpy_helper.from_array(np.array(1.0, np.float32), "one"),
        numpy_helper.from_array(np.array(BIG, np.float32), "big"),
        numpy_helper.from_array(np.array(-BIG, np.float32), "negbig"),
        numpy_helper.from_array(np.array(0.0, np.float32), "zero"),
        numpy_helper.from_array(np.array([0], np.int64), "c0s"),
        numpy_helper.from_array(np.array([1], np.int64), "c1e"),
        numpy_helper.from_array(np.array([2], np.int64), "c2s"),
        numpy_helper.from_array(np.array([3], np.int64), "c3e"),
        numpy_helper.from_array(np.array([1], np.int64), "ax1"),
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
    # precompute R+-0.5, C+-0.5
    nodes += [
        n("Add", ["R", "half"], ["Rp"]), n("Sub", ["R", "half"], ["Rm"]),
        n("Add", ["C", "half"], ["Cp"]), n("Sub", ["C", "half"], ["Cm"]),
        # masks: content, bg(ch0), is2(ch2), structure
        n("ReduceSum", ["input"], ["content"], axes=[1], keepdims=1),
        n("Slice", ["input", "c0s", "c1e", "ax1"], ["bg"]),
        n("Slice", ["input", "c2s", "c3e", "ax1"], ["inbox0"]),    # is2
        n("Sub", ["content", "bg"], ["nonbg"]),
        n("Sub", ["nonbg", "inbox0"], ["isstruct"]),
        # init bound fields
        n("Sub", ["one", "inbox0"], ["notbox0"]),
        n("Mul", ["R", "inbox0"], ["Rin"]), n("Mul", ["C", "inbox0"], ["Cin"]),
        n("Mul", ["big", "notbox0"], ["bigout"]), n("Mul", ["negbig", "notbox0"], ["negout"]),
        n("Add", ["Rin", "bigout"], ["minr0"]), n("Add", ["Rin", "negout"], ["maxr0"]),
        n("Add", ["Cin", "bigout"], ["minc0"]), n("Add", ["Cin", "negout"], ["maxc0"]),
    ]
    inbox, minr, maxr, minc, maxc = "inbox0", "minr0", "maxr0", "minc0", "maxc0"
    for k in range(NITER):
        nbs = []; cminr_in = []; cmaxr_in = []; cminc_in = []; cmaxc_in = []
        for (dr, dc) in DIRS:
            nbs.append(read_shift(inbox, dr, dc, "zero"))
            cminr_in.append(read_shift(minr, dr, dc, "big"))
            cmaxr_in.append(read_shift(maxr, dr, dc, "negbig"))
            cminc_in.append(read_shift(minc, dr, dc, "big"))
            cmaxc_in.append(read_shift(maxc, dr, dc, "negbig"))
        a = f"adj{k}"; cmnr = f"cmnr{k}"; cmxr = f"cmxr{k}"; cmnc = f"cmnc{k}"; cmxc = f"cmxc{k}"
        nodes += [
            n("Max", nbs, [a]),
            n("Min", cminr_in, [cmnr]), n("Max", cmaxr_in, [cmxr]),
            n("Min", cminc_in, [cmnc]), n("Max", cmaxc_in, [cmxc]),
        ]
        # within = Rp>cmnr & Rm<cmxr & Cp>cmnc & Cm<cmxc
        w = f"w{k}"
        nodes += [
            n("Greater", ["Rp", cmnr], [w + "a_b"]), n("Cast", [w + "a_b"], [w + "a"], to=F),
            n("Less", ["Rm", cmxr], [w + "b_b"]), n("Cast", [w + "b_b"], [w + "b"], to=F),
            n("Greater", ["Cp", cmnc], [w + "c_b"]), n("Cast", [w + "c_b"], [w + "c"], to=F),
            n("Less", ["Cm", cmxc], [w + "d_b"]), n("Cast", [w + "d_b"], [w + "d"], to=F),
            n("Mul", [w + "a", w + "b"], [w + "ab"]), n("Mul", [w + "c", w + "d"], [w + "cd"]),
            n("Mul", [w + "ab", w + "cd"], [w]),
            n("Sub", ["one", inbox], [f"notin{k}"]),
            n("Mul", [a, w], [f"aw{k}"]), n("Mul", [f"aw{k}", f"notin{k}"], [f"new{k}"]),
        ]
        nb_inbox = f"inbox{k+1}"; nmnr = f"minr{k+1}"; nmxr = f"maxr{k+1}"; nmnc = f"minc{k+1}"; nmxc = f"maxc{k+1}"
        new = f"new{k}"; notnew = f"nn{k}"
        nodes += [
            n("Max", [inbox, new], [nb_inbox]),
            n("Sub", ["one", new], [notnew]),
            n("Mul", [minr, notnew], [nmnr + "k"]), n("Mul", [cmnr, new], [nmnr + "n"]), n("Add", [nmnr + "k", nmnr + "n"], [nmnr]),
            n("Mul", [maxr, notnew], [nmxr + "k"]), n("Mul", [cmxr, new], [nmxr + "n"]), n("Add", [nmxr + "k", nmxr + "n"], [nmxr]),
            n("Mul", [minc, notnew], [nmnc + "k"]), n("Mul", [cmnc, new], [nmnc + "n"]), n("Add", [nmnc + "k", nmnc + "n"], [nmnc]),
            n("Mul", [maxc, notnew], [nmxc + "k"]), n("Mul", [cmxc, new], [nmxc + "n"]), n("Add", [nmxc + "k", nmxc + "n"], [nmxc]),
        ]
        inbox, minr, maxr, minc, maxc = nb_inbox, nmnr, nmxr, nmnc, nmxc
    nodes += [
        n("Mul", ["isstruct", inbox], ["paint"]),
        n("Sub", ["one", "paint"], ["keep"]),
        n("Mul", ["input", "keep"], ["kept"]),
        n("Mul", ["e4", "paint"], ["addc"]),
        n("Add", ["kept", "addc"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "bbox_fill",
                              [helper.make_tensor_value_info("input", F, FULL)],
                              [helper.make_tensor_value_info("output", F, FULL)],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_bbox_fill(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
