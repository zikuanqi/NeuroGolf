"""Solver: satellite cells shoot diagonal rays away from a block (task 190).

A solid 2x2 block of one colour has one or more isolated "satellite" cells of
the same colour sitting diagonally off its corners.  Each satellite emits a
diagonal ray (same colour) continuing away from the block to the grid edge::

    3 3 .            3 3 . . . .
    3 3 .      ->    3 3 . . . .
    . . 3            . . 3 . . .
    . . . .          . . . 3 . .
    . . . .          . . . . 3 .

The ray direction is purely local: a satellite (an off-block cell with no
orthogonal same-colour neighbour) has exactly one diagonal same-colour neighbour
(the block corner); the ray heads the opposite way.

Build: ``nz`` = non-background; ``sat`` = nz cells with no orthogonal nz
neighbour; four diagonal-neighbour tests split the satellites by direction; each
group is propagated to the edge with a log-step doubling scan, then the union is
painted the (runtime) shape colour over background cells.
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


def _ref(g: np.ndarray) -> Optional[np.ndarray]:
    H, W = g.shape
    vals, cnts = np.unique(g, return_counts=True)
    bg = vals[cnts.argmax()]
    nz = g != bg
    if nz.sum() == 0:
        return None
    C = int(g[nz][0])
    if (g[nz] != C).any():
        return None

    def has_nbr(m):
        P = np.pad(m, 1); acc = np.zeros_like(m)
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            acc |= P[1 + dr:1 + dr + H, 1 + dc:1 + dc + W]
        return acc

    blk = nz & has_nbr(nz)
    sat = nz & ~blk
    if blk.sum() == 0 or sat.sum() == 0:
        return None
    by, bx = np.where(blk); bcr, bcc = by.mean(), bx.mean()
    out = g.copy()
    sy, sx = np.where(sat)
    for r, c in zip(sy, sx):
        dr = 1 if r > bcr else -1
        dc = 1 if c > bcc else -1
        rr, cc = r + dr, c + dc
        while 0 <= rr < H and 0 <= cc < W:
            if g[rr, cc] == bg:
                out[rr, cc] = C
            rr += dr; cc += dc
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

    ch0mask = np.ones((1, CHANNELS, 1, 1), np.float32); ch0mask[0, 0] = 0.0
    init = [
        numpy_helper.from_array(ch0mask, "ch0mask"),
        numpy_helper.from_array(np.array(0.5, np.float32), "half"),
        numpy_helper.from_array(np.array(1.0, np.float32), "one"),
        numpy_helper.from_array(np.array([0], np.int64), "c0s"),
        numpy_helper.from_array(np.array([1], np.int64), "c1e"),
        numpy_helper.from_array(np.array([1], np.int64), "ax1"),
        numpy_helper.from_array(np.array([2, 3], np.int64), "ax23"),
    ]
    nodes = []
    seen = {"pad": set(), "sl": set()}
    ctr = [0]

    def newid(p):
        ctr[0] += 1
        return f"{p}{ctr[0]}"

    def read_shift(x, ar, ac):
        """y[r,c] = x[r-ar, c-ac] (zero-filled)."""
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
        pid, oid = newid("ps"), newid("rs")
        nodes.append(n("Pad", [x, pname], [pid], mode="constant"))
        nodes.append(n("Slice", [pid, sname, ename, "ax23"], [oid]))
        return oid

    def propagate(seed, dr, dc):
        acc = seed
        for k in (1, 2, 4, 8, 16):
            sh = read_shift(acc, k * dr, k * dc)
            nid = newid("pr")
            nodes.append(n("Max", [acc, sh], [nid]))
            acc = nid
        return acc

    # base masks
    nodes += [
        n("ReduceSum", ["input"], ["content"], axes=[1], keepdims=1),
        n("Slice", ["input", "c0s", "c1e", "ax1"], ["bg"]),
        n("Sub", ["content", "bg"], ["nz"]),
    ]
    # orthogonal-neighbour presence
    up = read_shift("nz", 1, 0); dn = read_shift("nz", -1, 0)
    lf = read_shift("nz", 0, 1); rt = read_shift("nz", 0, -1)
    nodes += [
        n("Max", [up, dn], ["orv"]), n("Max", [lf, rt], ["orh"]),
        n("Max", ["orv", "orh"], ["orth"]),
        n("Sub", ["one", "orth"], ["noorth"]),
        n("Mul", ["nz", "noorth"], ["sat"]),
    ]
    # diagonal-neighbour tests -> per-direction satellites
    nz_ul = read_shift("nz", 1, 1)    # block up-left  -> ray down-right
    nz_ur = read_shift("nz", 1, -1)   # block up-right -> ray down-left
    nz_dl = read_shift("nz", -1, 1)   # block dn-left  -> ray up-right
    nz_dr = read_shift("nz", -1, -1)  # block dn-right -> ray up-left
    nodes += [
        n("Mul", ["sat", nz_ul], ["sat_dr"]),
        n("Mul", ["sat", nz_ur], ["sat_dl"]),
        n("Mul", ["sat", nz_dl], ["sat_ur"]),
        n("Mul", ["sat", nz_dr], ["sat_ul"]),
    ]
    r_dr = propagate("sat_dr", 1, 1)
    r_dl = propagate("sat_dl", 1, -1)
    r_ur = propagate("sat_ur", -1, 1)
    r_ul = propagate("sat_ul", -1, -1)
    nodes += [
        n("Max", [r_dr, r_dl], ["ra"]), n("Max", [r_ur, r_ul], ["rb"]),
        n("Max", ["ra", "rb"], ["allray"]),
        n("Mul", ["allray", "bg"], ["ray"]),          # background cells only
        # runtime shape colour one-hot
        n("ReduceSum", ["input"], ["hist"], axes=[2, 3], keepdims=1),
        n("Mul", ["hist", "ch0mask"], ["nbhist"]),
        n("Greater", ["nbhist", "half"], ["cvec_b"]), n("Cast", ["cvec_b"], ["cvec"], to=F),
        # paint
        n("Sub", ["one", "ray"], ["inv"]), n("Mul", ["input", "inv"], ["kept"]),
        n("Mul", ["cvec", "ray"], ["addc"]), n("Add", ["kept", "addc"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "diag_shoot",
                              [helper.make_tensor_value_info("input", F, FULL)],
                              [helper.make_tensor_value_info("output", F, FULL)],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_diag_shoot(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
