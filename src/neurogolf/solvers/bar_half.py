"""Solver: recolour the bottom half of each vertical 2-bar to 8 (task 320).

Each vertical bar of colour 2 has its bottom ``floor(height/2)`` cells recoloured
to 8 (the top half stays 2)::

    2          2            2          2
    2          2            2          2
    2    ->    8     (h=5,  2    ->    8    (h=4, bottom 2 -> 8)
    2          8      bottom 2         8
    2          8       2 -> 8)

Build: per-cell ``minr`` / ``maxr`` of each bar come from min/max floods through
the 2-mask; ``half = floor((maxr-minr+1)/2)``; a 2-cell at row ``r`` becomes 8
iff ``maxr - r < half`` (it is among the bottom ``half`` rows).
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


def _shift(a, dr, dc, fill):
    H, W = a.shape; r = np.full_like(a, fill)
    ys = slice(max(0, dr), H + min(0, dr)); yd = slice(max(0, -dr), H + min(0, -dr))
    xs = slice(max(0, dc), W + min(0, dc)); xd = slice(max(0, -dc), W + min(0, -dc))
    r[yd, xd] = a[ys, xs]; return r


def _ref(g: np.ndarray) -> Optional[np.ndarray]:
    H, W = g.shape
    is2 = (g == 2)
    if not is2.any():
        return None
    R = np.broadcast_to(np.arange(H).reshape(-1, 1), (H, W)).astype(float)
    minr = np.where(is2, R, BIG); maxr = np.where(is2, R, -BIG)
    for _ in range(80):
        a = minr.copy(); b = maxr.copy()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a = np.minimum(a, _shift(minr, dr, dc, BIG))
            b = np.maximum(b, _shift(maxr, dr, dc, -BIG))
        a = np.where(is2, a, BIG); b = np.where(is2, b, -BIG)
        if np.array_equal(a, minr) and np.array_equal(b, maxr):
            break
        minr, maxr = a, b
    half = np.floor((maxr - minr + 1) / 2.0)
    is8 = is2 & ((maxr - R) < half)
    out = g.copy(); out[is8] = 8
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
    e8 = np.zeros((1, CHANNELS, 1, 1), np.float32); e8[0, 8] = 1.0
    init = [
        numpy_helper.from_array(Rarr, "R"),
        numpy_helper.from_array(e8, "e8"),
        numpy_helper.from_array(np.array(0.5, np.float32), "half"),
        numpy_helper.from_array(np.array(1.0, np.float32), "one"),
        numpy_helper.from_array(np.array(BIG, np.float32), "big"),
        numpy_helper.from_array(np.array(-BIG, np.float32), "negbig"),
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
    nodes += [
        n("Slice", ["input", "c2s", "c3e", "ax1"], ["is2"]),
        n("Sub", ["one", "is2"], ["not2"]),
        n("Mul", ["R", "is2"], ["Rin"]),
        n("Mul", ["big", "not2"], ["bigout"]), n("Mul", ["negbig", "not2"], ["negout"]),
        n("Add", ["Rin", "bigout"], ["minr0"]), n("Add", ["Rin", "negout"], ["maxr0"]),
    ]
    minr, maxr = "minr0", "maxr0"
    for k in range(NITER):
        ir = [minr]; xr = [maxr]
        for (dr, dc) in DIRS:
            ir.append(read_shift(minr, dr, dc, "big"))
            xr.append(read_shift(maxr, dr, dc, "negbig"))
        amnr = f"amnr{k}"; amxr = f"amxr{k}"
        nodes += [n("Min", ir, [amnr]), n("Max", xr, [amxr])]
        nmnr = f"minr{k+1}"; nmxr = f"maxr{k+1}"
        nodes += [
            n("Mul", [amnr, "is2"], [nmnr + "s"]), n("Add", [nmnr + "s", "bigout"], [nmnr]),
            n("Mul", [amxr, "is2"], [nmxr + "s"]), n("Add", [nmxr + "s", "negout"], [nmxr]),
        ]
        minr, maxr = nmnr, nmxr
    nodes += [
        n("Sub", [maxr, minr], ["hm1"]), n("Add", ["hm1", "one"], ["h"]),
        n("Mul", ["h", "half"], ["hh"]), n("Floor", ["hh"], ["halfh"]),
        n("Sub", [maxr, "R"], ["dist"]),
        n("Less", ["dist", "halfh"], ["lo_b"]), n("Cast", ["lo_b"], ["lo"], to=F),
        n("Mul", ["is2", "lo"], ["is8"]),
        n("Sub", ["one", "is8"], ["keep"]),
        n("Mul", ["input", "keep"], ["kept"]),
        n("Mul", ["e8", "is8"], ["addc"]),
        n("Add", ["kept", "addc"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "bar_half",
                              [helper.make_tensor_value_info("input", F, FULL)],
                              [helper.make_tensor_value_info("output", F, FULL)],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_bar_half(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
