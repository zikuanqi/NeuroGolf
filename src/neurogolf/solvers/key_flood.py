"""Solver: flood each 5-block with the row-0 key colour of the column it spans (task 354).

Row 0 holds key markers at scattered columns; the grid also has blocks of colour
``5``.  Each block is recoloured to the key colour whose column falls inside the
block (the key colour floods through the block)::

    . 2 . . 6 .          . 2 . . 6 .
    . . . . . .          . . . . . .
    . 5 5 . 5 5    ->    . 2 2 . 6 6
    . 5 5 . 5 5          . 2 2 . 6 6

Build: the key colour *value* is read off row 0 (``Σ k·channel``) and broadcast
down each column, seeded only on 5-cells; a max-dilation flood spreads each
seed value through its connected 5-component; the value is converted back to a
one-hot and painted.  Flooding the scalar value (not 10 channels) keeps memory
low.
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
NITER = 20


def _ref(g: np.ndarray) -> Optional[np.ndarray]:
    H, W = g.shape
    is5 = (g == 5)
    if not is5.any():
        return None
    keyrow = g[0]
    seed = np.zeros((H, W), int)
    for c in range(W):
        if keyrow[c] not in (0, 5):
            seed[:, c] = keyrow[c]
    acc = np.where(is5, seed, 0)
    while True:
        new = acc.copy()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            sh = np.zeros_like(acc)
            ys = slice(max(0, dr), H + min(0, dr)); yd = slice(max(0, -dr), H + min(0, -dr))
            xs = slice(max(0, dc), W + min(0, dc)); xd = slice(max(0, -dc), W + min(0, -dc))
            sh[yd, xd] = acc[ys, xs]
            new = np.where((new == 0) & is5 & (sh != 0), sh, new)
        if np.array_equal(new, acc):
            break
        acc = new
    out = g.copy(); out[is5] = acc[is5]
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

    chramp = np.arange(CHANNELS, dtype=np.float32).reshape(1, CHANNELS, 1, 1)
    init = [
        numpy_helper.from_array(chramp, "chramp"),
        numpy_helper.from_array(np.array(0.5, np.float32), "half"),
        numpy_helper.from_array(np.array(1.0, np.float32), "one"),
        numpy_helper.from_array(np.array([5], np.int64), "c5s"),
        numpy_helper.from_array(np.array([6], np.int64), "c6e"),
        numpy_helper.from_array(np.array([1], np.int64), "ax1"),
        numpy_helper.from_array(np.array([0], np.int64), "r0s"),
        numpy_helper.from_array(np.array([1], np.int64), "r1e"),
        numpy_helper.from_array(np.array([2], np.int64), "ax2"),
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

    nodes += [
        n("Slice", ["input", "c5s", "c6e", "ax1"], ["is5"]),                # (1,1,H,W)
        n("Slice", ["input", "r0s", "r1e", "ax2"], ["row0"]),               # (1,C,1,W)
        n("Mul", ["row0", "chramp"], ["r0v"]),
        n("ReduceSum", ["r0v"], ["valrow"], axes=[1], keepdims=1),          # (1,1,1,W)
        n("Mul", ["valrow", "is5"], ["acc_0"]),                             # seed values
    ]
    acc = "acc_0"
    for k in range(NITER):
        u = read_shift(acc, 1, 0); d = read_shift(acc, -1, 0)
        l = read_shift(acc, 0, 1); r = read_shift(acc, 0, -1)
        dil = f"dil_{k}"
        nodes.append(n("Max", [acc, u, d, l, r], [dil]))
        nxt = f"acc_{k+1}"
        nodes.append(n("Mul", [dil, "is5"], [nxt]))
        acc = nxt
    nodes += [
        # value -> one-hot
        n("Sub", [acc, "chramp"], ["vdiff"]),                               # (1,C,H,W)
        n("Abs", ["vdiff"], ["vad"]),
        n("Less", ["vad", "half"], ["oh_b"]), n("Cast", ["oh_b"], ["oh"], to=F),
        n("Mul", ["oh", "is5"], ["paint"]),
        n("Sub", ["one", "is5"], ["keep"]),
        n("Mul", ["input", "keep"], ["kept"]),
        n("Add", ["kept", "paint"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "key_flood",
                              [helper.make_tensor_value_info("input", F, FULL)],
                              [helper.make_tensor_value_info("output", F, FULL)],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_key_flood(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
