"""Solver: recolour small blobs by their size (task 369).

On a solid field, the off-field cells form tiny connected blobs (size 1, 2 or 3);
each blob is recoloured by its size -- 1 -> 3, 2 -> 2, 3 -> 1 (i.e. ``4 - size``).

For a line / L blob, ``size = (max 4-neighbour degree) + 1``, so
``colour = 3 - maxdeg`` -- and the component's max degree is obtained with a
max-dilation flood of the per-cell degree (no cell counting / labelling needed).

Build: most-common colour = field; the rest is the blob mask ``nz``; degree =
count of ``nz`` 4-neighbours; max-flood the degree through ``nz``; paint
``3 - maxdeg`` as a one-hot.
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
NITER = 8
F = TensorProto.FLOAT


def _bgcol(g):
    v, c = np.unique(g, return_counts=True)
    return v[c.argmax()]


def _ref(g: np.ndarray) -> Optional[np.ndarray]:
    bg = _bgcol(g); H, W = g.shape
    nz = g != bg
    if not nz.any():
        return None
    deg = np.zeros((H, W), int); P = np.pad(nz, 1)
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        deg += P[1 + dr:1 + dr + H, 1 + dc:1 + dc + W].astype(int)
    deg = deg * nz
    acc = deg.copy()
    while True:
        new = acc.copy(); Pa = np.pad(acc, 1)
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            new = np.maximum(new, Pa[1 + dr:1 + dr + H, 1 + dc:1 + dc + W] * nz)
        new = new * nz
        if np.array_equal(new, acc):
            break
        acc = new
    color = 3 - acc
    out = g.copy(); out[nz] = color[nz]
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
    chramp = np.arange(CHANNELS, dtype=np.float32).reshape(1, CHANNELS, 1, 1)
    init = [
        numpy_helper.from_array(chramp, "chramp"),
        numpy_helper.from_array(np.array(0.5, np.float32), "half"),
        numpy_helper.from_array(np.array(1.0, np.float32), "one"),
        numpy_helper.from_array(np.array(3.0, np.float32), "three"),
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
        n("ReduceSum", ["input"], ["content"], axes=[1], keepdims=1),
        n("ReduceSum", ["input"], ["hist"], axes=[2, 3], keepdims=1),
        n("ReduceMax", ["hist"], ["mccount"], axes=[1], keepdims=1),
        n("Sub", ["mccount", "half"], ["mchalf"]),
        n("Greater", ["hist", "mchalf"], ["mcsel_b"]), n("Cast", ["mcsel_b"], ["mcsel"], to=F),
        n("Mul", ["input", "mcsel"], ["imc"]),
        n("ReduceSum", ["imc"], ["is_mc"], axes=[1], keepdims=1),
        n("Sub", ["content", "is_mc"], ["nz"]),
    ]
    # degree = nz 4-neighbour count, masked to nz
    u = read_shift("nz", 1, 0); d = read_shift("nz", -1, 0)
    l = read_shift("nz", 0, 1); r = read_shift("nz", 0, -1)
    nodes += [
        n("Add", [u, d], ["dg1"]), n("Add", [l, r], ["dg2"]),
        n("Add", ["dg1", "dg2"], ["dgsum"]),
        n("Mul", ["dgsum", "nz"], ["acc_0"]),
    ]
    acc = "acc_0"
    for k in range(NITER):
        su = read_shift(acc, 1, 0); sd = read_shift(acc, -1, 0)
        sl = read_shift(acc, 0, 1); sr = read_shift(acc, 0, -1)
        dil = f"dil_{k}"
        nodes.append(n("Max", [acc, su, sd, sl, sr], [dil]))
        nxt = f"acc_{k+1}"
        nodes.append(n("Mul", [dil, "nz"], [nxt]))
        acc = nxt
    nodes += [
        n("Sub", ["three", acc], ["colorval"]),                 # 3 - maxdeg
        n("Sub", ["colorval", "chramp"], ["cdiff"]),
        n("Abs", ["cdiff"], ["cad"]),
        n("Less", ["cad", "half"], ["oh_b"]), n("Cast", ["oh_b"], ["oh"], to=F),
        n("Mul", ["oh", "nz"], ["paint"]),
        n("Sub", ["one", "nz"], ["keep"]),
        n("Mul", ["input", "keep"], ["kept"]),
        n("Add", ["kept", "paint"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "blob_size_color",
                              [helper.make_tensor_value_info("input", F, FULL)],
                              [helper.make_tensor_value_info("output", F, FULL)],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_blob_size_color(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
