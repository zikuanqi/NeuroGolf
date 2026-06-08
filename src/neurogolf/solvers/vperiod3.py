"""Solver: extend a vertically period-3 pattern to fill the grid (task 215).

A texture with a vertical period of 3 rows is shown in part of the grid; the
output tiles it across every row (the single non-background colour is preserved)::

    . . .            8 . 8
    . . .            . 8 .     (the period-3 block repeats up and down
    . 8 .      ->    8 . 8      to fill the whole grid)
    8 . 8            . 8 .
    . 8 .            8 . 8

Build: ``nz`` = non-background mask; OR it with vertical shifts of every
multiple of 3 via a log-step doubling scan (steps 3, 6, 12, 24, both
directions); paint the runtime colour on the resulting mask over the real grid.
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
    nz = g != 0
    if nz.sum() == 0:
        return None
    vals = np.unique(g[nz])
    if len(vals) != 1:
        return None
    val = int(vals[0])
    acc = nz.copy()
    for sh in range(-H, H + 1):
        if sh == 0 or sh % 3 != 0:
            continue
        s = np.zeros_like(nz)
        if sh > 0:
            s[sh:, :] = nz[:H - sh, :]
        else:
            s[:H + sh, :] = nz[-sh:, :]
        acc |= s
    out = g.copy()
    out[acc] = val
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

    def read_shift(x, ar):
        """y[r,c] = x[r-ar, c] (vertical shift, zero-filled)."""
        pt, pb = max(ar, 0), max(-ar, 0)
        pname = f"pad_{ar}"
        if pname not in seen["pad"]:
            init.append(numpy_helper.from_array(
                np.array([0, 0, pt, 0, 0, 0, pb, 0], np.int64), pname))
            seen["pad"].add(pname)
        rs = max(-ar, 0)
        sname, ename = f"sst_{rs}", f"sen_{rs}"
        if sname not in seen["sl"]:
            init.append(numpy_helper.from_array(np.array([rs, 0], np.int64), sname))
            init.append(numpy_helper.from_array(
                np.array([rs + HEIGHT, WIDTH], np.int64), ename))
            seen["sl"].add(sname)
        ctr[0] += 1
        pid, oid = f"ps{ctr[0]}", f"rs{ctr[0]}"
        nodes.append(n("Pad", [x, pname], [pid], mode="constant"))
        nodes.append(n("Slice", [pid, sname, ename, "ax23"], [oid]))
        return oid

    nodes += [
        n("ReduceSum", ["input"], ["content"], axes=[1], keepdims=1),
        n("Slice", ["input", "c0s", "c1e", "ax1"], ["bg"]),
        n("Sub", ["content", "bg"], ["acc0"]),
    ]
    acc = "acc0"
    for step in (3, 6, 12, 24):
        up = read_shift(acc, step)
        dn = read_shift(acc, -step)
        ctr[0] += 1
        m1 = f"mx{ctr[0]}a"; m2 = f"mx{ctr[0]}b"
        nodes.append(n("Max", [up, dn], [m1]))
        nodes.append(n("Max", [acc, m1], [m2]))
        acc = m2
    nodes += [
        n("Mul", [acc, "content"], ["paint"]),
        # runtime colour
        n("ReduceSum", ["input"], ["hist"], axes=[2, 3], keepdims=1),
        n("Mul", ["hist", "ch0mask"], ["nbhist"]),
        n("Greater", ["nbhist", "half"], ["cvec_b"]), n("Cast", ["cvec_b"], ["cvec"], to=F),
        n("Sub", ["one", "paint"], ["inv"]), n("Mul", ["input", "inv"], ["kept"]),
        n("Mul", ["cvec", "paint"], ["addc"]), n("Add", ["kept", "addc"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "vperiod3",
                              [helper.make_tensor_value_info("input", F, FULL)],
                              [helper.make_tensor_value_info("output", F, FULL)],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_vperiod3(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
