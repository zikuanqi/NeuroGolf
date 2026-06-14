"""Solver: drill each 0-hole across its colour stripe (task 202).

The grid is solid colour stripes (all horizontal, or all vertical).  Each
stripe may contain 0-holes; every hole is extended into a full 0-line spanning
its stripe, perpendicular to the stripe direction::

    5 5 5 5 5            5 5 5 5 5
    4 4 4 0 4    ->      4 4 4 0 4      (horizontal stripes: the hole drills
    4 4 4 4 4            4 4 4 0 4        a vertical 0-line down its band)
    8 8 8 8 8            8 8 8 8 8

Implemented as a band-gated vertical flood: a hole spreads up/down only while
the neighbouring row shares its band colour (``bandVec[r]·bandVec[r±1]``), so it
fills the band's thickness in that column and stops at stripe boundaries.  The
vertical-stripe case is the same flood on the transpose; results are blended by
an orientation gate (``max band-colours per row``).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from ..grids import CHANNELS, HEIGHT, WIDTH, all_examples

OPSET = 11
IR_VERSION = 8
F = TensorProto.FLOAT
NITER = 16


def _drill_h(g: np.ndarray) -> np.ndarray:
    H, W = g.shape
    bc = g.max(axis=1)
    out = g.copy()
    r = 0
    while r < H:
        r2 = r
        while r2 < H and bc[r2] == bc[r]:
            r2 += 1
        for c in range(W):
            if (g[r:r2, c] == 0).any():
                out[r:r2, c] = 0
        r = r2
    return out


def _ref(g: np.ndarray) -> Optional[np.ndarray]:
    rowcols = max(len({v for v in g[r] if v != 0}) for r in range(g.shape[0]))
    colcols = max(len({v for v in g[:, c] if v != 0}) for c in range(g.shape[1]))
    if rowcols <= 1 and colcols > 1:
        out = _drill_h(g)
    elif colcols <= 1 and rowcols > 1:
        out = _drill_h(g.T).T
    else:
        return None
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
    e0 = np.zeros((1, CHANNELS, 1, 1), np.float32); e0[0, 0] = 1.0
    notbg = np.ones((1, CHANNELS, 1, 1), np.float32); notbg[0, 0] = 0.0
    SD = np.zeros((HEIGHT, HEIGHT), np.float32)
    SU = np.zeros((HEIGHT, HEIGHT), np.float32)
    for r in range(HEIGHT):
        if r - 1 >= 0:
            SD[r, r - 1] = 1.0
        if r + 1 < HEIGHT:
            SU[r, r + 1] = 1.0
    init = [
        numpy_helper.from_array(e0, "e0"),
        numpy_helper.from_array(notbg, "notbg"),
        numpy_helper.from_array(SD, "SD"),
        numpy_helper.from_array(SU, "SU"),
        numpy_helper.from_array(np.array(1.0, np.float32), "one"),
        numpy_helper.from_array(np.array(1.5, np.float32), "oneh"),
        numpy_helper.from_array(np.array([0], np.int64), "c0s"),
        numpy_helper.from_array(np.array([1], np.int64), "c1e"),
        numpy_helper.from_array(np.array([1], np.int64), "ax1"),
    ]
    nodes: list = []

    def flood(x: str, p: str) -> str:
        rh, bv, bu, pr, su, sd, m = (f"{p}rh", f"{p}bv", f"{p}bu", f"{p}pr",
                                     f"{p}su", f"{p}sd", f"{p}m0")
        nodes.extend([
            n("ReduceMax", [x], [rh], axes=[3], keepdims=1),       # (1,10,30,1)
            n("Mul", [rh, "notbg"], [bv]),                         # band one-hot per row
            n("MatMul", ["SD", bv], [bu]),                         # bandVec[r-1]
            n("Mul", [bv, bu], [pr]),
            n("ReduceSum", [pr, ], [su], axes=[1], keepdims=1),    # sameUp (1,1,30,1)
            n("MatMul", ["SU", su], [sd]),                         # sameDown = sameUp[r+1]
            n("Slice", [x, "c0s", "c1e", "ax1"], [m]),             # seed = 0-cells
        ])
        cur = m
        for k in range(NITER):
            mu, gu, md, gd, nx = (f"{p}mu{k}", f"{p}gu{k}", f"{p}md{k}",
                                  f"{p}gd{k}", f"{p}m{k + 1}")
            nodes.extend([
                n("MatMul", ["SD", cur], [mu]), n("Mul", [mu, su], [gu]),
                n("MatMul", ["SU", cur], [md]), n("Mul", [md, sd], [gd]),
                n("Max", [cur, gu], [f"{p}t{k}"]),
                n("Max", [f"{p}t{k}", gd], [nx]),
            ])
            cur = nx
        inv, a, b, out = f"{p}inv", f"{p}a", f"{p}b", f"{p}out"
        nodes.extend([
            n("Sub", ["one", cur], [inv]),
            n("Mul", [x, inv], [a]),
            n("Mul", [cur, "e0"], [b]),
            n("Add", [a, b], [out]),
        ])
        return out

    hout = flood("input", "h")
    nodes.append(n("Transpose", ["input"], ["xt"], perm=[0, 1, 3, 2]))
    vout_t = flood("xt", "v")
    nodes.append(n("Transpose", [vout_t], ["vfinal"], perm=[0, 1, 3, 2]))
    # orientation gate: max band-colours per row of input
    nodes.extend([
        n("ReduceMax", ["input"], ["rh0"], axes=[3], keepdims=1),
        n("Mul", ["rh0", "notbg"], ["bv0"]),
        n("ReduceSum", ["bv0"], ["rmul"], axes=[1], keepdims=1),
        n("ReduceMax", ["rmul"], ["mxmul"], axes=[2, 3], keepdims=1),
        n("Less", ["mxmul", "oneh"], ["ih_b"]), n("Cast", ["ih_b"], ["isH"], to=F),
        n("Mul", [hout, "isH"], ["hsel"]),
        n("Sub", ["one", "isH"], ["isV"]),
        n("Mul", ["vfinal", "isV"], ["vsel"]),
        n("Add", ["hsel", "vsel"], ["output"]),
    ])
    graph = helper.make_graph(nodes, "band_drill",
                              [helper.make_tensor_value_info("input", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              [helper.make_tensor_value_info("output", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_band_drill(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
