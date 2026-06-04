"""Solver: connect same-colour pairs with diagonal lines (task 37).

Each colour appears exactly twice, the two cells lying on a diagonal; the output
draws the diagonal segment between them in that colour. Vectorised over the 9
colour channels: four diagonal cumulative-max passes (down-right / up-left /
down-left / up-right) by log-doubling shift-and-max, then a cell is on a main
diagonal segment where DR and UL both reach it, and on an anti-diagonal segment
where DL and UR both reach it.
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
NC = CHANNELS - 1
DIRS = {"DR": (1, 1), "UL": (-1, -1), "DL": (1, -1), "UR": (-1, 1)}


def _ref(g: np.ndarray) -> Optional[np.ndarray]:
    from collections import defaultdict
    H, W = g.shape
    pos = defaultdict(list)
    for r in range(H):
        for c in range(W):
            if g[r, c]:
                pos[int(g[r, c])].append((r, c))
    out = g.copy()
    for col, pts in pos.items():
        if len(pts) != 2:
            return None
        (r0, c0), (r1, c1) = pts
        dr, dc = r1 - r0, c1 - c0
        if abs(dr) != abs(dc) or dr == 0:
            return None
        sr, sc = (1 if dr > 0 else -1), (1 if dc > 0 else -1)
        for k in range(abs(dr) + 1):
            out[r0 + sr * k, c0 + sc * k] = col
    return out


def _detect(task: dict) -> bool:
    saw = False
    for ex in all_examples(task):
        i, o = ex["input"], ex["output"]
        if not i or not i[0] or len(i) > HEIGHT or len(i[0]) > WIDTH:
            return False
        ref = _ref(np.array(i))
        if ref is None or not np.array_equal(ref, np.array(o)):
            return False
        saw = True
    return saw


def _i64(name, v): return numpy_helper.from_array(np.array(v, np.int64), name)


def _build() -> onnx.ModelProto:
    F = TensorProto.FLOAT
    n = helper.make_node
    init = [
        _i64("ch1_s", [0, 1, 0, 0]), _i64("ch1_e", [1, CHANNELS, HEIGHT, WIDTH]),
        _i64("ax4", [0, 1, 2, 3]),
    ]
    vi = []
    c9 = [1, NC, HEIGHT, WIDTH]
    def V(nm, sh=c9, dt=F): vi.append(helper.make_tensor_value_info(nm, dt, sh))
    nodes = [n("Slice", ["input", "ch1_s", "ch1_e", "ax4"], ["ch19"])]
    V("ch19")

    cumnames = {}
    for nm, (dr, dc) in DIRS.items():
        cur = "ch19"
        for i in range(5):
            sr, scl = dr * (2 ** i), dc * (2 ** i)
            ptop, pbot = max(sr, 0), max(-sr, 0)
            plef, prig = max(scl, 0), max(-scl, 0)
            pn = f"pad_{nm}_{i}"; ss = f"ss_{nm}_{i}"; se = f"se_{nm}_{i}"
            init += [
                _i64(pn, [0, 0, ptop, plef, 0, 0, pbot, prig]),
                _i64(ss, [0, 0, pbot, prig]),
                _i64(se, [1, NC, pbot + HEIGHT, prig + WIDTH]),
            ]
            po = f"p_{nm}_{i}"; sho = f"sh_{nm}_{i}"; mo = f"cur_{nm}_{i}"
            nodes += [
                n("Pad", [cur, pn], [po], mode="constant"),
                n("Slice", [po, ss, se, "ax4"], [sho]),
                n("Max", [cur, sho], [mo]),
            ]
            V(po, [1, NC, HEIGHT + pbot + ptop, WIDTH + plef + prig])
            V(sho); V(mo)
            cur = mo
        cumnames[nm] = cur

    nodes += [
        n("Mul", [cumnames["DR"], cumnames["UL"]], ["main"]),
        n("Mul", [cumnames["DL"], cumnames["UR"]], ["anti"]),
        n("Max", ["main", "anti"], ["seg"]),
        # background channel 0 where no segment, within the grid
        n("ReduceSum", ["input"], ["content"], axes=[1], keepdims=1),
        n("ReduceMax", ["seg"], ["anyseg"], axes=[1], keepdims=1),
        n("Sub", ["content", "anyseg"], ["bg"]),
        n("Concat", ["bg", "seg"], ["output"], axis=1),
    ]
    V("main"); V("anti"); V("seg")
    s1 = [1, 1, HEIGHT, WIDTH]
    vi += [helper.make_tensor_value_info("content", F, s1),
           helper.make_tensor_value_info("anyseg", F, s1),
           helper.make_tensor_value_info("bg", F, s1)]

    graph = helper.make_graph(nodes, "diag_connect",
                              [helper.make_tensor_value_info("input", F, FULL)],
                              [helper.make_tensor_value_info("output", F, FULL)],
                              initializer=init, value_info=vi)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_diag_connect(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
