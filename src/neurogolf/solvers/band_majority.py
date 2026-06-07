"""Solver: denoise striped bands to each line's majority colour (task 359).

The grid is made of solid colour bands along one axis (rows or columns) with a
little noise.  The output makes every line uniform with its majority colour.

Build: per-row and per-col channel histograms give, via `ArgMax`, each line's
majority colour (turned into a one-hot by comparing the arg to a channel-index
ramp).  The band axis is whichever has the larger summed per-line max count;
the two candidates are blended by that flag and masked to the grid.
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


def _linemaj(g, axis):
    out = g.copy()
    if axis == 0:
        for r in range(g.shape[0]):
            v, c = np.unique(g[r], return_counts=True)
            out[r] = v[np.argmax(c)]
    else:
        for j in range(g.shape[1]):
            v, c = np.unique(g[:, j], return_counts=True)
            out[:, j] = v[np.argmax(c)]
    return out


def _strength(g, axis):
    lines = g if axis == 0 else g.T
    return sum(int(np.unique(ln, return_counts=True)[1].max()) for ln in lines)


def _ref(g: np.ndarray) -> Optional[np.ndarray]:
    is_row = _strength(g, 0) >= _strength(g, 1)
    pred = _linemaj(g, 0) if is_row else _linemaj(g, 1)
    if np.array_equal(pred, g):
        return None
    return pred


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

    def cf(b, name):
        return n("Cast", [b], [name], to=F)

    chidx = np.arange(CHANNELS, dtype=np.float32).reshape(1, CHANNELS, 1, 1)
    init = [
        numpy_helper.from_array(chidx, "chidx"),
        numpy_helper.from_array(np.array(0.5, np.float32), "half"),
        numpy_helper.from_array(np.array(1.0, np.float32), "one"),
    ]
    nodes = [
        # per-row majority
        n("ReduceSum", ["input"], ["rc"], axes=[3], keepdims=1),          # (1,10,H,1)
        n("ArgMax", ["rc"], ["rarg_i"], axis=1, keepdims=1),              # (1,1,H,1)
        cf("rarg_i", "rarg"),
        n("Sub", ["chidx", "rarg"], ["rd"]), n("Abs", ["rd"], ["rda"]),
        n("Less", ["rda", "half"], ["roh_b"]), cf("roh_b", "roh"),       # (1,10,H,1)
        n("ReduceMax", ["rc"], ["rmax"], axes=[1], keepdims=1),          # (1,1,H,1)
        n("ReduceSum", ["rmax"], ["rstr"], keepdims=0),                  # scalar
        # per-col majority
        n("ReduceSum", ["input"], ["cc"], axes=[2], keepdims=1),          # (1,10,1,W)
        n("ArgMax", ["cc"], ["carg_i"], axis=1, keepdims=1),             # (1,1,1,W)
        cf("carg_i", "carg"),
        n("Sub", ["chidx", "carg"], ["cd"]), n("Abs", ["cd"], ["cda"]),
        n("Less", ["cda", "half"], ["coh_b"]), cf("coh_b", "coh"),       # (1,10,1,W)
        n("ReduceMax", ["cc"], ["cmax"], axes=[1], keepdims=1),          # (1,1,1,W)
        n("ReduceSum", ["cmax"], ["cstr"], keepdims=0),
        # axis flag: rows win ties (>=)
        n("Add", ["rstr", "half"], ["rstr_h"]),
        n("Greater", ["rstr_h", "cstr"], ["isrow_b"]), cf("isrow_b", "isrow"),
        # blend (broadcast roh over W, coh over H)
        n("Mul", ["roh", "isrow"], ["rpick"]),
        n("Sub", ["one", "isrow"], ["niso"]),
        n("Mul", ["coh", "niso"], ["cpick"]),
        n("Add", ["rpick", "cpick"], ["maj"]),                           # (1,10,H,W)
        # mask to grid
        n("ReduceSum", ["input"], ["grid"], axes=[1], keepdims=1),       # (1,1,H,W)
        n("Mul", ["maj", "grid"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "band_majority",
                              [helper.make_tensor_value_info("input", F, FULL)],
                              [helper.make_tensor_value_info("output", F, FULL)],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_band_majority(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
