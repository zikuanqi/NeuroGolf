"""Solver: horizontal periodic extension (task 231).

The input is an H x W grid whose columns are periodic with a small fundamental
period p (1, 2 or 3); the output is H x 2W, continuing that period:
`out[r][c] = in[r][c % p]`.

We never need dynamic modulo: for each candidate period P in {1,2,3} the tiling
`in[:, c % P]` is a *fixed* `Gather` along the width axis. We pick the
fundamental (smallest) period whose tiling reproduces the input over its real
columns, then mask the result to the first 2W columns.
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
PERIODS = (1, 2, 3)


def _fund_period(g: np.ndarray) -> int:
    h, w = g.shape
    for p in range(1, w + 1):
        if all(np.array_equal(g[:, c], g[:, c - p]) for c in range(p, w)):
            return p
    return w


def _detect(task: dict) -> bool:
    saw = False
    for ex in all_examples(task):
        i, o = ex["input"], ex["output"]
        if not i or not i[0] or not o or not o[0]:
            return False
        H, W = len(i), len(i[0])
        if H > HEIGHT or W > WIDTH:
            continue  # scorer skips oversized examples
        OH, OW = len(o), len(o[0])
        if OH != H or OW != 2 * W or OW > WIDTH:
            return False
        g = np.array(i)
        p = _fund_period(g)
        if p > max(PERIODS):
            return False
        rec = g[:, [c % p for c in range(OW)]]
        if not np.array_equal(rec, np.array(o)):
            return False
        saw = True
    return saw


def _build() -> onnx.ModelProto:
    F = TensorProto.FLOAT
    n_ = helper.make_node
    col_ar = np.arange(WIDTH, dtype=np.float32).reshape(1, 1, 1, WIDTH)
    init = [
        numpy_helper.from_array(col_ar, "col_ar"),
        numpy_helper.from_array(np.array([0.5], dtype=np.float32), "half"),
        numpy_helper.from_array(np.array([1.0], dtype=np.float32), "one"),
        numpy_helper.from_array(np.array([2.0], dtype=np.float32), "two"),
    ]
    for P in PERIODS:
        idx = np.array([c % P for c in range(WIDTH)], dtype=np.int64)
        init.append(numpy_helper.from_array(idx, f"idx{P}"))

    nodes = [
        n_("ReduceSum", ["input"], ["content"], axes=[1], keepdims=1),
        n_("ReduceMax", ["content"], ["col_pres"], axes=[2], keepdims=1),
        n_("Mul", ["col_pres", "col_ar"], ["col_pos"]),
        n_("ReduceMax", ["col_pos"], ["maxcol"], axes=[3], keepdims=1),
        n_("Add", ["maxcol", "one"], ["W_f"]),
        n_("Mul", ["W_f", "two"], ["twoW_f"]),
        n_("Less", ["col_ar", "W_f"], ["maskW_b"]),
        n_("Cast", ["maskW_b"], ["maskW"], to=F),
        n_("Less", ["col_ar", "twoW_f"], ["mask2W_b"]),
        n_("Cast", ["mask2W_b"], ["mask2W"], to=F),
    ]

    for P in PERIODS:
        nodes += [
            n_("Gather", ["input", f"idx{P}"], [f"cand{P}"], axis=3),
            n_("Sub", [f"cand{P}", "input"], [f"diff{P}"]),
            n_("Mul", [f"diff{P}", f"diff{P}"], [f"sq{P}"]),
            n_("Mul", [f"sq{P}", "maskW"], [f"msq{P}"]),
            n_("ReduceSum", [f"msq{P}"], [f"err{P}"], axes=[0, 1, 2, 3],
               keepdims=1),
            n_("Less", [f"err{P}", "half"], [f"vb{P}"]),
            n_("Cast", [f"vb{P}"], [f"valid{P}"], to=F),
        ]

    # fundamental = smallest valid period
    nodes += [
        n_("Sub", ["one", "valid1"], ["inv1"]),
        n_("Sub", ["one", "valid2"], ["inv2"]),
        n_("Mul", ["valid2", "inv1"], ["w2"]),
        n_("Mul", ["valid3", "inv1"], ["w3a"]),
        n_("Mul", ["w3a", "inv2"], ["w3"]),
        n_("Mul", ["cand1", "valid1"], ["t1"]),
        n_("Mul", ["cand2", "w2"], ["t2"]),
        n_("Mul", ["cand3", "w3"], ["t3"]),
        n_("Add", ["t1", "t2"], ["tile_ab"]),
        n_("Add", ["tile_ab", "t3"], ["tile"]),
        n_("Mul", ["tile", "mask2W"], ["output"]),
    ]
    for nd in nodes:
        nd.name = nd.output[0]

    def vi(name, shape, dt=F):
        return helper.make_tensor_value_info(name, dt, shape)

    s1 = [1, 1, HEIGHT, WIDTH]
    ccol = [1, 1, 1, WIDTH]
    sc = [1, 1, 1, 1]
    value_info = [
        vi("content", s1), vi("col_pres", ccol), vi("col_pos", ccol),
        vi("maxcol", sc), vi("W_f", sc), vi("twoW_f", sc),
        vi("maskW_b", ccol, TensorProto.BOOL), vi("maskW", ccol),
        vi("mask2W_b", ccol, TensorProto.BOOL), vi("mask2W", ccol),
        vi("inv1", sc), vi("inv2", sc), vi("w2", sc), vi("w3a", sc), vi("w3", sc),
        vi("t1", FULL), vi("t2", FULL), vi("t3", FULL),
        vi("tile_ab", FULL), vi("tile", FULL),
    ]
    for P in PERIODS:
        value_info += [
            vi(f"cand{P}", FULL), vi(f"diff{P}", FULL), vi(f"sq{P}", FULL),
            vi(f"msq{P}", FULL), vi(f"err{P}", sc),
            vi(f"vb{P}", sc, TensorProto.BOOL), vi(f"valid{P}", sc),
        ]

    graph = helper.make_graph(nodes, "period_extend_h", [vi("input", FULL)],
                              [vi("output", FULL)], initializer=init,
                              value_info=value_info)
    return helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
        ir_version=IR_VERSION)


def solve_period_extend_h(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
