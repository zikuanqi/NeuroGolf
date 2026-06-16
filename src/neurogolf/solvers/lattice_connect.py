"""Solver: connect same-colour cells across a lattice along rows/columns (task 9).

On a lattice of cells separated by full grid lines, two same-colour cells in the
same row (or column) get the cells between them filled with that colour.  The
lattice line colour itself is excluded.

For every channel: ``CumSum`` (exclusive, forward and reverse) detects a same
colour strictly to the left and right (and above/below); a background cell
between such a pair is painted that colour.
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


def _ref(g: np.ndarray) -> Optional[np.ndarray]:
    H, W = g.shape
    cols = [c for c in range(W) if len(set(g[:, c].tolist())) == 1 and g[0, c] != 0]
    linecol = g[0, cols[0]] if cols else -1
    out = g.copy()
    changed = False
    for k in range(1, 10):
        if k == linecol:
            continue
        kk = (g == k)
        for r in range(H):
            xs = np.where(kk[r])[0]
            if len(xs) >= 2:
                for c in range(xs.min(), xs.max() + 1):
                    if g[r, c] == 0:
                        out[r, c] = k; changed = True
        for c in range(W):
            ys = np.where(kk[:, c])[0]
            if len(ys) >= 2:
                for r in range(ys.min(), ys.max() + 1):
                    if g[r, c] == 0:
                        out[r, c] = k; changed = True
    return out if changed else None


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
    init = [
        numpy_helper.from_array(e0, "e0"),
        numpy_helper.from_array(np.array(2, np.int64), "ax2v"),
        numpy_helper.from_array(np.array(3, np.int64), "ax3v"),
        numpy_helper.from_array(np.array(0.5, np.float32), "half"),
        numpy_helper.from_array(np.array([0], np.int64), "c0s"),
        numpy_helper.from_array(np.array([1], np.int64), "c1e"),
        numpy_helper.from_array(np.array([1], np.int64), "ax1"),
    ]
    nodes = [
        n("ReduceSum", ["input"], ["occ"], axes=[1], keepdims=1),
        n("Slice", ["input", "c0s", "c1e", "ax1"], ["is0"]),
        # row between
        n("CumSum", ["input", "ax3v"], ["rL"], exclusive=1),
        n("Greater", ["rL", "half"], ["hL_b"]), n("Cast", ["hL_b"], ["hL"], to=F),
        n("CumSum", ["input", "ax3v"], ["rR"], exclusive=1, reverse=1),
        n("Greater", ["rR", "half"], ["hR_b"]), n("Cast", ["hR_b"], ["hR"], to=F),
        n("Mul", ["hL", "hR"], ["rowBet"]),
        # col between
        n("CumSum", ["input", "ax2v"], ["cU"], exclusive=1),
        n("Greater", ["cU", "half"], ["hU_b"]), n("Cast", ["hU_b"], ["hU"], to=F),
        n("CumSum", ["input", "ax2v"], ["cD"], exclusive=1, reverse=1),
        n("Greater", ["cD", "half"], ["hD_b"]), n("Cast", ["hD_b"], ["hD"], to=F),
        n("Mul", ["hU", "hD"], ["colBet"]),
        n("Max", ["rowBet", "colBet"], ["between"]),
        n("Mul", ["between", "is0"], ["fillRaw"]),
        # line colour (forms a full row)
        n("ReduceSum", ["input"], ["cnt"], axes=[3], keepdims=1),     # (1,C,H,1)
        n("ReduceSum", ["occ"], ["rowW"], axes=[3], keepdims=1),      # (1,1,H,1)
        n("Sub", ["cnt", "rowW"], ["dcr"]), n("Mul", ["dcr", "dcr"], ["dcr2"]),
        n("Less", ["dcr2", "half"], ["eqW_b"]), n("Cast", ["eqW_b"], ["eqW"], to=F),
        n("Greater", ["rowW", "half"], ["rwpos_b"]), n("Cast", ["rwpos_b"], ["rwpos"], to=F),
        n("Mul", ["eqW", "rwpos"], ["allK"]),
        n("ReduceMax", ["allK"], ["lineV"], axes=[2], keepdims=1),    # (1,C,1,1)
    ]
    # notLine = 1 - lineV ; need a "one" const broadcastable to (1,C,1,1)
    init.append(numpy_helper.from_array(np.ones((1, CHANNELS, 1, 1), np.float32), "onesC"))
    nodes += [
        n("Sub", ["onesC", "lineV"], ["notLine"]),
        n("Mul", ["fillRaw", "notLine"], ["fill"]),
        n("ReduceSum", ["fill"], ["fillAny"], axes=[1], keepdims=1),
        n("Mul", ["fillAny", "e0"], ["rmBg"]),
        n("Add", ["input", "fill"], ["t1"]),
        n("Sub", ["t1", "rmBg"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "lattice_connect",
                              [helper.make_tensor_value_info("input", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              [helper.make_tensor_value_info("output", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_lattice_connect(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
