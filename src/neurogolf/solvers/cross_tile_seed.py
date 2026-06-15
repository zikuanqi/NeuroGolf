"""Solver: tile a vertical and a horizontal colour-seed across their lines (task 358).

One column holds a short vertical run of colours and one row holds a horizontal
run; each run is repeated periodically along its whole line (period = run
length), keeping phase.  Everything else is background.

Per line: the seed column/row is the one with the most foreground cells; its run
extent gives the period ``P`` and start; the tiled value at offset ``d`` is the
seed at ``d mod P`` (computed with ``Div``/``Floor``) and read back with a
``Gather``.
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
    out = g.copy()
    changed = False
    colcnt = (g != 0).sum(0); rowcnt = (g != 0).sum(1)
    vc = int(np.argmax(colcnt)); hr = int(np.argmax(rowcnt))
    rows = np.where(g[:, vc] != 0)[0]
    if len(rows) >= 2:
        r0, r1 = rows.min(), rows.max(); seed = g[r0:r1 + 1, vc]; P = len(seed)
        for r in range(H):
            out[r, vc] = seed[(r - r0) % P]
        changed = True
    cols = np.where(g[hr, :] != 0)[0]
    if len(cols) >= 2:
        c0, c1 = cols.min(), cols.max(); seed = g[hr, c0:c1 + 1]; P = len(seed)
        for c in range(W):
            out[hr, c] = seed[(c - c0) % P]
        changed = True
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
    notbg = np.ones((1, CHANNELS, 1, 1), np.float32); notbg[0, 0] = 0.0
    rowidx = np.arange(HEIGHT, dtype=np.float32).reshape(1, 1, HEIGHT, 1)
    colidx = np.arange(WIDTH, dtype=np.float32).reshape(1, 1, 1, WIDTH)
    init = [
        numpy_helper.from_array(e0, "e0"),
        numpy_helper.from_array(notbg, "notbg"),
        numpy_helper.from_array(rowidx, "rowidx"),
        numpy_helper.from_array(colidx, "colidx"),
        numpy_helper.from_array(np.array(1.0, np.float32), "one"),
        numpy_helper.from_array(np.array(0.5, np.float32), "half"),
        numpy_helper.from_array(np.array(1.5, np.float32), "onehalf"),
        numpy_helper.from_array(np.array(1e4, np.float32), "big"),
        numpy_helper.from_array(np.array([0], np.int64), "c0s"),
        numpy_helper.from_array(np.array([1], np.int64), "c1e"),
        numpy_helper.from_array(np.array([1], np.int64), "ax1"),
        numpy_helper.from_array(np.array([HEIGHT], np.int64), "to30"),
    ]
    nodes = [
        n("ReduceSum", ["input"], ["occ"], axes=[1], keepdims=1),
        n("Slice", ["input", "c0s", "c1e", "ax1"], ["is0"]),
        n("Sub", ["occ", "is0"], ["nonbg"]),
        n("ReduceSum", ["nonbg"], ["colCnt"], axes=[2], keepdims=1),   # (1,1,1,W)
        n("ReduceSum", ["nonbg"], ["rowCnt"], axes=[3], keepdims=1),   # (1,1,H,1)
        n("ReduceMax", ["colCnt"], ["maxCol"], axes=[3], keepdims=1),
        n("ReduceMax", ["rowCnt"], ["maxRow"], axes=[2], keepdims=1),
        n("Sub", ["colCnt", "maxCol"], ["vcd"]), n("Mul", ["vcd", "vcd"], ["vcd2"]),
        n("Less", ["vcd2", "half"], ["vc_b"]), n("Cast", ["vc_b"], ["vcMask"], to=F),   # (1,1,1,W)
        n("Sub", ["rowCnt", "maxRow"], ["hrd"]), n("Mul", ["hrd", "hrd"], ["hrd2"]),
        n("Less", ["hrd2", "half"], ["hr_b"]), n("Cast", ["hr_b"], ["hrMask"], to=F),   # (1,1,H,1)
        # ---- vertical seed (column vc) ----
        n("Mul", ["input", "vcMask"], ["vcol_x"]),
        n("ReduceSum", ["vcol_x"], ["vcolC"], axes=[3], keepdims=1),    # (1,C,H,1)
        n("Mul", ["vcolC", "notbg"], ["vcolNb"]),
        n("ReduceMax", ["vcolNb"], ["vseedRow"], axes=[1], keepdims=1), # (1,1,H,1)
        n("Mul", ["vseedRow", "rowidx"], ["vsr"]),
        n("ReduceMax", ["vsr"], ["vr1"], axes=[2], keepdims=1),
        n("Sub", ["one", "vseedRow"], ["vinv"]), n("Mul", ["vinv", "big"], ["vinvB"]),
        n("Add", ["vsr", "vinvB"], ["vmincand"]),
        n("ReduceMin", ["vmincand"], ["vr0"], axes=[2], keepdims=1),
        n("Sub", ["vr1", "vr0"], ["vspan"]), n("Add", ["vspan", "one"], ["vP"]),
        n("Sub", ["rowidx", "vr0"], ["vrdiff"]),
        n("Div", ["vrdiff", "vP"], ["vqf"]), n("Floor", ["vqf"], ["vq"]),
        n("Mul", ["vq", "vP"], ["vqp"]), n("Sub", ["vrdiff", "vqp"], ["vmod"]),
        n("Add", ["vr0", "vmod"], ["vsrcF"]),
        n("Cast", ["vsrcF"], ["vsrcI"], to=TensorProto.INT64),
        n("Reshape", ["vsrcI", "to30"], ["vsrc1d"]),
        n("Gather", ["vcolC", "vsrc1d"], ["tiledCol"], axis=2),        # (1,C,H,1)
        n("Greater", ["vP", "onehalf"], ["vgate_b"]), n("Cast", ["vgate_b"], ["vgate"], to=F),
        n("Mul", ["tiledCol", "vcMask"], ["tcolP0"]),
        n("Mul", ["tcolP0", "vgate"], ["tcolP"]),                       # (1,C,H,W)
        # ---- horizontal seed (row hr) ----
        n("Mul", ["input", "hrMask"], ["hrow_x"]),
        n("ReduceSum", ["hrow_x"], ["hrowC"], axes=[2], keepdims=1),    # (1,C,1,W)
        n("Mul", ["hrowC", "notbg"], ["hrowNb"]),
        n("ReduceMax", ["hrowNb"], ["hseedCol"], axes=[1], keepdims=1), # (1,1,1,W)
        n("Mul", ["hseedCol", "colidx"], ["hsc"]),
        n("ReduceMax", ["hsc"], ["hc1"], axes=[3], keepdims=1),
        n("Sub", ["one", "hseedCol"], ["hinv"]), n("Mul", ["hinv", "big"], ["hinvB"]),
        n("Add", ["hsc", "hinvB"], ["hmincand"]),
        n("ReduceMin", ["hmincand"], ["hc0"], axes=[3], keepdims=1),
        n("Sub", ["hc1", "hc0"], ["hspan"]), n("Add", ["hspan", "one"], ["hP"]),
        n("Sub", ["colidx", "hc0"], ["hcdiff"]),
        n("Div", ["hcdiff", "hP"], ["hqf"]), n("Floor", ["hqf"], ["hq"]),
        n("Mul", ["hq", "hP"], ["hqp"]), n("Sub", ["hcdiff", "hqp"], ["hmod"]),
        n("Add", ["hc0", "hmod"], ["hsrcF"]),
        n("Cast", ["hsrcF"], ["hsrcI"], to=TensorProto.INT64),
        n("Reshape", ["hsrcI", "to30"], ["hsrc1d"]),
        n("Gather", ["hrowC", "hsrc1d"], ["tiledRow"], axis=3),        # (1,C,1,W)
        n("Greater", ["hP", "onehalf"], ["hgate_b"]), n("Cast", ["hgate_b"], ["hgate"], to=F),
        n("Mul", ["tiledRow", "hrMask"], ["trowP0"]),
        n("Mul", ["trowP0", "hgate"], ["trowP"]),                       # (1,C,H,W)
        # ---- combine ----
        n("Max", ["tcolP", "trowP"], ["colorL0"]),
        n("Mul", ["colorL0", "occ"], ["colorL"]),
        n("ReduceSum", ["colorL"], ["painted"], axes=[1], keepdims=1),
        n("Sub", ["occ", "painted"], ["bgv"]),
        n("Mul", ["bgv", "e0"], ["bgL"]),
        n("Add", ["colorL", "bgL"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "cross_tile_seed",
                              [helper.make_tensor_value_info("input", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              [helper.make_tensor_value_info("output", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_cross_tile_seed(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
