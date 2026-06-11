"""Solver: stack scattered markers against a solid 5-band (task 93).

A solid band of colour 5 spans the whole grid in one direction (full rows or
full columns).  Scattered markers (one other colour) gravitate toward the band
and pack against its edge, each becoming a 5; the originals vanish::

    . m . m .            . . . . .          (markers fall onto the band and
    5 5 5 5 5    ->       5 5 5 5 5           stack: two in a column -> two
    . . m . .            5 . 5 . .           rows of 5 against that edge)

The band orientation is detected from full-of-5 rows/cols (measured within the
grid width, ignoring padding); per perpendicular line the marker count gives
how many cells to fill outward from the band edge.  Horizontal and vertical
results are both computed and blended by an orientation gate.
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
    fr = [r for r in range(H) if np.all(g[r] == 5)]
    fc = [c for c in range(W) if np.all(g[:, c] == 5)]
    mk = [v for v in np.unique(g) if v not in (0, 5)]
    if len(mk) != 1 or (not fr and not fc):
        return None
    m = mk[0]
    M = g == m
    out = g.copy()
    out[out == m] = 0
    if fr:
        t0, t1 = min(fr), max(fr)
        for c in range(W):
            for k in range(int(M[:t0, c].sum())):
                if t0 - 1 - k >= 0:
                    out[t0 - 1 - k, c] = 5
            for k in range(int(M[t1 + 1:, c].sum())):
                if t1 + 1 + k < H:
                    out[t1 + 1 + k, c] = 5
    else:
        c0, c1 = min(fc), max(fc)
        for r in range(H):
            for k in range(int(M[r, :c0].sum())):
                if c0 - 1 - k >= 0:
                    out[r, c0 - 1 - k] = 5
            for k in range(int(M[r, c1 + 1:].sum())):
                if c1 + 1 + k < W:
                    out[r, c1 + 1 + k] = 5
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
    e5 = np.zeros((1, CHANNELS, 1, 1), np.float32); e5[0, 5] = 1.0
    e0 = np.zeros((1, CHANNELS, 1, 1), np.float32); e0[0, 0] = 1.0
    init = [
        numpy_helper.from_array(e5, "e5"),
        numpy_helper.from_array(e0, "e0"),
        numpy_helper.from_array(np.arange(HEIGHT, dtype=np.float32).reshape(1, 1, HEIGHT, 1), "ah"),
        numpy_helper.from_array(np.arange(WIDTH, dtype=np.float32).reshape(1, 1, 1, WIDTH), "aw"),
        numpy_helper.from_array(np.array(0.5, np.float32), "half"),
        numpy_helper.from_array(np.array([5], np.int64), "c5"),
        numpy_helper.from_array(np.array([6], np.int64), "c6"),
        numpy_helper.from_array(np.array([0], np.int64), "c0"),
        numpy_helper.from_array(np.array([1], np.int64), "c1"),
        numpy_helper.from_array(np.array([1], np.int64), "ax1"),
    ]

    def cast(b, o):
        return n("Cast", [b], [o], to=F)

    nodes = [
        n("ReduceSum", ["input"], ["occ"], axes=[1], keepdims=1),
        n("Slice", ["input", "c0", "c1", "ax1"], ["is0"]),
        n("Slice", ["input", "c5", "c6", "ax1"], ["is5"]),
        n("Sub", ["occ", "is0"], ["nb0"]),
        n("Sub", ["nb0", "is5"], ["M"]),                          # marker mask
        # --- band rows ---
        n("ReduceSum", ["is5"], ["rs5"], axes=[3], keepdims=1),
        n("ReduceSum", ["occ"], ["rocc"], axes=[3], keepdims=1),
        n("Sub", ["rocc", "rs5"], ["rdiff"]),
        n("Less", ["rdiff", "half"], ["rfull_b"]), cast("rfull_b", "rfull"),
        n("Greater", ["rocc", "half"], ["rhas_b"]), cast("rhas_b", "rhas"),
        n("Mul", ["rfull", "rhas"], ["bandRow"]),                 # (1,1,30,1)
        n("ArgMax", ["bandRow"], ["t0i"], axis=2, keepdims=1), n("Cast", ["t0i"], ["t0"], to=F),
        n("Mul", ["bandRow", "ah"], ["brpos"]),
        n("ReduceMax", ["brpos"], ["t1"], axes=[2], keepdims=1),
        # above / below stacking
        n("Less", ["ah", "t0"], ["ltT0_b"]), cast("ltT0_b", "ltT0"),
        n("Mul", ["M", "ltT0"], ["Mabove"]),
        n("ReduceSum", ["Mabove"], ["aboveC"], axes=[2], keepdims=1),   # (1,1,1,30)
        n("Sub", ["t0", "ah"], ["t0mr"]),
        n("Add", ["aboveC", "half"], ["aboveCh"]),
        n("Less", ["t0mr", "aboveCh"], ["fa_b"]), cast("fa_b", "fa0"),
        n("Mul", ["fa0", "ltT0"], ["fillA"]),
        n("Greater", ["ah", "t1"], ["gtT1_b"]), cast("gtT1_b", "gtT1"),
        n("Mul", ["M", "gtT1"], ["Mbelow"]),
        n("ReduceSum", ["Mbelow"], ["belowC"], axes=[2], keepdims=1),
        n("Sub", ["ah", "t1"], ["rmt1"]),
        n("Add", ["belowC", "half"], ["belowCh"]),
        n("Less", ["rmt1", "belowCh"], ["fb_b"]), cast("fb_b", "fb0"),
        n("Mul", ["fb0", "gtT1"], ["fillB"]),
        n("Add", ["is5", "fillA"], ["h5a"]), n("Add", ["h5a", "fillB"], ["h5"]),
        n("Mul", ["h5", "e5"], ["Hp5"]),
        n("Sub", ["occ", "h5"], ["hbg"]), n("Mul", ["hbg", "e0"], ["Hp0"]),
        n("Add", ["Hp5", "Hp0"], ["Hout"]),
        # --- band cols (vertical) ---
        n("ReduceSum", ["is5"], ["cs5"], axes=[2], keepdims=1),
        n("ReduceSum", ["occ"], ["cocc"], axes=[2], keepdims=1),
        n("Sub", ["cocc", "cs5"], ["cdiff"]),
        n("Less", ["cdiff", "half"], ["cfull_b"]), cast("cfull_b", "cfull"),
        n("Greater", ["cocc", "half"], ["chas_b"]), cast("chas_b", "chas"),
        n("Mul", ["cfull", "chas"], ["bandCol"]),                 # (1,1,1,30)
        n("ArgMax", ["bandCol"], ["k0i"], axis=3, keepdims=1), n("Cast", ["k0i"], ["k0"], to=F),
        n("Mul", ["bandCol", "aw"], ["bcpos"]),
        n("ReduceMax", ["bcpos"], ["k1"], axes=[3], keepdims=1),
        n("Less", ["aw", "k0"], ["ltK0_b"]), cast("ltK0_b", "ltK0"),
        n("Mul", ["M", "ltK0"], ["Mleft"]),
        n("ReduceSum", ["Mleft"], ["leftC"], axes=[3], keepdims=1),     # (1,1,30,1)
        n("Sub", ["k0", "aw"], ["k0mc"]),
        n("Add", ["leftC", "half"], ["leftCh"]),
        n("Less", ["k0mc", "leftCh"], ["fl_b"]), cast("fl_b", "fl0"),
        n("Mul", ["fl0", "ltK0"], ["fillL"]),
        n("Greater", ["aw", "k1"], ["gtK1_b"]), cast("gtK1_b", "gtK1"),
        n("Mul", ["M", "gtK1"], ["Mright"]),
        n("ReduceSum", ["Mright"], ["rightC"], axes=[3], keepdims=1),
        n("Sub", ["aw", "k1"], ["cmk1"]),
        n("Add", ["rightC", "half"], ["rightCh"]),
        n("Less", ["cmk1", "rightCh"], ["fr_b"]), cast("fr_b", "fr0"),
        n("Mul", ["fr0", "gtK1"], ["fillR"]),
        n("Add", ["is5", "fillL"], ["v5a"]), n("Add", ["v5a", "fillR"], ["v5"]),
        n("Mul", ["v5", "e5"], ["Vp5"]),
        n("Sub", ["occ", "v5"], ["vbg"]), n("Mul", ["vbg", "e0"], ["Vp0"]),
        n("Add", ["Vp5", "Vp0"], ["Vout"]),
        # --- blend by orientation: output = hasH*Hout + (1-hasH)*Vout ---
        n("ReduceSum", ["bandRow"], ["nBandRow"], axes=[2, 3], keepdims=1),
        n("Greater", ["nBandRow", "half"], ["hasH_b"]), cast("hasH_b", "hasH"),
        n("Mul", ["Hout", "hasH"], ["Hsel"]),
        n("Sub", ["one", "hasH"], ["invH"]),
        n("Mul", ["Vout", "invH"], ["Vsel"]),
        n("Add", ["Hsel", "Vsel"], ["output"]),
    ]
    init.append(numpy_helper.from_array(np.array(1.0, np.float32), "one"))
    graph = helper.make_graph(nodes, "stack_to_band",
                              [helper.make_tensor_value_info("input", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              [helper.make_tensor_value_info("output", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_stack_to_band(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
