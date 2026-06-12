"""Solver: marked bar shoots 8-rays; the other bar echoes them (task 148).

Two vertical colour-2 bars stand in different columns.  Colour-8 markers sit in
some rows of one bar ("marked").  In each marked row the span between the bar
and the marker fills with ``8`` and the marker itself turns ``4``; the other
bar repeats the same row pattern at the same *relative* offsets, filling those
entire rows with ``8``::

    2 . . 8 .        2 8 8 4 .
    2 . . . .   ->   2 . . . .      (and the partner bar fills its row 0
    ...                              completely with 8)

Everything is closed-form: per-row bar/marker columns via ``ArgMax``, the
marked bar identified by the column shared with the 8-rows, and the offset
transfer done with a 30x30 row-vs-row comparison matrix (``relu == relm``).
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
    bars = []
    for c in range(W):
        rs = np.where(g[:, c] == 2)[0]
        if len(rs) == 0:
            continue
        if not np.all(np.diff(rs) == 1):
            return None
        bars.append((c, int(rs.min()), int(rs.max())))
    if len(bars) != 2 or bars[0][0] == bars[1][0]:
        return None
    r8 = {r: np.where(g[r] == 8)[0] for r in range(H) if (g[r] == 8).any()}
    if not r8:
        return None
    marked = [b for b in bars if any(b[1] <= m <= b[2] for m in r8)]
    if len(marked) != 1:
        return None
    mb = marked[0]
    ub = [b for b in bars if b != mb][0]
    offsets = []
    for r, cols8 in r8.items():
        if len(cols8) != 1 or not (mb[1] <= r <= mb[2]):
            return None
        b, e = mb[0], int(cols8[0])
        lo, hi = min(b, e), max(b, e)
        out[r, lo + 1:hi] = 8
        out[r, e] = 4
        offsets.append(r - mb[1])
    for o in offsets:
        ur = ub[1] + o
        if not (ub[1] <= ur <= ub[2]):
            return None
        row = out[ur]
        row[g[ur] == 0] = 8
        out[ur] = row
    return out


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
    e4 = np.zeros((1, CHANNELS, 1, 1), np.float32); e4[0, 4] = 1.0
    e8 = np.zeros((1, CHANNELS, 1, 1), np.float32); e8[0, 8] = 1.0
    init = [
        numpy_helper.from_array(e0, "e0"),
        numpy_helper.from_array(e4, "e4"),
        numpy_helper.from_array(e8, "e8"),
        numpy_helper.from_array(np.arange(HEIGHT, dtype=np.float32).reshape(1, 1, HEIGHT, 1), "ah"),
        numpy_helper.from_array(np.arange(WIDTH, dtype=np.float32).reshape(1, 1, 1, WIDTH), "aw"),
        numpy_helper.from_array(np.array(0.5, np.float32), "half"),
        numpy_helper.from_array(np.array(1.0, np.float32), "one"),
        numpy_helper.from_array(np.array([0], np.int64), "c0s"),
        numpy_helper.from_array(np.array([1], np.int64), "c1e"),
        numpy_helper.from_array(np.array([2], np.int64), "c2s"),
        numpy_helper.from_array(np.array([3], np.int64), "c3e"),
        numpy_helper.from_array(np.array([8], np.int64), "c8s"),
        numpy_helper.from_array(np.array([9], np.int64), "c9e"),
        numpy_helper.from_array(np.array([1], np.int64), "ax1"),
    ]
    nodes = [
        n("Slice", ["input", "c0s", "c1e", "ax1"], ["is0"]),
        n("Slice", ["input", "c2s", "c3e", "ax1"], ["is2"]),
        n("Slice", ["input", "c8s", "c9e", "ax1"], ["is8"]),
        n("ReduceMax", ["is2"], ["has2"], axes=[3], keepdims=1),       # (1,1,30,1)
        n("ReduceMax", ["is8"], ["has8"], axes=[3], keepdims=1),
        n("ArgMax", ["is2"], ["bcol_i"], axis=3, keepdims=1), n("Cast", ["bcol_i"], ["bcol"], to=F),
        n("ArgMax", ["is8"], ["ecol_i"], axis=3, keepdims=1), n("Cast", ["ecol_i"], ["ecol"], to=F),
        # marked bar column = bar col of any 8-row
        n("Mul", ["has8", "bcol"], ["mc8"]),
        n("ReduceMax", ["mc8"], ["mc"], axes=[2], keepdims=1),
        # marked / unmarked bar row masks
        n("Sub", ["bcol", "mc"], ["bd"]), n("Abs", ["bd"], ["bda"]),
        n("Less", ["bda", "half"], ["bm_b"]), n("Cast", ["bm_b"], ["bm0"], to=F),
        n("Mul", ["bm0", "has2"], ["mrow"]),
        n("Sub", ["has2", "mrow"], ["urow"]),
        # bar tops
        n("ArgMax", ["mrow"], ["mtop_i"], axis=2, keepdims=1), n("Cast", ["mtop_i"], ["mtop"], to=F),
        n("ArgMax", ["urow"], ["utop_i"], axis=2, keepdims=1), n("Cast", ["utop_i"], ["utop"], to=F),
        # marked-row fill between bar and marker
        n("Min", ["bcol", "ecol"], ["lo"]),
        n("Max", ["bcol", "ecol"], ["hi"]),
        n("Greater", ["aw", "lo"], ["gl_b"]), n("Cast", ["gl_b"], ["gl"], to=F),
        n("Less", ["aw", "hi"], ["lh_b"]), n("Cast", ["lh_b"], ["lh"], to=F),
        n("Mul", ["gl", "lh"], ["between"]),
        n("Mul", ["between", "has8"], ["fillm"]),
        # offset transfer: relu(u) == relm(m) for some 8-row m
        n("Sub", ["ah", "mtop"], ["relm"]),
        n("Sub", ["ah", "utop"], ["relu"]),
        n("Mul", ["relm", "has8"], ["relm8"]),
        n("Sub", ["relm8", "has8"], ["relm8a"]),     # 8-rows: relm-1 ; others: 0
        n("Transpose", ["relm8a"], ["relmT"], perm=[0, 1, 3, 2]),     # (1,1,1,30)
        n("Transpose", ["has8"], ["has8T"], perm=[0, 1, 3, 2]),
        n("Sub", ["relu", "one"], ["relu1"]),
        n("Sub", ["relu1", "relmT"], ["dmat"]),       # (1,1,30,30)
        n("Abs", ["dmat"], ["dabs"]),
        n("Less", ["dabs", "half"], ["dm_b"]), n("Cast", ["dm_b"], ["dm"], to=F),
        n("Mul", ["dm", "has8T"], ["dmg"]),
        n("ReduceMax", ["dmg"], ["uflag0"], axes=[3], keepdims=1),
        n("Mul", ["uflag0", "urow"], ["uflag"]),
        # unmarked fill: whole row on background
        n("Mul", ["uflag", "is0"], ["fillu"]),
        n("Mul", ["fillm", "is0"], ["fillm0"]),
        n("Max", ["fillm0", "fillu"], ["fill8"]),
        # paint: fill8 -> 8 ; markers (is8) -> 4
        n("Mul", ["fill8", "e8"], ["a8"]),
        n("Mul", ["fill8", "e0"], ["s0"]),
        n("Mul", ["is8", "e4"], ["a4"]),
        n("Mul", ["is8", "e8"], ["s8"]),
        n("Add", ["input", "a8"], ["t1"]),
        n("Sub", ["t1", "s0"], ["t2"]),
        n("Add", ["t2", "a4"], ["t3"]),
        n("Sub", ["t3", "s8"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "bar_echo",
                              [helper.make_tensor_value_info("input", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              [helper.make_tensor_value_info("output", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_bar_echo(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
