"""Solver: stretch a hollow box until its edge reaches a marker (task 281).

A rectangular frame of border colour B encloses an interior of colour I; a lone
marker M sits outside the frame, aligned with one of its sides.  The box is
stretched toward the marker so that side moves out to the marker's row/column,
the frame and interior redrawn at the new size and the marker removed.

Build (all at runtime): B is the most-common non-background channel and its
bbox the frame; the marker is the other non-B colour lying outside that bbox
(the interior colour lies inside).  One of the four stretch directions is
selected from the marker's position, the new bbox edges are computed, and the
output paints the perimeter B, the interior I and the remaining grid cells 0.
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
    cols = [int(c) for c in np.unique(g) if c != 0]
    if not cols:
        return None
    counts = {c: int((g == c).sum()) for c in cols}
    B = max(counts, key=lambda c: counts[c])
    ys, xs = np.where(g == B)
    r0, r1, c0, c1 = ys.min(), ys.max(), xs.min(), xs.max()
    M = I = None
    for c in cols:
        if c == B:
            continue
        cy, cx = int(np.where(g == c)[0][0]), int(np.where(g == c)[1][0])
        if r0 <= cy <= r1 and c0 <= cx <= c1:
            I = c
        else:
            M = c
    if M is None:
        return None
    my, mx = int(np.where(g == M)[0][0]), int(np.where(g == M)[1][0])
    if I is None:
        I = 0
    out = g.copy()
    out[g == M] = 0
    nr0, nr1, nc0, nc1 = r0, r1, c0, c1
    if my > r1 and c0 <= mx <= c1:
        nr1 = my
    elif my < r0 and c0 <= mx <= c1:
        nr0 = my
    elif mx > c1 and r0 <= my <= r1:
        nc1 = mx
    elif mx < c0 and r0 <= my <= r1:
        nc0 = mx
    else:
        return None
    out[nr0:nr1 + 1, nc0:nc1 + 1] = I
    out[nr0:nr1 + 1, nc0] = B; out[nr0:nr1 + 1, nc1] = B
    out[nr0, nc0:nc1 + 1] = B; out[nr1, nc0:nc1 + 1] = B
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
    F = TensorProto.FLOAT
    n = helper.make_node
    row_idx = np.arange(HEIGHT, dtype=np.float32).reshape(1, 1, HEIGHT, 1)
    col_idx = np.arange(WIDTH, dtype=np.float32).reshape(1, 1, 1, WIDTH)
    note0 = np.ones((1, CHANNELS, 1, 1), np.float32); note0[0, 0] = 0.0
    e0 = np.zeros((1, CHANNELS, 1, 1), np.float32); e0[0, 0] = 1.0
    init = [
        numpy_helper.from_array(row_idx, "row_idx"),
        numpy_helper.from_array(col_idx, "col_idx"),
        numpy_helper.from_array(note0, "note0"),
        numpy_helper.from_array(e0, "e0"),
        numpy_helper.from_array(np.array(0.5, np.float32), "half"),
        numpy_helper.from_array(np.array(1.0, np.float32), "one"),
        numpy_helper.from_array(np.array(1000.0, np.float32), "big"),
    ]

    def cf(b, name):
        return n("Cast", [b], [name], to=F)

    def ge(a, b, name):  # a >= b  (a tensor, b scalar)
        return [n("Sub", [b, "half"], [name + "_t"]), n("Greater", [a, name + "_t"], [name + "_b"]), cf(name + "_b", name)]

    def le(a, b, name):  # a <= b
        return [n("Add", [b, "half"], [name + "_t"]), n("Less", [a, name + "_t"], [name + "_b"]), cf(name + "_b", name)]

    def eqs(a, b, name):  # |a-b| < 0.5
        return [n("Sub", [a, b], [name + "_d"]), n("Abs", [name + "_d"], [name + "_a"]),
                n("Less", [name + "_a", "half"], [name + "_b"]), cf(name + "_b", name)]

    nodes = [
        # B = most-common non-bg channel
        n("ReduceSum", ["input"], ["tot"], axes=[2, 3], keepdims=1),
        n("Mul", ["tot", "note0"], ["totnb"]),
        n("ReduceMax", ["totnb"], ["maxnb"], axes=[1], keepdims=1),
        n("Sub", ["maxnb", "half"], ["maxnbm"]),
        n("Greater", ["totnb", "maxnbm"], ["eB_b"]), cf("eB_b", "eB"),
        n("Mul", ["input", "eB"], ["Bc"]),
        n("ReduceSum", ["Bc"], ["Bmask"], axes=[1], keepdims=1),       # (1,1,H,W)
        n("ReduceMax", ["Bmask"], ["rhB"], axes=[3], keepdims=1),      # (1,1,H,1)
        n("ReduceMax", ["Bmask"], ["chB"], axes=[2], keepdims=1),      # (1,1,1,W)
        # B bbox r0,r1,c0,c1
        n("Mul", ["rhB", "row_idx"], ["rB"]),
        n("ReduceMax", ["rB"], ["r1"], axes=[2], keepdims=1),
        n("Sub", ["one", "rhB"], ["nrhB"]), n("Mul", ["nrhB", "big"], ["nrhBb"]),
        n("Add", ["rB", "nrhBb"], ["rBlo"]), n("ReduceMin", ["rBlo"], ["r0"], axes=[2], keepdims=1),
        n("Mul", ["chB", "col_idx"], ["cB"]),
        n("ReduceMax", ["cB"], ["c1"], axes=[3], keepdims=1),
        n("Sub", ["one", "chB"], ["nchB"]), n("Mul", ["nchB", "big"], ["nchBb"]),
        n("Add", ["cB", "nchBb"], ["cBlo"]), n("ReduceMin", ["cBlo"], ["c0"], axes=[3], keepdims=1),
        # other non-bg, non-B colours
        n("ReduceMax", ["input"], ["hc"], axes=[2, 3], keepdims=1),
        n("Greater", ["hc", "half"], ["hcb"]), cf("hcb", "hcf"),
        n("Sub", ["one", "eB"], ["noteB"]),
        n("Mul", ["hcf", "note0"], ["o1"]), n("Mul", ["o1", "noteB"], ["eOther"]),
        n("Mul", ["input", "eOther"], ["otherc"]),
        n("ReduceSum", ["otherc"], ["othermask"], axes=[1], keepdims=1),   # (1,1,H,W)
        # inside-box mask (old bbox)
        *ge("row_idx", "r0", "ge_r0"), *le("row_idx", "r1", "le_r1"),
        n("Mul", ["ge_r0", "le_r1"], ["rin0"]),
        *ge("col_idx", "c0", "ge_c0"), *le("col_idx", "c1", "le_c1"),
        n("Mul", ["ge_c0", "le_c1"], ["cin0"]),
        n("Mul", ["rin0", "cin0"], ["inbox0"]),
        n("Sub", ["one", "inbox0"], ["outbox0"]),
        # marker = other cell outside the box
        n("Mul", ["othermask", "outbox0"], ["Mcell"]),
        n("Mul", ["input", "Mcell"], ["Mfield"]),
        n("ReduceSum", ["Mfield"], ["eM"], axes=[2, 3], keepdims=1),    # (1,10,1,1)
        n("Sub", ["eOther", "eM"], ["eI"]),
        n("Mul", ["Mcell", "row_idx"], ["myf"]), n("ReduceSum", ["myf"], ["my"]),
        n("Mul", ["Mcell", "col_idx"], ["mxf"]), n("ReduceSum", ["mxf"], ["mx"]),
        # stretch direction selectors (scalars)
        *ge("mx", "c0", "mx_ge_c0"), *le("mx", "c1", "mx_le_c1"),
        n("Mul", ["mx_ge_c0", "mx_le_c1"], ["col_align"]),
        *ge("my", "r0", "my_ge_r0"), *le("my", "r1", "my_le_r1"),
        n("Mul", ["my_ge_r0", "my_le_r1"], ["row_align"]),
        n("Sub", ["my", "r1"], ["my_m_r1"]), n("Greater", ["my_m_r1", "half"], ["dn_b"]), cf("dn_b", "dn0"),
        n("Mul", ["dn0", "col_align"], ["is_dn"]),
        n("Sub", ["r0", "my"], ["r0_m_my"]), n("Greater", ["r0_m_my", "half"], ["up_b"]), cf("up_b", "up0"),
        n("Mul", ["up0", "col_align"], ["is_up"]),
        n("Sub", ["mx", "c1"], ["mx_m_c1"]), n("Greater", ["mx_m_c1", "half"], ["rt_b"]), cf("rt_b", "rt0"),
        n("Mul", ["rt0", "row_align"], ["is_rt"]),
        n("Sub", ["c0", "mx"], ["c0_m_mx"]), n("Greater", ["c0_m_mx", "half"], ["lf_b"]), cf("lf_b", "lf0"),
        n("Mul", ["lf0", "row_align"], ["is_lf"]),
        # new bbox edges
        n("Sub", ["my", "r1"], ["dr1"]), n("Mul", ["is_dn", "dr1"], ["adr1"]), n("Add", ["r1", "adr1"], ["nr1"]),
        n("Sub", ["my", "r0"], ["dr0"]), n("Mul", ["is_up", "dr0"], ["adr0"]), n("Add", ["r0", "adr0"], ["nr0"]),
        n("Sub", ["mx", "c1"], ["dc1"]), n("Mul", ["is_rt", "dc1"], ["adc1"]), n("Add", ["c1", "adc1"], ["nc1"]),
        n("Sub", ["mx", "c0"], ["dc0"]), n("Mul", ["is_lf", "dc0"], ["adc0"]), n("Add", ["c0", "adc0"], ["nc0"]),
        # new inbox
        *ge("row_idx", "nr0", "ge_nr0"), *le("row_idx", "nr1", "le_nr1"),
        n("Mul", ["ge_nr0", "le_nr1"], ["rin"]),
        *ge("col_idx", "nc0", "ge_nc0"), *le("col_idx", "nc1", "le_nc1"),
        n("Mul", ["ge_nc0", "le_nc1"], ["cin"]),
        n("Mul", ["rin", "cin"], ["inbox"]),
        # frame = inbox AND on an edge
        *eqs("row_idx", "nr0", "e_nr0"), *eqs("row_idx", "nr1", "e_nr1"),
        n("Add", ["e_nr0", "e_nr1"], ["edge_r"]),
        *eqs("col_idx", "nc0", "e_nc0"), *eqs("col_idx", "nc1", "e_nc1"),
        n("Add", ["e_nc0", "e_nc1"], ["edge_c"]),
        n("Add", ["edge_r", "edge_c"], ["edge_sum"]),
        n("Greater", ["edge_sum", "half"], ["on_edge_b"]), cf("on_edge_b", "on_edge"),
        n("Mul", ["inbox", "on_edge"], ["frame"]),
        n("Sub", ["inbox", "frame"], ["interior"]),
        # background = grid cells outside the new box
        n("ReduceSum", ["input"], ["content"], axes=[1], keepdims=1),
        n("Sub", ["one", "inbox"], ["notin"]),
        n("Mul", ["content", "notin"], ["bgmask"]),
        # paint
        n("Mul", ["eB", "frame"], ["pB"]),
        n("Mul", ["eI", "interior"], ["pI"]),
        n("Mul", ["e0", "bgmask"], ["pBg"]),
        n("Add", ["pB", "pI"], ["pp"]),
        n("Add", ["pp", "pBg"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "box_stretch",
                              [helper.make_tensor_value_info("input", F, FULL)],
                              [helper.make_tensor_value_info("output", F, FULL)],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_box_stretch(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
