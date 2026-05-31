"""Solver: bridge the two dots in each row, meeting at a colour-5 midpoint.

Every row that holds exactly two marker cells gets the span between them filled:
cells left of the midpoint take the left dot's colour, cells right of it take the
right dot's colour, and the single midpoint cell (floor((c1+c2)/2)) becomes
colour 5. Rows without a pair are left untouched (task 60).

Per-row arithmetic on the (1, 10, 30, 30) one-hot tensor:
  c1, c2  = leftmost / rightmost marker column   (ReduceMax over column ramps)
  s       = c1 + c2
  between = c1 <= c <= c2
  d       = s - 2c   ->  left (d>=2), midpoint (d in {0,1}), right (d<0)
  a, b    = colours gathered at the c1 / c2 cells
  output  = a on left span + b on right span + 5 at midpoint + background
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from ..grids import CHANNELS, HEIGHT, WIDTH, all_examples

OPSET = 11
IR_VERSION = 8


def _transform(grid):
    g = np.array(grid)
    H, W = g.shape
    out = [row[:] for row in g.tolist()]
    for r in range(H):
        nz = [(c, int(g[r, c])) for c in range(W) if g[r, c] != 0]
        if not nz:
            continue
        if len(nz) != 2:
            return None
        (c1, a), (c2, b) = sorted(nz)
        mid = (c1 + c2) // 2
        for c in range(c1, c2 + 1):
            out[r][c] = a if c < mid else (b if c > mid else 5)
    return out


def _detect(task: dict) -> bool:
    examples = list(all_examples(task))
    if not examples:
        return False
    changed = False
    for ex in examples:
        inp, out = ex["input"], ex["output"]
        if len(inp) > 30 or len(inp[0]) > 30:
            continue
        if len(inp) != len(out) or len(inp[0]) != len(out[0]):
            return False
        if _transform(inp) != out:
            return False
        if inp != out:
            changed = True
    return changed


def _build() -> onnx.ModelProto:
    F = TensorProto.FLOAT
    B = TensorProto.BOOL

    def f32(name, arr):
        return numpy_helper.from_array(arr.astype(np.float32), name)

    colidx = np.arange(WIDTH, dtype=np.float32).reshape(1, 1, 1, WIDTH)
    colp1 = (np.arange(WIDTH) + 1).astype(np.float32).reshape(1, 1, 1, WIDTH)
    colrev = (WIDTH - np.arange(WIDTH)).astype(np.float32).reshape(1, 1, 1, WIDTH)
    e0 = np.zeros((1, CHANNELS, 1, 1), dtype=np.float32); e0[0, 0] = 1.0
    e5 = np.zeros((1, CHANNELS, 1, 1), dtype=np.float32); e5[0, 5] = 1.0

    init = [
        f32("colidx", colidx), f32("colp1", colp1), f32("colrev", colrev),
        f32("e0", e0), f32("e5", e5),
        f32("two", np.array([[[[2.0]]]])), f32("one", np.array([[[[1.0]]]])),
        f32("Wf", np.array([[[[float(WIDTH)]]]])),
        f32("n0p5", np.array([[[[-0.5]]]])),
        f32("p0p5", np.array([[[[0.5]]]])),
        f32("p1p5", np.array([[[[1.5]]]])),
        numpy_helper.from_array(np.array([0], dtype=np.int64), "idx0"),
    ]

    n = helper.make_node
    nodes = [
        n("ReduceSum", ["input"], ["content"], axes=[1], keepdims=1),
        n("Gather", ["input", "idx0"], ["ch0"], axis=1),
        n("Sub", ["content", "ch0"], ["nonbg"]),
        n("Mul", ["colidx", "two"], ["twoC"]),
        # rightmost / leftmost marker columns
        n("Mul", ["nonbg", "colp1"], ["pos_r"]),
        n("ReduceMax", ["pos_r"], ["c2p1"], axes=[3], keepdims=1),
        n("Sub", ["c2p1", "one"], ["c2"]),
        n("Mul", ["nonbg", "colrev"], ["pos_l"]),
        n("ReduceMax", ["pos_l"], ["maxl"], axes=[3], keepdims=1),
        n("Sub", ["Wf", "maxl"], ["c1"]),
        n("Add", ["c1", "c2"], ["s"]),
        # region + side selectors
        n("Sub", ["s", "twoC"], ["d"]),
        n("Sub", ["colidx", "c1"], ["d1"]),
        n("Sub", ["c2", "colidx"], ["d2"]),
        n("Greater", ["d1", "n0p5"], ["ge1_b"]), n("Cast", ["ge1_b"], ["ge1"], to=F),
        n("Greater", ["d2", "n0p5"], ["le2_b"]), n("Cast", ["le2_b"], ["le2"], to=F),
        n("Mul", ["ge1", "le2"], ["between"]),
        n("Greater", ["d", "p1p5"], ["lft_b"]), n("Cast", ["lft_b"], ["lft"], to=F),
        n("Less", ["d", "n0p5"], ["rgt_b"]), n("Cast", ["rgt_b"], ["rgt"], to=F),
        n("Greater", ["d", "n0p5"], ["mlo_b"]), n("Cast", ["mlo_b"], ["mlo"], to=F),
        n("Less", ["d", "p1p5"], ["mhi_b"]), n("Cast", ["mhi_b"], ["mhi"], to=F),
        n("Mul", ["mlo", "mhi"], ["mid_b"]),
        n("Mul", ["between", "lft"], ["left"]),
        n("Mul", ["between", "rgt"], ["right"]),
        n("Mul", ["between", "mid_b"], ["mid"]),
        # colours at the two endpoints
        n("Less", ["d1", "p0p5"], ["e1a_b"]), n("Cast", ["e1a_b"], ["e1a"], to=F),
        n("Mul", ["ge1", "e1a"], ["eq1"]),
        n("Mul", ["nonbg", "eq1"], ["lm_mask"]),
        n("Mul", ["input", "lm_mask"], ["a_in"]),
        n("ReduceMax", ["a_in"], ["a_chan"], axes=[3], keepdims=1),
        n("Less", ["d2", "p0p5"], ["e2a_b"]), n("Cast", ["e2a_b"], ["e2a"], to=F),
        n("Mul", ["le2", "e2a"], ["eq2"]),
        n("Mul", ["nonbg", "eq2"], ["rm_mask"]),
        n("Mul", ["input", "rm_mask"], ["b_in"]),
        n("ReduceMax", ["b_in"], ["b_chan"], axes=[3], keepdims=1),
        # assemble
        n("Mul", ["a_chan", "left"], ["out_left"]),
        n("Mul", ["b_chan", "right"], ["out_right"]),
        n("Mul", ["e5", "mid"], ["out_mid"]),
        n("Sub", ["one", "between"], ["notbetween"]),
        n("Mul", ["content", "notbetween"], ["real_nb"]),
        n("Mul", ["e0", "real_nb"], ["out_bg"]),
        n("Add", ["out_left", "out_right"], ["t1"]),
        n("Add", ["t1", "out_mid"], ["t2"]),
        n("Add", ["t2", "out_bg"], ["output"]),
    ]

    def vi(name, shape, dt=F):
        return helper.make_tensor_value_info(name, dt, shape)

    g4 = [1, CHANNELS, HEIGHT, WIDTH]
    s1 = [1, 1, HEIGHT, WIDTH]
    col = [1, 1, HEIGHT, 1]
    chcol = [1, CHANNELS, HEIGHT, 1]
    f_s1 = ["content", "ch0", "nonbg", "pos_r", "pos_l", "d", "d1", "d2",
            "ge1", "le2", "between", "lft", "rgt", "mlo", "mhi", "mid_b",
            "left", "right", "mid", "e1a", "eq1", "lm_mask", "e2a", "eq2",
            "rm_mask", "notbetween", "real_nb"]
    b_s1 = ["ge1_b", "le2_b", "lft_b", "rgt_b", "mlo_b", "mhi_b",
            "e1a_b", "e2a_b"]
    value_info = [vi(x, s1) for x in f_s1] + [vi(x, s1, B) for x in b_s1] + [
        vi("twoC", [1, 1, 1, WIDTH]),
        vi("c2p1", col), vi("c2", col), vi("maxl", col), vi("c1", col), vi("s", col),
        vi("a_chan", chcol), vi("b_chan", chcol),
        vi("a_in", g4), vi("b_in", g4),
        vi("out_left", g4), vi("out_right", g4), vi("out_mid", g4),
        vi("out_bg", g4), vi("t1", g4), vi("t2", g4),
    ]

    inputs = [vi("input", g4)]
    outputs = [vi("output", g4)]
    graph = helper.make_graph(nodes, "endpoint_bridge", inputs, outputs,
                              initializer=init, value_info=value_info)
    return helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
        ir_version=IR_VERSION)


def solve_endpoint_bridge(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
