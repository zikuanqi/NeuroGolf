"""Shape-aware odd-one-out panel solver (task 65).

Same idea as `solve_odd_panel` — four panels in a 2x2 layout separated by a
one-cell divider row & column, three identical and the output is the unique
fourth — but here the grid size varies across examples (5x5, 7x7, 11x11), so
the baked-corner static solver declines.

The input is (2n+1) x (2n+1); the divider sits at row/col n. We detect n at
runtime from the content extent, reposition each of the four n x n panels to
the top-left with index-shifted `Gather`s (shift toward the origin by n+1),
mask to the n x n region, then pick the panel that agrees with none of the
others (identical selection logic to the static version) and emit it.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from ..grids import CHANNELS, HEIGHT, WIDTH, all_examples
from .odd_panel import _panels, _odd_index

OPSET = 11
IR_VERSION = 8
FULL = [1, CHANNELS, HEIGHT, WIDTH]


def _detect(task: dict) -> bool:
    examples = list(all_examples(task))
    if not examples:
        return False
    sizes = set()
    for ex in examples:
        g = np.array(ex["input"])
        h, w = g.shape
        if h != w or h % 2 == 0 or h < 3 or h > HEIGHT:
            return False
        n = (h - 1) // 2
        oh, ow = len(ex["output"]), len(ex["output"][0])
        if oh != n or ow != n:
            return False
        # separated layout only (divider row/col at index n)
        if h != 2 * n + 1:
            return False
        cs = _panels(g, n, n)
        if cs is None or cs != [(0, 0), (0, n + 1), (n + 1, 0), (n + 1, n + 1)]:
            return False
        ps = [g[r:r + n, c:c + n] for (r, c) in cs]
        idx = _odd_index(ps)
        if idx is None or not np.array_equal(ps[idx], np.array(ex["output"])):
            return False
        sizes.add(n)
    return len(sizes) >= 1


def _build() -> onnx.ModelProto:
    F = TensorProto.FLOAT
    B = TensorProto.BOOL
    n_ = helper.make_node

    row_ar = np.arange(HEIGHT, dtype=np.float32).reshape(1, 1, HEIGHT, 1)
    col_ar = np.arange(WIDTH, dtype=np.float32).reshape(1, 1, 1, WIDTH)
    init = [
        numpy_helper.from_array(row_ar, "row_ar"),
        numpy_helper.from_array(col_ar, "col_ar"),
        numpy_helper.from_array(np.array(0.5, dtype=np.float32), "half"),
        numpy_helper.from_array(np.array(1.0, dtype=np.float32), "one"),
        numpy_helper.from_array(np.array(0.0, dtype=np.float32), "zero"),
        numpy_helper.from_array(np.array(float(HEIGHT - 1), dtype=np.float32),
                                "cmax"),
        numpy_helper.from_array(np.array([HEIGHT], dtype=np.int64), "flatH"),
        numpy_helper.from_array(np.array([WIDTH], dtype=np.int64), "flatW"),
    ]

    nodes = [
        # --- detect n and shift amount k = n + 1 ---
        n_("ReduceSum", ["input"], ["content"], axes=[1], keepdims=1),
        n_("ReduceMax", ["content"], ["row_pres"], axes=[3], keepdims=1),
        n_("Mul", ["row_pres", "row_ar"], ["row_pos"]),
        n_("ReduceMax", ["row_pos"], ["maxidx"], axes=[2], keepdims=1),
        n_("Mul", ["maxidx", "half"], ["n_f"]),
        n_("Add", ["n_f", "one"], ["k_f"]),
        # --- masks for the n x n top-left region ---
        n_("Less", ["row_ar", "n_f"], ["rmask_b"]),
        n_("Cast", ["rmask_b"], ["rmask"], to=F),
        n_("Less", ["col_ar", "n_f"], ["cmask_b"]),
        n_("Cast", ["cmask_b"], ["cmask"], to=F),
        # --- shift indices (toward origin by k) ---
        n_("Add", ["row_ar", "k_f"], ["ridx_f"]),
        n_("Clip", ["ridx_f", "zero", "cmax"], ["ridx_c"]),
        n_("Cast", ["ridx_c"], ["ridx_i"], to=TensorProto.INT64),
        n_("Reshape", ["ridx_i", "flatH"], ["ridx"]),
        n_("Add", ["col_ar", "k_f"], ["cidx_f"]),
        n_("Clip", ["cidx_f", "zero", "cmax"], ["cidx_c"]),
        n_("Cast", ["cidx_c"], ["cidx_i"], to=TensorProto.INT64),
        n_("Reshape", ["cidx_i", "flatW"], ["cidx"]),
        # --- the four panels, each repositioned to the top-left n x n ---
        n_("Mul", ["input", "rmask"], ["p0a"]),
        n_("Mul", ["p0a", "cmask"], ["P0"]),
        n_("Gather", ["input", "cidx"], ["shl"], axis=3),
        n_("Mul", ["shl", "rmask"], ["p1a"]),
        n_("Mul", ["p1a", "cmask"], ["P1"]),
        n_("Gather", ["input", "ridx"], ["shu"], axis=2),
        n_("Mul", ["shu", "rmask"], ["p2a"]),
        n_("Mul", ["p2a", "cmask"], ["P2"]),
        n_("Gather", ["shl", "ridx"], ["shlu"], axis=2),
        n_("Mul", ["shlu", "rmask"], ["p3a"]),
        n_("Mul", ["p3a", "cmask"], ["P3"]),
    ]

    # --- pairwise equality of the four aligned panels ---
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    for a, b in pairs:
        nodes += [
            n_("Sub", [f"P{a}", f"P{b}"], [f"d{a}{b}"]),
            n_("Abs", [f"d{a}{b}"], [f"ad{a}{b}"]),
            n_("ReduceSum", [f"ad{a}{b}"], [f"s{a}{b}"], axes=[1, 2, 3],
               keepdims=1),
            n_("Less", [f"s{a}{b}", "half"], [f"eqb{a}{b}"]),
            n_("Cast", [f"eqb{a}{b}"], [f"eq{a}{b}"], to=F),
        ]

    def eq(a, b):
        return f"eq{a}{b}" if a < b else f"eq{b}{a}"

    for i in range(4):
        o = [j for j in range(4) if j != i]
        nodes += [
            n_("Add", [eq(i, o[0]), eq(i, o[1])], [f"ag{i}_0"]),
            n_("Add", [f"ag{i}_0", eq(i, o[2])], [f"ag{i}"]),
            n_("Less", [f"ag{i}", "half"], [f"wb{i}"]),
            n_("Cast", [f"wb{i}"], [f"w{i}"], to=F),
            n_("Mul", [f"P{i}", f"w{i}"], [f"sel{i}"]),
        ]
    nodes += [
        n_("Add", ["sel0", "sel1"], ["sum01"]),
        n_("Add", ["sel2", "sel3"], ["sum23"]),
        n_("Add", ["sum01", "sum23"], ["output"]),
    ]
    for nd in nodes:
        nd.name = nd.output[0]

    def vi(name, shape, dt=F):
        return helper.make_tensor_value_info(name, dt, shape)

    s1 = [1, 1, HEIGHT, WIDTH]
    rcol = [1, 1, HEIGHT, 1]
    ccol = [1, 1, 1, WIDTH]
    sc = [1, 1, 1, 1]
    value_info = [
        vi("content", s1), vi("row_pres", rcol), vi("row_pos", rcol),
        vi("maxidx", sc), vi("n_f", sc), vi("k_f", sc),
        vi("rmask_b", rcol, B), vi("rmask", rcol),
        vi("cmask_b", ccol, B), vi("cmask", ccol),
        vi("ridx_f", rcol), vi("ridx_c", rcol), vi("ridx_i", rcol, TensorProto.INT64),
        vi("ridx", [HEIGHT], TensorProto.INT64),
        vi("cidx_f", ccol), vi("cidx_c", ccol), vi("cidx_i", ccol, TensorProto.INT64),
        vi("cidx", [WIDTH], TensorProto.INT64),
        vi("shl", FULL), vi("shu", FULL), vi("shlu", FULL),
    ]
    for k in range(4):
        value_info += [vi(f"p{k}a", FULL) if k else vi("p0a", FULL),
                       vi(f"P{k}", FULL), vi(f"ag{k}_0", sc), vi(f"ag{k}", sc),
                       vi(f"wb{k}", sc, B), vi(f"w{k}", sc), vi(f"sel{k}", FULL)]
    for a, b in pairs:
        value_info += [vi(f"d{a}{b}", FULL), vi(f"ad{a}{b}", FULL),
                       vi(f"s{a}{b}", sc), vi(f"eqb{a}{b}", sc, B),
                       vi(f"eq{a}{b}", sc)]
    value_info += [vi("sum01", FULL), vi("sum23", FULL)]

    inputs = [vi("input", FULL)]
    outputs = [vi("output", FULL)]
    graph = helper.make_graph(nodes, "odd_panel_aware", inputs, outputs,
                              initializer=init, value_info=value_info)
    return helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
        ir_version=IR_VERSION)


def solve_odd_panel_aware(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
