"""Solver: count the bands of a lattice and emit a solid block (task 21).

The input is a rectangular grid criss-crossed by full monochromatic divider
lines.  D_r horizontal dividers split it into R = D_r + 1 row-bands and D_c
vertical dividers into C = D_c + 1 column-bands.  The output is a solid
R x C block painted in the grid's majority colour.

Everything is computed at runtime (no baked sizes):

  * a divider row is a row whose every real cell shares one colour, detected
    as `max_channel_count == content_width` (compared against the content mask
    so the 30x30 padding is ignored);
  * R = 1 + sum(divider rows), C = 1 + sum(divider cols);
  * the fill colour is the global per-channel arg-max;
  * the block is `(row_index < R) x (col_index < C)` painted with that colour,
    leaving the rest of the canvas empty so the scorer crops to R x C.
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
    H, W = g.shape
    Dr = sum(1 for r in range(H) if len(set(g[r].tolist())) == 1)
    Dc = sum(1 for c in range(W) if len(set(g[:, c].tolist())) == 1)
    R, C = Dr + 1, Dc + 1
    if R > HEIGHT or C > WIDTH:
        return None
    vals, cnts = np.unique(g, return_counts=True)
    M = int(vals[np.argmax(cnts)])
    return np.full((R, C), M, dtype=int)


def _detect(task: dict) -> bool:
    saw = False
    for ex in all_examples(task):
        i, o = ex["input"], ex["output"]
        if not i or not i[0]:
            continue
        if len(i) > HEIGHT or len(i[0]) > WIDTH:
            continue
        ref = _ref(np.array(i))
        if ref is None or not np.array_equal(ref, np.array(o)):
            return False
        saw = True
    return saw


def _build() -> onnx.ModelProto:
    F = TensorProto.FLOAT
    B = TensorProto.BOOL
    n = helper.make_node
    row_idx = np.arange(HEIGHT, dtype=np.float32).reshape(1, 1, HEIGHT, 1)
    col_idx = np.arange(WIDTH, dtype=np.float32).reshape(1, 1, 1, WIDTH)
    init = [
        numpy_helper.from_array(row_idx, "row_idx"),
        numpy_helper.from_array(col_idx, "col_idx"),
        numpy_helper.from_array(np.array(0.5, np.float32), "half"),
        numpy_helper.from_array(np.array(1.0, np.float32), "one"),
    ]
    rc = [1, 1, HEIGHT, 1]; cc = [1, 1, 1, WIDTH]
    ch_rc = [1, CHANNELS, HEIGHT, 1]; ch_cc = [1, CHANNELS, 1, WIDTH]
    chan = [1, CHANNELS, 1, 1]; sc = [1, 1, 1, 1]; s1 = [1, 1, HEIGHT, WIDTH]

    def cast(b, name):
        return n("Cast", [b], [name], to=F)

    nodes = [
        n("ReduceSum", ["input"], ["content"], axes=[1], keepdims=1),     # (1,1,H,W)
        # ---- divider rows ----
        n("ReduceSum", ["input"], ["cnt_row"], axes=[3], keepdims=1),     # (1,10,H,1)
        n("ReduceMax", ["cnt_row"], ["maxr"], axes=[1], keepdims=1),      # (1,1,H,1)
        n("ReduceSum", ["content"], ["cw"], axes=[3], keepdims=1),        # (1,1,H,1)
        n("Sub", ["cw", "maxr"], ["dr_diff"]),
        n("Less", ["dr_diff", "half"], ["dr_eq_b"]), cast("dr_eq_b", "dr_eq"),
        n("Greater", ["cw", "half"], ["rpos_b"]), cast("rpos_b", "rpos"),
        n("Mul", ["dr_eq", "rpos"], ["mono_row"]),                        # (1,1,H,1)
        n("ReduceSum", ["mono_row"], ["Dr"]),                             # scalar
        n("Add", ["Dr", "one"], ["R"]),
        # ---- divider cols ----
        n("ReduceSum", ["input"], ["cnt_col"], axes=[2], keepdims=1),     # (1,10,1,W)
        n("ReduceMax", ["cnt_col"], ["maxc"], axes=[1], keepdims=1),      # (1,1,1,W)
        n("ReduceSum", ["content"], ["ch_"], axes=[2], keepdims=1),       # (1,1,1,W)
        n("Sub", ["ch_", "maxc"], ["dc_diff"]),
        n("Less", ["dc_diff", "half"], ["dc_eq_b"]), cast("dc_eq_b", "dc_eq"),
        n("Greater", ["ch_", "half"], ["cpos_b"]), cast("cpos_b", "cpos"),
        n("Mul", ["dc_eq", "cpos"], ["mono_col"]),                        # (1,1,1,W)
        n("ReduceSum", ["mono_col"], ["Dc"]),
        n("Add", ["Dc", "one"], ["C"]),
        # ---- fill colour = global per-channel arg-max ----
        n("ReduceSum", ["input"], ["tot"], axes=[2, 3], keepdims=1),      # (1,10,1,1)
        n("ReduceMax", ["tot"], ["maxtot"], axes=[1], keepdims=1),        # (1,1,1,1)
        n("Sub", ["maxtot", "half"], ["maxtot_m"]),
        n("Greater", ["tot", "maxtot_m"], ["eM_b"]), cast("eM_b", "eM"),  # (1,10,1,1)
        # ---- block ----
        n("Less", ["row_idx", "R"], ["rowsel_b"]), cast("rowsel_b", "rowsel"),   # (1,1,H,1)
        n("Less", ["col_idx", "C"], ["colsel_b"]), cast("colsel_b", "colsel"),   # (1,1,1,W)
        n("Mul", ["rowsel", "colsel"], ["block"]),                        # (1,1,H,W)
        n("Mul", ["eM", "block"], ["output"]),                            # (1,10,H,W)
    ]

    def vif(name, shp):
        return helper.make_tensor_value_info(name, F, shp)

    def vib(name, shp):
        return helper.make_tensor_value_info(name, B, shp)

    vi = [
        vif("content", s1),
        vif("cnt_row", ch_rc), vif("maxr", rc), vif("cw", rc), vif("dr_diff", rc),
        vib("dr_eq_b", rc), vif("dr_eq", rc), vib("rpos_b", rc), vif("rpos", rc),
        vif("mono_row", rc), vif("Dr", sc), vif("R", sc),
        vif("cnt_col", ch_cc), vif("maxc", cc), vif("ch_", cc), vif("dc_diff", cc),
        vib("dc_eq_b", cc), vif("dc_eq", cc), vib("cpos_b", cc), vif("cpos", cc),
        vif("mono_col", cc), vif("Dc", sc), vif("C", sc),
        vif("tot", chan), vif("maxtot", sc), vif("maxtot_m", sc),
        vib("eM_b", chan), vif("eM", chan),
        vib("rowsel_b", rc), vif("rowsel", rc), vib("colsel_b", cc), vif("colsel", cc),
        vif("block", s1),
    ]
    graph = helper.make_graph(nodes, "lattice_count",
                              [vif("input", FULL)], [vif("output", FULL)],
                              initializer=init, value_info=vi)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_lattice_count(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
