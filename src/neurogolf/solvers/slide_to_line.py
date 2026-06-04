"""Solver: markers slide to their matching-colour line (task 25).

The grid contains a few full lines (each a complete row or column of one
colour) plus scattered single markers. Every marker slides, along the axis
perpendicular to the line of its own colour, until it sits directly adjacent to
that line (its other coordinate preserved). Markers whose colour has no line are
removed; the lines stay. Each example uses one orientation (all rows or all
columns).

The whole transform vectorises over the 9 colour channels: a row that is fully
one colour is found with `ReduceSum == W`; `CumSum` splits each colour's cells
into "above the line" / "below the line"; a per-column `ReduceMax` projects them,
and an outer product with the line shifted by ±1 drops them adjacent. The
vertical case reuses the same block on the transposed grid; an edge flag selects
the orientation.
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
NC = CHANNELS - 1  # 9 colour channels


# ---------- reference ----------

def _lines(g):
    H, W = g.shape
    res = {}
    for r in range(H):
        s = set(g[r].tolist())
        if len(s) == 1 and 0 not in s:
            res[int(g[r, 0])] = ('H', r)
    for c in range(W):
        s = set(g[:, c].tolist())
        if len(s) == 1 and 0 not in s:
            res[int(g[0, c])] = ('V', c)
    return res


def _ref(g: np.ndarray) -> Optional[np.ndarray]:
    H, W = g.shape
    ln = _lines(g)
    if not ln:
        return None
    orient = set(o for o, _ in ln.values())
    if len(orient) != 1:
        return None
    out = np.zeros_like(g)
    for col, (o, idx) in ln.items():
        if o == 'H':
            out[idx, :W] = col
        else:
            out[:H, idx] = col
    for r in range(H):
        for c in range(W):
            v = int(g[r, c])
            if v == 0 or v not in ln:
                continue
            o, idx = ln[v]
            if o == 'H':
                if r == idx:
                    continue
                out[idx - 1 if r < idx else idx + 1, c] = v
            else:
                if c == idx:
                    continue
                out[r, idx - 1 if c < idx else idx + 1] = v
    return out


def _detect(task: dict) -> bool:
    saw = False
    for ex in all_examples(task):
        i, o = ex["input"], ex["output"]
        if not i or not i[0] or len(i) > HEIGHT or len(i[0]) > WIDTH:
            if i and i[0] and (len(i) > HEIGHT or len(i[0]) > WIDTH):
                continue
            return False
        ref = _ref(np.array(i))
        if ref is None or not np.array_equal(ref, np.array(o)):
            return False
        saw = True
    return saw


# ---------- graph ----------

def _i64(name, v): return numpy_helper.from_array(np.array(v, np.int64), name)
def _f32(name, v): return numpy_helper.from_array(np.array(v, np.float32), name)


def _emit_hlogic(in_name: str, sfx: str, init: list, vi: list) -> tuple:
    """Emit horizontal-line logic on `in_name`; return (out19, line_exists)."""
    F = TensorProto.FLOAT
    n_ = helper.make_node
    g4 = FULL
    c9 = [1, NC, HEIGHT, WIDTH]
    rc = [1, NC, HEIGHT, 1]
    cc = [1, NC, 1, WIDTH]
    nodes = [
        # grid width W from content
        n_("ReduceSum", [in_name], [f"pres_{sfx}"], axes=[1], keepdims=1),
        n_("ReduceMax", [f"pres_{sfx}"], [f"colp_{sfx}"], axes=[2], keepdims=1),
        n_("Mul", [f"colp_{sfx}", "col_ar1"], [f"cpos_{sfx}"]),
        n_("ReduceMax", [f"cpos_{sfx}"], [f"cmax_{sfx}"], axes=[3], keepdims=1),
        n_("Add", [f"cmax_{sfx}", "one"], [f"W_{sfx}"]),
        n_("Sub", [f"W_{sfx}", "half"], [f"Wm_{sfx}"]),
        # channels 1..9
        n_("Slice", [in_name, "ch1_s", "ch1_e", "ax4"], [f"ch19_{sfx}"]),
        n_("ReduceSum", [f"ch19_{sfx}"], [f"rowsum_{sfx}"], axes=[3], keepdims=1),
        n_("Greater", [f"rowsum_{sfx}", f"Wm_{sfx}"], [f"lineb_{sfx}"]),
        n_("Cast", [f"lineb_{sfx}"], [f"line_{sfx}"], to=F),
        # above / below regions via cumulative sums along rows
        n_("CumSum", [f"line_{sfx}", "ax2"], [f"revcum_{sfx}"], reverse=1),
        n_("CumSum", [f"line_{sfx}", "ax2"], [f"fwdcum_{sfx}"], reverse=0),
        n_("Sub", [f"revcum_{sfx}", f"line_{sfx}"], [f"above_{sfx}"]),
        n_("Sub", [f"fwdcum_{sfx}", f"line_{sfx}"], [f"below_{sfx}"]),
        n_("Mul", [f"ch19_{sfx}", f"above_{sfx}"], [f"amk_{sfx}"]),
        n_("Mul", [f"ch19_{sfx}", f"below_{sfx}"], [f"bmk_{sfx}"]),
        n_("ReduceMax", [f"amk_{sfx}"], [f"aproj_{sfx}"], axes=[2], keepdims=1),
        n_("ReduceMax", [f"bmk_{sfx}"], [f"bproj_{sfx}"], axes=[2], keepdims=1),
        # shift line up / down by 1 to get the adjacent rows
        n_("Pad", [f"line_{sfx}", "pad_b1"], [f"lpb_{sfx}"], mode="constant"),
        n_("Slice", [f"lpb_{sfx}", "s1", "s31", "ax2"], [f"adja_{sfx}"]),
        n_("Pad", [f"line_{sfx}", "pad_t1"], [f"lpt_{sfx}"], mode="constant"),
        n_("Slice", [f"lpt_{sfx}", "s0", "s30", "ax2"], [f"adjb_{sfx}"]),
        n_("Mul", [f"aproj_{sfx}", f"adja_{sfx}"], [f"rabove_{sfx}"]),
        n_("Mul", [f"bproj_{sfx}", f"adjb_{sfx}"], [f"rbelow_{sfx}"]),
        # full lines, clipped to the grid width
        n_("Less", ["col_ar1", f"W_{sfx}"], [f"cmaskb_{sfx}"]),
        n_("Cast", [f"cmaskb_{sfx}"], [f"cmask_{sfx}"], to=F),
        n_("Mul", [f"line_{sfx}", f"cmask_{sfx}"], [f"lfull_{sfx}"]),
        n_("Add", [f"lfull_{sfx}", f"rabove_{sfx}"], [f"o1_{sfx}"]),
        n_("Add", [f"o1_{sfx}", f"rbelow_{sfx}"], [f"o2_{sfx}"]),
        n_("Clip", [f"o2_{sfx}", "zero", "one"], [f"out19_{sfx}"]),
        n_("ReduceMax", [f"line_{sfx}"], [f"lex_{sfx}"], axes=[0, 1, 2, 3], keepdims=1),
    ]
    B = TensorProto.BOOL
    vi += [
        helper.make_tensor_value_info(f"pres_{sfx}", F, [1, 1, HEIGHT, WIDTH]),
        helper.make_tensor_value_info(f"colp_{sfx}", F, [1, 1, 1, WIDTH]),
        helper.make_tensor_value_info(f"cpos_{sfx}", F, [1, 1, 1, WIDTH]),
        helper.make_tensor_value_info(f"cmax_{sfx}", F, [1, 1, 1, 1]),
        helper.make_tensor_value_info(f"W_{sfx}", F, [1, 1, 1, 1]),
        helper.make_tensor_value_info(f"Wm_{sfx}", F, [1, 1, 1, 1]),
        helper.make_tensor_value_info(f"ch19_{sfx}", F, c9),
        helper.make_tensor_value_info(f"rowsum_{sfx}", F, rc),
        helper.make_tensor_value_info(f"lineb_{sfx}", B, rc),
        helper.make_tensor_value_info(f"line_{sfx}", F, rc),
        helper.make_tensor_value_info(f"revcum_{sfx}", F, rc),
        helper.make_tensor_value_info(f"fwdcum_{sfx}", F, rc),
        helper.make_tensor_value_info(f"above_{sfx}", F, rc),
        helper.make_tensor_value_info(f"below_{sfx}", F, rc),
        helper.make_tensor_value_info(f"amk_{sfx}", F, c9),
        helper.make_tensor_value_info(f"bmk_{sfx}", F, c9),
        helper.make_tensor_value_info(f"aproj_{sfx}", F, cc),
        helper.make_tensor_value_info(f"bproj_{sfx}", F, cc),
        helper.make_tensor_value_info(f"lpb_{sfx}", F, [1, NC, HEIGHT + 1, 1]),
        helper.make_tensor_value_info(f"adja_{sfx}", F, rc),
        helper.make_tensor_value_info(f"lpt_{sfx}", F, [1, NC, HEIGHT + 1, 1]),
        helper.make_tensor_value_info(f"adjb_{sfx}", F, rc),
        helper.make_tensor_value_info(f"rabove_{sfx}", F, c9),
        helper.make_tensor_value_info(f"rbelow_{sfx}", F, c9),
        helper.make_tensor_value_info(f"cmaskb_{sfx}", B, [1, 1, 1, WIDTH]),
        helper.make_tensor_value_info(f"cmask_{sfx}", F, [1, 1, 1, WIDTH]),
        helper.make_tensor_value_info(f"lfull_{sfx}", F, c9),
        helper.make_tensor_value_info(f"o1_{sfx}", F, c9),
        helper.make_tensor_value_info(f"o2_{sfx}", F, c9),
        helper.make_tensor_value_info(f"out19_{sfx}", F, c9),
        helper.make_tensor_value_info(f"lex_{sfx}", F, [1, 1, 1, 1]),
    ]
    return f"out19_{sfx}", f"lex_{sfx}", nodes


def _build() -> onnx.ModelProto:
    F = TensorProto.FLOAT
    n_ = helper.make_node
    col_ar1 = np.arange(WIDTH, dtype=np.float32).reshape(1, 1, 1, WIDTH)
    init = [
        _f32("col_ar1", col_ar1), _f32("one", 1.0), _f32("half", 0.5),
        _f32("zero", 0.0),
        _i64("ch1_s", [0, 1, 0, 0]), _i64("ch1_e", [1, CHANNELS, HEIGHT, WIDTH]),
        _i64("ax4", [0, 1, 2, 3]), _i64("ax2", [2]),
        _i64("pad_b1", [0, 0, 0, 0, 0, 0, 1, 0]),
        _i64("pad_t1", [0, 0, 1, 0, 0, 0, 0, 0]),
        _i64("s1", [1]), _i64("s31", [HEIGHT + 1]), _i64("s0", [0]), _i64("s30", [HEIGHT]),
    ]
    vi: list = []
    h_out, h_lex, hn = _emit_hlogic("input", "h", init, vi)
    nodes = list(hn)
    nodes.append(n_("Transpose", ["input"], ["tin"], perm=[0, 1, 3, 2]))
    v_out, v_lex, vn = _emit_hlogic("tin", "v", init, vi)
    nodes += vn
    nodes.append(n_("Transpose", [v_out], ["vout19"], perm=[0, 1, 3, 2]))

    # orientation select by which axis has a line
    nodes += [
        n_("Mul", [h_out, h_lex], ["hsel"]),
        n_("Sub", ["one", h_lex], ["nh"]),
        n_("Mul", ["vout19", "nh"], ["vsel"]),
        n_("Add", ["hsel", "vsel"], ["out19"]),
        # background channel 0 within the grid extent
        n_("ReduceSum", ["input"], ["pres0"], axes=[1], keepdims=1),
        n_("Greater", ["pres0", "zero"], ["extb"]),
        n_("Cast", ["extb"], ["extent"], to=F),
        n_("ReduceSum", ["out19"], ["colsum"], axes=[1], keepdims=1),
        n_("Greater", ["colsum", "zero"], ["hascolb"]),
        n_("Cast", ["hascolb"], ["hascol"], to=F),
        n_("Sub", ["one", "hascol"], ["nocol"]),
        n_("Mul", ["extent", "nocol"], ["bg"]),
        n_("Concat", ["bg", "out19"], ["output"], axis=1),
    ]
    B = TensorProto.BOOL
    c9 = [1, NC, HEIGHT, WIDTH]
    s1 = [1, 1, HEIGHT, WIDTH]
    vi += [
        helper.make_tensor_value_info("tin", F, FULL),
        helper.make_tensor_value_info("vout19", F, c9),
        helper.make_tensor_value_info("hsel", F, c9),
        helper.make_tensor_value_info("nh", F, [1, 1, 1, 1]),
        helper.make_tensor_value_info("vsel", F, c9),
        helper.make_tensor_value_info("out19", F, c9),
        helper.make_tensor_value_info("pres0", F, s1),
        helper.make_tensor_value_info("extb", B, s1),
        helper.make_tensor_value_info("extent", F, s1),
        helper.make_tensor_value_info("colsum", F, s1),
        helper.make_tensor_value_info("hascolb", B, s1),
        helper.make_tensor_value_info("hascol", F, s1),
        helper.make_tensor_value_info("nocol", F, s1),
        helper.make_tensor_value_info("bg", F, s1),
    ]
    graph = helper.make_graph(nodes, "slide_to_line",
                              [helper.make_tensor_value_info("input", F, FULL)],
                              [helper.make_tensor_value_info("output", F, FULL)],
                              initializer=init, value_info=vi)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_slide_to_line(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
