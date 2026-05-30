"""Solver: periodic in-painting.

The output is a fully periodic tiling and the input is that tiling with some
cells erased to background 0. Recover the tiling by detecting the period and
copying known cells across it (tasks 17, 61, 110, 305).

Per-example the period (Ph, Pw) varies, so it can't be baked into the graph.
Instead the fill is separable (rows then columns) and, for each axis, every
candidate period p is enumerated statically:

  * validity(p): shifting the colour field by p leaves no conflict on the
    overlap (a cell and its p-shifted partner are never two *different*
    non-background colours) while overlapping on at least one cell;
  * fill(p): propagate known colours along the axis in steps of p by repeated
    shift-and-Max (one-hot OR), which is exact because a valid period makes
    every residue class single-coloured.

Candidates are applied from the largest p down to the smallest with a
`Where(valid_p, fill_p, current)`, so the smallest valid period wins - the
true fundamental period.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from ..grids import CHANNELS, HEIGHT, WIDTH, all_examples

OPSET = 11
IR_VERSION = 8


# ----- reference NumPy implementation (also used for detection) -------------

def _shift_max(g: np.ndarray, p: int, n: int, axis: int) -> np.ndarray:
    out = g.copy().astype(float)
    for _ in range(n // p + 1):
        up = np.zeros_like(out)
        dn = np.zeros_like(out)
        if axis == 0:
            up[:-p] = out[p:]
            dn[p:] = out[:-p]
        else:
            up[:, :-p] = out[:, p:]
            dn[:, p:] = out[:, :-p]
        out = np.maximum(out, np.maximum(up, dn))
    return out


def _valid(g: np.ndarray, p: int, axis: int) -> bool:
    if axis == 0:
        a, b = g[:-p], g[p:]
    else:
        a, b = g[:, :-p], g[:, p:]
    both = (a != 0) & (b != 0)
    return bool(((a != b) & both).sum() == 0 and ((a == b) & both).sum() > 0)


def _fill_axis(g: np.ndarray, axis: int, n: int) -> np.ndarray:
    out = g.copy().astype(float)
    for p in range(n - 1, 0, -1):       # smallest valid p wins (applied last)
        if _valid(g, p, axis):
            out = _shift_max(g, p, n, axis)
    return out


def _periodic_fill(grid) -> list:
    g = np.array(grid)
    g = _fill_axis(g, 0, g.shape[0])
    g = _fill_axis(g, 1, g.shape[1])
    return g.astype(int).tolist()


def _detect(task: dict) -> bool:
    examples = list(all_examples(task))
    if not examples:
        return False
    saw_fill = False
    for ex in examples:
        inp, out = ex["input"], ex["output"]
        if len(inp) != len(out) or len(inp[0]) != len(out[0]):
            return False
        # non-background input cells must be preserved
        for r in range(len(inp)):
            for c in range(len(inp[0])):
                if inp[r][c] != 0 and inp[r][c] != out[r][c]:
                    return False
        if _periodic_fill(inp) != out:
            return False
        if any(inp[r][c] == 0 and out[r][c] != 0
               for r in range(len(inp)) for c in range(len(inp[0]))):
            saw_fill = True
    return saw_fill


# ----- ONNX graph -----------------------------------------------------------

def _build() -> onnx.ModelProto:
    def i64(name, data):
        return numpy_helper.from_array(np.array(data, dtype=np.int64), name)

    def f32(name, data):
        return numpy_helper.from_array(np.array(data, dtype=np.float32), name)

    init = [
        i64("c1", [1]), i64("c10", [CHANNELS]), i64("ax1", [1]), i64("st1", [1]),
        f32("zero", [0.0]), f32("one_f", [1.0]),
        i64("idx0", [0]),
    ]
    nodes = [
        helper.make_node("Slice", ["input", "c1", "c10", "ax1", "st1"],
                         ["cur0"]),                # colour channels [1,9,H,W]
    ]

    def emit_axis(cur: str, axis: int, n: int, prefix: str) -> str:
        """Emit period-enumeration fill along axis (2=rows, 3=cols)."""
        dim = HEIGHT if axis == 2 else WIDTH
        base = cur                       # fixed axis-entry field (matches numpy)
        for p in range(n - 1, 0, -1):
            tag = f"{prefix}{p}"
            # --- propagate known cells in steps of p by doubling the shift
            # distance (p, 2p, 4p, ...), so O(log) Max steps cover the axis.
            filled = base
            d = p
            s = 0
            while d < dim:
                # down-shift by d: pad d at the low side, slice [0:dim]
                # up-shift by d:   pad d at the high side, slice [d:dim+d]
                if axis == 2:
                    pad_dn = [0, 0, d, 0, 0, 0, 0, 0]
                    pad_up = [0, 0, 0, 0, 0, 0, d, 0]
                    sax = [2]
                else:
                    pad_dn = [0, 0, 0, d, 0, 0, 0, 0]
                    pad_up = [0, 0, 0, 0, 0, 0, 0, d]
                    sax = [3]
                pdn = f"pdn_{tag}_{s}"
                pup = f"pup_{tag}_{s}"
                init.append(i64(f"padv_dn_{tag}_{s}", pad_dn))
                init.append(i64(f"padv_up_{tag}_{s}", pad_up))
                init.append(i64(f"slo0_{tag}_{s}", [0]))
                init.append(i64(f"shi0_{tag}_{s}", [dim]))
                init.append(i64(f"slop_{tag}_{s}", [d]))
                init.append(i64(f"ship_{tag}_{s}", [dim + d]))
                init.append(i64(f"sax_{tag}_{s}", sax))
                nodes.append(helper.make_node(
                    "Pad", [filled, f"padv_dn_{tag}_{s}", "zero"], [pdn + "_p"]))
                nodes.append(helper.make_node(
                    "Slice", [pdn + "_p", f"slo0_{tag}_{s}", f"shi0_{tag}_{s}",
                              f"sax_{tag}_{s}", "st1"], [pdn]))
                nodes.append(helper.make_node(
                    "Pad", [filled, f"padv_up_{tag}_{s}", "zero"], [pup + "_p"]))
                nodes.append(helper.make_node(
                    "Slice", [pup + "_p", f"slop_{tag}_{s}", f"ship_{tag}_{s}",
                              f"sax_{tag}_{s}", "st1"], [pup]))
                m1 = f"m1_{tag}_{s}"
                nodes.append(helper.make_node("Max", [filled, pdn], [m1]))
                filled = f"fl_{tag}_{s}"
                nodes.append(helper.make_node("Max", [m1, pup], [filled]))
                d *= 2
                s += 1

            # --- validity(p): no conflicting overlap when base shifted by p
            sd = f"vd_{tag}"
            init.append(i64(f"vpad_{tag}", (
                [0, 0, p, 0, 0, 0, 0, 0] if axis == 2
                else [0, 0, 0, p, 0, 0, 0, 0])))
            init.append(i64(f"vslo_{tag}", [0]))
            init.append(i64(f"vshi_{tag}", [dim]))
            init.append(i64(f"vsax_{tag}", [axis]))
            nodes.append(helper.make_node(
                "Pad", [base, f"vpad_{tag}", "zero"], [sd + "_p"]))
            nodes.append(helper.make_node(
                "Slice", [sd + "_p", f"vslo_{tag}", f"vshi_{tag}",
                          f"vsax_{tag}", "st1"], [sd]))
            # overlap-and-same: dot product of one-hot over channel == same col
            same = f"same_{tag}"
            nodes.append(helper.make_node("Mul", [base, sd], [same + "_m"]))
            nodes.append(helper.make_node("ReduceSum", [same + "_m"], [same],
                                          axes=[1], keepdims=1))   # [1,1,H,W]
            # both non-bg: (sum_c base)*(sum_c sd)
            ca = f"ca_{tag}"
            cb = f"cb_{tag}"
            nodes.append(helper.make_node("ReduceSum", [base], [ca], axes=[1],
                                          keepdims=1))
            nodes.append(helper.make_node("ReduceSum", [sd], [cb], axes=[1],
                                          keepdims=1))
            both = f"both_{tag}"
            nodes.append(helper.make_node("Mul", [ca, cb], [both]))
            # conflict cell = both==1 and same==0 ; count conflicts
            confcell = f"conf_{tag}"
            nodes.append(helper.make_node("Sub", [both, same], [confcell]))
            confsum = f"confsum_{tag}"
            nodes.append(helper.make_node("ReduceSum", [confcell], [confsum]))
            matchsum = f"match_{tag}"
            nodes.append(helper.make_node("ReduceSum", [same], [matchsum]))
            # valid = (confsum == 0) and (matchsum > 0)
            noconf = f"noconf_{tag}"
            nodes.append(helper.make_node("LessOrEqual" if False else "Less",
                                          [confsum, "one_f"], [noconf]))  # <1 -> ==0
            hasmatch = f"hasm_{tag}"
            nodes.append(helper.make_node("Greater", [matchsum, "zero"],
                                          [hasmatch]))
            validb = f"valid_{tag}"
            nodes.append(helper.make_node("And", [noconf, hasmatch], [validb]))
            # broadcast valid scalar to [1,9,H,W] via Where
            newcur = f"cur_{prefix}{p}"
            nodes.append(helper.make_node("Where", [validb, filled, cur],
                                          [newcur]))
            cur = newcur
        return cur

    cur = emit_axis("cur0", 2, HEIGHT, "r")
    cur = emit_axis(cur, 3, WIDTH, "c")

    # in-grid mask from the ORIGINAL input: a cell is inside the real grid iff
    # its row and its column both contain some input content. Outside that box
    # the canvas stays all-zero so the scorer crops back to the true size.
    nodes += [
        helper.make_node("ReduceSum", ["input"], ["in_content"], axes=[1],
                         keepdims=1),                              # [1,1,H,W]
        helper.make_node("ReduceMax", ["in_content"], ["row_has"], axes=[3],
                         keepdims=1),                              # [1,1,H,1]
        helper.make_node("ReduceMax", ["in_content"], ["col_has"], axes=[2],
                         keepdims=1),                              # [1,1,1,W]
        helper.make_node("Mul", ["row_has", "col_has"], ["in_grid"]),  # [1,1,H,W]
        # shift-max propagation can push colours past the real grid edge; clip
        # the colour channels back to the in-grid box (broadcast over channels)
        helper.make_node("Mul", [cur, "in_grid"], ["cur_clip"]),
        # background = in-grid AND not a colour
        helper.make_node("ReduceSum", ["cur_clip"], ["csum"], axes=[1],
                         keepdims=1),
        helper.make_node("Sub", ["one_f", "csum"], ["not_color"]),
        helper.make_node("Mul", ["in_grid", "not_color"], ["bg"]),
        helper.make_node("Concat", ["bg", "cur_clip"], ["output"], axis=1),
    ]

    inputs = [helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, CHANNELS, HEIGHT, WIDTH])]
    outputs = [helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, CHANNELS, HEIGHT, WIDTH])]
    graph = helper.make_graph(nodes, "periodic_fill", inputs, outputs,
                              initializer=init)
    return helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
        ir_version=IR_VERSION)


def solve_periodic_fill(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
