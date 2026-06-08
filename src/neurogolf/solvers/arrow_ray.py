"""Solver: a triangle/arrow shoots a ray from its apex (task 51).

A solid triangle of one colour points along a cardinal direction; a single
marker cell of a second colour sits at the centre of its base.  A ray of the
marker's colour is emitted from the apex onward, filling background cells to the
grid edge::

    . 2 .            . 2 . . . . .
    2 2 .      ->     2 2 . . . . .
    1 2 2            1 2 2 1 1 1 1   (ray to the right edge)
    2 2 .            2 2 . . . . .
    . 2 .            . 2 . . . . .

Build: a colour histogram (``ReduceSum`` over space) gives, per cell, the count
of its own colour; the marker is the unique non-background cell with count 1.
Its position and one-hot colour, plus the shape centroid, fix the pointing
direction; the half-line in that direction is painted the marker's colour over
background cells only.
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
    vals, cnts = np.unique(g, return_counts=True)
    bg = vals[cnts.argmax()]
    nb = sorted([(int(v), int(c)) for v, c in zip(vals, cnts) if v != bg], key=lambda vc: vc[1])
    if len(nb) != 2 or nb[0][1] != 1:
        return None
    mcol, scol = nb[0][0], nb[1][0]
    ys, xs = np.where(g == mcol)
    if len(ys) != 1:
        return None
    mr, mc = int(ys[0]), int(xs[0])
    sy, sx = np.where(g == scol)
    dr = sy.mean() - mr; dc = sx.mean() - mc
    out = g.copy()
    if abs(dr) >= abs(dc):
        step = 1 if dr > 0 else -1
        r = mr + step
        while 0 <= r < H:
            if g[r, mc] == bg:
                out[r, mc] = mcol
            r += step
    else:
        step = 1 if dc > 0 else -1
        c = mc + step
        while 0 <= c < W:
            if g[mr, c] == bg:
                out[mr, c] = mcol
            c += step
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
    F = TensorProto.FLOAT
    n = helper.make_node

    row_idx = np.arange(HEIGHT, dtype=np.float32).reshape(1, 1, HEIGHT, 1)
    col_idx = np.arange(WIDTH, dtype=np.float32).reshape(1, 1, 1, WIDTH)
    init = [
        numpy_helper.from_array(row_idx, "row_idx"),
        numpy_helper.from_array(col_idx, "col_idx"),
        numpy_helper.from_array(np.array(0.5, np.float32), "half"),
        numpy_helper.from_array(np.array(1.5, np.float32), "onehalf"),
        numpy_helper.from_array(np.array(1.0, np.float32), "one"),
        numpy_helper.from_array(np.array(0.0, np.float32), "zero"),
        numpy_helper.from_array(np.array([0], np.int64), "c0s"),
        numpy_helper.from_array(np.array([1], np.int64), "c1e"),
        numpy_helper.from_array(np.array([1], np.int64), "ax1"),
    ]
    nodes = [
        n("ReduceSum", ["input"], ["content"], axes=[1], keepdims=1),
        n("Slice", ["input", "c0s", "c1e", "ax1"], ["bg"]),
        # per-cell count of own colour via histogram
        n("ReduceSum", ["input"], ["hist"], axes=[2, 3], keepdims=1),   # (1,C,1,1)
        n("Mul", ["input", "hist"], ["ih"]),
        n("ReduceSum", ["ih"], ["cellcount"], axes=[1], keepdims=1),    # (1,1,H,W)
        n("Greater", ["cellcount", "half"], ["mk_gt_b"]), n("Cast", ["mk_gt_b"], ["mk_gt"], to=F),
        n("Less", ["cellcount", "onehalf"], ["mk_lt_b"]), n("Cast", ["mk_lt_b"], ["mk_lt"], to=F),
        n("Mul", ["mk_gt", "mk_lt"], ["marker"]),                       # (1,1,H,W)
        # marker position + one-hot colour
        n("Mul", ["marker", "row_idx"], ["mkr"]), n("ReduceSum", ["mkr"], ["mr"], keepdims=0),
        n("Mul", ["marker", "col_idx"], ["mkc"]), n("ReduceSum", ["mkc"], ["mc"], keepdims=0),
        n("Mul", ["input", "marker"], ["imk"]),
        n("ReduceSum", ["imk"], ["mcolor"], axes=[2, 3], keepdims=1),   # (1,C,1,1)
        # shape mask + centroid
        n("Sub", ["content", "bg"], ["nonbg"]),
        n("Sub", ["nonbg", "marker"], ["shape"]),
        n("ReduceSum", ["shape"], ["scnt"], keepdims=0),
        n("Mul", ["shape", "row_idx"], ["shr"]), n("ReduceSum", ["shr"], ["ssr"], keepdims=0),
        n("Div", ["ssr", "scnt"], ["scr"]),
        n("Mul", ["shape", "col_idx"], ["shc"]), n("ReduceSum", ["shc"], ["ssc"], keepdims=0),
        n("Div", ["ssc", "scnt"], ["scc"]),
        # direction
        n("Sub", ["scr", "mr"], ["dr"]), n("Abs", ["dr"], ["adr"]),
        n("Sub", ["scc", "mc"], ["dc"]), n("Abs", ["dc"], ["adc"]),
        n("Less", ["adr", "adc"], ["hor_b"]), n("Cast", ["hor_b"], ["ishor"], to=F),
        n("Sub", ["one", "ishor"], ["isver"]),
        n("Greater", ["dr", "zero"], ["drp_b"]), n("Cast", ["drp_b"], ["drp"], to=F),
        n("Less", ["dr", "zero"], ["drn_b"]), n("Cast", ["drn_b"], ["drn"], to=F),
        n("Greater", ["dc", "zero"], ["dcp_b"]), n("Cast", ["dcp_b"], ["dcp"], to=F),
        n("Less", ["dc", "zero"], ["dcn_b"]), n("Cast", ["dcn_b"], ["dcn"], to=F),
        n("Mul", ["isver", "drn"], ["is_up"]), n("Mul", ["isver", "drp"], ["is_down"]),
        n("Mul", ["ishor", "dcp"], ["is_right"]), n("Mul", ["ishor", "dcn"], ["is_left"]),
        # half-line masks
        n("Sub", ["row_idx", "mr"], ["rr"]), n("Abs", ["rr"], ["arr"]),
        n("Less", ["arr", "half"], ["onr_b"]), n("Cast", ["onr_b"], ["onrow"], to=F),
        n("Sub", ["col_idx", "mc"], ["cc"]), n("Abs", ["cc"], ["acc"]),
        n("Less", ["acc", "half"], ["onc_b"]), n("Cast", ["onc_b"], ["oncol"], to=F),
        n("Greater", ["col_idx", "mc"], ["gtc_b"]), n("Cast", ["gtc_b"], ["gtc"], to=F),
        n("Less", ["col_idx", "mc"], ["ltc_b"]), n("Cast", ["ltc_b"], ["ltc"], to=F),
        n("Greater", ["row_idx", "mr"], ["gtr_b"]), n("Cast", ["gtr_b"], ["gtr"], to=F),
        n("Less", ["row_idx", "mr"], ["ltr_b"]), n("Cast", ["ltr_b"], ["ltr"], to=F),
        n("Mul", ["onrow", "gtc"], ["right_half"]),
        n("Mul", ["onrow", "ltc"], ["left_half"]),
        n("Mul", ["oncol", "ltr"], ["up_half"]),
        n("Mul", ["oncol", "gtr"], ["down_half"]),
        n("Mul", ["is_right", "right_half"], ["sr"]),
        n("Mul", ["is_left", "left_half"], ["sl"]),
        n("Mul", ["is_up", "up_half"], ["su"]),
        n("Mul", ["is_down", "down_half"], ["sd"]),
        n("Add", ["sr", "sl"], ["sa"]), n("Add", ["su", "sd"], ["sb"]),
        n("Add", ["sa", "sb"], ["sel"]),
        n("Mul", ["sel", "bg"], ["ray"]),                              # background only
        n("Sub", ["one", "ray"], ["inv"]), n("Mul", ["input", "inv"], ["kept"]),
        n("Mul", ["mcolor", "ray"], ["addc"]), n("Add", ["kept", "addc"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "arrow_ray",
                              [helper.make_tensor_value_info("input", F, FULL)],
                              [helper.make_tensor_value_info("output", F, FULL)],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_arrow_ray(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
