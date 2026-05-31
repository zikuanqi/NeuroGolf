"""Solver: recolour each marker to the colour of the nearer of two facing walls.

Two solid full-length walls of distinct non-background colours sit on opposite
edges of the grid - either the left and right columns, or the top and bottom
rows. Every other non-background cell ("marker") is repainted with the colour
of whichever wall is closer to it; the walls and background are untouched
(task 40).

The same network must cope with both orientations, so the graph computes the
vertical-wall result and the horizontal-wall result independently and blends
them with a data-derived orientation flag (exactly one wall pair is solid).

One-hot channel arithmetic on the (1, 10, 30, 30) tensor:
  content  = sum over channels                (1 on real cells, 0 on padding)
  nonbg    = content - channel-0              (1 on coloured cells)
  W, H     = real width / height              (sum of any-content col / row)
  A_chan   = colour of the near wall (col 0 / row 0)   one-hot over channels
  B_chan   = colour of the far  wall (col W-1 / row H-1, gathered by index)
  leftmask = columns whose index is left of centre  (2*c < W-1)
  vert     = output where nonbg cells take A on the left half, B on the right
  horiz    = the row-wise analogue
  output   = is_vert * vert + is_horiz * horiz  +  background passthrough
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from ..grids import CHANNELS, HEIGHT, WIDTH, all_examples

OPSET = 11
IR_VERSION = 8


def _expected(grid):
    g = np.array(grid)
    H, W = g.shape
    left = g[:, 0]
    right = g[:, W - 1]
    top = g[0, :]
    bot = g[H - 1, :]
    vert = (len(set(left.tolist())) == 1 and len(set(right.tolist())) == 1
            and left[0] != 0 and right[0] != 0 and left[0] != right[0])
    horiz = (len(set(top.tolist())) == 1 and len(set(bot.tolist())) == 1
             and top[0] != 0 and bot[0] != 0 and top[0] != bot[0])
    if vert == horiz:                 # need exactly one orientation
        return None
    out = g.copy()
    if vert:
        A, B = int(left[0]), int(right[0])
        for r in range(H):
            for c in range(1, W - 1):
                if g[r, c] != 0:
                    dA, dB = c, (W - 1) - c
                    out[r, c] = A if dA < dB else (B if dB < dA else g[r, c])
    else:
        A, B = int(top[0]), int(bot[0])
        for r in range(1, H - 1):
            for c in range(W):
                if g[r, c] != 0:
                    dA, dB = r, (H - 1) - r
                    out[r, c] = A if dA < dB else (B if dB < dA else g[r, c])
    return out.tolist()


def _detect(task: dict) -> bool:
    examples = list(all_examples(task))
    if not examples:
        return False
    changed = False
    for ex in examples:
        inp, out = ex["input"], ex["output"]
        if len(inp) > 30 or len(inp[0]) > 30:
            continue                  # scorer skips oversized grids
        if len(inp) != len(out) or len(inp[0]) != len(out[0]):
            return False
        res = _expected(inp)
        if res is None or res != out:
            return False
        if inp != out:
            changed = True
    return changed


def _build() -> onnx.ModelProto:
    F = TensorProto.FLOAT
    B = TensorProto.BOOL
    I = TensorProto.INT64

    def f32(name, arr):
        return numpy_helper.from_array(arr.astype(np.float32), name)

    def i64(name, arr):
        return numpy_helper.from_array(np.array(arr, dtype=np.int64), name)

    idx_c = np.arange(WIDTH, dtype=np.float32).reshape(1, 1, 1, WIDTH)
    idx_r = np.arange(HEIGHT, dtype=np.float32).reshape(1, 1, HEIGHT, 1)
    e0 = np.zeros((1, CHANNELS, 1, 1), dtype=np.float32)
    e0[0, 0, 0, 0] = 1.0

    init = [
        f32("idx_c", idx_c), f32("idx_r", idx_r), f32("e0", e0),
        f32("one_f", np.array([[[[1.0]]]])),
        f32("half", np.array([[[[0.5]]]])),
        f32("two_f", np.array([[[[2.0]]]])),
        i64("idx0", [0]), i64("neg1", [-1]),
    ]

    n = helper.make_node
    nodes = [
        n("ReduceSum", ["input"], ["content"], axes=[1], keepdims=1),
        n("Gather", ["input", "idx0"], ["ch0"], axis=1),
        n("Sub", ["content", "ch0"], ["nonbg"]),
        # real width / height
        n("ReduceMax", ["content"], ["col_real"], axes=[2], keepdims=1),
        n("ReduceSum", ["col_real"], ["Wf"], axes=[3], keepdims=1),
        n("ReduceMax", ["content"], ["row_real"], axes=[3], keepdims=1),
        n("ReduceSum", ["row_real"], ["Hf"], axes=[2], keepdims=1),
        n("Sub", ["Wf", "one_f"], ["Wm1f"]),
        n("Sub", ["Hf", "one_f"], ["Hm1f"]),
        # far-wall indices (W-1, H-1) as 1-D gather indices
        n("Cast", ["Wm1f"], ["Wm1_i"], to=I),
        n("Reshape", ["Wm1_i", "neg1"], ["Wm1_idx"]),
        n("Cast", ["Hm1f"], ["Hm1_i"], to=I),
        n("Reshape", ["Hm1_i", "neg1"], ["Hm1_idx"]),
        # wall columns / rows and their colours
        n("Gather", ["input", "idx0"], ["left_col"], axis=3),
        n("Gather", ["input", "Wm1_idx"], ["right_col"], axis=3),
        n("Gather", ["input", "idx0"], ["top_row"], axis=2),
        n("Gather", ["input", "Hm1_idx"], ["bot_row"], axis=2),
        n("ReduceMax", ["left_col"], ["A_chan"], axes=[2], keepdims=1),
        n("ReduceMax", ["right_col"], ["B_chan"], axes=[2], keepdims=1),
        n("ReduceMax", ["top_row"], ["A2_chan"], axes=[3], keepdims=1),
        n("ReduceMax", ["bot_row"], ["B2_chan"], axes=[3], keepdims=1),
        # orientation flags: a single channel fills the whole near wall
        n("ReduceSum", ["left_col"], ["leftcol_cnt"], axes=[2], keepdims=1),
        n("ReduceMax", ["leftcol_cnt"], ["left_maxcnt"], axes=[1], keepdims=1),
        n("Sub", ["Hf", "half"], ["Hm_half"]),
        n("Greater", ["left_maxcnt", "Hm_half"], ["is_vert_b"]),
        n("Cast", ["is_vert_b"], ["is_vert"], to=F),
        n("ReduceSum", ["top_row"], ["toprow_cnt"], axes=[3], keepdims=1),
        n("ReduceMax", ["toprow_cnt"], ["top_maxcnt"], axes=[1], keepdims=1),
        n("Sub", ["Wf", "half"], ["Wm_half"]),
        n("Greater", ["top_maxcnt", "Wm_half"], ["is_horiz_b"]),
        n("Cast", ["is_horiz_b"], ["is_horiz"], to=F),
        # column / row half masks
        n("Mul", ["idx_c", "two_f"], ["twoC"]),
        n("Less", ["twoC", "Wm1f"], ["leftmask_b"]),
        n("Cast", ["leftmask_b"], ["leftmask"], to=F),
        n("Greater", ["twoC", "Wm1f"], ["rightmask_b"]),
        n("Cast", ["rightmask_b"], ["rightmask"], to=F),
        n("Mul", ["idx_r", "two_f"], ["twoR"]),
        n("Less", ["twoR", "Hm1f"], ["topmask_b"]),
        n("Cast", ["topmask_b"], ["topmask"], to=F),
        n("Greater", ["twoR", "Hm1f"], ["botmask_b"]),
        n("Cast", ["botmask_b"], ["botmask"], to=F),
        # marker selectors per region
        n("Mul", ["nonbg", "leftmask"], ["leftsel"]),
        n("Mul", ["nonbg", "rightmask"], ["rightsel"]),
        n("Mul", ["nonbg", "topmask"], ["topsel"]),
        n("Mul", ["nonbg", "botmask"], ["botsel"]),
        # paint each region with its wall colour
        n("Mul", ["A_chan", "leftsel"], ["A_contrib"]),
        n("Mul", ["B_chan", "rightsel"], ["B_contrib"]),
        n("Mul", ["A2_chan", "topsel"], ["A2_contrib"]),
        n("Mul", ["B2_chan", "botsel"], ["B2_contrib"]),
        n("Mul", ["e0", "ch0"], ["bg_contrib"]),
        n("Add", ["A_contrib", "B_contrib"], ["V_ab"]),
        n("Add", ["V_ab", "bg_contrib"], ["V"]),
        n("Add", ["A2_contrib", "B2_contrib"], ["H_ab"]),
        n("Add", ["H_ab", "bg_contrib"], ["Hh"]),
        # blend the orientation that actually holds
        n("Mul", ["V", "is_vert"], ["Vsel"]),
        n("Mul", ["Hh", "is_horiz"], ["Hsel"]),
        n("Add", ["Vsel", "Hsel"], ["output"]),
    ]

    def vi(name, shape, dt=F):
        return helper.make_tensor_value_info(name, dt, shape)

    g4 = [1, CHANNELS, HEIGHT, WIDTH]
    s1 = [1, 1, HEIGHT, WIDTH]
    value_info = [
        vi("content", s1), vi("ch0", s1), vi("nonbg", s1),
        vi("col_real", [1, 1, 1, WIDTH]), vi("Wf", [1, 1, 1, 1]),
        vi("row_real", [1, 1, HEIGHT, 1]), vi("Hf", [1, 1, 1, 1]),
        vi("Wm1f", [1, 1, 1, 1]), vi("Hm1f", [1, 1, 1, 1]),
        vi("Wm1_i", [1, 1, 1, 1], I), vi("Wm1_idx", [1], I),
        vi("Hm1_i", [1, 1, 1, 1], I), vi("Hm1_idx", [1], I),
        vi("left_col", [1, CHANNELS, HEIGHT, 1]),
        vi("right_col", [1, CHANNELS, HEIGHT, 1]),
        vi("top_row", [1, CHANNELS, 1, WIDTH]),
        vi("bot_row", [1, CHANNELS, 1, WIDTH]),
        vi("A_chan", [1, CHANNELS, 1, 1]), vi("B_chan", [1, CHANNELS, 1, 1]),
        vi("A2_chan", [1, CHANNELS, 1, 1]), vi("B2_chan", [1, CHANNELS, 1, 1]),
        vi("leftcol_cnt", [1, CHANNELS, 1, 1]), vi("left_maxcnt", [1, 1, 1, 1]),
        vi("Hm_half", [1, 1, 1, 1]), vi("is_vert_b", [1, 1, 1, 1], B),
        vi("is_vert", [1, 1, 1, 1]),
        vi("toprow_cnt", [1, CHANNELS, 1, 1]), vi("top_maxcnt", [1, 1, 1, 1]),
        vi("Wm_half", [1, 1, 1, 1]), vi("is_horiz_b", [1, 1, 1, 1], B),
        vi("is_horiz", [1, 1, 1, 1]),
        vi("twoC", [1, 1, 1, WIDTH]), vi("leftmask_b", [1, 1, 1, WIDTH], B),
        vi("leftmask", [1, 1, 1, WIDTH]), vi("rightmask_b", [1, 1, 1, WIDTH], B),
        vi("rightmask", [1, 1, 1, WIDTH]),
        vi("twoR", [1, 1, HEIGHT, 1]), vi("topmask_b", [1, 1, HEIGHT, 1], B),
        vi("topmask", [1, 1, HEIGHT, 1]), vi("botmask_b", [1, 1, HEIGHT, 1], B),
        vi("botmask", [1, 1, HEIGHT, 1]),
        vi("leftsel", s1), vi("rightsel", s1), vi("topsel", s1), vi("botsel", s1),
        vi("A_contrib", g4), vi("B_contrib", g4),
        vi("A2_contrib", g4), vi("B2_contrib", g4), vi("bg_contrib", g4),
        vi("V_ab", g4), vi("V", g4), vi("H_ab", g4), vi("Hh", g4),
        vi("Vsel", g4), vi("Hsel", g4),
    ]

    inputs = [vi("input", g4)]
    outputs = [vi("output", g4)]
    graph = helper.make_graph(nodes, "nearest_wall", inputs, outputs,
                              initializer=init, value_info=value_info)
    return helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
        ir_version=IR_VERSION)


def solve_nearest_wall(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
