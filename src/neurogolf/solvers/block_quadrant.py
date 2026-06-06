"""Solver: fill a 2x2 block with the four quadrant markers (task 342).

A 2x2 block of colour 8 sits among four single-cell markers, one in each
quadrant of the grid.  The output keeps only the 2x2 block, painting each of its
cells with the marker from the matching quadrant (top-left marker -> block's
top-left cell, ...); everything else becomes background.

Build: the block centre (rc, cc) is the colour-8 centroid.  Quadrant masks
(row<rc / >rc, col<cc / >cc) times the marker channels give the four corner
colours; index masks at the block's four cells place them, with background
elsewhere in the grid.
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
    ys, xs = np.where(g == 8)
    if len(ys) != 4:
        return None
    r0, c0 = ys.min(), xs.min()
    if ys.max() - r0 != 1 or xs.max() - c0 != 1:
        return None
    rc, cc = r0 + 0.5, c0 + 0.5
    markers = [(r, c, g[r, c]) for r in range(H) for c in range(W)
               if g[r, c] not in (0, 8)]
    if len(markers) != 4:
        return None
    out = np.zeros_like(g)
    for (mr, mc, col) in markers:
        out[r0 if mr < rc else r0 + 1, c0 if mc < cc else c0 + 1] = col
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

    def cf(b, name):
        return n("Cast", [b], [name], to=F)

    row_idx = np.arange(HEIGHT, dtype=np.float32).reshape(1, 1, HEIGHT, 1)
    col_idx = np.arange(WIDTH, dtype=np.float32).reshape(1, 1, 1, WIDTH)
    e0 = np.zeros((1, CHANNELS, 1, 1), np.float32); e0[0, 0] = 1.0
    note08 = np.ones((1, CHANNELS, 1, 1), np.float32); note08[0, 0] = 0.0; note08[0, 8] = 0.0
    init = [
        numpy_helper.from_array(row_idx, "row_idx"),
        numpy_helper.from_array(col_idx, "col_idx"),
        numpy_helper.from_array(e0, "e0"),
        numpy_helper.from_array(note08, "note08"),
        numpy_helper.from_array(np.array(0.5, np.float32), "half"),
        numpy_helper.from_array(np.array([0, 8, 0, 0], np.int64), "s8"),
        numpy_helper.from_array(np.array([1, 9, HEIGHT, WIDTH], np.int64), "e8"),
        numpy_helper.from_array(np.array([0, 1, 2, 3], np.int64), "ax4"),
    ]
    nodes = [
        n("Slice", ["input", "s8", "e8", "ax4"], ["ch8"]),               # (1,1,H,W)
        n("ReduceSum", ["ch8"], ["cnt"], keepdims=0),
        n("Mul", ["ch8", "row_idx"], ["c8r"]), n("ReduceSum", ["c8r"], ["sr"], keepdims=0),
        n("Mul", ["ch8", "col_idx"], ["c8c"]), n("ReduceSum", ["c8c"], ["sc"], keepdims=0),
        n("Div", ["sr", "cnt"], ["rc"]), n("Div", ["sc", "cnt"], ["cc"]),
        n("Sub", ["rc", "half"], ["r0"]), n("Add", ["rc", "half"], ["r1"]),
        n("Sub", ["cc", "half"], ["c0"]), n("Add", ["cc", "half"], ["c1"]),
        n("Mul", ["input", "note08"], ["markers"]),                      # (1,10,H,W)
        # quadrant masks relative to the block centre
        n("Less", ["row_idx", "rc"], ["top_b"]), cf("top_b", "top"),
        n("Greater", ["row_idx", "rc"], ["bot_b"]), cf("bot_b", "bot"),
        n("Less", ["col_idx", "cc"], ["lft_b"]), cf("lft_b", "lft"),
        n("Greater", ["col_idx", "cc"], ["rgt_b"]), cf("rgt_b", "rgt"),
        n("Mul", ["top", "lft"], ["TLr"]), n("Mul", ["top", "rgt"], ["TRr"]),
        n("Mul", ["bot", "lft"], ["BLr"]), n("Mul", ["bot", "rgt"], ["BRr"]),
        n("Mul", ["markers", "TLr"], ["mTL"]), n("ReduceSum", ["mTL"], ["TLc"], axes=[2, 3], keepdims=1),
        n("Mul", ["markers", "TRr"], ["mTR"]), n("ReduceSum", ["mTR"], ["TRc"], axes=[2, 3], keepdims=1),
        n("Mul", ["markers", "BLr"], ["mBL"]), n("ReduceSum", ["mBL"], ["BLc"], axes=[2, 3], keepdims=1),
        n("Mul", ["markers", "BRr"], ["mBR"]), n("ReduceSum", ["mBR"], ["BRc"], axes=[2, 3], keepdims=1),
        # block cell index masks
        n("Sub", ["row_idx", "r0"], ["dr0"]), n("Abs", ["dr0"], ["adr0"]), n("Less", ["adr0", "half"], ["rs0_b"]), cf("rs0_b", "rs0"),
        n("Sub", ["row_idx", "r1"], ["dr1"]), n("Abs", ["dr1"], ["adr1"]), n("Less", ["adr1", "half"], ["rs1_b"]), cf("rs1_b", "rs1"),
        n("Sub", ["col_idx", "c0"], ["dc0"]), n("Abs", ["dc0"], ["adc0"]), n("Less", ["adc0", "half"], ["cs0_b"]), cf("cs0_b", "cs0"),
        n("Sub", ["col_idx", "c1"], ["dc1"]), n("Abs", ["dc1"], ["adc1"]), n("Less", ["adc1", "half"], ["cs1_b"]), cf("cs1_b", "cs1"),
        n("Mul", ["rs0", "cs0"], ["cTL"]), n("Mul", ["rs0", "cs1"], ["cTR"]),
        n("Mul", ["rs1", "cs0"], ["cBL"]), n("Mul", ["rs1", "cs1"], ["cBR"]),
        n("Mul", ["TLc", "cTL"], ["pTL"]), n("Mul", ["TRc", "cTR"], ["pTR"]),
        n("Mul", ["BLc", "cBL"], ["pBL"]), n("Mul", ["BRc", "cBR"], ["pBR"]),
        n("Add", ["pTL", "pTR"], ["pa"]), n("Add", ["pBL", "pBR"], ["pb"]), n("Add", ["pa", "pb"], ["colors"]),
        # background everywhere else in the grid
        n("ReduceSum", ["input"], ["grid"], axes=[1], keepdims=1),
        n("Add", ["cTL", "cTR"], ["bca"]), n("Add", ["cBL", "cBR"], ["bcb"]), n("Add", ["bca", "bcb"], ["blockcells"]),
        n("Sub", ["grid", "blockcells"], ["bgreg"]),
        n("Mul", ["e0", "bgreg"], ["pBg"]),
        n("Add", ["colors", "pBg"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "block_quadrant",
                              [helper.make_tensor_value_info("input", F, FULL)],
                              [helper.make_tensor_value_info("output", F, FULL)],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_block_quadrant(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
