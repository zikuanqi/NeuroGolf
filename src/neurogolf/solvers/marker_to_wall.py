"""Solver: each interior marker slides to its matching-colour wall (task 340).

The grid has a 4-coloured border frame (top/bottom/left/right each a colour).
Each interior marker moves to sit just inside the wall whose colour it matches,
keeping its other coordinate; the interior is otherwise cleared, the frame kept.

Wall colours are read off the four border lines; markers are projected to the
fixed inner rows/cols (row1 / col1) or to the grid's second-last row/col (found
by shifting the last grid line in by one).
"""
from __future__ import annotations

from collections import Counter
from typing import Optional

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from ..grids import CHANNELS, HEIGHT, WIDTH, all_examples

OPSET = 11
IR_VERSION = 8
F = TensorProto.FLOAT


def _wallcol(line):
    vals = [v for v in line if v != 0]
    return Counter(vals).most_common(1)[0][0] if vals else 0


def _ref(g: np.ndarray) -> Optional[np.ndarray]:
    H, W = g.shape
    if H < 3 or W < 3:
        return None
    top, bot = _wallcol(g[0, :]), _wallcol(g[H - 1, :])
    left, right = _wallcol(g[:, 0]), _wallcol(g[:, W - 1])
    if len({top, bot, left, right}) < 4 or 0 in (top, bot, left, right):
        return None
    out = np.zeros_like(g)
    out[0, :] = g[0, :]; out[H - 1, :] = g[H - 1, :]
    out[:, 0] = g[:, 0]; out[:, W - 1] = g[:, W - 1]
    changed = False
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            v = g[r, c]
            if v == 0:
                continue
            if v == top:
                out[1, c] = v
            elif v == bot:
                out[H - 2, c] = v
            elif v == left:
                out[r, 1] = v
            elif v == right:
                out[r, W - 2] = v
            else:
                continue
            changed = True
    return out if changed else None


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
    notbg = np.ones((1, CHANNELS, 1, 1), np.float32); notbg[0, 0] = 0.0
    row0 = np.zeros((1, 1, HEIGHT, 1), np.float32); row0[0, 0, 0, 0] = 1.0
    row1 = np.zeros((1, 1, HEIGHT, 1), np.float32); row1[0, 0, 1, 0] = 1.0
    col0 = np.zeros((1, 1, 1, WIDTH), np.float32); col0[0, 0, 0, 0] = 1.0
    col1 = np.zeros((1, 1, 1, WIDTH), np.float32); col1[0, 0, 0, 1] = 1.0
    init = [
        numpy_helper.from_array(e0, "e0"),
        numpy_helper.from_array(notbg, "notbg"),
        numpy_helper.from_array(row0, "row0"),
        numpy_helper.from_array(row1, "row1"),
        numpy_helper.from_array(col0, "col0"),
        numpy_helper.from_array(col1, "col1"),
        numpy_helper.from_array(np.array(1.0, np.float32), "one"),
        numpy_helper.from_array(np.array([1], np.int64), "s1"),
        numpy_helper.from_array(np.array([HEIGHT + 1], np.int64), "sN1"),
        numpy_helper.from_array(np.array([2], np.int64), "ax2"),
        numpy_helper.from_array(np.array([3], np.int64), "ax3"),
        numpy_helper.from_array(np.array([0, 0, 0, 0, 0, 0, 1, 0], np.int64), "padBot"),
        numpy_helper.from_array(np.array([0, 0, 0, 0, 0, 0, 0, 1], np.int64), "padRgt"),
    ]
    nodes = [
        n("ReduceSum", ["input"], ["occ"], axes=[1], keepdims=1),
        n("ReduceMax", ["occ"], ["gridRow"], axes=[3], keepdims=1),    # (1,1,H,1)
        n("ReduceMax", ["occ"], ["gridCol"], axes=[2], keepdims=1),    # (1,1,1,W)
        # last grid row / second-last
        n("Pad", ["gridRow", "padBot"], ["grP"]), n("Slice", ["grP", "s1", "sN1", "ax2"], ["gru"]),
        n("Sub", ["one", "gru"], ["gruInv"]), n("Mul", ["gridRow", "gruInv"], ["lastRow"]),
        n("Pad", ["lastRow", "padBot"], ["lrP"]), n("Slice", ["lrP", "s1", "sN1", "ax2"], ["secondLastRow"]),
        # last grid col / second-last
        n("Pad", ["gridCol", "padRgt"], ["gcP"]), n("Slice", ["gcP", "s1", "sN1", "ax3"], ["gcl"]),
        n("Sub", ["one", "gcl"], ["gclInv"]), n("Mul", ["gridCol", "gclInv"], ["lastCol"]),
        n("Pad", ["lastCol", "padRgt"], ["lcP"]), n("Slice", ["lcP", "s1", "sN1", "ax3"], ["secondLastCol"]),
        # wall colours (1,C,1,1)
        n("Mul", ["input", "row0"], ["x_top"]), n("ReduceMax", ["x_top"], ["topV0"], axes=[2, 3], keepdims=1), n("Mul", ["topV0", "notbg"], ["topV"]),
        n("Mul", ["input", "lastRow"], ["x_bot"]), n("ReduceMax", ["x_bot"], ["botV0"], axes=[2, 3], keepdims=1), n("Mul", ["botV0", "notbg"], ["botV"]),
        n("Mul", ["input", "col0"], ["x_lft"]), n("ReduceMax", ["x_lft"], ["leftV0"], axes=[2, 3], keepdims=1), n("Mul", ["leftV0", "notbg"], ["leftV"]),
        n("Mul", ["input", "lastCol"], ["x_rgt"]), n("ReduceMax", ["x_rgt"], ["rightV0"], axes=[2, 3], keepdims=1), n("Mul", ["rightV0", "notbg"], ["rightV"]),
        # border / interior masks
        n("Max", ["row0", "lastRow"], ["borderRow"]),
        n("Max", ["col0", "lastCol"], ["borderCol"]),
        n("Max", ["borderRow", "borderCol"], ["borderMask"]),   # (1,1,H,W)
        n("Sub", ["one", "borderMask"], ["intInv"]), n("Mul", ["occ", "intInv"], ["interior"]),
        # markers per wall colour
        n("Mul", ["input", "topV"], ["mt0"]), n("ReduceSum", ["mt0"], ["isTop"], axes=[1], keepdims=1),
        n("Mul", ["isTop", "interior"], ["topCells"]),
        n("Mul", ["input", "botV"], ["mb0"]), n("ReduceSum", ["mb0"], ["isBot"], axes=[1], keepdims=1),
        n("Mul", ["isBot", "interior"], ["botCells"]),
        n("Mul", ["input", "leftV"], ["ml0"]), n("ReduceSum", ["ml0"], ["isLeft"], axes=[1], keepdims=1),
        n("Mul", ["isLeft", "interior"], ["leftCells"]),
        n("Mul", ["input", "rightV"], ["mr0"]), n("ReduceSum", ["mr0"], ["isRight"], axes=[1], keepdims=1),
        n("Mul", ["isRight", "interior"], ["rightCells"]),
        # project to walls
        n("ReduceMax", ["topCells"], ["topCols"], axes=[2], keepdims=1),
        n("Mul", ["topCols", "row1"], ["topP"]), n("Mul", ["topP", "topV"], ["topLayer"]),
        n("ReduceMax", ["botCells"], ["botCols"], axes=[2], keepdims=1),
        n("Mul", ["botCols", "secondLastRow"], ["botP"]), n("Mul", ["botP", "botV"], ["botLayer"]),
        n("ReduceMax", ["leftCells"], ["leftRows"], axes=[3], keepdims=1),
        n("Mul", ["leftRows", "col1"], ["leftP"]), n("Mul", ["leftP", "leftV"], ["leftLayer"]),
        n("ReduceMax", ["rightCells"], ["rightRows"], axes=[3], keepdims=1),
        n("Mul", ["rightRows", "secondLastCol"], ["rightP"]), n("Mul", ["rightP", "rightV"], ["rightLayer"]),
        # frame + combine
        n("Mul", ["input", "borderMask"], ["frameLayer"]),
        n("Add", ["frameLayer", "topLayer"], ["a1"]),
        n("Add", ["a1", "botLayer"], ["a2"]),
        n("Add", ["a2", "leftLayer"], ["a3"]),
        n("Add", ["a3", "rightLayer"], ["colorL"]),
        n("ReduceSum", ["colorL"], ["painted"], axes=[1], keepdims=1),
        n("Sub", ["occ", "painted"], ["bgv"]),
        n("Mul", ["bgv", "e0"], ["bgL"]),
        n("Add", ["colorL", "bgL"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "marker_to_wall",
                              [helper.make_tensor_value_info("input", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              [helper.make_tensor_value_info("output", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_marker_to_wall(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
