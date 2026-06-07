"""Solver: draw a plus at the midpoint of two markers (task 371).

The grid holds exactly two isolated ``1`` cells, aligned on a row or column with
an even gap.  A 5-cell plus (centre + 4 orthogonal neighbours) of colour ``3`` is
drawn at their midpoint; the markers stay::

    1 . . . . . . 1     ->     . . . 3 . . .
                               1 . 3 3 3 . 1
                               . . . 3 . . .

Build: centroid of the ``1`` mask = ``(sum idx) / count`` (an exact integer since
the gap is even); index masks ``|row-mid_r|`` / ``|col-mid_c|`` give the plus
(on-axis & within-1 on the other axis); paint ``e_3`` over the real grid.
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
    ys, xs = np.where(g == 1)
    if len(ys) < 2:
        return None
    mr = int(round(ys.mean())); mc = int(round(xs.mean()))
    out = g.copy()
    for dr, dc in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
        r, c = mr + dr, mc + dc
        if 0 <= r < g.shape[0] and 0 <= c < g.shape[1]:
            out[r, c] = 3
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
    e3 = np.zeros((1, CHANNELS, 1, 1), np.float32); e3[0, 3] = 1.0
    init = [
        numpy_helper.from_array(row_idx, "row_idx"),
        numpy_helper.from_array(col_idx, "col_idx"),
        numpy_helper.from_array(e3, "e3"),
        numpy_helper.from_array(np.array(0.5, np.float32), "half"),
        numpy_helper.from_array(np.array(1.5, np.float32), "onehalf"),
        numpy_helper.from_array(np.array(1.0, np.float32), "one"),
        numpy_helper.from_array(np.array([1], np.int64), "c1s"),
        numpy_helper.from_array(np.array([2], np.int64), "c2e"),
        numpy_helper.from_array(np.array([1], np.int64), "ax1"),
    ]
    nodes = [
        n("Slice", ["input", "c1s", "c2e", "ax1"], ["is1"]),              # (1,1,H,W)
        n("ReduceSum", ["is1"], ["cnt"], keepdims=0),                     # scalar count
        n("Mul", ["is1", "row_idx"], ["ir"]),
        n("ReduceSum", ["ir"], ["sumr"], keepdims=0),
        n("Div", ["sumr", "cnt"], ["mid_r"]),                            # scalar
        n("Mul", ["is1", "col_idx"], ["ic"]),
        n("ReduceSum", ["ic"], ["sumc"], keepdims=0),
        n("Div", ["sumc", "cnt"], ["mid_c"]),
        # distances
        n("Sub", ["row_idx", "mid_r"], ["dr"]), n("Abs", ["dr"], ["adr"]),
        n("Sub", ["col_idx", "mid_c"], ["dc"]), n("Abs", ["dc"], ["adc"]),
        n("Less", ["adr", "half"], ["onr_b"]), n("Cast", ["onr_b"], ["onr"], to=F),
        n("Less", ["adr", "onehalf"], ["nr_b"]), n("Cast", ["nr_b"], ["nearr"], to=F),
        n("Less", ["adc", "half"], ["onc_b"]), n("Cast", ["onc_b"], ["onc"], to=F),
        n("Less", ["adc", "onehalf"], ["nc_b"]), n("Cast", ["nc_b"], ["nearc"], to=F),
        n("Mul", ["onr", "nearc"], ["armA"]),                            # (1,1,H,W)
        n("Mul", ["nearr", "onc"], ["armB"]),
        n("Max", ["armA", "armB"], ["plus0"]),
        n("ReduceSum", ["input"], ["content"], axes=[1], keepdims=1),
        n("Mul", ["plus0", "content"], ["paint"]),                       # clip to real grid
        n("Sub", ["one", "paint"], ["inv"]),
        n("Mul", ["input", "inv"], ["kept"]),
        n("Mul", ["e3", "paint"], ["addc"]),
        n("Add", ["kept", "addc"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "midpoint_plus",
                              [helper.make_tensor_value_info("input", F, FULL)],
                              [helper.make_tensor_value_info("output", F, FULL)],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_midpoint_plus(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
