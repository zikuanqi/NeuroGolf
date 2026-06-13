"""Solver: ring the crossing of a full row and a full column with 4 (task 151).

One full horizontal line and one full vertical line (any colours) cross; the
eight neighbours of the intersection cell are painted ``4``, the intersection
itself is untouched::

    . 3 . .          4 4 4 .
    2 2 2 2    ->    4 2 4 2
    . 3 . .          4 4 4 .
    . 3 . .          . 3 . .

The full row/column are found by comparing per-line non-background counts to
the grid width/height (recovered from the occupancy mask), and the ring is a
Chebyshev-distance-1 test around ``(ArgMax(fullrow), ArgMax(fullcol))``.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from ..grids import CHANNELS, HEIGHT, WIDTH, all_examples

OPSET = 11
IR_VERSION = 8
F = TensorProto.FLOAT


def _ref(g: np.ndarray) -> Optional[np.ndarray]:
    H, W = g.shape
    rows = [r for r in range(H) if np.all(g[r] != 0)]
    cols = [c for c in range(W) if np.all(g[:, c] != 0)]
    if len(rows) != 1 or len(cols) != 1:
        return None
    r0, c0 = rows[0], cols[0]
    out = g.copy()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if (dr, dc) == (0, 0):
                continue
            rr, cc = r0 + dr, c0 + dc
            if 0 <= rr < H and 0 <= cc < W:
                out[rr, cc] = 4
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
    n = helper.make_node
    e4 = np.zeros((1, CHANNELS, 1, 1), np.float32); e4[0, 4] = 1.0
    init = [
        numpy_helper.from_array(e4, "e4"),
        numpy_helper.from_array(np.arange(HEIGHT, dtype=np.float32).reshape(1, 1, HEIGHT, 1), "ah"),
        numpy_helper.from_array(np.arange(WIDTH, dtype=np.float32).reshape(1, 1, 1, WIDTH), "aw"),
        numpy_helper.from_array(np.array(0.5, np.float32), "half"),
        numpy_helper.from_array(np.array(1.5, np.float32), "oneh"),
        numpy_helper.from_array(np.array(1.0, np.float32), "one"),
        numpy_helper.from_array(np.array([0], np.int64), "c0s"),
        numpy_helper.from_array(np.array([1], np.int64), "c1e"),
        numpy_helper.from_array(np.array([1], np.int64), "ax1"),
    ]
    nodes = [
        n("ReduceSum", ["input"], ["occ"], axes=[1], keepdims=1),
        n("Slice", ["input", "c0s", "c1e", "ax1"], ["is0"]),
        n("Sub", ["occ", "is0"], ["nbg"]),
        # grid width/height
        n("ReduceMax", ["occ"], ["rowData"], axes=[3], keepdims=1),
        n("ReduceMax", ["occ"], ["colData"], axes=[2], keepdims=1),
        n("ReduceSum", ["colData"], ["W"], axes=[3], keepdims=1),
        n("ReduceSum", ["rowData"], ["Hh"], axes=[2], keepdims=1),
        # full row: nonbg count == W (and row in grid)
        n("ReduceSum", ["nbg"], ["rcnt"], axes=[3], keepdims=1),
        n("Sub", ["W", "half"], ["Wm"]),
        n("Greater", ["rcnt", "Wm"], ["fr_b"]), n("Cast", ["fr_b"], ["fullrow"], to=F),
        n("ReduceSum", ["nbg"], ["ccnt"], axes=[2], keepdims=1),
        n("Sub", ["Hh", "half"], ["Hm"]),
        n("Greater", ["ccnt", "Hm"], ["fc_b"]), n("Cast", ["fc_b"], ["fullcol"], to=F),
        n("ArgMax", ["fullrow"], ["r0i"], axis=2, keepdims=1), n("Cast", ["r0i"], ["r0"], to=F),
        n("ArgMax", ["fullcol"], ["c0i"], axis=3, keepdims=1), n("Cast", ["c0i"], ["c0"], to=F),
        # ring = chebyshev<=1 minus centre, in grid
        n("Sub", ["ah", "r0"], ["dr"]), n("Abs", ["dr"], ["adr"]),
        n("Sub", ["aw", "c0"], ["dc"]), n("Abs", ["dc"], ["adc"]),
        n("Less", ["adr", "oneh"], ["nr_b"]), n("Cast", ["nr_b"], ["nr"], to=F),
        n("Less", ["adc", "oneh"], ["nc_b"]), n("Cast", ["nc_b"], ["nc"], to=F),
        n("Mul", ["nr", "nc"], ["near"]),
        n("Add", ["adr", "adc"], ["dist"]),
        n("Greater", ["dist", "half"], ["nz_b"]), n("Cast", ["nz_b"], ["nz"], to=F),
        n("Mul", ["near", "nz"], ["ring0"]),
        n("Mul", ["ring0", "occ"], ["ring"]),
        # paint ring with 4
        n("Sub", ["one", "ring"], ["keep"]),
        n("Mul", ["input", "keep"], ["kept"]),
        n("Mul", ["ring", "e4"], ["add4"]),
        n("Add", ["kept", "add4"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "cross_ring",
                              [helper.make_tensor_value_info("input", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              [helper.make_tensor_value_info("output", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_cross_ring(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
