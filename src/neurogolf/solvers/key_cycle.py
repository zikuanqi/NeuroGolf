"""Solver: cycle a key row into solid bands below a separator (task 297).

Row 0 holds a key of ``W`` colours; row 1 is a separator.  Every row below is
filled solid with the key colours cycled vertically -- row ``r`` (``r>=2``) takes
``key[(r-2) mod W]``::

    2 1 4          2 1 4
    5 5 5    ->    5 5 5
    . . .          2 2 2
    . . .          1 1 1
    . . .          4 4 4
    . . .          2 2 2  ...

Build: transpose row 0 into a colour column; compute per-row indices
``(r-2) mod W`` with ``W`` = real grid width; ``Gather`` the column and paint it
across rows ``>=2`` of the real grid.
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
    if H < 3:
        return None
    key = g[0]
    out = g.copy()
    for r in range(2, H):
        out[r, :] = key[(r - 2) % W]
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
    I = TensorProto.INT64
    n = helper.make_node

    row_idx = np.arange(HEIGHT, dtype=np.float32).reshape(1, 1, HEIGHT, 1)
    col_idx = np.arange(WIDTH, dtype=np.float32).reshape(1, 1, 1, WIDTH)
    rvals = np.arange(HEIGHT, dtype=np.float32) - 2.0   # (H,)
    init = [
        numpy_helper.from_array(row_idx, "row_idx"),
        numpy_helper.from_array(col_idx, "col_idx"),
        numpy_helper.from_array(rvals, "rvals"),
        numpy_helper.from_array(np.array(1.0, np.float32), "one"),
        numpy_helper.from_array(np.array(1.5, np.float32), "onehalf"),
        numpy_helper.from_array(np.array([0], np.int64), "r0s"),
        numpy_helper.from_array(np.array([1], np.int64), "r1e"),
        numpy_helper.from_array(np.array([2], np.int64), "ax2"),
    ]
    nodes = [
        n("ReduceSum", ["input"], ["content"], axes=[1], keepdims=1),
        # real grid width W = last real col + 1
        n("ReduceMax", ["content"], ["realcol"], axes=[2], keepdims=1),    # (1,1,1,W)
        n("Mul", ["col_idx", "realcol"], ["cc"]),
        n("ReduceMax", ["cc"], ["clast"], keepdims=0),                     # scalar
        n("Add", ["clast", "one"], ["W"]),
        # per-row index (r-2) mod W
        n("Div", ["rvals", "W"], ["q0"]),
        n("Floor", ["q0"], ["q"]),
        n("Mul", ["q", "W"], ["qW"]),
        n("Sub", ["rvals", "qW"], ["idxf"]),                              # (H,)
        n("Cast", ["idxf"], ["idx"], to=I),
        # key column = transpose of row 0
        n("Slice", ["input", "r0s", "r1e", "ax2"], ["keyrow"]),           # (1,C,1,W)
        n("Transpose", ["keyrow"], ["keycol"], perm=[0, 1, 3, 2]),        # (1,C,W,1)
        n("Gather", ["keycol", "idx"], ["gathered"], axis=2),             # (1,C,H,1)
        # fill mask: rows >= 2 of the real grid
        n("Greater", ["row_idx", "onehalf"], ["ge2_b"]), n("Cast", ["ge2_b"], ["ge2"], to=F),
        n("Mul", ["ge2", "content"], ["fillmask"]),                       # (1,1,H,W)
        n("Sub", ["one", "fillmask"], ["keepmask"]),
        n("Mul", ["input", "keepmask"], ["kept"]),
        n("Mul", ["gathered", "fillmask"], ["addc"]),
        n("Add", ["kept", "addc"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "key_cycle",
                              [helper.make_tensor_value_info("input", F, FULL)],
                              [helper.make_tensor_value_info("output", F, FULL)],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_key_cycle(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
