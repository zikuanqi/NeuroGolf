"""Solver: anti-diagonal + bottom row from a column-0 line (task 84).

The input is a square grid whose only content is a colour-C line down column 0.
The output keeps that line and adds, over the background, an anti-diagonal of
colour 2 (top-right corner down to just left of the line) and a bottom row of
colour 4.  Both are independent of C.

Build: the grid side H is `sum(row_has)`; the anti-diagonal is the cells where
`row_index + col_index == H-1` and the bottom row is `row_index == H-1`, each
restricted to columns >= 1 and to real grid cells (content mask).
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
DIAG, BOT = 2, 4


def _ref(g: np.ndarray) -> Optional[np.ndarray]:
    H, W = g.shape
    if H != W:
        return None
    nz = np.argwhere(g != 0)
    if len(nz) == 0 or set(nz[:, 1].tolist()) != {0}:
        return None
    out = g.copy()
    for r in range(H):
        for c in range(1, W):
            if r + c == H - 1:
                out[r, c] = DIAG
            if r == H - 1:
                out[r, c] = BOT
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


def _edelta(col: int) -> np.ndarray:
    v = np.zeros((1, CHANNELS, 1, 1), np.float32)
    v[0, col] = 1.0
    v[0, 0] = -1.0
    return v


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
        numpy_helper.from_array(_edelta(DIAG), "e_diag"),
        numpy_helper.from_array(_edelta(BOT), "e_bot"),
    ]
    rc = [1, 1, HEIGHT, 1]; cc = [1, 1, 1, WIDTH]; s1 = [1, 1, HEIGHT, WIDTH]
    sc = [1, 1, 1, 1]
    nodes = [
        n("ReduceSum", ["input"], ["content"], axes=[1], keepdims=1),     # (1,1,H,W)
        n("ReduceMax", ["content"], ["row_has"], axes=[3], keepdims=1),   # (1,1,H,1)
        n("ReduceSum", ["row_has"], ["Hsz"]),                             # scalar
        n("Sub", ["Hsz", "one"], ["Hm1"]),
        n("Add", ["row_idx", "col_idx"], ["rcsum"]),                      # (1,1,H,W)
        # anti-diagonal: rcsum == Hm1
        n("Sub", ["rcsum", "Hm1"], ["addiff"]),
        n("Abs", ["addiff"], ["adabs"]),
        n("Less", ["adabs", "half"], ["adeq_b"]), n("Cast", ["adeq_b"], ["adeq"], to=F),
        # bottom row: row_idx == Hm1
        n("Sub", ["row_idx", "Hm1"], ["brdiff"]),
        n("Abs", ["brdiff"], ["brabs"]),
        n("Less", ["brabs", "half"], ["breq_b"]), n("Cast", ["breq_b"], ["breq"], to=F),
        # col >= 1
        n("Greater", ["col_idx", "half"], ["cge1_b"]), n("Cast", ["cge1_b"], ["cge1"], to=F),
        # masks restricted to columns>=1 and real grid cells
        n("Mul", ["adeq", "cge1"], ["m2a"]),
        n("Mul", ["m2a", "content"], ["mask2"]),
        n("Mul", ["breq", "cge1"], ["m4a"]),
        n("Mul", ["m4a", "content"], ["mask4"]),
        # paint
        n("Mul", ["e_diag", "mask2"], ["d2"]),
        n("Mul", ["e_bot", "mask4"], ["d4"]),
        n("Add", ["d2", "d4"], ["delta"]),
        n("Add", ["input", "delta"], ["output"]),
    ]
    vi = [
        helper.make_tensor_value_info("content", F, s1),
        helper.make_tensor_value_info("row_has", F, rc),
        helper.make_tensor_value_info("Hsz", F, sc),
        helper.make_tensor_value_info("Hm1", F, sc),
        helper.make_tensor_value_info("rcsum", F, s1),
        helper.make_tensor_value_info("addiff", F, s1),
        helper.make_tensor_value_info("adabs", F, s1),
        helper.make_tensor_value_info("adeq_b", B, s1),
        helper.make_tensor_value_info("adeq", F, s1),
        helper.make_tensor_value_info("brdiff", F, rc),
        helper.make_tensor_value_info("brabs", F, rc),
        helper.make_tensor_value_info("breq_b", B, rc),
        helper.make_tensor_value_info("breq", F, rc),
        helper.make_tensor_value_info("cge1_b", B, cc),
        helper.make_tensor_value_info("cge1", F, cc),
        helper.make_tensor_value_info("m2a", F, s1),
        helper.make_tensor_value_info("mask2", F, s1),
        helper.make_tensor_value_info("m4a", F, s1),
        helper.make_tensor_value_info("mask4", F, s1),
        helper.make_tensor_value_info("d2", F, FULL),
        helper.make_tensor_value_info("d4", F, FULL),
        helper.make_tensor_value_info("delta", F, FULL),
    ]
    graph = helper.make_graph(nodes, "corner_rays",
                              [helper.make_tensor_value_info("input", F, FULL)],
                              [helper.make_tensor_value_info("output", F, FULL)],
                              initializer=init, value_info=vi)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_corner_rays(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
