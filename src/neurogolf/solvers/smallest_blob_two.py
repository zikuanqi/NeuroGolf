"""Solver: recolour the smallest connected blob to 2, the rest to 1 (task 277).

Every 8-connected non-background blob is repainted ``1`` except the one with the
fewest cells, which becomes ``2``.

Graph: (1) label blobs by iterated 3x3 ``MaxPool`` max-propagation of a unique
per-cell id, re-masked to the foreground each step (8-connected); (2) per-blob
size via an all-pairs label match (reshape to 900, ``Equal``, ``ReduceSum``);
(3) the global minimum size selects the ``2`` blob.
"""
from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from ..grids import CHANNELS, HEIGHT, WIDTH, all_examples

OPSET = 11
IR_VERSION = 8
F = TensorProto.FLOAT
B = TensorProto.BOOL
N_ITERS = 30
NCELL = HEIGHT * WIDTH


def _comps(mask):
    H, W = mask.shape
    seen = np.zeros((H, W), bool)
    res = []
    nbs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    for r in range(H):
        for c in range(W):
            if mask[r, c] and not seen[r, c]:
                cs = []; q = deque([(r, c)]); seen[r, c] = True
                while q:
                    y, x = q.popleft(); cs.append((y, x))
                    for dy, dx in nbs:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True; q.append((ny, nx))
                res.append(cs)
    return res


def _ref(g: np.ndarray) -> Optional[np.ndarray]:
    out = np.zeros_like(g)
    cs = _comps(g != 0)
    if len(cs) < 2:
        return None
    mn = min(len(c) for c in cs)
    if sum(1 for c in cs if len(c) == mn) != 1:
        return None  # ambiguous smallest
    for c in cs:
        col = 2 if len(c) == mn else 1
        for y, x in c:
            out[y, x] = col
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


def _ev(k):
    v = np.zeros((1, CHANNELS, 1, 1), np.float32); v[0, k] = 1.0
    return v


def _build() -> onnx.ModelProto:
    n = helper.make_node
    cellid = (np.arange(NCELL).reshape(1, 1, HEIGHT, WIDTH) + 1).astype(np.float32)
    init = [
        numpy_helper.from_array(cellid, "cellid"),
        numpy_helper.from_array(_ev(0), "e0"),
        numpy_helper.from_array(_ev(1), "e1"),
        numpy_helper.from_array(_ev(2), "e2"),
        numpy_helper.from_array(np.array(1.0, np.float32), "one"),
        numpy_helper.from_array(np.array(1e6, np.float32), "big"),
        numpy_helper.from_array(np.array([0], np.int64), "idx0"),
        numpy_helper.from_array(np.array([1, 1, NCELL, 1], np.int64), "to_col"),
        numpy_helper.from_array(np.array([1, 1, 1, NCELL], np.int64), "to_row"),
        numpy_helper.from_array(np.array([1, 1, HEIGHT, WIDTH], np.int64), "to_grid"),
    ]
    nodes = [
        n("ReduceSum", ["input"], ["content"], axes=[1], keepdims=1),
        n("Gather", ["input", "idx0"], ["ch0"], axis=1),
        n("Sub", ["content", "ch0"], ["nonbg"]),
        n("Mul", ["cellid", "nonbg"], ["label0"]),
    ]
    cur = "label0"
    for it in range(N_ITERS):
        nodes.append(n("MaxPool", [cur], [f"mp_{it}"], kernel_shape=[3, 3],
                       pads=[1, 1, 1, 1], strides=[1, 1]))
        nxt = f"label_{it+1}"
        nodes.append(n("Mul", [f"mp_{it}", "nonbg"], [nxt]))
        cur = nxt
    label = cur
    nodes += [
        n("Reshape", [label, "to_col"], ["lcol"]),
        n("Reshape", [label, "to_row"], ["lrow"]),
        n("Equal", ["lcol", "lrow"], ["eqm"]),
        n("Cast", ["eqm"], ["eqmf"], to=F),
        n("ReduceSum", ["eqmf"], ["szcol"], axes=[3], keepdims=1),
        n("Reshape", ["szcol", "to_grid"], ["size"]),
        # min size over foreground (bg pushed up by big)
        n("Sub", ["one", "nonbg"], ["bgmask"]),
        n("Mul", ["bgmask", "big"], ["bgbig"]),
        n("Add", ["size", "bgbig"], ["sizeMasked"]),
        n("ReduceMin", ["sizeMasked"], ["minSize"], axes=[2, 3], keepdims=1),
        n("Equal", ["size", "minSize"], ["eqMin"]),
        n("Cast", ["eqMin"], ["eqMinf"], to=F),
        n("Mul", ["eqMinf", "nonbg"], ["isMin"]),    # smallest blob cells
        n("Sub", ["nonbg", "isMin"], ["other"]),     # the rest
        n("Mul", ["ch0", "e0"], ["L0"]),
        n("Mul", ["isMin", "e2"], ["L2"]),
        n("Mul", ["other", "e1"], ["L1"]),
        n("Add", ["L0", "L2"], ["t1"]),
        n("Add", ["t1", "L1"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "smallest_blob_two",
                              [helper.make_tensor_value_info("input", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              [helper.make_tensor_value_info("output", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_smallest_blob_two(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
