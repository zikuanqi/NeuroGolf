"""Solver: stamp the reference pattern onto every same-size 5-block (task 368).

One multi-colour block is the reference; every solid ``5`` block (same size) is
replaced by a copy of it.

The reference is cropped to a 5x5 runtime ``ConvTranspose`` kernel; an impulse at
each 5-block's top-left corner stamps the kernel back across the grid.
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
K = 5


def _comps(mask):
    H, W = mask.shape
    seen = np.zeros((H, W), bool); res = []
    for r in range(H):
        for c in range(W):
            if mask[r, c] and not seen[r, c]:
                cs = []; q = deque([(r, c)]); seen[r, c] = True
                while q:
                    y, x = q.popleft(); cs.append((y, x))
                    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True; q.append((ny, nx))
                res.append(cs)
    return res


def _ref(g: np.ndarray) -> Optional[np.ndarray]:
    H, W = g.shape
    ref5 = (g == 5)
    refcells = (g != 0) & (g != 5)
    if not refcells.any() or not ref5.any():
        return None
    ys, xs = np.where(refcells)
    R0, R1, C0, C1 = ys.min(), ys.max(), xs.min(), xs.max()
    P = g[R0:R1 + 1, C0:C1 + 1].copy()
    if (P == 0).any() or P.shape[0] > K or P.shape[1] > K:
        return None
    out = g.copy()
    for comp in _comps(ref5):
        ys2 = [y for y, x in comp]; xs2 = [x for y, x in comp]
        br, bc = min(ys2), min(xs2)
        if (max(ys2) - br + 1, max(xs2) - bc + 1) != P.shape:
            return None
        for y, x in comp:
            out[y, x] = 0
        out[br:br + P.shape[0], bc:bc + P.shape[1]] = P
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
    rowidx = np.arange(HEIGHT, dtype=np.float32).reshape(1, 1, HEIGHT, 1)
    colidx = np.arange(WIDTH, dtype=np.float32).reshape(1, 1, 1, WIDTH)
    init = [
        numpy_helper.from_array(rowidx, "rowidx"),
        numpy_helper.from_array(colidx, "colidx"),
        numpy_helper.from_array(np.array(1.0, np.float32), "one"),
        numpy_helper.from_array(np.array(1e4, np.float32), "big"),
        numpy_helper.from_array(np.array(0.0, np.float32), "cmin"),
        numpy_helper.from_array(np.array(float(HEIGHT - 1), np.float32), "cmax"),
        numpy_helper.from_array(np.array([0], np.int64), "z0"),
        numpy_helper.from_array(np.array([1], np.int64), "z1"),
        numpy_helper.from_array(np.array([5], np.int64), "z5"),
        numpy_helper.from_array(np.array([6], np.int64), "z6"),
        numpy_helper.from_array(np.array([HEIGHT], np.int64), "z30"),
        numpy_helper.from_array(np.array([1], np.int64), "ax1"),
        numpy_helper.from_array(np.array([2], np.int64), "ax2"),
        numpy_helper.from_array(np.array([3], np.int64), "ax3"),
        numpy_helper.from_array(np.array([HEIGHT], np.int64), "to30"),
        numpy_helper.from_array(np.array([0, 0, 1, 0, 0, 0, 0, 0], np.int64), "padT"),
        numpy_helper.from_array(np.array([0, 0, 0, 1, 0, 0, 0, 0], np.int64), "padL"),
    ]
    nodes = [
        n("ReduceSum", ["input"], ["occ"], axes=[1], keepdims=1),
        n("Slice", ["input", "z0", "z1", "ax1"], ["is0"]),
        n("Slice", ["input", "z5", "z6", "ax1"], ["is5"]),
        n("Sub", ["occ", "is0"], ["nonbg"]),
        n("Sub", ["nonbg", "is5"], ["refcells"]),     # non-bg, non-5
        # reference bbox
        n("ReduceMax", ["refcells"], ["refRowHas"], axes=[3], keepdims=1),
        n("ReduceMax", ["refcells"], ["refColHas"], axes=[2], keepdims=1),
        n("Mul", ["refRowHas", "rowidx"], ["rr"]),
        n("Sub", ["one", "refRowHas"], ["rrinv"]), n("Mul", ["rrinv", "big"], ["rrb"]),
        n("Add", ["rr", "rrb"], ["rrmin"]), n("ReduceMin", ["rrmin"], ["refTop"], axes=[2], keepdims=1),
        n("ReduceMax", ["rr"], ["refBot"], axes=[2], keepdims=1),
        n("Sub", ["refBot", "refTop"], ["rHspan"]), n("Add", ["rHspan", "one"], ["rH"]),
        n("Mul", ["refColHas", "colidx"], ["cc"]),
        n("Sub", ["one", "refColHas"], ["ccinv"]), n("Mul", ["ccinv", "big"], ["ccb"]),
        n("Add", ["cc", "ccb"], ["ccmin"]), n("ReduceMin", ["ccmin"], ["refLeft"], axes=[3], keepdims=1),
        n("Mul", ["refColHas", "colidx"], ["cc2"]), n("ReduceMax", ["cc2"], ["refRight"], axes=[3], keepdims=1),
        n("Sub", ["refRight", "refLeft"], ["rWspan"]), n("Add", ["rWspan", "one"], ["rW"]),
        # crop reference to top-left
        n("Add", ["rowidx", "refTop"], ["rgi0"]), n("Clip", ["rgi0", "cmin", "cmax"], ["rgic"]),
        n("Cast", ["rgic"], ["rgii"], to=TensorProto.INT64), n("Reshape", ["rgii", "to30"], ["rgi"]),
        n("Add", ["colidx", "refLeft"], ["cgi0"]), n("Clip", ["cgi0", "cmin", "cmax"], ["cgic"]),
        n("Cast", ["cgic"], ["cgii"], to=TensorProto.INT64), n("Reshape", ["cgii", "to30"], ["cgi"]),
        n("Gather", ["input", "rgi"], ["refR"], axis=2),
        n("Gather", ["refR", "cgi"], ["refCrop"], axis=3),
        # mask to ref bbox
        n("Less", ["rowidx", "rH"], ["mrb"]), n("Cast", ["mrb"], ["mr"], to=F),
        n("Less", ["colidx", "rW"], ["mcb"]), n("Cast", ["mcb"], ["mc"], to=F),
        n("Mul", ["mr", "mc"], ["refMask"]),
        n("Mul", ["refCrop", "refMask"], ["refCropM"]),
        # 5x5 kernel
        n("Slice", ["refCropM", "z0", "z5", "ax2"], ["kr"]),
        n("Slice", ["kr", "z0", "z5", "ax3"], ["W"]),         # (1,10,5,5)
        # 5-block top-left impulse
        n("Pad", ["is5", "padT"], ["upP"]), n("Slice", ["upP", "z0", "z30", "ax2"], ["up5"]),
        n("Pad", ["is5", "padL"], ["lfP"]), n("Slice", ["lfP", "z0", "z30", "ax3"], ["lf5"]),
        n("Sub", ["one", "up5"], ["nup"]), n("Sub", ["one", "lf5"], ["nlf"]),
        n("Mul", ["is5", "nup"], ["t0"]), n("Mul", ["t0", "nlf"], ["impulse"]),
        # stamp via ConvTranspose
        n("ConvTranspose", ["impulse", "W"], ["stamp0"], strides=[1, 1], kernel_shape=[K, K]),
        n("Slice", ["stamp0", "z0", "z30", "ax2"], ["stampR"]),
        n("Slice", ["stampR", "z0", "z30", "ax3"], ["stamp"]),
        # combine: keep non-5, add stamp
        n("Sub", ["one", "is5"], ["not5"]),
        n("Mul", ["input", "not5"], ["keep"]),
        n("Add", ["keep", "stamp"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "stamp_ref_on_blocks",
                              [helper.make_tensor_value_info("input", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              [helper.make_tensor_value_info("output", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_stamp_ref_on_blocks(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
