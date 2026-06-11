"""Solver: slide every colour block vertically to the 1-block's row (task 30).

The grid holds several solid rectangular blocks in disjoint columns.  The
block coloured ``1`` is the anchor; every block (including the others) is
translated vertically so that its top row coincides with the anchor's top
row, keeping its columns, colour and height::

    2 2 . . . .          . . . . . .
    2 2 . 1 1 .   ->     2 2 . 1 1 .
    . . . 1 1 .          2 2 . 1 1 .
    . . . . . .          . . . . . .   (the 2-block slid down to meet 1)

Per colour ``k`` the shift is ``delta_k = top(1) - top(k)``.  A runtime shift
matrix ``S`` with ``S[r,j] = 1  iff  r - j = delta_k`` applied as ``MatMul(S,
mask_k)`` performs the vertical translation (rows pushed off-grid vanish).
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
F = TensorProto.FLOAT


def _ref(g: np.ndarray) -> Optional[np.ndarray]:
    y1, _ = np.where(g == 1)
    if len(y1) == 0:
        return None
    t1 = y1.min()
    out = np.zeros_like(g)
    for k in range(1, 10):
        ys, xs = np.where(g == k)
        if len(ys) == 0:
            continue
        dt = t1 - ys.min()
        rr = ys + dt
        ok = (rr >= 0) & (rr < g.shape[0])
        out[rr[ok], xs[ok]] = k
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


def _top_nodes(n, mask, has, cand, top, dst):
    """Emit nodes computing dst = min row index where `mask` has a cell."""
    return [
        n("ReduceMax", [mask], [has], axes=[3], keepdims=1),
        n("Sub", ["one", has], [has + "_n"]),
        n("Mul", [has + "_n", "BIG"], [has + "_b"]),
        n("Add", ["rowidx", has + "_b"], [cand]),
        n("ReduceMin", [cand], [top], axes=[2], keepdims=1),
    ]


def _build() -> onnx.ModelProto:
    n = helper.make_node
    rowidx = np.arange(HEIGHT, dtype=np.float32).reshape(1, 1, HEIGHT, 1)
    idx = np.arange(HEIGHT)
    D = (idx[:, None] - idx[None, :]).astype(np.float32)  # D[r,j] = r - j
    init = [
        numpy_helper.from_array(rowidx, "rowidx"),
        numpy_helper.from_array(D, "D"),
        numpy_helper.from_array(np.array(1000.0, np.float32), "BIG"),
        numpy_helper.from_array(np.array(1.0, np.float32), "one"),
        numpy_helper.from_array(np.array(0.5, np.float32), "half"),
        numpy_helper.from_array(np.array([1], np.int64), "ax1"),
    ]
    for k in range(1, CHANNELS):
        init.append(numpy_helper.from_array(np.array([k], np.int64), f"s{k}"))
        init.append(numpy_helper.from_array(np.array([k + 1], np.int64), f"e{k}"))

    nodes = []
    shifted = []
    for k in range(1, CHANNELS):
        mk = f"m{k}"
        nodes.append(n("Slice", ["input", f"s{k}", f"e{k}", "ax1"], [mk]))
        nodes += _top_nodes(n, mk, f"h{k}", f"cand{k}", f"top{k}", "")
        nodes += [
            n("Sub", ["top1", f"top{k}"], [f"delta{k}"]),
            n("Sub", ["D", f"delta{k}"], [f"diff{k}"]),
            n("Abs", [f"diff{k}"], [f"ad{k}"]),
            n("Less", [f"ad{k}", "half"], [f"ltb{k}"]),
            n("Cast", [f"ltb{k}"], [f"S{k}"], to=F),
            n("MatMul", [f"S{k}", mk], [f"sh{k}"]),
        ]
        shifted.append(f"sh{k}")

    # channel 0 (background) = 1 - sum of shifted colour channels
    acc = shifted[0]
    for j, s in enumerate(shifted[1:], 1):
        nodes.append(n("Add", [acc, s], [f"acc{j}"]))
        acc = f"acc{j}"
    # mask channel-0 by input occupancy so padding (all-zero) stays all-zero
    nodes.append(n("ReduceSum", ["input"], ["occ"], axes=[1], keepdims=1))
    nodes.append(n("Sub", ["one", acc], ["bg"]))
    nodes.append(n("Mul", ["bg", "occ"], ["ch0"]))
    nodes.append(n("Concat", ["ch0"] + shifted, ["output"], axis=1))

    graph = helper.make_graph(nodes, "align_to_anchor",
                              [helper.make_tensor_value_info("input", F, FULL)],
                              [helper.make_tensor_value_info("output", F, FULL)],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_align_to_anchor(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
