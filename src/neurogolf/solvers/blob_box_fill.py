"""Solver: fill each 4-blob's 3x3 bounding box with 7 (task 139).

Two 8-connected colour-4 blobs sit in the grid, each with a tight 3x3
bounding box.  Every background cell inside a blob's box becomes ``7``::

    4 4 4            4 4 4
    4 . 4     ->     4 7 4
    . 4 4            7 4 4

Because each box is exactly 3x3, its centre is ``(minr+1, minc+1)`` and every
blob cell lies within Chebyshev distance 1 of that centre.  So: min-flood the
row/col indices through the 8-connected blob mask (``-MaxPool(-V)`` per step,
re-masked so values can't leak through background); each blob cell then knows
its component's ``(minr, minc)``.  A cell is a box centre iff one of its nine
neighbours points at it (``minr+1 == r`` and ``minc+1 == c``), and the filled
box is the centre mask dilated once with a 3x3 ``MaxPool``.
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
NITER = 8
BIG = 1000.0

_OFFS8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    H, W = mask.shape
    seen = np.zeros((H, W), bool)
    comps = []
    for r in range(H):
        for c in range(W):
            if mask[r, c] and not seen[r, c]:
                q = deque([(r, c)])
                seen[r, c] = True
                cells = []
                while q:
                    y, x = q.popleft()
                    cells.append((y, x))
                    for dr, dc in _OFFS8:
                        yy, xx = y + dr, x + dc
                        if 0 <= yy < H and 0 <= xx < W and mask[yy, xx] and not seen[yy, xx]:
                            seen[yy, xx] = True
                            q.append((yy, xx))
                comps.append(cells)
    return comps


def _ref(g: np.ndarray) -> Optional[np.ndarray]:
    comps = _components(g == 4)
    if not comps:
        return None
    out = g.copy()
    for cells in comps:
        ys = [y for y, _ in cells]
        xs = [x for _, x in cells]
        r0, r1, c0, c1 = min(ys), max(ys), min(xs), max(xs)
        if r1 - r0 != 2 or c1 - c0 != 2:
            return None
        sub = out[r0:r1 + 1, c0:c1 + 1]
        sub[sub == 0] = 7
        out[r0:r1 + 1, c0:c1 + 1] = sub
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
    e0 = np.zeros((1, CHANNELS, 1, 1), np.float32); e0[0, 0] = 1.0
    e7 = np.zeros((1, CHANNELS, 1, 1), np.float32); e7[0, 7] = 1.0
    init = [
        numpy_helper.from_array(e0, "e0"),
        numpy_helper.from_array(e7, "e7"),
        numpy_helper.from_array(np.arange(HEIGHT, dtype=np.float32).reshape(1, 1, HEIGHT, 1), "R"),
        numpy_helper.from_array(np.arange(WIDTH, dtype=np.float32).reshape(1, 1, 1, WIDTH), "C"),
        numpy_helper.from_array(np.array(BIG, np.float32), "BIGc"),
        numpy_helper.from_array(np.array(1.0, np.float32), "one"),
        numpy_helper.from_array(np.array(0.5, np.float32), "half"),
        numpy_helper.from_array(np.array([0], np.int64), "c0s"),
        numpy_helper.from_array(np.array([1], np.int64), "c1e"),
        numpy_helper.from_array(np.array([4], np.int64), "c4s"),
        numpy_helper.from_array(np.array([5], np.int64), "c5e"),
        numpy_helper.from_array(np.array([1], np.int64), "ax1"),
        numpy_helper.from_array(np.array([2, 3], np.int64), "ax23"),
        numpy_helper.from_array(np.array([0, 0, 1, 1, 0, 0, 1, 1], np.int64), "pad1"),
    ]
    nodes = [
        n("Slice", ["input", "c4s", "c5e", "ax1"], ["m"]),
        n("Slice", ["input", "c0s", "c1e", "ax1"], ["is0"]),
        n("Sub", ["one", "m"], ["invm"]),
        n("Mul", ["invm", "BIGc"], ["biginv"]),
        n("Mul", ["R", "m"], ["vr0m"]), n("Add", ["vr0m", "biginv"], ["Vr0"]),
        n("Mul", ["C", "m"], ["vc0m"]), n("Add", ["vc0m", "biginv"], ["Vc0"]),
    ]
    for V0, pre in (("Vr0", "r"), ("Vc0", "c")):
        cur = V0
        for it in range(NITER):
            ng, mp, ps, kp, nx = (f"{pre}n{it}", f"{pre}mp{it}", f"{pre}ps{it}",
                                  f"{pre}kp{it}", f"{pre}V{it + 1}")
            nodes += [
                n("Neg", [cur], [ng]),
                n("MaxPool", [ng], [mp], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
                n("Neg", [mp], [ps]),
                n("Mul", [ps, "m"], [kp]),
                n("Add", [kp, "biginv"], [nx]),
            ]
            cur = nx
    nodes += [
        n("Add", [f"rV{NITER}", "one"], ["cr"]),
        n("Add", [f"cV{NITER}", "one"], ["cc"]),
        n("Pad", ["cr", "pad1", "BIGc"], ["crP"], mode="constant"),
        n("Pad", ["cc", "pad1", "BIGc"], ["ccP"], mode="constant"),
    ]
    acc = None
    for k, (dr, dc) in enumerate([(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)]):
        st, en = f"st{k}", f"en{k}"
        init.append(numpy_helper.from_array(np.array([1 + dr, 1 + dc], np.int64), st))
        init.append(numpy_helper.from_array(np.array([31 + dr, 31 + dc], np.int64), en))
        nodes += [
            n("Slice", ["crP", st, en, "ax23"], [f"scr{k}"]),
            n("Slice", ["ccP", st, en, "ax23"], [f"scc{k}"]),
            n("Sub", [f"scr{k}", "R"], [f"dr{k}"]), n("Abs", [f"dr{k}"], [f"ar{k}"]),
            n("Sub", [f"scc{k}", "C"], [f"dc{k}"]), n("Abs", [f"dc{k}"], [f"ac{k}"]),
            n("Add", [f"ar{k}", f"ac{k}"], [f"dist{k}"]),
            n("Less", [f"dist{k}", "half"], [f"hb{k}"]),
            n("Cast", [f"hb{k}"], [f"hit{k}"], to=F),
        ]
        if acc is None:
            acc = f"hit{k}"
        else:
            nodes.append(n("Max", [acc, f"hit{k}"], [f"acc{k}"]))
            acc = f"acc{k}"
    nodes += [
        n("MaxPool", [acc], ["bbm"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        n("Mul", ["bbm", "is0"], ["fill"]),
        n("Mul", ["fill", "e7"], ["add7"]),
        n("Mul", ["fill", "e0"], ["sub0"]),
        n("Add", ["input", "add7"], ["t1"]),
        n("Sub", ["t1", "sub0"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "blob_box_fill",
                              [helper.make_tensor_value_info("input", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              [helper.make_tensor_value_info("output", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_blob_box_fill(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
