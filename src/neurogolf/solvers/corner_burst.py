"""Solver: a 2-marker bursts into four fixed-colour diagonal corners (task 266).

A single ``2`` cell is replaced by four cells on its diagonals -- up-left ``3``,
up-right ``6``, down-left ``8``, down-right ``7`` -- and the ``2`` itself is
cleared to background (corners that fall off the grid are dropped)::

    . . .          3 . 6
    . 2 .    ->    . . .
    . . .          8 . 7

Build: ``is2`` = channel-2 mask; four diagonal Pad+Slice shifts place each corner
colour (clipped to the real grid), and the ``2`` cell is reset to ``e_0``.
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
    ys, xs = np.where(g == 2)
    if len(ys) == 0:
        return None
    out = g.copy()
    for r, c in zip(ys, xs):
        out[r, c] = 0
        for dr, dc, col in ((-1, -1, 3), (-1, 1, 6), (1, -1, 8), (1, 1, 7)):
            R, C = r + dr, c + dc
            if 0 <= R < H and 0 <= C < W:
                out[R, C] = col
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

    def onehot(k):
        v = np.zeros((1, CHANNELS, 1, 1), np.float32); v[0, k] = 1.0
        return v

    init = [
        numpy_helper.from_array(onehot(0), "e0"),
        numpy_helper.from_array(onehot(3), "e3"),
        numpy_helper.from_array(onehot(6), "e6"),
        numpy_helper.from_array(onehot(7), "e7"),
        numpy_helper.from_array(onehot(8), "e8"),
        numpy_helper.from_array(np.array(1.0, np.float32), "one"),
        numpy_helper.from_array(np.array([2], np.int64), "c2s"),
        numpy_helper.from_array(np.array([3], np.int64), "c3e"),
        numpy_helper.from_array(np.array([1], np.int64), "ax1"),
        numpy_helper.from_array(np.array([2, 3], np.int64), "ax23"),
    ]
    nodes = []
    seen = {"pad": set(), "sl": set()}
    ctr = [0]

    def read_shift(x, ar, ac):
        """y[r,c] = x[r-ar, c-ac] (zero-filled)."""
        pt, pl, pb, pr = max(ar, 0), max(ac, 0), max(-ar, 0), max(-ac, 0)
        pname = f"pad_{ar}_{ac}"
        if pname not in seen["pad"]:
            init.append(numpy_helper.from_array(
                np.array([0, 0, pt, pl, 0, 0, pb, pr], np.int64), pname))
            seen["pad"].add(pname)
        rs, cs = max(-ar, 0), max(-ac, 0)
        sname, ename = f"sst_{rs}_{cs}", f"sen_{rs}_{cs}"
        if sname not in seen["sl"]:
            init.append(numpy_helper.from_array(np.array([rs, cs], np.int64), sname))
            init.append(numpy_helper.from_array(
                np.array([rs + HEIGHT, cs + WIDTH], np.int64), ename))
            seen["sl"].add(sname)
        ctr[0] += 1
        pid, oid = f"ps{ctr[0]}", f"rs{ctr[0]}"
        nodes.append(n("Pad", [x, pname], [pid], mode="constant"))
        nodes.append(n("Slice", [pid, sname, ename, "ax23"], [oid]))
        return oid

    nodes += [
        n("ReduceSum", ["input"], ["content"], axes=[1], keepdims=1),
        n("Slice", ["input", "c2s", "c3e", "ax1"], ["is2"]),
        # clear the 2 -> background
        n("Sub", ["one", "is2"], ["not2"]),
        n("Mul", ["input", "not2"], ["base0"]),
        n("Mul", ["e0", "is2"], ["addbg"]),
        n("Add", ["base0", "addbg"], ["base"]),
    ]
    ul = read_shift("is2", -1, -1)   # up-left corner cell -> 3
    ur = read_shift("is2", -1, 1)    # up-right -> 6
    dl = read_shift("is2", 1, -1)    # down-left -> 8
    dr = read_shift("is2", 1, 1)     # down-right -> 7
    nodes += [
        n("Mul", [ul, "content"], ["p_ul"]),
        n("Mul", [ur, "content"], ["p_ur"]),
        n("Mul", [dl, "content"], ["p_dl"]),
        n("Mul", [dr, "content"], ["p_dr"]),
        n("Mul", ["e3", "p_ul"], ["a_ul"]),
        n("Mul", ["e6", "p_ur"], ["a_ur"]),
        n("Mul", ["e8", "p_dl"], ["a_dl"]),
        n("Mul", ["e7", "p_dr"], ["a_dr"]),
        n("Add", ["a_ul", "a_ur"], ["a1"]), n("Add", ["a_dl", "a_dr"], ["a2"]),
        n("Add", ["a1", "a2"], ["addc"]),
        n("Add", ["p_ul", "p_ur"], ["s1"]), n("Add", ["p_dl", "p_dr"], ["s2"]),
        n("Add", ["s1", "s2"], ["csum"]),
        n("Sub", ["one", "csum"], ["keep"]),
        n("Mul", ["base", "keep"], ["kept"]),
        n("Add", ["kept", "addc"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "corner_burst",
                              [helper.make_tensor_value_info("input", F, FULL)],
                              [helper.make_tensor_value_info("output", F, FULL)],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_corner_burst(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
