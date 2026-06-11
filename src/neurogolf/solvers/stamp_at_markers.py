"""Solver: stamp the top-left 3x3 template at every 1-marker (task 75).

A 3x3 colour template sits at the top-left corner; a colour-5 divider column
separates it from a field that holds ``1`` markers.  Each marker is replaced by
a copy of the template, centred on the marker; the markers themselves vanish::

    T T T | . . . . .          T T T | . . . . .
    T T T | . 1 . . .   ->     T T T | T T T . .
    T T T | . . . . .          T T T | T T T . .

For each of the 9 template offsets ``(dr, dc)`` the marker mask is shifted by
``(dr, dc)`` and multiplied by that template cell's colour; summing the nine
colour-weighted, shifted marker masks paints every stamp at once.
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
    if H < 3 or W < 3:
        return None
    tmpl = g[0:3, 0:3].copy()
    out = g.copy()
    out[out == 1] = 0
    ys, xs = np.where(g == 1)
    if len(ys) == 0:
        return None
    for r, c in zip(ys, xs):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < H and 0 <= cc < W:
                    out[rr, cc] = tmpl[1 + dr, 1 + dc]
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
    e1me0 = np.zeros((1, CHANNELS, 1, 1), np.float32); e1me0[0, 1] = 1.0; e1me0[0, 0] = -1.0
    init = [
        numpy_helper.from_array(e1me0, "e1me0"),
        numpy_helper.from_array(np.array(1.0, np.float32), "one"),
        numpy_helper.from_array(np.array([1], np.int64), "c1"),
        numpy_helper.from_array(np.array([2], np.int64), "c2"),
        numpy_helper.from_array(np.array([1], np.int64), "ax1"),
        numpy_helper.from_array(np.array([2, 3], np.int64), "ax23"),
        numpy_helper.from_array(np.array([0, 0, 1, 1, 0, 0, 1, 1], np.int64), "padm"),
    ]
    nodes = [
        n("Slice", ["input", "c1", "c2", "ax1"], ["is1"]),
        n("Pad", ["is1", "padm"], ["padM"], mode="constant"),       # (1,1,32,32)
        n("Mul", ["is1", "e1me0"], ["mrm"]),
        n("Sub", ["input", "mrm"], ["cleared"]),                    # markers -> background
    ]
    contribs, smasks = [], []
    k = 0
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            ms, me = f"ms{k}", f"me{k}"
            ts, te = f"ts{k}", f"te{k}"
            init.append(numpy_helper.from_array(np.array([1 - dr, 1 - dc], np.int64), ms))
            init.append(numpy_helper.from_array(np.array([31 - dr, 31 - dc], np.int64), me))
            init.append(numpy_helper.from_array(np.array([1 + dr, 1 + dc], np.int64), ts))
            init.append(numpy_helper.from_array(np.array([2 + dr, 2 + dc], np.int64), te))
            sm, tc, cb = f"sm{k}", f"tc{k}", f"cb{k}"
            nodes.append(n("Slice", ["padM", ms, me, "ax23"], [sm]))    # shifted marker
            nodes.append(n("Slice", ["input", ts, te, "ax23"], [tc]))   # template colour
            nodes.append(n("Mul", [sm, tc], [cb]))
            contribs.append(cb); smasks.append(sm)
            k += 1
    # stamp layer = sum of contributions ; stamp mask = sum of shifted markers
    acc = contribs[0]
    for j, cb in enumerate(contribs[1:], 1):
        nodes.append(n("Add", [acc, cb], [f"sl{j}"])); acc = f"sl{j}"
    mracc = smasks[0]
    for j, sm in enumerate(smasks[1:], 1):
        nodes.append(n("Add", [mracc, sm], [f"mk{j}"])); mracc = f"mk{j}"
    nodes += [
        n("Sub", ["one", mracc], ["inv"]),
        n("Mul", ["cleared", "inv"], ["base"]),
        n("Add", ["base", acc], ["output"]),
    ]
    graph = helper.make_graph(nodes, "stamp_at_markers",
                              [helper.make_tensor_value_info("input", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              [helper.make_tensor_value_info("output", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_stamp_at_markers(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
