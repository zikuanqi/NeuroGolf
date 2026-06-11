"""Solver: fill the panel(s) holding the most markers (task 59).

An 11x11 grid is split by colour-5 lines (rows/cols 3 and 7) into a 3x3 array
of 3x3 panels.  Scattered marker cells (the single non-bg, non-5 colour) are
distributed across the panels; the output clears every marker and fills the
panel(s) containing the **most** markers solid with the marker colour (ties
fill all the winners)::

    markers per panel  ->  the max-count panel becomes a solid block

The panel layout is fixed, so per-panel counts come from a stride-4 ``Conv``
with a 3x3 ones kernel, and the winning flags are expanded back to a fill mask
with a matching stride-4 ``ConvTranspose``.
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
_BANDS = [(0, 3), (4, 7), (8, 11)]


def _ref(g: np.ndarray) -> Optional[np.ndarray]:
    if g.shape != (11, 11):
        return None
    if not (np.all(g[3] == 5) and np.all(g[7] == 5)
            and np.all(g[:, 3] == 5) and np.all(g[:, 7] == 5)):
        return None
    marks = [v for v in np.unique(g) if v not in (0, 5)]
    if len(marks) != 1:
        return None
    m = marks[0]
    cnt = {}
    for r0, r1 in _BANDS:
        for c0, c1 in _BANDS:
            cnt[(r0, c0)] = int((g[r0:r1, c0:c1] == m).sum())
    mx = max(cnt.values())
    out = g.copy()
    out[out == m] = 0
    if mx > 0:
        for (r0, c0), v in cnt.items():
            if v == mx:
                out[r0:r0 + 3, c0:c0 + 3] = m
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
    ones3 = np.ones((1, 1, 3, 3), np.float32)
    init = [
        numpy_helper.from_array(e0, "e0"),
        numpy_helper.from_array(ones3, "ones3"),
        numpy_helper.from_array(np.array(1.0, np.float32), "one"),
        numpy_helper.from_array(np.array(0.5, np.float32), "half"),
        numpy_helper.from_array(np.array([0], np.int64), "c0"),
        numpy_helper.from_array(np.array([1], np.int64), "c1"),
        numpy_helper.from_array(np.array([5], np.int64), "c5"),
        numpy_helper.from_array(np.array([6], np.int64), "c6"),
        numpy_helper.from_array(np.array([1], np.int64), "ax1"),
        numpy_helper.from_array(np.array([0, 0, 0, 0, 0, 0, 3, 3], np.int64), "padTo30"),
    ]
    nodes = [
        n("ReduceSum", ["input"], ["occ"], axes=[1], keepdims=1),
        n("Slice", ["input", "c0", "c1", "ax1"], ["is0"]),
        n("Slice", ["input", "c5", "c6", "ax1"], ["is5"]),
        n("Sub", ["occ", "is0"], ["t0"]),
        n("Sub", ["t0", "is5"], ["mark"]),                       # marker mask
        n("Mul", ["input", "mark"], ["prod"]),
        n("ReduceMax", ["prod"], ["eM"], axes=[2, 3], keepdims=1),  # marker colour one-hot
        n("Conv", ["mark", "ones3"], ["counts"], strides=[4, 4]),   # (1,1,7,7) panel sums
        n("ReduceMax", ["counts"], ["mx"], axes=[2, 3], keepdims=1),
        n("Sub", ["mx", "half"], ["mx_h"]),
        n("Greater", ["counts", "mx_h"], ["gt"]), n("Cast", ["gt"], ["flags0"], to=F),
        n("Greater", ["mx", "half"], ["mg"]), n("Cast", ["mg"], ["maxgate"], to=F),
        n("Mul", ["flags0", "maxgate"], ["flags"]),
        n("ConvTranspose", ["flags", "ones3"], ["fill27"], strides=[4, 4]),  # (1,1,27,27)
        n("Pad", ["fill27", "padTo30"], ["fillMask"], mode="constant"),
        n("Mul", ["mark", "eM"], ["mm_eM"]),
        n("Mul", ["mark", "e0"], ["mm_e0"]),
        n("Sub", ["input", "mm_eM"], ["c_tmp"]),
        n("Add", ["c_tmp", "mm_e0"], ["cleared"]),               # markers -> background
        n("Sub", ["one", "fillMask"], ["inv"]),
        n("Mul", ["cleared", "inv"], ["a"]),
        n("Mul", ["eM", "fillMask"], ["b"]),
        n("Add", ["a", "b"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "panel_max_fill",
                              [helper.make_tensor_value_info("input", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              [helper.make_tensor_value_info("output", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_panel_max_fill(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
