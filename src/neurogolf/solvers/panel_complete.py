"""Solver: complete every panel to the union template (task 33).

A 17x17 grid is split by coloured divider lines (rows/cols 5 and 11) into a
3x3 array of 5x5 panels.  Across the panels the same shape appears, but each
copy may be partial.  The complete template is the union (OR) of all nine
panels; every panel is then completed - the cells present in the template but
missing in that panel are painted with the **divider colour**, while the cells
already drawn keep their original colour::

    template (union)      a partial panel  ->  completed (missing cells = D)
       . X .                  . X .                . X .
       X X X                  X . X                X D X
       . X .                  . X .                . X .

The divider layout is fixed (always rows/cols 5 & 11), so the panels are tiled
with a padded 6x6 template; the fill is masked to background cells only.
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

_BANDS = [(0, 5), (6, 11), (12, 17)]


def _ref(g: np.ndarray) -> Optional[np.ndarray]:
    if g.shape != (17, 17):
        return None
    d = g[5, 0]
    if not (np.all(g[5] == d) and np.all(g[11] == d)
            and np.all(g[:, 5] == d) and np.all(g[:, 11] == d)):
        return None
    tmpl = np.zeros((5, 5), bool)
    for r0, r1 in _BANDS:
        for c0, c1 in _BANDS:
            tmpl |= g[r0:r1, c0:c1] != 0
    out = g.copy()
    for r0, r1 in _BANDS:
        for c0, c1 in _BANDS:
            p = out[r0:r1, c0:c1]
            p[tmpl & (p == 0)] = d
            out[r0:r1, c0:c1] = p
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
    init = [
        numpy_helper.from_array(e0, "e0"),
        numpy_helper.from_array(np.array(1.0, np.float32), "one"),
        numpy_helper.from_array(np.array([0], np.int64), "c0"),
        numpy_helper.from_array(np.array([1], np.int64), "c1"),
        numpy_helper.from_array(np.array([5], np.int64), "five"),
        numpy_helper.from_array(np.array([6], np.int64), "six"),
        numpy_helper.from_array(np.array([1], np.int64), "ax1"),
        numpy_helper.from_array(np.array([2], np.int64), "ax2"),
        numpy_helper.from_array(np.array([3], np.int64), "ax3"),
        numpy_helper.from_array(np.array([2, 3], np.int64), "ax23"),
        numpy_helper.from_array(np.array([0, 0, 0, 0, 0, 0, 1, 1], np.int64), "pad6"),
        numpy_helper.from_array(np.array([1, 1, 5, 5], np.int64), "rep"),
    ]
    nodes = [
        n("Slice", ["input", "c0", "c1", "ax1"], ["is0"]),
        n("Sub", ["one", "is0"], ["sm"]),            # shape mask (non-bg)
    ]
    panels = []
    for r0, r1 in _BANDS:
        for c0, c1 in _BANDS:
            st, en, nm = f"st{r0}_{c0}", f"en{r0}_{c0}", f"p{r0}_{c0}"
            init.append(numpy_helper.from_array(np.array([r0, c0], np.int64), st))
            init.append(numpy_helper.from_array(np.array([r1, c1], np.int64), en))
            nodes.append(n("Slice", ["sm", st, en, "ax23"], [nm]))
            panels.append(nm)
    acc = panels[0]
    for j, p in enumerate(panels[1:], 1):
        nodes.append(n("Max", [acc, p], [f"tm{j}"]))
        acc = f"tm{j}"
    nodes += [
        n("Pad", [acc, "pad6"], ["tp"], mode="constant"),     # (1,1,6,6)
        n("Tile", ["tp", "rep"], ["tfull"]),                  # (1,1,30,30)
        n("Mul", ["tfull", "is0"], ["fill"]),                 # background cells only
        n("Slice", ["input", "five", "six", "ax2"], ["drow"]),     # row 5
        n("Slice", ["drow", "c0", "c1", "ax3"], ["eD"]),           # col 0 -> (1,10,1,1)
        n("Mul", ["fill", "eD"], ["addD"]),
        n("Mul", ["fill", "e0"], ["subBg"]),
        n("Add", ["input", "addD"], ["t1"]),
        n("Sub", ["t1", "subBg"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "panel_complete",
                              [helper.make_tensor_value_info("input", F, FULL)],
                              [helper.make_tensor_value_info("output", F, FULL)],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_panel_complete(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
