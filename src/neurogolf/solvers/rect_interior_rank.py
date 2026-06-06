"""Solver: recolour two stacked rectangles' interiors by size rank (task 156).

Two solid rectangles sit one above the other.  Each rectangle's one-cell-eroded
interior is repainted: the smaller rectangle's interior becomes colour 1, the
larger's becomes colour 2; borders and background are untouched.

Build: the interior mask is the 4-neighbour same-colour erosion (as in
`interior_recolor`).  Per-row foreground counts give a `rowhas` indicator; a
`CumSum` over its 0->1 transitions labels the top band (1) and bottom band (2),
so each band's area is a masked `ReduceSum`.  Comparing the two areas selects
which colour fills which band.  Detection requires exactly two row-separable
rectangles with distinct areas.
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
FULL = [1, CHANNELS, HEIGHT, WIDTH]


def _components(fg: np.ndarray):
    H, W = fg.shape
    lab = np.zeros((H, W), int)
    comps = []
    k = 0
    for r in range(H):
        for c in range(W):
            if fg[r, c] and lab[r, c] == 0:
                k += 1
                cells = [(r, c)]
                lab[r, c] = k
                dq = deque([(r, c)])
                while dq:
                    y, x = dq.popleft()
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < H and 0 <= nx < W and fg[ny, nx] and lab[ny, nx] == 0:
                            lab[ny, nx] = k
                            dq.append((ny, nx))
                            cells.append((ny, nx))
                comps.append(cells)
    return comps


def _ref(g: np.ndarray) -> Optional[np.ndarray]:
    H, W = g.shape
    comps = _components(g != 0)
    if len(comps) != 2:
        return None
    info = []
    for cells in comps:
        ys = [y for y, _ in cells]
        xs = [x for _, x in cells]
        info.append((len(cells), min(ys), max(ys), min(xs), max(xs), cells))
    a, b = info
    # row-separable (one strictly above the other)
    if not (a[2] < b[1] or b[2] < a[1]):
        return None
    if a[0] == b[0]:
        return None
    out = g.copy()
    for area, y0, y1, x0, x1, cells in info:
        colour = 1 if area == min(a[0], b[0]) else 2
        cset = set(cells)
        for (y, x) in cells:
            if all((y + dy, x + dx) in cset for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1))):
                out[y, x] = colour
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


def _build() -> onnx.ModelProto:
    F = TensorProto.FLOAT
    n = helper.make_node
    e1 = np.zeros((1, CHANNELS, 1, 1), np.float32); e1[0, 1] = 1.0
    e2 = np.zeros((1, CHANNELS, 1, 1), np.float32); e2[0, 2] = 1.0
    init = [
        numpy_helper.from_array(e1, "e1"),
        numpy_helper.from_array(e2, "e2"),
        numpy_helper.from_array(np.array(1.0, np.float32), "one"),
        numpy_helper.from_array(np.array(0.5, np.float32), "half"),
        numpy_helper.from_array(np.array(1.5, np.float32), "oneh"),
        numpy_helper.from_array(np.array(2, np.int64), "cumax"),
        numpy_helper.from_array(np.array([1], np.int64), "st1"),
        numpy_helper.from_array(np.array([0.0], np.float32), "zero"),
        numpy_helper.from_array(np.array([0, 0, 1, 0, 0, 0, 0, 0], np.int64), "pad_prev"),
        numpy_helper.from_array(np.array([0], np.int64), "h0"),
        numpy_helper.from_array(np.array([HEIGHT], np.int64), "hH"),
        numpy_helper.from_array(np.array([2], np.int64), "axr"),
    ]
    # 4-neighbour same-colour erosion -> interior mask
    shifts = {
        "up":    ([0, 0, 1, 0, 0, 0, 0, 0], [0], [HEIGHT], 2),
        "down":  ([0, 0, 0, 0, 0, 0, 1, 0], [1], [HEIGHT + 1], 2),
        "left":  ([0, 0, 0, 1, 0, 0, 0, 0], [0], [WIDTH], 3),
        "right": ([0, 0, 0, 0, 0, 0, 0, 1], [1], [WIDTH + 1], 3),
    }
    nodes = []
    sames = []
    for name, (pads, slo, shi, ax) in shifts.items():
        init.append(numpy_helper.from_array(np.array(pads, np.int64), f"pad_{name}"))
        init.append(numpy_helper.from_array(np.array(slo, np.int64), f"slo_{name}"))
        init.append(numpy_helper.from_array(np.array(shi, np.int64), f"shi_{name}"))
        init.append(numpy_helper.from_array(np.array([ax], np.int64), f"ax_{name}"))
        nodes.append(n("Pad", ["input", f"pad_{name}", "zero"], [f"p_{name}"]))
        nodes.append(n("Slice", [f"p_{name}", f"slo_{name}", f"shi_{name}",
                                 f"ax_{name}", "st1"], [f"nb_{name}"]))
        nodes.append(n("Mul", ["input", f"nb_{name}"], [f"m_{name}"]))
        nodes.append(n("ReduceSum", [f"m_{name}"], [f"same_{name}"], axes=[1], keepdims=1))
        sames.append(f"same_{name}")
    nodes += [
        n("Mul", [sames[0], sames[1]], ["i1"]),
        n("Mul", [sames[2], sames[3]], ["i2"]),
        n("Mul", ["i1", "i2"], ["interior_raw"]),                   # (1,1,H,W) incl. background
        n("ReduceSum", ["input"], ["content"], axes=[1], keepdims=1),
        n("Slice", ["input", "h0c", "h0e", "ax4"], ["ch0"]),
        n("Sub", ["content", "ch0"], ["fg"]),                       # rectangle mask (1,1,H,W)
        n("Mul", ["interior_raw", "fg"], ["interior"]),             # restrict to foreground
        # per-row foreground indicator
        n("ReduceSum", ["fg"], ["rowsum"], axes=[3], keepdims=1),   # (1,1,H,1)
        n("Greater", ["rowsum", "half"], ["rowhas_b"]),
        n("Cast", ["rowhas_b"], ["rowhas"], to=F),
        # previous row (shift down by 1 along axis 2)
        n("Pad", ["rowhas", "pad_prev", "zero"], ["rh_pad"]),
        n("Slice", ["rh_pad", "h0", "hH", "axr", "st1"], ["rowprev"]),
        n("Sub", ["one", "rowprev"], ["notprev"]),
        n("Mul", ["rowhas", "notprev"], ["trans"]),                 # 0->1 transitions
        n("CumSum", ["trans", "cumax"], ["bandid"]),                # 1 in top band, 2 in bottom
        # top band: bandid == 1  (|bandid-1|<0.5  ==  bandid<1.5 and bandid>0.5)
        n("Less", ["bandid", "oneh"], ["lt15"]),
        n("Cast", ["lt15"], ["lt15f"], to=F),
        n("Mul", ["rowhas", "lt15f"], ["istop"]),                   # (1,1,H,1)
        n("Sub", ["rowhas", "istop"], ["isbot"]),                   # remaining band
        # band areas
        n("Mul", ["fg", "istop"], ["fgtop"]),
        n("ReduceSum", ["fgtop"], ["atop"], keepdims=0),            # scalar
        n("Mul", ["fg", "isbot"], ["fgbot"]),
        n("ReduceSum", ["fgbot"], ["abot"], keepdims=0),
        n("Less", ["atop", "abot"], ["topsm_b"]),
        n("Cast", ["topsm_b"], ["topsm"], to=F),                    # scalar 1 if top smaller
        n("Sub", ["one", "topsm"], ["topbig"]),
        # fill colours per band
        n("Mul", ["e1", "topsm"], ["t1"]), n("Mul", ["e2", "topbig"], ["t2"]),
        n("Add", ["t1", "t2"], ["topfill"]),                        # (1,10,1,1)
        n("Mul", ["e2", "topsm"], ["b2"]), n("Mul", ["e1", "topbig"], ["b1"]),
        n("Add", ["b2", "b1"], ["botfill"]),
        # paint
        n("Mul", ["interior", "istop"], ["itop"]),                  # (1,1,H,W)
        n("Mul", ["interior", "isbot"], ["ibot"]),
        n("Sub", ["one", "interior"], ["keep"]),
        n("Mul", ["input", "keep"], ["kept"]),
        n("Mul", ["topfill", "itop"], ["ptop"]),
        n("Mul", ["botfill", "ibot"], ["pbot"]),
        n("Add", ["kept", "ptop"], ["s1"]),
        n("Add", ["s1", "pbot"], ["output"]),
    ]
    init.append(numpy_helper.from_array(np.array([0, 0, 0, 0], np.int64), "h0c"))
    init.append(numpy_helper.from_array(np.array([1, 1, HEIGHT, WIDTH], np.int64), "h0e"))
    init.append(numpy_helper.from_array(np.array([0, 1, 2, 3], np.int64), "ax4"))
    graph = helper.make_graph(nodes, "rect_interior_rank",
                              [helper.make_tensor_value_info("input", F, FULL)],
                              [helper.make_tensor_value_info("output", F, FULL)],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_rect_interior_rank(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
