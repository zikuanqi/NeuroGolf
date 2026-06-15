"""Solver: 2-markers pierce their bounding 8-lines with 3x3 boxes (task 379).

The grid holds full straight ``8``-lines (all rows or all columns).  Each ``2``
shoots a ray to the nearest line on each side (the bounding lines of its band);
the ray is painted ``2``, each pierced line bulges into a 3x3 box of ``8`` and
the pierce cell itself is ``2``.

The ray is found with a prefix line-count (`CumSum`): a cell is in the ray if it
is in the marker's band (same cumulative line count) on the correct side, plus
the adjacent line cell, gated by whether a line actually exists on that side.
Both orientations are handled by also running on the transpose and blending by a
"has a full 8-row" gate.
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
    rows8 = [r for r in range(H) if np.all(g[r] == 8)]
    cols8 = [c for c in range(W) if np.all(g[:, c] == 8)]
    twos = list(zip(*np.where(g == 2)))
    if not twos or (rows8 and cols8) or (not rows8 and not cols8):
        return None
    out = g.copy()
    if rows8:
        for mr, mc in twos:
            tgt = []
            below = [L for L in rows8 if L > mr]
            above = [L for L in rows8 if L < mr]
            if below:
                tgt.append(min(below))
            if above:
                tgt.append(max(above))
            for LR in tgt:
                out[min(mr, LR):max(mr, LR) + 1, mc] = 2
            for LR in tgt:
                out[max(0, LR - 1):LR + 2, max(0, mc - 1):mc + 2] = 8
                out[LR, mc] = 2
    else:
        for mr, mc in twos:
            tgt = []
            right = [L for L in cols8 if L > mc]
            left = [L for L in cols8 if L < mc]
            if right:
                tgt.append(min(right))
            if left:
                tgt.append(max(left))
            for LC in tgt:
                out[mr, min(mc, LC):max(mc, LC) + 1] = 2
            for LC in tgt:
                out[max(0, mr - 1):mr + 2, max(0, LC - 1):LC + 2] = 8
                out[mr, LC] = 2
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


def _ev(k):
    v = np.zeros((1, CHANNELS, 1, 1), np.float32); v[0, k] = 1.0
    return v


def _build() -> onnx.ModelProto:
    n = helper.make_node
    rowidx = np.arange(HEIGHT, dtype=np.float32).reshape(1, 1, HEIGHT, 1)
    init = [
        numpy_helper.from_array(_ev(0), "e0"),
        numpy_helper.from_array(_ev(2), "e2"),
        numpy_helper.from_array(_ev(8), "e8"),
        numpy_helper.from_array(rowidx, "rowidx"),
        numpy_helper.from_array(np.array(0.5, np.float32), "half"),
        numpy_helper.from_array(np.array(-0.5, np.float32), "neghalf"),
        numpy_helper.from_array(np.array(1.0, np.float32), "one"),
        numpy_helper.from_array(np.array(2, np.int64), "axis2"),
        numpy_helper.from_array(np.array([0], np.int64), "c0s"),
        numpy_helper.from_array(np.array([1], np.int64), "c1"),
        numpy_helper.from_array(np.array([2], np.int64), "c2"),
        numpy_helper.from_array(np.array([3], np.int64), "c3"),
        numpy_helper.from_array(np.array([8], np.int64), "c8"),
        numpy_helper.from_array(np.array([9], np.int64), "c9"),
        numpy_helper.from_array(np.array([1], np.int64), "ax1"),
    ]
    nodes: list = []
    lineRow_names = {}

    def emit(p: str, src: str) -> str:
        def nm(s):
            return f"{p}_{s}"
        nd = [
            n("ReduceSum", [src], [nm("occ")], axes=[1], keepdims=1),
            n("Slice", [src, "c2", "c3", "ax1"], [nm("is2")]),
            n("Slice", [src, "c8", "c9", "ax1"], [nm("is8")]),
            # a "line row" = every grid cell in the row is 8 (count8 == row width)
            n("ReduceSum", [nm("is8")], [nm("rs8")], axes=[3], keepdims=1),
            n("ReduceSum", [nm("occ")], [nm("rsOcc")], axes=[3], keepdims=1),
            n("Sub", [nm("rs8"), nm("rsOcc")], [nm("rsd")]),
            n("Mul", [nm("rsd"), nm("rsd")], [nm("rsd2")]),
            n("Less", [nm("rsd2"), "half"], [nm("eqln_b")]), n("Cast", [nm("eqln_b")], [nm("eqln")], to=F),
            n("Greater", [nm("rsOcc"), "half"], [nm("hasc_b")]), n("Cast", [nm("hasc_b")], [nm("hasc")], to=F),
            n("Mul", [nm("eqln"), nm("hasc")], [nm("lineRow")]),   # (1,1,H,1)
            n("CumSum", [nm("lineRow"), "axis2"], [nm("DL")]),
            n("ReduceMax", [nm("DL")], [nm("total")], axes=[2], keepdims=1),       # (1,1,1,1)
            n("Mul", [nm("is2"), nm("DL")], [nm("is2DL")]),
            n("ReduceSum", [nm("is2DL")], [nm("DLm")], axes=[2], keepdims=1),      # (1,1,1,W) DL at marker
            n("Mul", [nm("is2"), "rowidx"], [nm("is2row")]),
            n("ReduceSum", [nm("is2row")], [nm("mrVal")], axes=[2], keepdims=1),   # (1,1,1,W)
            n("ReduceMax", [nm("is2")], [nm("colHas")], axes=[2], keepdims=1),     # (1,1,1,W)
            # side masks
            n("Sub", ["rowidx", nm("mrVal")], [nm("rdiff")]),                      # (1,1,H,W)
            n("Greater", [nm("rdiff"), "neghalf"], [nm("ge_b")]), n("Cast", [nm("ge_b")], [nm("ge")], to=F),
            n("Less", [nm("rdiff"), "half"], [nm("le_b")]), n("Cast", [nm("le_b")], [nm("le")], to=F),
            # band membership: DL == DLm
            n("Sub", [nm("DL"), nm("DLm")], [nm("dband")]),                        # (1,1,H,W)
            n("Mul", [nm("dband"), nm("dband")], [nm("dband2")]),
            n("Less", [nm("dband2"), "half"], [nm("eqM_b")]), n("Cast", [nm("eqM_b")], [nm("eqM")], to=F),
            # below-line cell: DL == DLm+1 and lineRow
            n("Sub", [nm("dband"), "one"], [nm("dbm1")]),
            n("Mul", [nm("dbm1"), nm("dbm1")], [nm("dbm1sq")]),
            n("Less", [nm("dbm1sq"), "half"], [nm("eqP1_b")]), n("Cast", [nm("eqP1_b")], [nm("eqP1")], to=F),
            n("Mul", [nm("eqP1"), nm("lineRow")], [nm("blcell")]),
            # exists gates (1,1,1,W)
            n("Sub", [nm("total"), nm("DLm")], [nm("rem")]),
            n("Greater", [nm("rem"), "half"], [nm("belowEx_b")]), n("Cast", [nm("belowEx_b")], [nm("belowEx")], to=F),
            n("Greater", [nm("DLm"), "half"], [nm("aboveEx_b")]), n("Cast", [nm("aboveEx_b")], [nm("aboveEx")], to=F),
            # ray down = (eqM*ge  OR  blcell*ge) * belowEx * colHas
            n("Mul", [nm("eqM"), nm("ge")], [nm("rdb")]),
            n("Mul", [nm("blcell"), nm("ge")], [nm("rdl")]),
            n("Max", [nm("rdb"), nm("rdl")], [nm("rd0")]),
            n("Mul", [nm("rd0"), nm("belowEx")], [nm("rd1")]),
            n("Mul", [nm("rd1"), nm("colHas")], [nm("rayD")]),
            # ray up = eqM*le * aboveEx * colHas
            n("Mul", [nm("eqM"), nm("le")], [nm("ru0")]),
            n("Mul", [nm("ru0"), nm("aboveEx")], [nm("ru1")]),
            n("Mul", [nm("ru1"), nm("colHas")], [nm("rayU")]),
            n("Max", [nm("rayD"), nm("rayU")], [nm("rayDU")]),
            n("Max", [nm("rayDU"), nm("is2")], [nm("rayRaw")]),
            n("Mul", [nm("rayRaw"), nm("occ")], [nm("ray")]),
            # pierce / box / center
            n("Mul", [nm("ray"), nm("lineRow")], [nm("pierce")]),
            n("MaxPool", [nm("pierce")], [nm("boxR")], kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]),
            n("Mul", [nm("boxR"), nm("occ")], [nm("box")]),
            n("Sub", [nm("box"), nm("pierce")], [nm("boxNC")]),
            n("Mul", [nm("ray"), nm("box")], [nm("rayBox")]),
            n("Sub", [nm("ray"), nm("rayBox")], [nm("rayOnly")]),
            n("Sub", [nm("is8"), nm("pierce")], [nm("is8mC")]),
            n("Add", [nm("pierce"), nm("rayOnly")], [nm("twoM")]),
            n("Max", [nm("boxNC"), nm("is8mC")], [nm("eightM")]),
            n("Add", [nm("twoM"), nm("eightM")], [nm("nonbg")]),
            n("Sub", [nm("occ"), nm("nonbg")], [nm("bgM")]),
            n("Mul", [nm("twoM"), "e2"], [nm("L2")]),
            n("Mul", [nm("eightM"), "e8"], [nm("L8")]),
            n("Mul", [nm("bgM"), "e0"], [nm("L0")]),
            n("Add", [nm("L2"), nm("L8")], [nm("t1")]),
            n("Add", [nm("t1"), nm("L0")], [nm("out")]),
        ]
        nodes.extend(nd)
        lineRow_names[p] = nm("lineRow")
        return nm("out")

    outH = emit("h", "input")
    nodes.append(n("Transpose", ["input"], ["tin"], perm=[0, 1, 3, 2]))
    outT = emit("v", "tin")
    nodes.append(n("Transpose", [outT], ["outV"], perm=[0, 1, 3, 2]))
    # gate: does the input have a full 8-row?
    nodes += [
        n("ReduceMax", [lineRow_names["h"]], ["gate"], axes=[2], keepdims=1),  # (1,1,1,1)
        n("Sub", ["one", "gate"], ["ngate"]),
        n("Mul", [outH, "gate"], ["oH"]),
        n("Mul", ["outV", "ngate"], ["oV"]),
        n("Add", ["oH", "oV"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "line_pierce_box",
                              [helper.make_tensor_value_info("input", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              [helper.make_tensor_value_info("output", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_line_pierce_box(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
