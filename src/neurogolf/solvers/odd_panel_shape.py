"""Solver: output the odd-shaped 3x3 panel (task 263).

The grid is a strip of 3x3 panels, stacked vertically (3N x 3) or horizontally
(3 x 3N).  All panels share one binary shape except a single "odd" one; the
output is that odd panel (its colour and shape).

Build: both orientations are computed and blended by an orientation flag (how
many columns carry content).  Per orientation the panels are gathered into a
(1,10,P,3,3) tensor; the per-cell majority shape over non-empty panels is found,
and the one non-empty panel that differs from the majority is selected and
emitted at the top-left.
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


def _panels(g: np.ndarray):
    H, W = g.shape
    if W == 3 and H % 3 == 0:
        return [g[3 * k:3 * k + 3, 0:3] for k in range(H // 3)]
    if H == 3 and W % 3 == 0:
        return [g[0:3, 3 * k:3 * k + 3] for k in range(W // 3)]
    return None


def _ref(g: np.ndarray) -> Optional[np.ndarray]:
    ps = _panels(g)
    if ps is None or len(ps) < 2:
        return None
    shapes = [(p != 0).astype(int) for p in ps]
    ne = [s.sum() > 0 for s in shapes]
    num = sum(ne)
    if num < 2:
        return None
    overlay = sum(s for s, e in zip(shapes, ne) if e)
    M = (overlay * 2 > num).astype(int)
    odd = [k for k in range(len(ps)) if ne[k] and (shapes[k] != M).any()]
    if len(odd) != 1:
        return None
    return ps[odd[0]]


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

    def cf(b, name):
        return n("Cast", [b], [name], to=F)

    note0 = np.ones((1, CHANNELS, 1, 1, 1), np.float32); note0[0, 0] = 0.0
    init = [
        numpy_helper.from_array(note0, "note0_5"),
        numpy_helper.from_array(np.array(0.5, np.float32), "half"),
        numpy_helper.from_array(np.array(2.0, np.float32), "two"),
        numpy_helper.from_array(np.array(1.0, np.float32), "one"),
        numpy_helper.from_array(np.array(3.5, np.float32), "threeh"),
        numpy_helper.from_array(np.array([0, 0, 0, 0], np.int64), "z4"),
        numpy_helper.from_array(np.array([1, CHANNELS, HEIGHT, 3], np.int64), "vend"),
        numpy_helper.from_array(np.array([1, CHANNELS, 3, WIDTH], np.int64), "hend"),
        numpy_helper.from_array(np.array([0, 1, 2, 3], np.int64), "ax4"),
        numpy_helper.from_array(np.array([1, CHANNELS, HEIGHT // 3, 3, 3], np.int64), "rv"),
        numpy_helper.from_array(np.array([1, CHANNELS, 3, WIDTH // 3, 3], np.int64), "rh"),
        numpy_helper.from_array(np.array([0, 0, 0, 0, 0, 0, HEIGHT - 3, WIDTH - 3], np.int64), "padout"),
        numpy_helper.from_array(np.array([0.0], np.float32), "zero"),
    ]
    nodes = [
        n("ReduceSum", ["input"], ["content"], axes=[1], keepdims=1),     # (1,1,H,W)
        n("ReduceMax", ["content"], ["colhas"], axes=[2], keepdims=1),    # (1,1,1,W)
        n("ReduceSum", ["colhas"], ["ncols"], keepdims=0),
        n("Less", ["ncols", "threeh"], ["isvert_b"]), cf("isvert_b", "isvert"),
        # vertical panels: (1,10,P,3,3)
        n("Slice", ["input", "z4", "vend", "ax4"], ["vstrip"]),
        n("Reshape", ["vstrip", "rv"], ["vpan"]),
        # horizontal panels: reshape then transpose panel axis to front
        n("Slice", ["input", "z4", "hend", "ax4"], ["hstrip"]),
        n("Reshape", ["hstrip", "rh"], ["hr"]),
        n("Transpose", ["hr"], ["hpan"], perm=[0, 1, 3, 2, 4]),
    ]

    def pipeline(pan, tag):
        nodes.extend([
            n("Mul", [pan, "note0_5"], [f"pm_{tag}"]),
            n("ReduceSum", [f"pm_{tag}"], [f"shp_{tag}"], axes=[1], keepdims=1),       # (1,1,P,3,3)
            n("ReduceMax", [f"shp_{tag}"], [f"ne_{tag}"], axes=[3, 4], keepdims=1),    # (1,1,P,1,1)
            n("ReduceSum", [f"ne_{tag}"], [f"num_{tag}"], axes=[2], keepdims=1),       # (1,1,1,1,1)
            n("ReduceSum", [f"shp_{tag}"], [f"ov_{tag}"], axes=[2], keepdims=1),       # (1,1,1,3,3)
            n("Mul", [f"ov_{tag}", "two"], [f"ov2_{tag}"]),
            n("Greater", [f"ov2_{tag}", f"num_{tag}"], [f"M_{tag}b"]), cf(f"M_{tag}b", f"M_{tag}"),
            n("Sub", [f"shp_{tag}", f"M_{tag}"], [f"sd_{tag}"]),                       # (1,1,P,3,3)
            n("Abs", [f"sd_{tag}"], [f"sda_{tag}"]),
            n("ReduceSum", [f"sda_{tag}"], [f"diff_{tag}"], axes=[3, 4], keepdims=1),  # (1,1,P,1,1)
            n("Greater", [f"diff_{tag}", "half"], [f"dh_{tag}b"]), cf(f"dh_{tag}b", f"dh_{tag}"),
            n("Mul", [f"ne_{tag}", f"dh_{tag}"], [f"odd_{tag}"]),                      # (1,1,P,1,1)
            n("Mul", [pan, f"odd_{tag}"], [f"oc5_{tag}"]),                             # (1,10,P,3,3)
            n("ReduceSum", [f"oc5_{tag}"], [f"oc_{tag}"], axes=[2], keepdims=0),       # (1,10,3,3)
        ])
        return f"oc_{tag}"

    ocv = pipeline("vpan", "v")
    och = pipeline("hpan", "h")
    nodes += [
        n("Mul", [ocv, "isvert"], ["pv"]),
        n("Sub", ["one", "isvert"], ["niv"]),
        n("Mul", [och, "niv"], ["ph"]),
        n("Add", ["pv", "ph"], ["outp"]),                                             # (1,10,3,3)
        n("Pad", ["outp", "padout", "zero"], ["output"]),                             # (1,10,30,30)
    ]
    graph = helper.make_graph(nodes, "odd_panel_shape",
                              [helper.make_tensor_value_info("input", F, FULL)],
                              [helper.make_tensor_value_info("output", F, FULL)],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_odd_panel_shape(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
