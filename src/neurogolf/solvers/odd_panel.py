"""Solver: output the odd-one-out among four panels (task 207).

The input divides into a 2x2 arrangement of equal panels — either an exact
2-split (H = 2h, W = 2w) or a separated split with a blank/uniform middle row
and column (H = 2h+1, W = 2w+1). Three of the four panels are identical; the
output is the unique fourth one.

This static version bakes the four panel corners (so it needs a constant input
shape). It extracts the panels with `Slice`, compares every pair via
`Sub`/`Abs`/`ReduceSum`, marks the panel that agrees with none of the others
(`Less(agree, 0.5)`), and emits it as a weighted sum, padded back to 30x30.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from ..grids import CHANNELS, HEIGHT, WIDTH, all_examples

OPSET = 11
IR_VERSION = 8


def _panels(g: np.ndarray, oh: int, ow: int):
    h, w = g.shape
    if h == 2 * oh and w == 2 * ow:
        rs, cs = [0, oh], [0, ow]
    elif h == 2 * oh + 1 and w == 2 * ow + 1:
        rs, cs = [0, oh + 1], [0, ow + 1]
    else:
        return None
    return [(r, c) for r in rs for c in cs]


def _odd_index(panels):
    eq = [[np.array_equal(panels[a], panels[b]) for b in range(4)]
          for a in range(4)]
    agree = [sum(eq[a][b] for b in range(4) if b != a) for a in range(4)]
    if sorted(agree) != [0, 2, 2, 2]:
        return None
    return agree.index(0)


def _detect(task: dict):
    examples = list(all_examples(task))
    if not examples:
        return None
    ishapes, oshapes = set(), set()
    for ex in examples:
        ishapes.add((len(ex["input"]), len(ex["input"][0])))
        oshapes.add((len(ex["output"]), len(ex["output"][0])))
    if len(ishapes) != 1 or len(oshapes) != 1:
        return None
    (H, W), (oh, ow) = next(iter(ishapes)), next(iter(oshapes))
    if H > HEIGHT or W > WIDTH or oh < 1 or ow < 1:
        return None

    corners = None
    for ex in examples:
        g = np.array(ex["input"])
        cs = _panels(g, oh, ow)
        if cs is None:
            return None
        corners = cs
        ps = [g[r:r + oh, c:c + ow] for (r, c) in cs]
        idx = _odd_index(ps)
        if idx is None or not np.array_equal(ps[idx], np.array(ex["output"])):
            return None
    return corners, oh, ow


def _build(corners, oh: int, ow: int) -> onnx.ModelProto:
    F = TensorProto.FLOAT
    B = TensorProto.BOOL
    n = helper.make_node
    init = [
        numpy_helper.from_array(np.array([0, 1, 2, 3], dtype=np.int64), "ax4"),
        numpy_helper.from_array(np.array(0.5, dtype=np.float32), "half"),
    ]
    nodes = []
    # extract the four panels
    for k, (r, c) in enumerate(corners):
        init += [
            numpy_helper.from_array(np.array([0, 0, r, c], dtype=np.int64),
                                    f"p{k}_s"),
            numpy_helper.from_array(
                np.array([1, CHANNELS, r + oh, c + ow], dtype=np.int64),
                f"p{k}_e"),
        ]
        nodes.append(n("Slice", ["input", f"p{k}_s", f"p{k}_e", "ax4"],
                       [f"P{k}"], name=f"slice_p{k}"))

    # pairwise equality (1 if panels identical)
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    for a, b in pairs:
        nodes += [
            n("Sub", [f"P{a}", f"P{b}"], [f"d{a}{b}"], name=f"sub{a}{b}"),
            n("Abs", [f"d{a}{b}"], [f"ad{a}{b}"], name=f"abs{a}{b}"),
            n("ReduceSum", [f"ad{a}{b}"], [f"s{a}{b}"], axes=[1, 2, 3],
              keepdims=1, name=f"rs{a}{b}"),
            n("Less", [f"s{a}{b}", "half"], [f"eqb{a}{b}"], name=f"lt{a}{b}"),
            n("Cast", [f"eqb{a}{b}"], [f"eq{a}{b}"], to=F, name=f"cst{a}{b}"),
        ]

    def eq(a, b):
        return f"eq{a}{b}" if a < b else f"eq{b}{a}"

    # agree_i = sum of equalities with the other three panels
    for i in range(4):
        others = [j for j in range(4) if j != i]
        nodes += [
            n("Add", [eq(i, others[0]), eq(i, others[1])], [f"ag{i}_0"],
              name=f"ag{i}a"),
            n("Add", [f"ag{i}_0", eq(i, others[2])], [f"ag{i}"], name=f"ag{i}b"),
            n("Less", [f"ag{i}", "half"], [f"wb{i}"], name=f"w{i}lt"),
            n("Cast", [f"wb{i}"], [f"w{i}"], to=F, name=f"w{i}cast"),
            n("Mul", [f"P{i}", f"w{i}"], [f"sel{i}"], name=f"sel{i}mul"),
        ]

    nodes += [
        n("Add", ["sel0", "sel1"], ["sum01"], name="add01"),
        n("Add", ["sel2", "sel3"], ["sum23"], name="add23"),
        n("Add", ["sum01", "sum23"], ["picked"], name="addpick"),
    ]
    init.append(numpy_helper.from_array(
        np.array([0, 0, 0, 0, 0, 0, HEIGHT - oh, WIDTH - ow], dtype=np.int64),
        "frame_pads"))
    nodes.append(n("Pad", ["picked", "frame_pads"], ["output"],
                   mode="constant", name="frame_pad"))

    def vi(name, shape, dt=F):
        return helper.make_tensor_value_info(name, dt, shape)

    panel_shape = [1, CHANNELS, oh, ow]
    sc = [1, 1, 1, 1]
    value_info = []
    for k in range(4):
        value_info += [vi(f"P{k}", panel_shape), vi(f"sel{k}", panel_shape),
                       vi(f"ag{k}_0", sc), vi(f"ag{k}", sc),
                       vi(f"wb{k}", sc, B), vi(f"w{k}", sc)]
    for a, b in pairs:
        value_info += [vi(f"d{a}{b}", panel_shape), vi(f"ad{a}{b}", panel_shape),
                       vi(f"s{a}{b}", sc), vi(f"eqb{a}{b}", sc, B),
                       vi(f"eq{a}{b}", sc)]
    value_info += [vi("sum01", panel_shape), vi("sum23", panel_shape),
                   vi("picked", panel_shape)]

    inputs = [vi("input", [1, CHANNELS, HEIGHT, WIDTH])]
    outputs = [vi("output", [1, CHANNELS, HEIGHT, WIDTH])]
    graph = helper.make_graph(nodes, "odd_panel", inputs, outputs,
                              initializer=init, value_info=value_info)
    return helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
        ir_version=IR_VERSION)


def solve_odd_panel(task: dict) -> Optional[onnx.ModelProto]:
    res = _detect(task)
    if res is None:
        return None
    return _build(*res)
