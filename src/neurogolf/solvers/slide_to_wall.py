"""Solver: slide an object until it touches a wall (task 8).

The grid holds a colour-2 "mover" object and a colour-8 "wall" object,
separated along one axis. The mover slides as a rigid block toward the wall
until the two are adjacent (gap closed); the wall stays put.

The graph reads the two objects from channels 2 and 8, takes their bounding
boxes, derives a signed row/column shift (one is zero) = the gap toward the
wall, translates the mover with index-shifted `Gather`s, then reassembles the
one-hot output (background where neither object lands).
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
MOVER, WALL = 2, 8


def _ref(grid) -> Optional[list]:
    g = np.array(grid)
    H, W = g.shape
    mm = (g == MOVER)
    wm = (g == WALL)
    if mm.sum() == 0 or wm.sum() == 0:
        return None
    mr = np.argwhere(mm)
    wr = np.argwhere(wm)
    mr0, mr1 = mr[:, 0].min(), mr[:, 0].max()
    mc0, mc1 = mr[:, 1].min(), mr[:, 1].max()
    wr0, wr1 = wr[:, 0].min(), wr[:, 0].max()
    wc0, wc1 = wr[:, 1].min(), wr[:, 1].max()
    dv = int(wr0 > mr1) - int(wr1 < mr0)
    dh = int(wc0 > mc1) - int(wc1 < mc0)
    if (dv != 0) == (dh != 0):       # need exactly one separated axis
        return None
    out = np.where(wm, WALL, 0)
    if dv != 0:
        sh = (wr0 - mr1 - 1) if dv > 0 else -(mr0 - wr1 - 1)
        for r, c in mr:
            out[r + sh, c] = MOVER
    else:
        sh = (wc0 - mc1 - 1) if dh > 0 else -(mc0 - wc1 - 1)
        for r, c in mr:
            out[r, c + sh] = MOVER
    return out.tolist()


def _detect(task: dict) -> bool:
    saw = False
    for ex in all_examples(task):
        i, o = ex["input"], ex["output"]
        if not i or not i[0]:
            return False
        if len(i) > HEIGHT or len(i[0]) > WIDTH:
            continue
        ref = _ref(i)
        if ref is None or ref != o:
            return False
        saw = True
    return saw


def _build() -> onnx.ModelProto:
    F = TensorProto.FLOAT
    I = TensorProto.INT64
    B = TensorProto.BOOL
    n_ = helper.make_node

    row_ar = np.arange(HEIGHT, dtype=np.float32).reshape(1, 1, HEIGHT, 1)
    col_ar = np.arange(WIDTH, dtype=np.float32).reshape(1, 1, 1, WIDTH)
    init = [
        numpy_helper.from_array(row_ar, "row_ar"),
        numpy_helper.from_array(col_ar, "col_ar"),
        numpy_helper.from_array(np.array(0.0, np.float32), "zero"),
        numpy_helper.from_array(np.array(1.0, np.float32), "one"),
        numpy_helper.from_array(np.array(-0.5, np.float32), "neghalf"),
        numpy_helper.from_array(np.array(29.5, np.float32), "c295"),
        numpy_helper.from_array(np.array(29.0, np.float32), "c29"),
        numpy_helper.from_array(np.array(1000.0, np.float32), "big"),
        numpy_helper.from_array(np.array([0, MOVER, 0, 0], np.int64), "m2_s"),
        numpy_helper.from_array(np.array([1, MOVER + 1, HEIGHT, WIDTH], np.int64), "m2_e"),
        numpy_helper.from_array(np.array([0, WALL, 0, 0], np.int64), "m8_s"),
        numpy_helper.from_array(np.array([1, WALL + 1, HEIGHT, WIDTH], np.int64), "m8_e"),
        numpy_helper.from_array(np.array([0, 1, 2, 3], np.int64), "ax4"),
        numpy_helper.from_array(np.array([HEIGHT], np.int64), "flatH"),
        numpy_helper.from_array(np.array([WIDTH], np.int64), "flatW"),
    ]

    nodes = [
        n_("Slice", ["input", "m2_s", "m2_e", "ax4"], ["m2"]),
        n_("Slice", ["input", "m8_s", "m8_e", "ax4"], ["m8"]),
    ]

    def bbox(mask, s):
        out = [
            n_("ReduceMax", [mask], [f"{s}_rows"], axes=[3], keepdims=1),
            n_("ReduceMax", [mask], [f"{s}_cols"], axes=[2], keepdims=1),
            n_("Mul", [f"{s}_rows", "row_ar"], [f"{s}_rp"]),
            n_("Mul", [f"{s}_cols", "col_ar"], [f"{s}_cp"]),
            n_("ReduceMax", [f"{s}_rp"], [f"{s}_r1"], axes=[2], keepdims=1),
            n_("ReduceMax", [f"{s}_cp"], [f"{s}_c1"], axes=[3], keepdims=1),
            n_("Sub", ["one", f"{s}_rows"], [f"{s}_rinv"]),
            n_("Sub", ["one", f"{s}_cols"], [f"{s}_cinv"]),
            n_("Mul", [f"{s}_rinv", "big"], [f"{s}_rb"]),
            n_("Mul", [f"{s}_cinv", "big"], [f"{s}_cb"]),
            n_("Add", [f"{s}_rp", f"{s}_rb"], [f"{s}_rpb"]),
            n_("Add", [f"{s}_cp", f"{s}_cb"], [f"{s}_cpb"]),
            n_("ReduceMin", [f"{s}_rpb"], [f"{s}_r0"], axes=[2], keepdims=1),
            n_("ReduceMin", [f"{s}_cpb"], [f"{s}_c0"], axes=[3], keepdims=1),
        ]
        return out

    nodes += bbox("m2", "m2")
    nodes += bbox("m8", "m8")

    # direction flags and signed shifts
    nodes += [
        n_("Greater", ["m8_r0", "m2_r1"], ["below_b"]),   # wall below mover
        n_("Less", ["m8_r1", "m2_r0"], ["above_b"]),
        n_("Greater", ["m8_c0", "m2_c1"], ["right_b"]),
        n_("Less", ["m8_c1", "m2_c0"], ["left_b"]),
        n_("Cast", ["below_b"], ["below"], to=F),
        n_("Cast", ["above_b"], ["above"], to=F),
        n_("Cast", ["right_b"], ["right"], to=F),
        n_("Cast", ["left_b"], ["left"], to=F),
        # gaps: wr0-mr1-1 ; mr0-wr1-1 ; wc0-mc1-1 ; mc0-wc1-1
        n_("Sub", ["m8_r0", "m2_r1"], ["gb0"]), n_("Sub", ["gb0", "one"], ["gap_below"]),
        n_("Sub", ["m2_r0", "m8_r1"], ["ga0"]), n_("Sub", ["ga0", "one"], ["gap_above"]),
        n_("Sub", ["m8_c0", "m2_c1"], ["gr0"]), n_("Sub", ["gr0", "one"], ["gap_right"]),
        n_("Sub", ["m2_c0", "m8_c1"], ["gl0"]), n_("Sub", ["gl0", "one"], ["gap_left"]),
        n_("Mul", ["below", "gap_below"], ["sr_b"]),
        n_("Mul", ["above", "gap_above"], ["sr_a"]),
        n_("Sub", ["sr_b", "sr_a"], ["sh_r"]),
        n_("Mul", ["right", "gap_right"], ["sc_r"]),
        n_("Mul", ["left", "gap_left"], ["sc_l"]),
        n_("Sub", ["sc_r", "sc_l"], ["sh_c"]),
    ]

    # shift mover by (sh_r, sh_c) with index-shifted Gathers + validity masks
    nodes += [
        n_("Sub", ["row_ar", "sh_r"], ["ridx"]),
        n_("Greater", ["ridx", "neghalf"], ["rge_b"]), n_("Cast", ["rge_b"], ["rge"], to=F),
        n_("Less", ["ridx", "c295"], ["rle_b"]), n_("Cast", ["rle_b"], ["rle"], to=F),
        n_("Mul", ["rge", "rle"], ["rvalid"]),
        n_("Clip", ["ridx", "zero", "c29"], ["ridx_c"]),
        n_("Cast", ["ridx_c"], ["ridx_i"], to=I),
        n_("Reshape", ["ridx_i", "flatH"], ["ridx1d"]),
        n_("Gather", ["m2", "ridx1d"], ["m2r"], axis=2),
        n_("Mul", ["m2r", "rvalid"], ["m2rv"]),
        n_("Sub", ["col_ar", "sh_c"], ["cidx"]),
        n_("Greater", ["cidx", "neghalf"], ["cge_b"]), n_("Cast", ["cge_b"], ["cge"], to=F),
        n_("Less", ["cidx", "c295"], ["cle_b"]), n_("Cast", ["cle_b"], ["cle"], to=F),
        n_("Mul", ["cge", "cle"], ["cvalid"]),
        n_("Clip", ["cidx", "zero", "c29"], ["cidx_c"]),
        n_("Cast", ["cidx_c"], ["cidx_i"], to=I),
        n_("Reshape", ["cidx_i", "flatW"], ["cidx1d"]),
        n_("Gather", ["m2rv", "cidx1d"], ["m2rc"], axis=3),
        n_("Mul", ["m2rc", "cvalid"], ["moved"]),
    ]

    # reassemble one-hot: ch0 background, ch2 moved, ch8 wall
    nodes += [
        n_("ReduceSum", ["input"], ["pres"], axes=[1], keepdims=1),
        n_("Greater", ["pres", "zero"], ["ext_b"]), n_("Cast", ["ext_b"], ["extent"], to=F),
        n_("Add", ["moved", "m8"], ["occ"]),
        n_("Sub", ["extent", "occ"], ["bg"]),
        n_("Mul", ["m2", "zero"], ["zp"]),
    ]
    planes = []
    for k in range(CHANNELS):
        if k == 0:
            planes.append("bg")
        elif k == MOVER:
            planes.append("moved")
        elif k == WALL:
            planes.append("m8")
        else:
            planes.append("zp")
    nodes.append(n_("Concat", planes, ["output"], axis=1))
    for nd in nodes:
        nd.name = nd.output[0]

    def vi(name, shape, dt=F):
        return helper.make_tensor_value_info(name, dt, shape)

    s1 = [1, 1, HEIGHT, WIDTH]
    rc = [1, 1, HEIGHT, 1]
    cc = [1, 1, 1, WIDTH]
    sc = [1, 1, 1, 1]
    value_info = [vi("m2", s1), vi("m8", s1)]
    for s in ("m2", "m8"):
        value_info += [
            vi(f"{s}_rows", rc), vi(f"{s}_cols", cc), vi(f"{s}_rp", rc),
            vi(f"{s}_cp", cc), vi(f"{s}_r1", sc), vi(f"{s}_c1", sc),
            vi(f"{s}_rinv", rc), vi(f"{s}_cinv", cc), vi(f"{s}_rb", rc),
            vi(f"{s}_cb", cc), vi(f"{s}_rpb", rc), vi(f"{s}_cpb", cc),
            vi(f"{s}_r0", sc), vi(f"{s}_c0", sc),
        ]
    for nm in ("below", "above", "right", "left"):
        value_info += [vi(f"{nm}_b", sc, B), vi(nm, sc)]
    for nm in ("gb0", "gap_below", "ga0", "gap_above", "gr0", "gap_right",
               "gl0", "gap_left", "sr_b", "sr_a", "sh_r", "sc_r", "sc_l", "sh_c"):
        value_info.append(vi(nm, sc))
    value_info += [
        vi("ridx", rc), vi("rge_b", rc, B), vi("rge", rc), vi("rle_b", rc, B),
        vi("rle", rc), vi("rvalid", rc), vi("ridx_c", rc), vi("ridx_i", rc, I), vi("ridx1d", [HEIGHT], I),
        vi("m2r", s1), vi("m2rv", s1),
        vi("cidx", cc), vi("cge_b", cc, B), vi("cge", cc), vi("cle_b", cc, B),
        vi("cle", cc), vi("cvalid", cc), vi("cidx_c", cc), vi("cidx_i", cc, I), vi("cidx1d", [WIDTH], I),
        vi("m2rc", s1), vi("moved", s1),
        vi("pres", s1), vi("ext_b", s1, B), vi("extent", s1),
        vi("occ", s1), vi("bg", s1), vi("zp", s1),
    ]

    graph = helper.make_graph(nodes, "slide_to_wall", [vi("input", FULL)],
                              [vi("output", FULL)], initializer=init,
                              value_info=value_info)
    return helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
        ir_version=IR_VERSION)


def solve_slide_to_wall(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
