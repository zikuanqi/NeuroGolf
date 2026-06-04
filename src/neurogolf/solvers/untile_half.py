"""Solver: un-tile a 2x repetition (task 188).

The input is one panel repeated twice — either side-by-side (left half == right
half) or stacked (top half == bottom half). The output is a single panel.

We detect the content extent (H, W), test whether the left half equals the
right half (an index-shifted compare), and emit the left half if so, else the
top half. The horizontal test is applied first, which matches the data's choice
on the few grids that tile both ways. Because the surviving half already sits in
the top-left, the whole transform is just a mask of the input.
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
    if W % 2 == 0 and np.array_equal(g[:, :W // 2], g[:, W // 2:]):
        return g[:, :W // 2]
    if H % 2 == 0 and np.array_equal(g[:H // 2, :], g[H // 2:, :]):
        return g[:H // 2, :]
    return None


def _detect(task: dict) -> bool:
    saw = False
    for ex in all_examples(task):
        i, o = ex["input"], ex["output"]
        if not i or not i[0] or not o or not o[0]:
            return False
        if len(i) > HEIGHT or len(i[0]) > WIDTH:
            continue
        ref = _ref(np.array(i))
        if ref is None or not np.array_equal(ref, np.array(o)):
            return False
        saw = True
    return saw


def _build() -> onnx.ModelProto:
    F = TensorProto.FLOAT
    n_ = helper.make_node
    row_ar = np.arange(HEIGHT, dtype=np.float32).reshape(1, 1, HEIGHT, 1)
    col_ar = np.arange(WIDTH, dtype=np.float32).reshape(1, 1, 1, WIDTH)
    init = [
        numpy_helper.from_array(row_ar, "row_ar"),
        numpy_helper.from_array(col_ar, "col_ar"),
        numpy_helper.from_array(np.array(0.0, np.float32), "zero"),
        numpy_helper.from_array(np.array(0.5, np.float32), "half"),
        numpy_helper.from_array(np.array(1.0, np.float32), "one"),
        numpy_helper.from_array(np.array(29.0, np.float32), "c29"),
        numpy_helper.from_array(np.array([WIDTH], np.int64), "flatW"),
    ]
    nodes = [
        n_("ReduceSum", ["input"], ["pres"], axes=[1], keepdims=1),
        n_("ReduceMax", ["pres"], ["rowpres"], axes=[3], keepdims=1),
        n_("ReduceMax", ["pres"], ["colpres"], axes=[2], keepdims=1),
        n_("Mul", ["rowpres", "row_ar"], ["rpos"]),
        n_("Mul", ["colpres", "col_ar"], ["cpos"]),
        n_("ReduceMax", ["rpos"], ["rmax"], axes=[2], keepdims=1),
        n_("ReduceMax", ["cpos"], ["cmax"], axes=[3], keepdims=1),
        n_("Add", ["rmax", "one"], ["H_f"]),
        n_("Add", ["cmax", "one"], ["W_f"]),
        n_("Mul", ["H_f", "half"], ["Hh"]),
        n_("Mul", ["W_f", "half"], ["Wh"]),
        # masks
        n_("Less", ["row_ar", "H_f"], ["rmaskH_b"]), n_("Cast", ["rmaskH_b"], ["rmaskH"], to=F),
        n_("Less", ["col_ar", "Wh"], ["cmaskWh_b"]), n_("Cast", ["cmaskWh_b"], ["cmaskWh"], to=F),
        n_("Less", ["row_ar", "Hh"], ["rmaskHh_b"]), n_("Cast", ["rmaskHh_b"], ["rmaskHh"], to=F),
        # shift input left by Wh and compare to detect horizontal tiling
        n_("Add", ["col_ar", "Wh"], ["cidx"]),
        n_("Clip", ["cidx", "zero", "c29"], ["cidx_c"]),
        n_("Cast", ["cidx_c"], ["cidx_i"], to=TensorProto.INT64),
        n_("Reshape", ["cidx_i", "flatW"], ["cidx1d"]),
        n_("Gather", ["input", "cidx1d"], ["rshift"], axis=3),
        n_("Sub", ["input", "rshift"], ["d"]),
        n_("Mul", ["d", "d"], ["sq"]),
        n_("Mul", ["sq", "cmaskWh"], ["sqm1"]),
        n_("Mul", ["sqm1", "rmaskH"], ["sqm2"]),
        n_("ReduceSum", ["sqm2"], ["hdiff"], axes=[0, 1, 2, 3], keepdims=1),
        n_("Less", ["hdiff", "half"], ["htiled_b"]), n_("Cast", ["htiled_b"], ["htiled"], to=F),
        # outputs: left half (h) or top half (v), select by flag
        n_("Mul", ["input", "cmaskWh"], ["lefthalf"]),
        n_("Mul", ["input", "rmaskHh"], ["tophalf"]),
        n_("Mul", ["lefthalf", "htiled"], ["lsel"]),
        n_("Sub", ["one", "htiled"], ["nh"]),
        n_("Mul", ["tophalf", "nh"], ["tsel"]),
        n_("Add", ["lsel", "tsel"], ["output"]),
    ]
    for nd in nodes:
        nd.name = nd.output[0]

    def vi(name, shape, dt=F):
        return helper.make_tensor_value_info(name, dt, shape)
    s1 = [1, 1, HEIGHT, WIDTH]; rc = [1, 1, HEIGHT, 1]; cc = [1, 1, 1, WIDTH]; sc = [1, 1, 1, 1]
    B = TensorProto.BOOL
    value_info = [
        vi("pres", s1), vi("rowpres", rc), vi("colpres", cc), vi("rpos", rc), vi("cpos", cc),
        vi("rmax", sc), vi("cmax", sc), vi("H_f", sc), vi("W_f", sc), vi("Hh", sc), vi("Wh", sc),
        vi("rmaskH_b", rc, B), vi("rmaskH", rc), vi("cmaskWh_b", cc, B), vi("cmaskWh", cc),
        vi("rmaskHh_b", rc, B), vi("rmaskHh", rc),
        vi("cidx", cc), vi("cidx_c", cc), vi("cidx_i", cc, TensorProto.INT64), vi("cidx1d", [WIDTH], TensorProto.INT64),
        vi("rshift", FULL), vi("d", FULL), vi("sq", FULL), vi("sqm1", FULL), vi("sqm2", FULL),
        vi("hdiff", sc), vi("htiled_b", sc, B), vi("htiled", sc),
        vi("lefthalf", FULL), vi("tophalf", FULL), vi("lsel", FULL), vi("nh", sc), vi("tsel", FULL),
    ]
    graph = helper.make_graph(nodes, "untile_half", [vi("input", FULL)],
                              [vi("output", FULL)], initializer=init, value_info=value_info)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_untile_half(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
