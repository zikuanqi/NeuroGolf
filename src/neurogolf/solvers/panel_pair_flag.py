"""Solver: flag the panels holding at least two 6s (task 149).

An 11x11 grid of 3x3 panels (colour-8 divider lines at rows/cols 3 and 7)
holds scattered colour-6 cells.  The output is a 3x3 grid with ``1`` in every
cell whose panel contains **two or more** 6s and ``0`` elsewhere::

    panel sixes:  2 1 2          1 0 1
                  2 1 1    ->    1 0 0
                  1 1 1          0 0 0

A stride-4 ones ``Conv`` sums each panel; a ``> 1.5`` threshold produces the
flags, which are zero-padded to the canvas as channel-1 (channel-0 carries the
complement so the 3x3 window decodes as 0/1).
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
    out = np.zeros((3, 3), int)
    for a, (r0, r1) in enumerate(_BANDS):
        for b, (c0, c1) in enumerate(_BANDS):
            if (g[r0:r1, c0:c1] == 6).sum() >= 2:
                out[a, b] = 1
    return out


def _detect(task: dict) -> bool:
    saw = False
    for ex in all_examples(task):
        i, o = ex["input"], ex["output"]
        if not i or not i[0] or len(i) > HEIGHT or len(i[0]) > WIDTH:
            continue
        r = _ref(np.array(i))
        if r is None or r.shape != np.array(o).shape or not np.array_equal(r, np.array(o)):
            return False
        saw = True
    return saw


def _build() -> onnx.ModelProto:
    n = helper.make_node
    e0 = np.zeros((1, CHANNELS, 1, 1), np.float32); e0[0, 0] = 1.0
    e1 = np.zeros((1, CHANNELS, 1, 1), np.float32); e1[0, 1] = 1.0
    init = [
        numpy_helper.from_array(e0, "e0"),
        numpy_helper.from_array(e1, "e1"),
        numpy_helper.from_array(np.ones((1, 1, 3, 3), np.float32), "ones3"),
        numpy_helper.from_array(np.array(1.5, np.float32), "th"),
        numpy_helper.from_array(np.array(1.0, np.float32), "one"),
        numpy_helper.from_array(np.array([6], np.int64), "c6s"),
        numpy_helper.from_array(np.array([7], np.int64), "c7e"),
        numpy_helper.from_array(np.array([1], np.int64), "ax1"),
        numpy_helper.from_array(np.array([0, 0, 0, 0, 0, 0, 27, 27], np.int64), "padTo30"),
        numpy_helper.from_array(np.array([0, 0], np.int64), "w0"),
        numpy_helper.from_array(np.array([3, 3], np.int64), "w3"),
        numpy_helper.from_array(np.array([2, 3], np.int64), "ax23"),
    ]
    nodes = [
        n("Slice", ["input", "c6s", "c7e", "ax1"], ["is6"]),
        n("Conv", ["is6", "ones3"], ["counts7"], strides=[4, 4]),    # (1,1,7,7) over canvas
        n("Slice", ["counts7", "w0", "w3", "ax23"], ["counts"]),     # panel sums (1,1,3,3)
        n("Greater", ["counts", "th"], ["gb"]), n("Cast", ["gb"], ["flag"], to=F),
        n("Sub", ["one", "flag"], ["invf"]),
        n("Pad", ["flag", "padTo30"], ["fpad"], mode="constant"),
        n("Pad", ["invf", "padTo30"], ["ipad"], mode="constant"),
        n("Mul", ["fpad", "e1"], ["a"]),
        n("Mul", ["ipad", "e0"], ["b"]),
        n("Add", ["a", "b"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "panel_pair_flag",
                              [helper.make_tensor_value_info("input", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              [helper.make_tensor_value_info("output", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_panel_pair_flag(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
