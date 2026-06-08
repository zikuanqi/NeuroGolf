"""Solver: fill each enclosed square hole by its side parity (task 204).

Boxes of colour 1 enclose square holes; each hole is filled with ``2`` if its
side is even, ``7`` if odd::

    1 1 1 1          1 1 1 1          1 1 1
    1 . . 1    ->    1 2 2 1   (2x2)  1 . 1   ->  1 7 1   (1x1)
    1 . . 1          1 2 2 1          1 1 1       1 1 1
    1 1 1 1          1 1 1 1

Shares the enclosure + masked run-length machinery with ``hole_size_fill`` (the
run-length of a square hole equals its side); here the side parity selects the
fill colour: ``fill = 2 + 5*(run mod 2)``.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from ..grids import CHANNELS, HEIGHT, WIDTH, all_examples
from .hole_size_fill import _bgcol, _enc, _runlen, _emit_common

OPSET = 11
IR_VERSION = 8
FULL = [1, CHANNELS, HEIGHT, WIDTH]
F = TensorProto.FLOAT


def _ref(g: np.ndarray) -> Optional[np.ndarray]:
    bg = _bgcol(g)
    enc = _enc(g, bg)
    if not enc.any():
        return None
    run = _runlen(enc)
    out = g.copy()
    for r, c in zip(*np.where(enc)):
        out[r, c] = 2 if run[r, c] % 2 == 0 else 7
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
    chramp = np.arange(CHANNELS, dtype=np.float32).reshape(1, CHANNELS, 1, 1)
    init = [
        numpy_helper.from_array(chramp, "chramp"),
        numpy_helper.from_array(np.array(0.5, np.float32), "half"),
        numpy_helper.from_array(np.array(1.0, np.float32), "one"),
        numpy_helper.from_array(np.array(2.0, np.float32), "two"),
        numpy_helper.from_array(np.array(5.0, np.float32), "five"),
        numpy_helper.from_array(np.array([2, 3], np.int64), "ax23"),
    ]
    nodes = []
    seen = {"pad": set(), "sl": set()}
    ctr = [0]
    _emit_common(n, init, nodes, seen, ctr)            # -> "enc", "run"
    nodes += [
        # parity = run - 2*floor(run/2)
        n("Mul", ["run", "half"], ["half_run"]),
        n("Floor", ["half_run"], ["fl"]),
        n("Add", ["fl", "fl"], ["twice"]),
        n("Sub", ["run", "twice"], ["parity"]),
        # fill = 2 + 5*parity   (even->2, odd->7)
        n("Mul", ["parity", "five"], ["p5"]),
        n("Add", ["p5", "two"], ["fillval"]),
        n("Sub", ["fillval", "chramp"], ["fdiff"]),
        n("Abs", ["fdiff"], ["fad"]),
        n("Less", ["fad", "half"], ["oh_b"]), n("Cast", ["oh_b"], ["oh"], to=F),
        n("Mul", ["oh", "enc"], ["paint"]),
        n("Sub", ["one", "enc"], ["keep"]),
        n("Mul", ["input", "keep"], ["kept"]),
        n("Add", ["kept", "paint"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "hole_parity_fill",
                              [helper.make_tensor_value_info("input", F, FULL)],
                              [helper.make_tensor_value_info("output", F, FULL)],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_hole_parity_fill(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
