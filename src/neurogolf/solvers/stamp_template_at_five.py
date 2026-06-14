"""Solver: stamp a colour template centred on the 5-marker (task 206).

A small multi-colour template sits somewhere on the grid; a lone ``5`` marks
where to drop a copy.  The output keeps the original template and adds a copy
translated so its bounding-box centre lands on the 5; the 5 itself is removed::

    . 2 .                  . 2 .
    2 2 1     . . 5  ->    2 2 1     . 2 .
    . 1 3                  . 1 3     2 2 1
                                     . 1 3

The template is every non-0, non-5 cell; the shift ``delta = 5-pos - bbox
centre`` drives a runtime 2D translation (row matrix ``SR`` then column matrix
``SC``, each ``|D - delta| < 0.5``).
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
    ys5, xs5 = np.where(g == 5)
    if len(ys5) != 1:
        return None
    mr, mc = ys5[0], xs5[0]
    ys, xs = np.where((g != 0) & (g != 5))
    if len(ys) == 0:
        return None
    tcr = (ys.min() + ys.max()) // 2
    tcc = (xs.min() + xs.max()) // 2
    dr, dc = mr - tcr, mc - tcc
    out = g.copy()
    out[g == 5] = 0
    for r, c in zip(ys, xs):
        rr, cc = r + dr, c + dc
        if 0 <= rr < H and 0 <= cc < W:
            out[rr, cc] = g[r, c]
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
    e5 = np.zeros((1, CHANNELS, 1, 1), np.float32); e5[0, 5] = 1.0
    ih = np.arange(HEIGHT); iw = np.arange(WIDTH)
    Drow = (ih[:, None] - ih[None, :]).astype(np.float32).reshape(1, 1, HEIGHT, HEIGHT)  # r-j
    Dcol = (iw[None, :] - iw[:, None]).astype(np.float32).reshape(1, 1, WIDTH, WIDTH)     # c-j
    init = [
        numpy_helper.from_array(e0, "e0"),
        numpy_helper.from_array(e5, "e5"),
        numpy_helper.from_array(Drow, "Drow"),
        numpy_helper.from_array(Dcol, "Dcol"),
        numpy_helper.from_array(np.arange(HEIGHT, dtype=np.float32).reshape(1, 1, HEIGHT, 1), "ah"),
        numpy_helper.from_array(np.arange(WIDTH, dtype=np.float32).reshape(1, 1, 1, WIDTH), "aw"),
        numpy_helper.from_array(np.array(0.5, np.float32), "half"),
        numpy_helper.from_array(np.array([0], np.int64), "c0s"),
        numpy_helper.from_array(np.array([1], np.int64), "c1e"),
        numpy_helper.from_array(np.array([5], np.int64), "c5s"),
        numpy_helper.from_array(np.array([6], np.int64), "c6e"),
        numpy_helper.from_array(np.array([1], np.int64), "ax1"),
    ]
    nodes = [
        n("ReduceSum", ["input"], ["occ"], axes=[1], keepdims=1),
        n("Slice", ["input", "c0s", "c1e", "ax1"], ["is0"]),
        n("Slice", ["input", "c5s", "c6e", "ax1"], ["is5"]),
        n("Sub", ["occ", "is0"], ["t0"]), n("Sub", ["t0", "is5"], ["tmask"]),
        # 5 position
        n("ReduceMax", ["is5"], ["r5"], axes=[3], keepdims=1),
        n("ArgMax", ["r5"], ["mr_i"], axis=2, keepdims=1), n("Cast", ["mr_i"], ["mr"], to=F),
        n("ReduceMax", ["is5"], ["c5"], axes=[2], keepdims=1),
        n("ArgMax", ["c5"], ["mc_i"], axis=3, keepdims=1), n("Cast", ["mc_i"], ["mc"], to=F),
        # template bbox centre
        n("ReduceMax", ["tmask"], ["rT"], axes=[3], keepdims=1),
        n("ArgMax", ["rT"], ["minr_i"], axis=2, keepdims=1), n("Cast", ["minr_i"], ["minr"], to=F),
        n("Mul", ["rT", "ah"], ["rTp"]), n("ReduceMax", ["rTp"], ["maxr"], axes=[2], keepdims=1),
        n("Add", ["minr", "maxr"], ["sr"]), n("Mul", ["sr", "half"], ["cr0"]), n("Floor", ["cr0"], ["tcr"]),
        n("ReduceMax", ["tmask"], ["cT"], axes=[2], keepdims=1),
        n("ArgMax", ["cT"], ["minc_i"], axis=3, keepdims=1), n("Cast", ["minc_i"], ["minc"], to=F),
        n("Mul", ["cT", "aw"], ["cTp"]), n("ReduceMax", ["cTp"], ["maxc"], axes=[3], keepdims=1),
        n("Add", ["minc", "maxc"], ["sc"]), n("Mul", ["sc", "half"], ["cc0"]), n("Floor", ["cc0"], ["tcc"]),
        n("Sub", ["mr", "tcr"], ["dr"]), n("Sub", ["mc", "tcc"], ["dc"]),
        # shift matrices
        n("Sub", ["Drow", "dr"], ["sdr"]), n("Abs", ["sdr"], ["adr"]),
        n("Less", ["adr", "half"], ["SR_b"]), n("Cast", ["SR_b"], ["SR"], to=F),
        n("Sub", ["Dcol", "dc"], ["sdc"]), n("Abs", ["sdc"], ["adc"]),
        n("Less", ["adc", "half"], ["SC_b"]), n("Cast", ["SC_b"], ["SC"], to=F),
        # translate template
        n("Mul", ["input", "tmask"], ["tonly"]),
        n("MatMul", ["SR", "tonly"], ["tr"]),
        n("MatMul", ["tr", "SC"], ["trans"]),
        # assemble: clear the 5, overlay the stamped template
        n("Mul", ["is5", "e5"], ["s5"]), n("Sub", ["input", "s5"], ["c1"]),
        n("Mul", ["is5", "e0"], ["a0"]), n("Add", ["c1", "a0"], ["cleared"]),
        n("ReduceSum", ["trans"], ["smask"], axes=[1], keepdims=1),
        n("Sub", ["one", "smask"], ["inv"]),
        n("Mul", ["cleared", "inv"], ["base"]),
        n("Add", ["base", "trans"], ["output"]),
    ]
    init.append(numpy_helper.from_array(np.array(1.0, np.float32), "one"))
    graph = helper.make_graph(nodes, "stamp_template_at_five",
                              [helper.make_tensor_value_info("input", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              [helper.make_tensor_value_info("output", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_stamp_template_at_five(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
