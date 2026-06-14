"""Solver: each marker shoots a ray right, then turns down the right edge (task 237).

For every isolated marker at ``(r, c)`` the marker colour is painted across row
``r`` from column ``c`` to the right edge, then down the rightmost grid column
from row ``r`` to the bottom.  Where several markers reach the right column, the
lower one wins (its downward segment overwrites).

* horizontal segment  – rightward solid fill, ``CumSum(markers) > 0``;
* right-column segment – a sample-and-hold "fill-down" of the per-row marker
  colour, implemented with Hillis-Steele doubling shifts so an empty cell
  inherits the nearest marker colour above it;
* the right column is located as the last grid column
  (``colHasGrid[c] and not colHasGrid[c+1]``).
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
    out = g.copy()
    ys, xs = np.where(g != 0)
    if len(ys) == 0:
        return None
    for r, c in sorted(zip(ys, xs)):
        col = int(g[r, c])
        out[r, c:W] = col
        out[r:H, W - 1] = col
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
    notbg = np.ones((1, CHANNELS, 1, 1), np.float32); notbg[0, 0] = 0.0
    init = [
        numpy_helper.from_array(e0, "e0"),
        numpy_helper.from_array(notbg, "notbg"),
        numpy_helper.from_array(np.array(0.5, np.float32), "half"),
        numpy_helper.from_array(np.array(1.0, np.float32), "one"),
        numpy_helper.from_array(np.array(3, np.int64), "axis3"),
        numpy_helper.from_array(np.array([0], np.int64), "c0s"),
        numpy_helper.from_array(np.array([1], np.int64), "c1e"),
        numpy_helper.from_array(np.array([1], np.int64), "ax1"),
        numpy_helper.from_array(np.array([0], np.int64), "s0"),
        numpy_helper.from_array(np.array([1], np.int64), "s1"),
        numpy_helper.from_array(np.array([HEIGHT], np.int64), "eH"),
        numpy_helper.from_array(np.array([2], np.int64), "ax2"),
        numpy_helper.from_array(np.array([3], np.int64), "ax3"),
        numpy_helper.from_array(np.array([0, 0, 0, 0, 0, 0, 0, 1], np.int64), "padEnd1"),
    ]
    for k in (1, 2, 4, 8, 16):
        init.append(numpy_helper.from_array(
            np.array([0, 0, k, 0, 0, 0, 0, 0], np.int64), f"padDown{k}"))

    nodes = [
        n("ReduceSum", ["input"], ["occ"], axes=[1], keepdims=1),
        n("Slice", ["input", "c0s", "c1e", "ax1"], ["is0"]),
        n("Sub", ["occ", "is0"], ["mark"]),
        # per-row marker colour one-hot (1,C,H,1)
        n("ReduceMax", ["input"], ["rowHas"], axes=[3], keepdims=1),
        n("Mul", ["rowHas", "notbg"], ["rowColor"]),
        # horizontal ray
        n("CumSum", ["mark", "axis3"], ["cumH"]),
        n("Greater", ["cumH", "half"], ["hact_b"]), n("Cast", ["hact_b"], ["hact"], to=F),
        n("Mul", ["hact", "occ"], ["hRayMask"]),
        n("Mul", ["hRayMask", "rowColor"], ["hRayLayer"]),
    ]
    # fill-down (sample & hold) of rowColor via doubling shifts -> FD
    prev = "rowColor"
    for k in (1, 2, 4, 8, 16):
        f, e, pad, sh, ct, nxt = (f"fl{k}", f"em{k}", f"pd{k}", f"sh{k}",
                                  f"ct{k}", f"FD{k}")
        nodes += [
            n("ReduceSum", [prev], [f], axes=[1], keepdims=1),
            n("Sub", ["one", f], [e]),
            n("Pad", [prev, f"padDown{k}"], [pad]),
            n("Slice", [pad, "s0", "eH", "ax2"], [sh]),
            n("Mul", [sh, e], [ct]),
            n("Add", [prev, ct], [nxt]),
        ]
        prev = nxt
    FD = prev
    # rightmost grid column mask
    nodes += [
        n("ReduceMax", ["occ"], ["colHas"], axes=[2], keepdims=1),
        n("Slice", ["colHas", "s1", "eH", "ax3"], ["colHasS"]),  # eH=30 ok as end
        n("Pad", ["colHasS", "padEnd1"], ["colHasL"]),
        n("Sub", ["one", "colHasL"], ["notColL"]),
        n("Mul", ["colHas", "notColL"], ["rightInd"]),
        n("Mul", ["occ", "rightInd"], ["rightColMask"]),
        n("Mul", [FD, "rightColMask"], ["rightColLayer"]),
        # combine
        n("Max", ["hRayLayer", "rightColLayer"], ["rawColor"]),
        n("ReduceSum", ["rawColor"], ["painted"], axes=[1], keepdims=1),
        n("Sub", ["occ", "painted"], ["bgv"]),
        n("Mul", ["bgv", "e0"], ["bgLayer"]),
        n("Add", ["rawColor", "bgLayer"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "right_then_down_ray",
                              [helper.make_tensor_value_info("input", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              [helper.make_tensor_value_info("output", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_right_then_down_ray(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
