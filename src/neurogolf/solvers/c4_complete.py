"""Solver: complete a partial pattern to 4-fold rotational symmetry (task 20).

The coloured pattern is C4-symmetric about its bounding-box centre but partly
missing; the output fills every cell from whichever of its four rotation images
is present.

With doubled sums ``sr=R0+R1``, ``sc=C0+C1``, ``A=(sr+sc)/2``, ``B=(sc-sr)/2``:
  180  = input[sr-r, sc-c]                (two-axis Gather)
   90  = input[A-c, r+B] = T[r+B, A-c]    (Gather on the transpose)
  270  = input[c-B, A-r] = T[A-r, c-B]
The non-background overlay of the four images is confined to the bbox.
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
    ys, xs = np.where(g != 0)
    if len(ys) == 0:
        return None
    R0, R1, C0, C1 = ys.min(), ys.max(), xs.min(), xs.max()
    sr, sc = R0 + R1, C0 + C1
    out = g.copy()
    changed = False
    for r in range(H):
        for c in range(W):
            a, b = 2 * r - sr, 2 * c - sc
            for (na, nb) in [(-b, a), (-a, -b), (b, -a)]:
                if (na + sr) % 2 or (nb + sc) % 2:
                    continue
                rr, cc = (na + sr) // 2, (nb + sc) // 2
                if 0 <= rr < H and 0 <= cc < W and g[rr, cc] != 0:
                    if out[r, c] == 0:
                        out[r, c] = g[rr, cc]; changed = True
                    break
    return out if changed else None


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
    rowidx = np.arange(HEIGHT, dtype=np.float32).reshape(1, 1, HEIGHT, 1)
    colidx = np.arange(WIDTH, dtype=np.float32).reshape(1, 1, 1, WIDTH)
    init = [
        numpy_helper.from_array(e0, "e0"),
        numpy_helper.from_array(notbg, "notbg"),
        numpy_helper.from_array(rowidx, "rowidx"),
        numpy_helper.from_array(colidx, "colidx"),
        numpy_helper.from_array(np.array(1.0, np.float32), "one"),
        numpy_helper.from_array(np.array(0.5, np.float32), "half"),
        numpy_helper.from_array(np.array(1e4, np.float32), "big"),
        numpy_helper.from_array(np.array(0.0, np.float32), "cmin"),
        numpy_helper.from_array(np.array(float(HEIGHT - 1), np.float32), "cmax"),
        numpy_helper.from_array(np.array([0], np.int64), "c0s"),
        numpy_helper.from_array(np.array([1], np.int64), "c1e"),
        numpy_helper.from_array(np.array([1], np.int64), "ax1"),
        numpy_helper.from_array(np.array([HEIGHT], np.int64), "to30"),
    ]

    def ci(src, out):
        return [
            n("Clip", [src, "cmin", "cmax"], [out + "_c"]),
            n("Cast", [out + "_c"], [out + "_i"], to=TensorProto.INT64),
            n("Reshape", [out + "_i", "to30"], [out]),
        ]

    nodes = [
        n("ReduceSum", ["input"], ["occ"], axes=[1], keepdims=1),
        n("Slice", ["input", "c0s", "c1e", "ax1"], ["is0"]),
        n("Sub", ["occ", "is0"], ["nonbg"]),
        n("ReduceMax", ["nonbg"], ["rowHas"], axes=[3], keepdims=1),   # (1,1,H,1)
        n("ReduceMax", ["nonbg"], ["colHas"], axes=[2], keepdims=1),   # (1,1,1,W)
        # bbox
        n("Mul", ["rowHas", "rowidx"], ["rh"]),
        n("ReduceMax", ["rh"], ["R1"], axes=[2], keepdims=1),
        n("Sub", ["one", "rowHas"], ["rhinv"]), n("Mul", ["rhinv", "big"], ["rhb"]),
        n("Add", ["rh", "rhb"], ["rhmin"]), n("ReduceMin", ["rhmin"], ["R0"], axes=[2], keepdims=1),
        n("Mul", ["colHas", "colidx"], ["ch"]),
        n("ReduceMax", ["ch"], ["C1"], axes=[3], keepdims=1),
        n("Sub", ["one", "colHas"], ["chinv"]), n("Mul", ["chinv", "big"], ["chb"]),
        n("Add", ["ch", "chb"], ["chmin"]), n("ReduceMin", ["chmin"], ["C0"], axes=[3], keepdims=1),
        # sr, sc, A, B
        n("Add", ["R0", "R1"], ["sr"]), n("Add", ["C0", "C1"], ["sc"]),
        n("Add", ["sr", "sc"], ["srsc"]), n("Mul", ["srsc", "half"], ["A"]),
        n("Sub", ["sc", "sr"], ["scsr"]), n("Mul", ["scsr", "half"], ["B"]),
        # transpose
        n("Transpose", ["input"], ["T"], perm=[0, 1, 3, 2]),
        # index sequences
        n("Sub", ["sr", "rowidx"], ["r180r0"]),       # sr - r  (1,1,H,1)
        n("Sub", ["sc", "colidx"], ["r180c0"]),       # sc - c
        n("Add", ["rowidx", "B"], ["r90r0"]),         # r + B
        n("Sub", ["A", "colidx"], ["r90c0"]),         # A - c
        n("Sub", ["A", "rowidx"], ["r270r0"]),        # A - r
        n("Sub", ["colidx", "B"], ["r270c0"]),        # c - B
    ]
    for src, out in (("r180r0", "i180r"), ("r180c0", "i180c"),
                     ("r90r0", "i90r"), ("r90c0", "i90c"),
                     ("r270r0", "i270r"), ("r270c0", "i270c")):
        nodes += ci(src, out)
    nodes += [
        n("Gather", ["input", "i180r"], ["g180a"], axis=2), n("Gather", ["g180a", "i180c"], ["rot180"], axis=3),
        n("Gather", ["T", "i90r"], ["g90a"], axis=2), n("Gather", ["g90a", "i90c"], ["rot90"], axis=3),
        n("Gather", ["T", "i270r"], ["g270a"], axis=2), n("Gather", ["g270a", "i270c"], ["rot270"], axis=3),
        # non-bg overlay
        n("Mul", ["input", "notbg"], ["inNb"]),
        n("Mul", ["rot90", "notbg"], ["nb90"]),
        n("Mul", ["rot180", "notbg"], ["nb180"]),
        n("Mul", ["rot270", "notbg"], ["nb270"]),
        n("Max", ["inNb", "nb90"], ["m1"]), n("Max", ["nb180", "nb270"], ["m2"]),
        n("Max", ["m1", "m2"], ["overlay"]),
    ]
    # bbox mask (rowidx in [R0,R1], colidx in [C0,C1])
    nodes += [
        n("Sub", ["rowidx", "R0"], ["d_r0"]), n("Add", ["half", "d_r0"], ["d_r0b"]),
        n("Greater", ["d_r0b", "cmin"], ["geR0_b"]), n("Cast", ["geR0_b"], ["geR0"], to=F),
        n("Sub", ["R1", "rowidx"], ["d_r1"]), n("Add", ["half", "d_r1"], ["d_r1b"]),
        n("Greater", ["d_r1b", "cmin"], ["leR1_b"]), n("Cast", ["leR1_b"], ["leR1"], to=F),
        n("Mul", ["geR0", "leR1"], ["rowIn"]),
        n("Sub", ["colidx", "C0"], ["d_c0"]), n("Add", ["half", "d_c0"], ["d_c0b"]),
        n("Greater", ["d_c0b", "cmin"], ["geC0_b"]), n("Cast", ["geC0_b"], ["geC0"], to=F),
        n("Sub", ["C1", "colidx"], ["d_c1"]), n("Add", ["half", "d_c1"], ["d_c1b"]),
        n("Greater", ["d_c1b", "cmin"], ["leC1_b"]), n("Cast", ["leC1_b"], ["leC1"], to=F),
        n("Mul", ["geC0", "leC1"], ["colIn"]),
        n("Mul", ["rowIn", "colIn"], ["bboxMask"]),
        n("Mul", ["overlay", "bboxMask"], ["colourL"]),
        n("ReduceSum", ["colourL"], ["fillAny"], axes=[1], keepdims=1),
        n("Sub", ["occ", "fillAny"], ["bgv"]),
        n("Mul", ["bgv", "e0"], ["bgL"]),
        n("Add", ["colourL", "bgL"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "c4_complete",
                              [helper.make_tensor_value_info("input", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              [helper.make_tensor_value_info("output", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_c4_complete(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
