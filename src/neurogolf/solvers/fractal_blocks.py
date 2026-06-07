"""Solver: self-fractal of a 3x3-downsampled block grid (task 195).

The content is a 3x3 arrangement of equal blocks of one colour (packed, no
separators).  Downsampling it to a 3x3 occupancy meta, the 9x9 output is the
self-fractal (Kronecker square) of that meta: block (a,b) of the output holds a
copy of the meta when meta[a,b] is set.

Build: the content bbox is split into thirds by computing each row/col's third
index (floor(3*(idx-min)/extent)); one-hot thirds and a double `MatMul` bin
pixel counts into a 3x3 occupancy meta; a reshape-broadcast Kronecker squares it
to 9x9, painted in the block colour and padded to the canvas.
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
    from collections import Counter
    ys, xs = np.where(g != 0)
    if len(ys) == 0:
        return None
    r0, r1, c0, c1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    sub = g[r0:r1, c0:c1]
    H, W = sub.shape
    if H % 3 or W % 3:
        return None
    bh, bw = H // 3, W // 3
    B = Counter(sub[sub != 0].tolist()).most_common(1)[0][0]
    meta = np.zeros((3, 3), int)
    for i in range(3):
        for j in range(3):
            if (sub[i * bh:(i + 1) * bh, j * bw:(j + 1) * bw] != 0).any():
                meta[i, j] = 1
    return np.kron(meta, meta) * B


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

    note0 = np.ones((1, CHANNELS, 1, 1), np.float32); note0[0, 0] = 0.0
    e0 = np.zeros((1, CHANNELS, 1, 1), np.float32); e0[0, 0] = 1.0
    bands3 = np.arange(3, dtype=np.float32).reshape(1, 1, 1, 3)
    row_idx = np.arange(HEIGHT, dtype=np.float32).reshape(1, 1, HEIGHT, 1)
    col_idx = np.arange(WIDTH, dtype=np.float32).reshape(1, 1, 1, WIDTH)
    init = [
        numpy_helper.from_array(note0, "note0"),
        numpy_helper.from_array(e0, "e0"),
        numpy_helper.from_array(bands3, "bands3"),
        numpy_helper.from_array(row_idx, "row_idx"),
        numpy_helper.from_array(col_idx, "col_idx"),
        numpy_helper.from_array(np.array(0.5, np.float32), "half"),
        numpy_helper.from_array(np.array(1.0, np.float32), "one"),
        numpy_helper.from_array(np.array(3.0, np.float32), "three"),
        numpy_helper.from_array(np.array([1, 1, WIDTH, 1], np.int64), "rsW1"),
        numpy_helper.from_array(np.array([3, 1, 3, 1], np.int64), "rsA"),
        numpy_helper.from_array(np.array([1, 3, 1, 3], np.int64), "rsC"),
        numpy_helper.from_array(np.array([1, 1, 9, 9], np.int64), "rs99"),
        numpy_helper.from_array(np.array([0, 0, 0, 0, 0, 0, HEIGHT - 9, WIDTH - 9], np.int64), "padout"),
        numpy_helper.from_array(np.array([0.0], np.float32), "zero"),
    ]
    nodes = [
        n("Mul", ["input", "note0"], ["inb"]),
        n("ReduceSum", ["inb"], ["content"], axes=[1], keepdims=1),          # (1,1,H,W)
        n("ReduceMax", ["content"], ["rowhas"], axes=[3], keepdims=1),       # (1,1,H,1)
        n("ReduceMax", ["content"], ["colhas"], axes=[2], keepdims=1),       # (1,1,1,W)
        # bbox extents
        n("ArgMax", ["rowhas"], ["r0_i"], axis=2, keepdims=1), cf("r0_i", "r0"),
        n("Mul", ["rowhas", "row_idx"], ["rr"]), n("ReduceMax", ["rr"], ["rmax"], axes=[2], keepdims=1),
        n("Sub", ["rmax", "r0"], ["rspan"]), n("Add", ["rspan", "one"], ["Hb"]),
        n("ArgMax", ["colhas"], ["c0_i"], axis=3, keepdims=1), cf("c0_i", "c0"),
        n("Mul", ["colhas", "col_idx"], ["cc"]), n("ReduceMax", ["cc"], ["cmax"], axes=[3], keepdims=1),
        n("Sub", ["cmax", "c0"], ["cspan"]), n("Add", ["cspan", "one"], ["Wb"]),
        # third indices (floor)
        n("Sub", ["row_idx", "r0"], ["rrel"]), n("Mul", ["rrel", "three"], ["rrel3"]),
        n("Div", ["rrel3", "Hb"], ["tr_f"]), n("Cast", ["tr_f"], ["tr_i"], to=TensorProto.INT64), cf("tr_i", "tr"),
        n("Sub", ["tr", "bands3"], ["trd"]), n("Abs", ["trd"], ["tra"]),
        n("Less", ["tra", "half"], ["mro_b"]), cf("mro_b", "mro"),           # (1,1,H,3)
        n("Sub", ["col_idx", "c0"], ["crel"]), n("Mul", ["crel", "three"], ["crel3"]),
        n("Div", ["crel3", "Wb"], ["tc_f"]), n("Cast", ["tc_f"], ["tc_i"], to=TensorProto.INT64), cf("tc_i", "tc0"),
        n("Reshape", ["tc0", "rsW1"], ["tc"]),                               # (1,1,W,1)
        n("Sub", ["tc", "bands3"], ["tcd"]), n("Abs", ["tcd"], ["tca"]),
        n("Less", ["tca", "half"], ["mco_b"]), cf("mco_b", "mco"),           # (1,1,W,3)
        # bin counts -> 3x3 meta
        n("MatMul", ["content", "mco"], ["step1"]),                          # (1,1,H,3)
        n("Transpose", ["mro"], ["mroT"], perm=[0, 1, 3, 2]),               # (1,1,3,H)
        n("MatMul", ["mroT", "step1"], ["count"]),                           # (1,1,3,3)
        n("Greater", ["count", "half"], ["meta_b"]), cf("meta_b", "meta"),   # (1,1,3,3)
        # kron(meta, meta) -> 9x9
        n("Reshape", ["meta", "rsA"], ["metaA"]),                            # (3,1,3,1)
        n("Reshape", ["meta", "rsC"], ["metaC"]),                            # (1,3,1,3)
        n("Mul", ["metaA", "metaC"], ["prod"]),                              # (3,3,3,3)
        n("Reshape", ["prod", "rs99"], ["kron"]),                            # (1,1,9,9)
        # block colour
        n("ReduceSum", ["inb"], ["hist"], axes=[2, 3], keepdims=1),          # (1,10,1,1)
        n("Greater", ["hist", "half"], ["eB_b"]), cf("eB_b", "eB"),
        n("Mul", ["eB", "kron"], ["paintB"]),                                # block colour at fractal cells
        n("Sub", ["one", "kron"], ["nkron"]),
        n("Mul", ["e0", "nkron"], ["paint0"]),                               # background-0 elsewhere in 9x9
        n("Add", ["paintB", "paint0"], ["out9"]),                            # (1,10,9,9)
        n("Pad", ["out9", "padout", "zero"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "fractal_blocks",
                              [helper.make_tensor_value_info("input", F, FULL)],
                              [helper.make_tensor_value_info("output", F, FULL)],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_fractal_blocks(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
