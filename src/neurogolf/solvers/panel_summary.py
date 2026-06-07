"""Solver: summarise an MxN grid of colour panels to their dominant colours (task 184).

The grid is an MxN arrangement of noisy solid-colour panels separated by blank
rows/columns.  The output is the MxN grid where each cell is its panel's
dominant (most frequent non-background) colour.

Build: forward `CumSum` over row/col "content-start" transitions gives each
row/col a band index.  One-hot band indicators (capacity 15) and a double
`MatMul` bin the per-channel pixel counts into a (10, 15, 15) tensor; `ArgMax`
over channels is the dominant colour per band-pair, masked to bands that carry
content and padded to the canvas.
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
B = 15   # max bands per axis


def _bands(mask):
    out = []; i = 0; n = len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            out.append((i, j)); i = j
        else:
            i += 1
    return out


def _ref(g: np.ndarray) -> Optional[np.ndarray]:
    H, W = g.shape
    rb = _bands([(g[r] != 0).any() for r in range(H)])
    cb = _bands([(g[:, c] != 0).any() for c in range(W)])
    if len(rb) < 1 or len(cb) < 1 or (len(rb) == 1 and len(cb) == 1):
        return None
    out = np.zeros((len(rb), len(cb)), int)
    for i, (r0, r1) in enumerate(rb):
        for j, (c0, c1) in enumerate(cb):
            blk = g[r0:r1, c0:c1]
            vals, cnts = np.unique(blk[blk != 0], return_counts=True)
            if len(vals) == 0:
                return None
            out[i, j] = vals[np.argmax(cnts)]
    return out


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
    chidx = np.arange(CHANNELS, dtype=np.float32).reshape(1, CHANNELS, 1, 1)
    bandr = np.arange(B, dtype=np.float32).reshape(1, 1, 1, B)
    init = [
        numpy_helper.from_array(note0, "note0"),
        numpy_helper.from_array(chidx, "chidx"),
        numpy_helper.from_array(bandr, "bandr"),
        numpy_helper.from_array(np.array(0.5, np.float32), "half"),
        numpy_helper.from_array(np.array(1.0, np.float32), "one"),
        numpy_helper.from_array(np.array(2, np.int64), "ax2"),
        numpy_helper.from_array(np.array(3, np.int64), "ax3"),
        numpy_helper.from_array(np.array([2], np.int64), "ax2s"),
        numpy_helper.from_array(np.array([3], np.int64), "ax3s"),
        numpy_helper.from_array(np.array([0, 0, 1, 0, 0, 0, 0, 0], np.int64), "padr"),
        numpy_helper.from_array(np.array([0, 0, 0, 1, 0, 0, 0, 0], np.int64), "padc"),
        numpy_helper.from_array(np.array([0.0], np.float32), "zero"),
        numpy_helper.from_array(np.array([0], np.int64), "z1"),
        numpy_helper.from_array(np.array([HEIGHT], np.int64), "hH"),
        numpy_helper.from_array(np.array([WIDTH], np.int64), "wW"),
        numpy_helper.from_array(np.array([1, 1, HEIGHT, 1], np.int64), "rshapeH"),
        numpy_helper.from_array(np.array([1, 1, WIDTH, 1], np.int64), "rshapeW"),
        numpy_helper.from_array(np.array([0, 0, 0, 0, 0, 0, HEIGHT - B, WIDTH - B], np.int64), "padout"),
    ]
    nodes = [
        n("Mul", ["input", "note0"], ["inb"]),                            # (1,10,H,W) non-bg
        n("ReduceSum", ["inb"], ["content"], axes=[1], keepdims=1),       # (1,1,H,W) non-bg presence
        # row band index
        n("ReduceMax", ["content"], ["rowhas"], axes=[3], keepdims=1),    # (1,1,H,1)
        n("Pad", ["rowhas", "padr", "zero"], ["rh_pad"]),
        n("Slice", ["rh_pad", "z1", "hH", "ax2s"], ["rprev"]),
        n("Sub", ["one", "rprev"], ["nrprev"]), n("Mul", ["rowhas", "nrprev"], ["rtrans"]),
        n("CumSum", ["rtrans", "ax2"], ["rcum"]),
        n("Sub", ["rcum", "one"], ["ridx"]),                              # (1,1,H,1)
        n("Sub", ["ridx", "bandr"], ["rdd"]), n("Abs", ["rdd"], ["rda"]),
        n("Less", ["rda", "half"], ["roh_b"]), cf("roh_b", "roh"),        # (1,1,H,B)
        # col band index
        n("ReduceMax", ["content"], ["colhas"], axes=[2], keepdims=1),    # (1,1,1,W)
        n("Pad", ["colhas", "padc", "zero"], ["ch_pad"]),
        n("Slice", ["ch_pad", "z1", "wW", "ax3s"], ["cprev"]),
        n("Sub", ["one", "cprev"], ["ncprev"]), n("Mul", ["colhas", "ncprev"], ["ctrans"]),
        n("CumSum", ["ctrans", "ax3"], ["ccum"]),
        n("Sub", ["ccum", "one"], ["cidx0"]),                            # (1,1,1,W)
        n("Reshape", ["cidx0", "rshapeW"], ["cidx"]),                    # (1,1,W,1)
        n("Sub", ["cidx", "bandr"], ["cdd"]), n("Abs", ["cdd"], ["cda"]),
        n("Less", ["cda", "half"], ["coh_b"]), cf("coh_b", "coh"),       # (1,1,W,B)
        # bin counts: count[k,i,j] = roh^T @ (inb @ coh)
        n("MatMul", ["inb", "coh"], ["step1"]),                          # (1,10,H,B)
        n("Transpose", ["roh"], ["rohT"], perm=[0, 1, 3, 2]),            # (1,1,B,H)
        n("MatMul", ["rohT", "step1"], ["count"]),                       # (1,10,B,B)
        n("ArgMax", ["count"], ["domarg_i"], axis=1, keepdims=1),        # (1,1,B,B)
        cf("domarg_i", "domarg"),
        n("ReduceMax", ["count"], ["cmax"], axes=[1], keepdims=1),       # (1,1,B,B)
        n("Greater", ["cmax", "half"], ["valid_b"]), cf("valid_b", "valid"),
        n("Sub", ["chidx", "domarg"], ["domd"]), n("Abs", ["domd"], ["doma"]),
        n("Less", ["doma", "half"], ["domoh_b"]), cf("domoh_b", "domoh"),  # (1,10,B,B)
        n("Mul", ["domoh", "valid"], ["dom_m"]),
        n("Pad", ["dom_m", "padout", "zero"], ["output"]),               # (1,10,30,30)
    ]
    graph = helper.make_graph(nodes, "panel_summary",
                              [helper.make_tensor_value_info("input", F, FULL)],
                              [helper.make_tensor_value_info("output", F, FULL)],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_panel_summary(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
