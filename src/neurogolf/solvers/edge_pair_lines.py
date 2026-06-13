"""Solver: connect matching border-end markers with full lines (task 161).

Marker cells of one colour sit at both ends of some rows (left+right border)
and/or columns (top+bottom border); scattered noise of other colours fills the
rest.  The output is a blank canvas with full lines drawn between each marker
pair, in the marker colour::

    3 . . . 3      ->     3 3 3 3 3        (noise cleared)

The marker colour is recognised by having **no cells besides its endpoints**:
``count(v) == 2 x lines(v)``.  All per-colour work happens on the channel axis
at once: endpoint products ``left*right`` / ``top*bottom`` give per-channel
line flags, ReduceSums give the gate, and the gated flags broadcast into rows
and columns (columns overwrite rows at crossings).
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
I64 = TensorProto.INT64


def _ref(g: np.ndarray) -> Optional[np.ndarray]:
    H, W = g.shape
    rows = [(r, int(g[r, 0])) for r in range(H) if g[r, 0] != 0 and g[r, 0] == g[r, W - 1]]
    cols = [(c, int(g[0, c])) for c in range(W) if g[0, c] != 0 and g[0, c] == g[H - 1, c]]
    out = np.zeros_like(g)
    drew = False
    for v in range(1, 10):
        nl = sum(1 for _, vv in rows if vv == v) + sum(1 for _, vv in cols if vv == v)
        if nl == 0 or int((g == v).sum()) != 2 * nl:
            continue
        for r, vv in rows:
            if vv == v:
                out[r, :] = v
        for c, vv in cols:
            if vv == v:
                out[:, c] = v
        drew = True
    return out if drew else None


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
        numpy_helper.from_array(np.array([0], np.int64), "z0"),
        numpy_helper.from_array(np.array([1], np.int64), "z1"),
        numpy_helper.from_array(np.array([2], np.int64), "ax2v"),
        numpy_helper.from_array(np.array([3], np.int64), "ax3v"),
        numpy_helper.from_array(np.array([1], np.int64), "shape1"),
    ]
    nodes = [
        n("ReduceSum", ["input"], ["occ"], axes=[1], keepdims=1),
        n("ReduceMax", ["occ"], ["rowData"], axes=[3], keepdims=1),
        n("ReduceMax", ["occ"], ["colData"], axes=[2], keepdims=1),
        n("ReduceSum", ["colData"], ["W"], axes=[3], keepdims=1),
        n("ReduceSum", ["rowData"], ["Hh"], axes=[2], keepdims=1),
        n("Sub", ["W", "one"], ["Wm1"]),
        n("Cast", ["Wm1"], ["Wm1i0"], to=I64), n("Reshape", ["Wm1i0", "shape1"], ["wi"]),
        n("Sub", ["Hh", "one"], ["Hm1"]),
        n("Cast", ["Hm1"], ["Hm1i0"], to=I64), n("Reshape", ["Hm1i0", "shape1"], ["hi"]),
        n("Slice", ["input", "z0", "z1", "ax3v"], ["L"]),          # (1,10,30,1)
        n("Gather", ["input", "wi"], ["R"], axis=3),               # (1,10,30,1)
        n("Slice", ["input", "z0", "z1", "ax2v"], ["T"]),          # (1,10,1,30)
        n("Gather", ["input", "hi"], ["B"], axis=2),               # (1,10,1,30)
        n("Mul", ["L", "R"], ["PL"]),                              # per-channel row pairs
        n("Mul", ["T", "B"], ["PT"]),                              # per-channel col pairs
        n("ReduceSum", ["PL"], ["nlr"], axes=[2, 3], keepdims=1),
        n("ReduceSum", ["PT"], ["nlc"], axes=[2, 3], keepdims=1),
        n("Add", ["nlr", "nlc"], ["nl"]),                          # (1,10,1,1)
        n("ReduceSum", ["input"], ["cnt"], axes=[2, 3], keepdims=1),
        n("Add", ["nl", "nl"], ["nl2"]),
        n("Sub", ["cnt", "nl2"], ["cd"]), n("Abs", ["cd"], ["cda"]),
        n("Less", ["cda", "half"], ["eq_b"]), n("Cast", ["eq_b"], ["eq"], to=F),
        n("Greater", ["nl", "half"], ["pos_b"]), n("Cast", ["pos_b"], ["pos"], to=F),
        n("Mul", ["eq", "pos"], ["g0"]),
        n("Mul", ["g0", "notbg"], ["gate"]),                       # (1,10,1,1)
        n("Mul", ["PL", "gate"], ["rowF"]),                        # gated row flags
        n("Mul", ["PT", "gate"], ["colF"]),
        n("Mul", ["rowF", "occ"], ["rowP"]),                       # broadcast across cols
        n("Mul", ["colF", "occ"], ["colP"]),                       # broadcast across rows
        n("ReduceSum", ["colP"], ["colM"], axes=[1], keepdims=1),
        n("Sub", ["one", "colM"], ["invCM"]),
        n("Mul", ["rowP", "invCM"], ["rowP2"]),
        n("Add", ["rowP2", "colP"], ["painted"]),
        n("ReduceSum", ["painted"], ["pm"], axes=[1], keepdims=1),
        n("Sub", ["occ", "pm"], ["bgm"]),
        n("Mul", ["bgm", "e0"], ["bgp"]),
        n("Add", ["painted", "bgp"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "edge_pair_lines",
                              [helper.make_tensor_value_info("input", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              [helper.make_tensor_value_info("output", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_edge_pair_lines(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
