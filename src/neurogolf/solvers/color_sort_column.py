"""Solver: list the colours sorted by cell-count, as a K x 1 column (task 393).

The grid holds a handful of coloured blobs.  The output is a single column that
names each colour once, ordered by how many cells it occupies (largest first).

Build: a per-channel histogram gives each colour's count (background zeroed).
A 10x10 pairwise Greater turns counts into a rank (#colours strictly larger),
so the biggest colour ranks 0, next 1, ...  Each colour channel is then painted
into the output row equal to its rank, column 0; the empty channels (count 0)
are masked out so only real colours appear.  Ties are rejected in detection.
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
    counts = {int(c): int((g == c).sum()) for c in np.unique(g) if c != 0}
    if len(counts) < 1:
        return None
    vals = list(counts.values())
    if len(set(vals)) != len(vals):       # ties not handled by the rank build
        return None
    order = sorted(counts, key=lambda c: -counts[c])
    return np.array(order, dtype=np.int64).reshape(-1, 1)


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
    note0 = np.ones((1, CHANNELS, 1, 1), np.float32)
    note0[0, 0] = 0.0
    rowi = np.arange(HEIGHT, dtype=np.float32).reshape(1, 1, HEIGHT, 1)
    col0 = np.zeros((1, 1, 1, WIDTH), np.float32)
    col0[0, 0, 0, 0] = 1.0
    init = [
        numpy_helper.from_array(note0, "note0"),
        numpy_helper.from_array(rowi, "rowi"),
        numpy_helper.from_array(col0, "col0"),
        numpy_helper.from_array(np.array(0.5, np.float32), "half"),
        numpy_helper.from_array(np.array([CHANNELS, 1], np.int64), "rsA"),
        numpy_helper.from_array(np.array([1, CHANNELS], np.int64), "rsB"),
        numpy_helper.from_array(np.array([1, CHANNELS, 1, 1], np.int64), "rsRk"),
    ]
    nodes = [
        n("ReduceSum", ["input"], ["hist"], axes=[2, 3], keepdims=1),   # (1,10,1,1)
        n("Mul", ["hist", "note0"], ["hc"]),
        n("Reshape", ["hc", "rsA"], ["hcA"]),                           # (10,1)
        n("Reshape", ["hc", "rsB"], ["hcB"]),                           # (1,10)
        n("Greater", ["hcB", "hcA"], ["gt_b"]),                         # (10,10) [i,j]=cnt[j]>cnt[i]
        n("Cast", ["gt_b"], ["gt"], to=F),
        n("ReduceSum", ["gt"], ["rsum"], axes=[1], keepdims=1),         # (10,1) rank per channel
        n("Reshape", ["rsum", "rsRk"], ["rk"]),                         # (1,10,1,1)
        n("Greater", ["hc", "half"], ["isc_b"]),
        n("Cast", ["isc_b"], ["isc"], to=F),                            # (1,10,1,1) real colours
        n("Sub", ["rk", "rowi"], ["d"]),                                # (1,10,H,1)
        n("Abs", ["d"], ["da"]),
        n("Less", ["da", "half"], ["mt_b"]),
        n("Cast", ["mt_b"], ["match"], to=F),                           # (1,10,H,1)
        n("Mul", ["match", "isc"], ["placed"]),                         # mask empty channels
        n("Mul", ["placed", "col0"], ["output"]),                       # column 0 only
    ]
    graph = helper.make_graph(nodes, "color_sort_column",
                              [helper.make_tensor_value_info("input", F, FULL)],
                              [helper.make_tensor_value_info("output", F, FULL)],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_color_sort_column(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
