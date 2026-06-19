"""Solver: output the 3x3 shape of the most-repeated colour (task 79).

Several small shapes appear scattered in different colours; the colour with the
most copies (connected components) wins, and its 3x3 template is emitted as the
(3x3) output.

Per-colour components are labelled by a channel-wise ``MaxPool`` max-cellid flood
(``MaxPool`` is per-channel, so all ten colours label at once); each component's
representative cell (label == cellid) is counted, the winning channel chosen by
``ReduceMax``.  The winner's first component is isolated by its flood label, its
bbox top-left found, and the colour shifted to the grid origin via clamped
``Gather``s, with background filled inside the 3x3 box.
"""
from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from ..grids import CHANNELS, HEIGHT, WIDTH, all_examples

OPSET = 11
IR_VERSION = 8
F = TensorProto.FLOAT


def _comps(mask):
    H, W = mask.shape
    seen = np.zeros((H, W), bool); res = []
    for r in range(H):
        for c in range(W):
            if mask[r, c] and not seen[r, c]:
                cells = []; q = deque([(r, c)]); seen[r, c] = True
                while q:
                    y, x = q.popleft(); cells.append((y, x))
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and not seen[ny, nx]:
                                seen[ny, nx] = True; q.append((ny, nx))
                res.append(cells)
    return res


def _ref(g: np.ndarray) -> Optional[np.ndarray]:
    cols = [c for c in np.unique(g) if c != 0]
    if not cols:
        return None
    cc = {c: _comps(g == c) for c in cols}
    win = max(cols, key=lambda c: (len(cc[c]), -int(c)))
    comp = cc[win][0]
    ys = [y for y, x in comp]; xs = [x for y, x in comp]
    r0, c0 = min(ys), min(xs)
    return g[r0:r0 + 3, c0:c0 + 3]


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
    cellid = (np.arange(HEIGHT * WIDTH).reshape(1, 1, HEIGHT, WIDTH) + 1).astype(np.float32)
    notbg = np.ones((1, CHANNELS, 1, 1), np.float32); notbg[0, 0] = 0.0
    e0 = np.zeros((1, CHANNELS, 1, 1), np.float32); e0[0, 0] = 1.0
    box3 = np.zeros((1, 1, HEIGHT, WIDTH), np.float32); box3[0, 0, :3, :3] = 1.0
    init = [
        numpy_helper.from_array(cellid, "cellid"),
        numpy_helper.from_array(notbg, "notbg"),
        numpy_helper.from_array(e0, "e0"),
        numpy_helper.from_array(box3, "box3"),
        numpy_helper.from_array(np.arange(HEIGHT, dtype=np.float32).reshape(1, 1, HEIGHT, 1), "rowidx"),
        numpy_helper.from_array(np.arange(WIDTH, dtype=np.float32).reshape(1, 1, 1, WIDTH), "colidx"),
        numpy_helper.from_array(np.array(1.0, np.float32), "one"),
        numpy_helper.from_array(np.array(1e4, np.float32), "big"),
        numpy_helper.from_array(np.array(0.5, np.float32), "half"),
        numpy_helper.from_array(np.array(0.0, np.float32), "cmin"),
        numpy_helper.from_array(np.array(float(HEIGHT - 1), np.float32), "cmax"),
        numpy_helper.from_array(np.array([HEIGHT], np.int64), "shp30"),
    ]
    nodes = [
        # per-channel connected-component labelling (max-cellid flood)
        n("Mul", ["input", "cellid"], ["lab0"]),
    ]
    cur = "lab0"
    for it in range(7):
        nodes += [
            n("MaxPool", [cur], ["mp%d" % it], kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]),
            n("Mul", ["mp%d" % it, "input"], ["lab%d" % (it + 1)]),
        ]
        cur = "lab%d" % (it + 1)
    nodes += [
        # component representatives (label == cellid) -> per-channel count
        n("Sub", [cur, "cellid"], ["dlab"]), n("Mul", ["dlab", "dlab"], ["dlab2"]),
        n("Less", ["dlab2", "half"], ["rep_b"]), n("Cast", ["rep_b"], ["repf"], to=F),
        n("Mul", ["repf", "input"], ["rep"]),
        n("ReduceSum", ["rep"], ["cc0"], axes=[2, 3], keepdims=1),
        n("Mul", ["cc0", "notbg"], ["cc"]),                         # (1,10,1,1)
        n("ReduceMax", ["cc"], ["mx"], axes=[1], keepdims=1),
        n("Sub", ["cc", "mx"], ["dC"]), n("Mul", ["dC", "dC"], ["dC2"]),
        n("Less", ["dC2", "half"], ["win_b"]), n("Cast", ["win_b"], ["winf"], to=F),
        n("Mul", ["winf", "notbg"], ["winnerOH"]),                  # (1,10,1,1)
        n("Mul", ["input", "winnerOH"], ["wsel"]),
        n("ReduceSum", ["wsel"], ["winnerMask"], axes=[1], keepdims=1),  # (1,1,H,W)
        n("Mul", [cur, "winnerOH"], ["lsel"]),
        n("ReduceSum", ["lsel"], ["winnerLab"], axes=[1], keepdims=1),
        # reading-first winner cell -> its component label
        n("Mul", ["cellid", "winnerMask"], ["cidW"]),
        n("Sub", ["one", "winnerMask"], ["notW"]), n("Mul", ["notW", "big"], ["notWb"]),
        n("Add", ["cidW", "notWb"], ["cidW2"]),
        n("ReduceMin", ["cidW2"], ["minCid"], axes=[2, 3], keepdims=1),
        n("Sub", ["cellid", "minCid"], ["dm"]), n("Mul", ["dm", "dm"], ["dm2"]),
        n("Less", ["dm2", "half"], ["fst_b"]), n("Cast", ["fst_b"], ["fstf"], to=F),
        n("Mul", ["fstf", "winnerMask"], ["firstM"]),
        n("Mul", ["winnerLab", "firstM"], ["lsq"]),
        n("ReduceSum", ["lsq"], ["Lc"], axes=[2, 3], keepdims=1),
        # isolate first component
        n("Sub", ["winnerLab", "Lc"], ["dl"]), n("Mul", ["dl", "dl"], ["dl2"]),
        n("Less", ["dl2", "half"], ["iso_b"]), n("Cast", ["iso_b"], ["isof"], to=F),
        n("Mul", ["isof", "winnerMask"], ["isolate"]),
        n("Mul", ["input", "isolate"], ["compOH"]),                 # (1,10,H,W)
        # bbox top-left of the component
        n("Mul", ["rowidx", "isolate"], ["ri"]),
        n("Sub", ["one", "isolate"], ["niso"]), n("Mul", ["niso", "big"], ["nisob"]),
        n("Add", ["ri", "nisob"], ["rim"]), n("ReduceMin", ["rim"], ["r0"], axes=[2, 3], keepdims=1),
        n("Mul", ["colidx", "isolate"], ["ci"]),
        n("Add", ["ci", "nisob"], ["cim"]), n("ReduceMin", ["cim"], ["c0"], axes=[2, 3], keepdims=1),
        # shift component to the origin via clamped gathers
        n("Add", ["rowidx", "r0"], ["rgi0"]), n("Clip", ["rgi0", "cmin", "cmax"], ["rgic"]),
        n("Cast", ["rgic"], ["rgii"], to=TensorProto.INT64), n("Reshape", ["rgii", "shp30"], ["rGi"]),
        n("Add", ["colidx", "c0"], ["cgi0"]), n("Clip", ["cgi0", "cmin", "cmax"], ["cgic"]),
        n("Cast", ["cgic"], ["cgii"], to=TensorProto.INT64), n("Reshape", ["cgii", "shp30"], ["cGi"]),
        n("Gather", ["compOH", "rGi"], ["rowsG"], axis=2),
        n("Gather", ["rowsG", "cGi"], ["shifted"], axis=3),
        # fill background inside the 3x3 box; zero (padding) outside
        n("ReduceSum", ["shifted"], ["sAny"], axes=[1], keepdims=1),
        n("Sub", ["one", "sAny"], ["sNone"]), n("Mul", ["box3", "sNone"], ["bgIn"]),
        n("Mul", ["bgIn", "e0"], ["bgL"]),
        n("Add", ["shifted", "bgL"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "majority_shape_crop",
                              [helper.make_tensor_value_info("input", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              [helper.make_tensor_value_info("output", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_majority_shape_crop(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
