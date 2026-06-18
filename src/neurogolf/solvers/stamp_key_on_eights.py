"""Solver: stamp the colour key onto every 8-block, removing the key (task 69).

A small multi-colour key defines a colour per relative position; every ``8``
component is repainted by the key (aligned to the component's bounding-box
top-left), and the key itself is erased.

Anchoring: a min-cellid flood gives one impulse per component (its reading-order
first cell); shifting that impulse left by the key's own reading-offset lands it
on the bbox top-left, so the key (a runtime 5x5 ``ConvTranspose`` kernel) aligns
exactly.  The stamp is masked back to the original 8-cells.
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
K = 5


def _comps(mask):
    H, W = mask.shape
    seen = np.zeros((H, W), bool); res = []
    for r in range(H):
        for c in range(W):
            if mask[r, c] and not seen[r, c]:
                cs = []; q = deque([(r, c)]); seen[r, c] = True
                while q:
                    y, x = q.popleft(); cs.append((y, x))
                    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True; q.append((ny, nx))
                res.append(cs)
    return res


def _ref(g: np.ndarray) -> Optional[np.ndarray]:
    is8 = (g == 8)
    key = (g != 0) & (g != 8)
    if not key.any() or not is8.any():
        return None
    ys, xs = np.where(key)
    R0, C0 = ys.min(), xs.min()
    P = g[R0:ys.max() + 1, C0:xs.max() + 1]
    if P.shape[0] > K or P.shape[1] > K:
        return None
    out = np.zeros_like(g)
    for comp in _comps(is8):
        bys = [y for y, x in comp]; bxs = [x for y, x in comp]
        br, bc = min(bys), min(bxs)
        for y, x in comp:
            ri, ci = y - br, x - bc
            if ri < P.shape[0] and ci < P.shape[1]:
                out[y, x] = P[ri, ci]
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
    init = [
        numpy_helper.from_array(e0, "e0"),
        numpy_helper.from_array(np.array(1.0, np.float32), "one"),
        numpy_helper.from_array(np.array(0.5, np.float32), "half"),
        numpy_helper.from_array(np.array(1e4, np.float32), "big"),
        numpy_helper.from_array(np.array(0.0, np.float32), "cmin"),
        numpy_helper.from_array(np.array(float(HEIGHT - 1), np.float32), "cmax"),
        numpy_helper.from_array(np.array(float(WIDTH), np.float32), "wf"),
        numpy_helper.from_array(np.array(1.0 / WIDTH, np.float32), "invw"),
        numpy_helper.from_array((np.arange(HEIGHT * WIDTH).reshape(1, 1, HEIGHT, WIDTH) + 1).astype(np.float32), "cellid"),
        numpy_helper.from_array(np.arange(HEIGHT, dtype=np.float32).reshape(1, 1, HEIGHT, 1), "rowidx"),
        numpy_helper.from_array(np.arange(WIDTH, dtype=np.float32).reshape(1, 1, 1, WIDTH), "colidx"),
        numpy_helper.from_array(np.array([0], np.int64), "z0"),
        numpy_helper.from_array(np.array([1], np.int64), "z1"),
        numpy_helper.from_array(np.array([5], np.int64), "z5"),
        numpy_helper.from_array(np.array([8], np.int64), "z8"),
        numpy_helper.from_array(np.array([9], np.int64), "z9"),
        numpy_helper.from_array(np.array([HEIGHT], np.int64), "z30"),
        numpy_helper.from_array(np.array([1], np.int64), "ax1"),
        numpy_helper.from_array(np.array([2], np.int64), "ax2"),
        numpy_helper.from_array(np.array([3], np.int64), "ax3"),
        numpy_helper.from_array(np.array([HEIGHT], np.int64), "to30"),
    ]
    nodes = [
        n("ReduceSum", ["input"], ["occ"], axes=[1], keepdims=1),
        n("Slice", ["input", "z0", "z1", "ax1"], ["is0"]),
        n("Slice", ["input", "z8", "z9", "ax1"], ["is8"]),
        n("Sub", ["occ", "is0"], ["nonbg"]),
        n("Sub", ["nonbg", "is8"], ["keycells"]),
        # key bbox (kR0, kC0) + extents
        n("ReduceMax", ["keycells"], ["kRowHas"], axes=[3], keepdims=1),
        n("ReduceMax", ["keycells"], ["kColHas"], axes=[2], keepdims=1),
        n("Mul", ["kRowHas", "rowidx"], ["krh"]),
        n("Sub", ["one", "kRowHas"], ["krInv"]), n("Mul", ["krInv", "big"], ["krB"]),
        n("Add", ["krh", "krB"], ["krmin"]), n("ReduceMin", ["krmin"], ["kR0"], axes=[2], keepdims=1),
        n("ReduceMax", ["krh"], ["kR1"], axes=[2], keepdims=1),
        n("Sub", ["kR1", "kR0"], ["kRspan"]), n("Add", ["kRspan", "one"], ["kRH"]),
        n("Mul", ["kColHas", "colidx"], ["kch"]),
        n("Sub", ["one", "kColHas"], ["kcInv"]), n("Mul", ["kcInv", "big"], ["kcB"]),
        n("Add", ["kch", "kcB"], ["kcmin"]), n("ReduceMin", ["kcmin"], ["kC0"], axes=[3], keepdims=1),
        n("ReduceMax", ["kch"], ["kC1"], axes=[3], keepdims=1),
        n("Sub", ["kC1", "kC0"], ["kCspan"]), n("Add", ["kCspan", "one"], ["kCW"]),
        # crop key to top-left -> W kernel
        n("Add", ["rowidx", "kR0"], ["kgi0"]), n("Clip", ["kgi0", "cmin", "cmax"], ["kgic"]),
        n("Cast", ["kgic"], ["kgii"], to=TensorProto.INT64), n("Reshape", ["kgii", "to30"], ["kgi"]),
        n("Add", ["colidx", "kC0"], ["cgi0"]), n("Clip", ["cgi0", "cmin", "cmax"], ["cgic"]),
        n("Cast", ["cgic"], ["cgii"], to=TensorProto.INT64), n("Reshape", ["cgii", "to30"], ["cgi"]),
        n("Gather", ["input", "kgi"], ["keyR"], axis=2),
        n("Gather", ["keyR", "cgi"], ["keyCrop"], axis=3),
        n("Less", ["rowidx", "kRH"], ["mr_b"]), n("Cast", ["mr_b"], ["mr"], to=F),
        n("Less", ["colidx", "kCW"], ["mc_b"]), n("Cast", ["mc_b"], ["mc"], to=F),
        n("Mul", ["mr", "mc"], ["kmask"]), n("Mul", ["keyCrop", "kmask"], ["keyM"]),
        n("Slice", ["keyM", "z0", "z5", "ax2"], ["wr"]), n("Slice", ["wr", "z0", "z5", "ax3"], ["W"]),
        # key reading-offset = (key first-col in min row) - kC0
        n("Mul", ["cellid", "keycells"], ["kcid"]),
        n("Sub", ["one", "keycells"], ["knot"]), n("Mul", ["knot", "big"], ["knotbig"]),
        n("Add", ["kcid", "knotbig"], ["kcid2"]),
        n("ReduceMin", ["kcid2"], ["kFirstCid"], axes=[2, 3], keepdims=1),
        n("Sub", ["kFirstCid", "one"], ["kfc0"]),
        n("Mul", ["kfc0", "invw"], ["kfcq"]), n("Floor", ["kfcq"], ["kfcf"]),
        n("Mul", ["kfcf", "wf"], ["kfcfw"]), n("Sub", ["kfc0", "kfcfw"], ["kFirstCol"]),
        n("Sub", ["kFirstCol", "kC0"], ["keyOff"]),
        # one impulse per 8-component (min-cellid = reading-order first cell)
        n("Mul", ["cellid", "is8"], ["cid8"]),
        n("Sub", ["one", "is8"], ["not8"]), n("Mul", ["not8", "big"], ["not8big"]),
        n("Add", ["cid8", "not8big"], ["lab0"]),
    ]
    for it in range(10):
        nodes += [
            n("Neg", ["lab%d" % it], ["neg%d" % it]),
            n("MaxPool", ["neg%d" % it], ["mp%d" % it], kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]),
            n("Neg", ["mp%d" % it], ["mn%d" % it]),
            n("Mul", ["mn%d" % it, "is8"], ["mn8_%d" % it]),
            n("Add", ["mn8_%d" % it, "not8big"], ["lab%d" % (it + 1)]),
        ]
    nodes += [
        n("Sub", ["cellid", "lab10"], ["cdiff"]), n("Mul", ["cdiff", "cdiff"], ["cdiff2"]),
        n("Less", ["cdiff2", "half"], ["imp_b"]), n("Cast", ["imp_b"], ["impf"], to=F),
        n("Mul", ["impf", "is8"], ["impulse"]),
        # shift impulse left by keyOff -> bbox top-left corner
        n("Add", ["colidx", "keyOff"], ["shc0"]), n("Clip", ["shc0", "cmin", "cmax"], ["shc1"]),
        n("Cast", ["shc1"], ["shc2"], to=TensorProto.INT64), n("Reshape", ["shc2", "to30"], ["shc"]),
        n("Gather", ["impulse", "shc"], ["cornerImp"], axis=3),
        # stamp
        n("ConvTranspose", ["cornerImp", "W"], ["stamp0"], strides=[1, 1], kernel_shape=[K, K]),
        n("Slice", ["stamp0", "z0", "z30", "ax2"], ["stampR"]),
        n("Slice", ["stampR", "z0", "z30", "ax3"], ["stamp"]),
        n("Mul", ["stamp", "is8"], ["stampM"]),
        n("Sub", ["occ", "is8"], ["bgv"]),
        n("Mul", ["bgv", "e0"], ["bgL"]),
        n("Add", ["stampM", "bgL"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "stamp_key_on_eights",
                              [helper.make_tensor_value_info("input", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              [helper.make_tensor_value_info("output", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_stamp_key_on_eights(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
