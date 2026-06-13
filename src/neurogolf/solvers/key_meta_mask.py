"""Solver: mask a colour key by the arrangement of big blocks (task 170).

The grid holds an ``N x N`` colour key (N = 3 or 4) and a meta-pattern of
solid ``s x s`` single-colour blocks arranged on an ``N x N`` grid.  The
output is the key with every cell zeroed whose meta-position holds no block::

    blocks:  X . X      key:  3 1 7      out:  3 0 7
             . X .            2 8 9            0 8 0
             X . X            3 4 6            3 0 6

The block colour B is the most frequent colour.  Because B may also appear
*inside* the key, the pattern blocks are isolated by **erode-then-dilate** of
the B-mask (a 3x3 erosion kills the key's B cells but keeps block interiors,
and one dilation restores each block exactly).  The key is then all non-bg
cells outside the pattern; its bbox gives N, and the blocks' bbox sampled at
block centres gives the occupancy mask.
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


def _erode(m: np.ndarray) -> np.ndarray:
    H, W = m.shape
    out = np.ones((H, W), bool)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            tmp = np.ones((H, W), bool)
            rs = slice(max(0, dr), H + min(0, dr)); rd = slice(max(0, -dr), H + min(0, -dr))
            cs = slice(max(0, dc), W + min(0, dc)); cd = slice(max(0, -dc), W + min(0, -dc))
            tmp[rd, cd] = m[rs, cs]
            out &= tmp
    return out & m


def _dilate(m: np.ndarray) -> np.ndarray:
    H, W = m.shape
    out = np.zeros((H, W), bool)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            tmp = np.zeros((H, W), bool)
            rs = slice(max(0, dr), H + min(0, dr)); rd = slice(max(0, -dr), H + min(0, -dr))
            cs = slice(max(0, dc), W + min(0, dc)); cd = slice(max(0, -dc), W + min(0, -dc))
            tmp[rd, cd] = m[rs, cs]
            out |= tmp
    return out


def _ref(g: np.ndarray) -> Optional[np.ndarray]:
    u, cn = np.unique(g[g != 0], return_counts=True)
    if len(u) < 2:
        return None
    B = int(u[np.argmax(cn)])
    pmask = _dilate(_erode(g == B))
    km = (g != 0) & ~pmask
    ys, xs = np.where(km)
    if len(ys) == 0:
        return None
    kr, kc = ys.min(), xs.min()
    kh, kw = ys.max() - kr + 1, xs.max() - kc + 1
    if kh != kw or kh > 4:
        return None
    N = kh
    key = g[kr:kr + N, kc:kc + N]
    by, bx = np.where(pmask)
    if len(by) == 0:
        return None
    br0, bc0 = by.min(), bx.min()
    h, w = by.max() - br0 + 1, bx.max() - bc0 + 1
    if h != w or h % N:
        return None
    s = h // N
    occ = np.zeros((N, N), bool)
    for a in range(N):
        for b in range(N):
            occ[a, b] = pmask[br0 + a * s + s // 2, bc0 + b * s + s // 2]
    return np.where(occ, key, 0)


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
    e0 = np.zeros((1, CHANNELS, 1, 1), np.float32); e0[0, 0] = 1.0
    init = [
        numpy_helper.from_array(e0, "e0"),
        numpy_helper.from_array(np.arange(HEIGHT, dtype=np.float32).reshape(1, 1, HEIGHT, 1), "ah"),
        numpy_helper.from_array(np.arange(4, dtype=np.float32).reshape(1, 1, 4, 1), "ar4"),
        numpy_helper.from_array(np.arange(4, dtype=np.float32).reshape(1, 1, 1, 4), "ac4"),
        numpy_helper.from_array(np.array(0.5, np.float32), "half"),
        numpy_helper.from_array(np.array(1.0, np.float32), "one"),
        numpy_helper.from_array(np.array(2.0, np.float32), "two"),
        numpy_helper.from_array(np.array(0.0, np.float32), "zero"),
        numpy_helper.from_array(np.array(29.0, np.float32), "max29"),
        numpy_helper.from_array(np.array(10000.0, np.float32), "BIG"),
        numpy_helper.from_array(np.array([0], np.int64), "c0s"),
        numpy_helper.from_array(np.array([1], np.int64), "c1e"),
        numpy_helper.from_array(np.array([4], np.int64), "shape4"),
        numpy_helper.from_array(np.array([1], np.int64), "ax1"),
        numpy_helper.from_array(np.array([0, 0, 0, 0, 0, 0, 26, 26], np.int64), "padOut"),
    ]

    def idx4(base, scale, off, pre):
        """indices = clip(base + ar4*scale + off) -> int64 (4,)"""
        return [
            n("Mul", ["ar4", scale], [f"{pre}_m"]),
            n("Add", [f"{pre}_m", base], [f"{pre}_a"]),
            n("Add", [f"{pre}_a", off], [f"{pre}_o"]) if off else
            n("Add", [f"{pre}_a", "zero"], [f"{pre}_o"]),
            n("Clip", [f"{pre}_o", "zero", "max29"], [f"{pre}_c"]),
            n("Cast", [f"{pre}_c"], [f"{pre}_i"], to=I64),
            n("Reshape", [f"{pre}_i", "shape4"], [f"{pre}"]),
        ]

    nodes = [
        n("ReduceSum", ["input"], ["occ"], axes=[1], keepdims=1),
        n("Slice", ["input", "c0s", "c1e", "ax1"], ["is0"]),
        n("Sub", ["occ", "is0"], ["nonbg"]),
        # block colour = most frequent channel (excluding bg)
        n("ReduceSum", ["input"], ["cnt"], axes=[2, 3], keepdims=1),
        n("Mul", ["e0", "BIG"], ["bigE0"]),
        n("Sub", ["cnt", "bigE0"], ["cntn"]),
        n("ReduceMax", ["cntn"], ["mxc"], axes=[1], keepdims=1),
        n("Sub", ["mxc", "half"], ["mxh"]),
        n("Greater", ["cntn", "mxh"], ["bs_b"]), n("Cast", ["bs_b"], ["Bsel"], to=F),
        n("Mul", ["input", "Bsel"], ["Bx"]),
        n("ReduceSum", ["Bx"], ["Bmask"], axes=[1], keepdims=1),
        # pattern = dilate(erode(Bmask))
        n("Neg", ["Bmask"], ["nB"]),
        n("MaxPool", ["nB"], ["mpB"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        n("Neg", ["mpB"], ["erB0"]),
        n("Mul", ["erB0", "Bmask"], ["erB"]),
        n("MaxPool", ["erB"], ["pmask"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        # key mask + bbox
        n("Sub", ["one", "pmask"], ["invp"]),
        n("Mul", ["nonbg", "invp"], ["km"]),
        n("ReduceMax", ["km"], ["rowK"], axes=[3], keepdims=1),
        n("ReduceMax", ["km"], ["colK"], axes=[2], keepdims=1),
        n("ArgMax", ["rowK"], ["kri"], axis=2, keepdims=1), n("Cast", ["kri"], ["kr"], to=F),
        n("ArgMax", ["colK"], ["kci"], axis=3, keepdims=1), n("Cast", ["kci"], ["kc"], to=F),
        n("Mul", ["rowK", "ah"], ["krp"]),
        n("ReduceMax", ["krp"], ["krM"], axes=[2], keepdims=1),
        n("Sub", ["krM", "kr"], ["Nm1"]),
        n("Add", ["Nm1", "one"], ["N"]),
        # pattern bbox + block size
        n("ReduceMax", ["pmask"], ["rowP"], axes=[3], keepdims=1),
        n("ReduceMax", ["pmask"], ["colP"], axes=[2], keepdims=1),
        n("ArgMax", ["rowP"], ["bri"], axis=2, keepdims=1), n("Cast", ["bri"], ["br0"], to=F),
        n("ArgMax", ["colP"], ["bci"], axis=3, keepdims=1), n("Cast", ["bci"], ["bc0"], to=F),
        n("Mul", ["rowP", "ah"], ["brp"]),
        n("ReduceMax", ["brp"], ["brM"], axes=[2], keepdims=1),
        n("Sub", ["brM", "br0"], ["hm1"]), n("Add", ["hm1", "one"], ["hP"]),
        n("Div", ["hP", "N"], ["s"]),
        n("Div", ["s", "two"], ["s_2"]), n("Floor", ["s_2"], ["s2"]),
    ]
    # key gather indices (rows kr+0..3, cols kc+0..3)
    nodes += idx4("kr", "one", None, "kRows")
    nodes += idx4("kc", "one", None, "kCols")
    # meta sample indices (br0 + a*s + s2)
    nodes += idx4("br0", "s", "s2", "mRows")
    nodes += idx4("bc0", "s", "s2", "mCols")
    nodes += [
        n("Gather", ["input", "kRows"], ["kg1"], axis=2),
        n("Gather", ["kg1", "kCols"], ["key44"], axis=3),       # (1,10,4,4)
        n("Gather", ["pmask", "mRows"], ["og1"], axis=2),
        n("Gather", ["og1", "mCols"], ["occ44"], axis=3),       # (1,1,4,4)
        # valid window (a < N) x (b < N)
        n("Sub", ["N", "half"], ["Nh"]),
        n("Less", ["ar4", "Nh"], ["vr_b"]), n("Cast", ["vr_b"], ["vr"], to=F),
        n("Less", ["ac4", "Nh"], ["vc_b"]), n("Cast", ["vc_b"], ["vc"], to=F),
        n("Mul", ["vr", "vc"], ["valid"]),
        n("Mul", ["occ44", "valid"], ["occm"]),
        n("Sub", ["valid", "occm"], ["bgm"]),
        n("Mul", ["key44", "occm"], ["paint"]),
        n("Mul", ["bgm", "e0"], ["bgp"]),
        n("Add", ["paint", "bgp"], ["out44"]),
        n("Pad", ["out44", "padOut"], ["output"], mode="constant"),
    ]
    graph = helper.make_graph(nodes, "key_meta_mask",
                              [helper.make_tensor_value_info("input", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              [helper.make_tensor_value_info("output", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_key_meta_mask(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
