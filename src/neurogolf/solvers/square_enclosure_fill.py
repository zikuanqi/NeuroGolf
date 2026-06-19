"""Solver: fill enclosed regions that are solid squares with 2 (task 102).

Background sealed off from the grid border becomes ``2`` only when the enclosed
region is a solid square (its cells fill a square bounding box); L-shapes,
non-square rectangles and holed regions are left untouched.

Pipeline: a 4-connected border flood marks reachable background (enclosed = bg
minus reached); an 8-connected ``MaxPool`` max-cellid flood labels the enclosed
components; per component the bounding box comes from min/max row/col floods and
the cell count from an all-pairs label match; a component qualifies when
``h == w`` and ``count == h*w``.
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
NREACH = 30
NLAB = 16
NCELL = HEIGHT * WIDTH


def _ref(g: np.ndarray) -> Optional[np.ndarray]:
    H, W = g.shape
    bg = (g == 0)
    reach = np.zeros((H, W), bool); q = deque()
    for r in range(H):
        for c in (0, W - 1):
            if bg[r, c] and not reach[r, c]:
                reach[r, c] = True; q.append((r, c))
    for c in range(W):
        for r in (0, H - 1):
            if bg[r, c] and not reach[r, c]:
                reach[r, c] = True; q.append((r, c))
    while q:
        y, x = q.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            yy, xx = y + dr, x + dc
            if 0 <= yy < H and 0 <= xx < W and bg[yy, xx] and not reach[yy, xx]:
                reach[yy, xx] = True; q.append((yy, xx))
    enc = bg & ~reach
    out = g.copy()
    seen = np.zeros((H, W), bool)
    for r in range(H):
        for c in range(W):
            if enc[r, c] and not seen[r, c]:
                cells = []; qq = deque([(r, c)]); seen[r, c] = True
                while qq:
                    y, x = qq.popleft(); cells.append((y, x))
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < H and 0 <= nx < W and enc[ny, nx] and not seen[ny, nx]:
                                seen[ny, nx] = True; qq.append((ny, nx))
                ys = [y for y, x in cells]; xs = [x for y, x in cells]
                h = max(ys) - min(ys) + 1; w = max(xs) - min(xs) + 1
                if h == w and len(cells) == h * w:
                    for y, x in cells:
                        out[y, x] = 2
    return out


def _detect(task: dict) -> bool:
    saw = False
    saw_fill = False
    for ex in all_examples(task):
        i, o = ex["input"], ex["output"]
        if not i or not i[0] or len(i) > HEIGHT or len(i[0]) > WIDTH:
            continue
        gi = np.array(i)
        r = _ref(gi)
        if not np.array_equal(r, np.array(o)):
            return False
        saw = True
        if not np.array_equal(r, gi):
            saw_fill = True
    return saw and saw_fill


def _build() -> onnx.ModelProto:
    n = helper.make_node
    e0 = np.zeros((1, CHANNELS, 1, 1), np.float32); e0[0, 0] = 1.0
    e2 = np.zeros((1, CHANNELS, 1, 1), np.float32); e2[0, 2] = 1.0
    cellid = (np.arange(NCELL).reshape(1, 1, HEIGHT, WIDTH) + 1).astype(np.float32)
    frame = np.zeros((1, 1, HEIGHT, WIDTH), np.float32)
    frame[0, 0, 0, :] = 1; frame[0, 0, HEIGHT - 1, :] = 1
    frame[0, 0, :, 0] = 1; frame[0, 0, :, WIDTH - 1] = 1
    init = [
        numpy_helper.from_array(e0, "e0"),
        numpy_helper.from_array(e2, "e2"),
        numpy_helper.from_array(cellid, "cellid"),
        numpy_helper.from_array(frame, "frame"),
        numpy_helper.from_array(np.arange(HEIGHT, dtype=np.float32).reshape(1, 1, HEIGHT, 1), "rowidx"),
        numpy_helper.from_array(np.arange(WIDTH, dtype=np.float32).reshape(1, 1, 1, WIDTH), "colidx"),
        numpy_helper.from_array(np.array(1.0, np.float32), "one"),
        numpy_helper.from_array(np.array(1e6, np.float32), "big"),
        numpy_helper.from_array(np.array(0.5, np.float32), "half"),
        numpy_helper.from_array(np.array([0], np.int64), "c0s"),
        numpy_helper.from_array(np.array([1], np.int64), "c1e"),
        numpy_helper.from_array(np.array([1], np.int64), "ax1"),
        numpy_helper.from_array(np.array([1, 1, NCELL, 1], np.int64), "to_col"),
        numpy_helper.from_array(np.array([1, 1, 1, NCELL], np.int64), "to_row"),
        numpy_helper.from_array(np.array([1, 1, HEIGHT, WIDTH], np.int64), "to_grid"),
    ]
    nodes = [
        n("ReduceSum", ["input"], ["occ"], axes=[1], keepdims=1),
        n("Slice", ["input", "c0s", "c1e", "ax1"], ["is0"]),
        n("Sub", ["one", "occ"], ["invocc"]),
        n("Add", ["is0", "invocc"], ["pass"]),
        n("Mul", ["pass", "frame"], ["reach0"]),
    ]
    cur = "reach0"
    for k in range(NREACH):
        h, v, d, nx = f"rh{k}", f"rv{k}", f"rd{k}", f"reach{k + 1}"
        nodes += [
            n("MaxPool", [cur], [h], kernel_shape=[1, 3], strides=[1, 1], pads=[0, 1, 0, 1]),
            n("MaxPool", [cur], [v], kernel_shape=[3, 1], strides=[1, 1], pads=[1, 0, 1, 0]),
            n("Max", [h, v], [d]), n("Mul", [d, "pass"], [nx]),
        ]
        cur = nx
    nodes += [
        n("Mul", [cur, "is0"], ["reachBg"]),
        n("Sub", ["is0", "reachBg"], ["enc"]),            # enclosed bg (1,1,H,W)
        n("Sub", ["one", "enc"], ["nenc"]), n("Mul", ["nenc", "big"], ["nencBig"]),
        # label enclosed components (max-cellid flood, 8-connected)
        n("Mul", ["cellid", "enc"], ["lab0"]),
    ]
    cur = "lab0"
    for it in range(NLAB):
        nodes += [
            n("MaxPool", [cur], [f"lmp{it}"], kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]),
            n("Mul", [f"lmp{it}", "enc"], [f"lab{it + 1}"]),
        ]
        cur = f"lab{it + 1}"
    label = cur
    # per-component count via all-pairs label match
    nodes += [
        n("Reshape", [label, "to_col"], ["lcol"]),
        n("Reshape", [label, "to_row"], ["lrow"]),
        n("Equal", ["lcol", "lrow"], ["eqm"]), n("Cast", ["eqm"], ["eqmf"], to=F),
        n("ReduceSum", ["eqmf"], ["szc"], axes=[3], keepdims=1),
        n("Reshape", ["szc", "to_grid"], ["size"]),
        # bbox via min/max row/col floods
        n("Mul", ["rowidx", "enc"], ["maxR0"]),
        n("Mul", ["colidx", "enc"], ["maxC0"]),
        n("Mul", ["rowidx", "enc"], ["minRe"]), n("Add", ["minRe", "nencBig"], ["minR0"]),
        n("Mul", ["colidx", "enc"], ["minCe"]), n("Add", ["minCe", "nencBig"], ["minC0"]),
    ]
    mxR, mxC, mnR, mnC = "maxR0", "maxC0", "minR0", "minC0"
    for it in range(NLAB):
        nodes += [
            n("MaxPool", [mxR], [f"aR{it}"], kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]),
            n("Mul", [f"aR{it}", "enc"], [f"maxR{it + 1}"]),
            n("MaxPool", [mxC], [f"aC{it}"], kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]),
            n("Mul", [f"aC{it}", "enc"], [f"maxC{it + 1}"]),
            n("Neg", [mnR], [f"nR{it}"]),
            n("MaxPool", [f"nR{it}"], [f"pR{it}"], kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]),
            n("Neg", [f"pR{it}"], [f"mnRr{it}"]), n("Mul", [f"mnRr{it}", "enc"], [f"mnRe{it}"]),
            n("Add", [f"mnRe{it}", "nencBig"], [f"minR{it + 1}"]),
            n("Neg", [mnC], [f"nC{it}"]),
            n("MaxPool", [f"nC{it}"], [f"pC{it}"], kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]),
            n("Neg", [f"pC{it}"], [f"mnCr{it}"]), n("Mul", [f"mnCr{it}", "enc"], [f"mnCe{it}"]),
            n("Add", [f"mnCe{it}", "nencBig"], [f"minC{it + 1}"]),
        ]
        mxR, mxC = f"maxR{it + 1}", f"maxC{it + 1}"
        mnR, mnC = f"minR{it + 1}", f"minC{it + 1}"
    nodes += [
        n("Sub", [mxR, mnR], ["hspan"]), n("Add", ["hspan", "one"], ["hh"]),
        n("Sub", [mxC, mnC], ["wspan"]), n("Add", ["wspan", "one"], ["ww"]),
        n("Mul", ["hh", "ww"], ["area"]),
        n("Sub", ["size", "area"], ["dA"]), n("Mul", ["dA", "dA"], ["dA2"]),
        n("Less", ["dA2", "half"], ["eqA_b"]), n("Cast", ["eqA_b"], ["eqA"], to=F),
        n("Sub", ["hh", "ww"], ["dHW"]), n("Mul", ["dHW", "dHW"], ["dHW2"]),
        n("Less", ["dHW2", "half"], ["eqHW_b"]), n("Cast", ["eqHW_b"], ["eqHW"], to=F),
        n("Mul", ["eqA", "eqHW"], ["q0"]), n("Mul", ["q0", "enc"], ["qualify"]),
        n("Sub", ["one", "qualify"], ["keepM"]), n("Mul", ["input", "keepM"], ["kept"]),
        n("Mul", ["qualify", "e2"], ["fillL"]), n("Add", ["kept", "fillL"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "square_enclosure_fill",
                              [helper.make_tensor_value_info("input", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              [helper.make_tensor_value_info("output", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_square_enclosure_fill(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
