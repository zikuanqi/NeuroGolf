"""Solver: fold a quadrant across a divider cross into a 4-fold symmetric grid (task 109).

The input has one full divider row and one full divider column of a colour L
that meet at the centre, plus a shape of colour S sitting in one quadrant.  The
output reflects the shape into all four quadrants (4-fold symmetry about the
divider crossing), recolours it to L, and drops the divider row and column so
the result is (H-1) x (W-1).

Everything runs at inference time: L is the channel that owns a full row and a
full column (content-aware, so 30x30 padding is ignored), S is the other
content colour, the divider position gives the reflection centre, the four
reflections are runtime-index `Gather`s combined with `Max`, the recolour adds
`e_L - e_S`, and two more `Gather`s skip the divider row/column.
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
    H, W = g.shape
    Ls = [c for c in np.unique(g) if c != 0
          and any((g[r] == c).all() for r in range(H))
          and any((g[:, x] == c).all() for x in range(W))]
    if len(Ls) != 1:
        return None
    L = Ls[0]
    rds = [r for r in range(H) if (g[r] == L).all()]
    cds = [x for x in range(W) if (g[:, x] == L).all()]
    if len(rds) != 1 or len(cds) != 1:
        return None
    rd, cd = rds[0], cds[0]
    Ss = [c for c in np.unique(g) if c not in (0, L)]
    if len(Ss) != 1:
        return None
    S = Ss[0]
    full = g.copy()
    for r in range(H):
        for c in range(W):
            for (rr, cc) in ((r, 2 * cd - c), (2 * rd - r, c), (2 * rd - r, 2 * cd - c)):
                if 0 <= rr < H and 0 <= cc < W:
                    full[rr, cc] = max(full[rr, cc], g[r, c])
    full[full == S] = L
    return np.delete(np.delete(full, rd, axis=0), cd, axis=1)


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
    F = TensorProto.FLOAT
    I = TensorProto.INT64
    n = helper.make_node
    rowf = np.arange(HEIGHT, dtype=np.float32).reshape(1, 1, HEIGHT, 1)
    colf = np.arange(WIDTH, dtype=np.float32).reshape(1, 1, 1, WIDTH)
    init = [
        numpy_helper.from_array(rowf, "rowf"),
        numpy_helper.from_array(colf, "colf"),
        numpy_helper.from_array(np.arange(HEIGHT, dtype=np.int64), "ar_r"),
        numpy_helper.from_array(np.arange(WIDTH, dtype=np.int64), "ar_c"),
        numpy_helper.from_array(np.array(0.5, np.float32), "half"),
        numpy_helper.from_array(np.array(1.0, np.float32), "one"),
        numpy_helper.from_array(np.array(2.0, np.float32), "two"),
        numpy_helper.from_array(np.array(0.0, np.float32), "zero"),
        numpy_helper.from_array(np.array(float(HEIGHT - 1), np.float32), "hmax"),
        numpy_helper.from_array(np.array(float(WIDTH - 1), np.float32), "wmax"),
        numpy_helper.from_array(np.array(0, np.int64), "i0"),
        numpy_helper.from_array(np.array(HEIGHT - 1, np.int64), "ihmax"),
        numpy_helper.from_array(np.array(WIDTH - 1, np.int64), "iwmax"),
        numpy_helper.from_array(np.array([1], np.int64), "shp1"),
        numpy_helper.from_array(
            np.array([1.0] + [0.0] * (CHANNELS - 1), np.float32).reshape(1, CHANNELS, 1, 1), "e0"),
    ]

    def cf(b, name):
        return n("Cast", [b], [name], to=F)

    nodes = [
        n("ReduceSum", ["input"], ["content"], axes=[1], keepdims=1),
        n("ReduceMax", ["content"], ["row_has"], axes=[3], keepdims=1),
        n("ReduceMax", ["content"], ["col_has"], axes=[2], keepdims=1),
        n("ReduceSum", ["row_has"], ["Hsz"]),
        n("ReduceSum", ["col_has"], ["Wsz"]),
        # per-channel / per-row counts
        n("ReduceSum", ["input"], ["cnt_r"], axes=[3], keepdims=1),    # (1,10,H,1)
        n("ReduceSum", ["content"], ["con_r"], axes=[3], keepdims=1),  # (1,1,H,1)
        n("Sub", ["con_r", "cnt_r"], ["d_r"]),
        n("Less", ["d_r", "half"], ["fr_b"]), cf("fr_b", "fr0"),       # row all-k
        n("Greater", ["con_r", "half"], ["rposr_b"]), cf("rposr_b", "rposr"),
        n("Mul", ["fr0", "rposr"], ["fr"]),                           # exclude empty rows
        n("ReduceMax", ["fr"], ["hfr"], axes=[2], keepdims=1),         # (1,10,1,1)
        n("ReduceSum", ["input"], ["cnt_c"], axes=[2], keepdims=1),
        n("ReduceSum", ["content"], ["con_c"], axes=[2], keepdims=1),
        n("Sub", ["con_c", "cnt_c"], ["d_c"]),
        n("Less", ["d_c", "half"], ["fc_b"]), cf("fc_b", "fc0"),
        n("Greater", ["con_c", "half"], ["cposc_b"]), cf("cposc_b", "cposc"),
        n("Mul", ["fc0", "cposc"], ["fc"]),                           # exclude empty cols
        n("ReduceMax", ["fc"], ["hfc"], axes=[3], keepdims=1),
        n("Mul", ["hfr", "hfc"], ["LhasRC"]),
        n("Sub", ["one", "e0"], ["note0"]),
        n("Mul", ["input", "note0"], ["ic"]),           # colour-only input (ch0 zeroed)
        n("Mul", ["LhasRC", "note0"], ["Lch"]),                       # (1,10,1,1) divider colour
        # shape colour = has content, not L, not bg
        n("ReduceMax", ["input"], ["hc"], axes=[2, 3], keepdims=1),    # (1,10,1,1)
        n("Sub", ["one", "Lch"], ["notL"]),
        n("Mul", ["hc", "notL"], ["s0"]),
        n("Mul", ["s0", "note0"], ["Sch"]),                           # (1,10,1,1)
        # divider cells / positions
        n("Mul", ["input", "Lch"], ["Lcells_c"]),
        n("ReduceSum", ["Lcells_c"], ["Lmask"], axes=[1], keepdims=1),  # (1,1,H,W)
        n("ReduceSum", ["Lmask"], ["Lrow_cnt"], axes=[3], keepdims=1),  # (1,1,H,1)
        n("Sub", ["con_r", "Lrow_cnt"], ["drd"]),
        n("Less", ["drd", "half"], ["isrd_b"]), cf("isrd_b", "isrd0"),
        n("Greater", ["con_r", "half"], ["rpos_b"]), cf("rpos_b", "rpos"),
        n("Mul", ["isrd0", "rpos"], ["isrd"]),                        # (1,1,H,1) one at rd
        n("Mul", ["isrd", "rowf"], ["rd_w"]),
        n("ReduceSum", ["rd_w"], ["rd"]),                             # scalar rd
        n("ReduceSum", ["Lmask"], ["Lcol_cnt"], axes=[2], keepdims=1),
        n("Sub", ["con_c", "Lcol_cnt"], ["dcd"]),
        n("Less", ["dcd", "half"], ["iscd_b"]), cf("iscd_b", "iscd0"),
        n("Greater", ["con_c", "half"], ["cpos_b"]), cf("cpos_b", "cpos"),
        n("Mul", ["iscd0", "cpos"], ["iscd"]),
        n("Mul", ["iscd", "colf"], ["cd_w"]),
        n("ReduceSum", ["cd_w"], ["cd"]),                             # scalar cd
        # 1-D float ramps
        n("Cast", ["ar_r"], ["ar_r_f"], to=F),
        n("Cast", ["ar_c"], ["ar_c_f"], to=F),
        # reflection indices (float)  idx_v = 2*rd - r,  idx_h = 2*cd - c
        n("Mul", ["rd", "two"], ["twrd"]), n("Reshape", ["twrd", "shp1"], ["twrd1"]),
        n("Sub", ["twrd1", "ar_r_f"], ["idxv_f"]),                    # (H,)
        n("Mul", ["cd", "two"], ["twcd"]), n("Reshape", ["twcd", "shp1"], ["twcd1"]),
        n("Sub", ["twcd1", "ar_c_f"], ["idxh_f"]),                    # (W,)
        # validity masks
        n("Greater", ["idxv_f", "neghalf"], ["vlo_b"]), cf("vlo_b", "vlo1"),
        n("Less", ["idxv_f", "hmaxh"], ["vhi_b"]), cf("vhi_b", "vhi1"),
        n("Mul", ["vlo1", "vhi1"], ["validv1"]), n("Reshape", ["validv1", "shp_r"], ["validv"]),
        n("Greater", ["idxh_f", "neghalf"], ["hlo_b"]), cf("hlo_b", "hlo1"),
        n("Less", ["idxh_f", "wmaxh"], ["hhi_b"]), cf("hhi_b", "hhi1"),
        n("Mul", ["hlo1", "hhi1"], ["validh1"]), n("Reshape", ["validh1", "shp_c"], ["validh"]),
        # clipped indices (float -> int)
        n("Max", ["idxv_f", "zero"], ["idxv_a"]), n("Min", ["idxv_a", "hmax"], ["idxv_cf"]),
        n("Cast", ["idxv_cf"], ["idxv_c"], to=I),
        n("Max", ["idxh_f", "zero"], ["idxh_a"]), n("Min", ["idxh_a", "wmax"], ["idxh_cf"]),
        n("Cast", ["idxh_cf"], ["idxh_c"], to=I),
        # four reflections
        n("Gather", ["ic", "idxv_c"], ["gv"], axis=2),
        n("Mul", ["gv", "validv"], ["vr"]),
        n("Gather", ["ic", "idxh_c"], ["gh"], axis=3),
        n("Mul", ["gh", "validh"], ["hr"]),
        n("Gather", ["vr", "idxh_c"], ["gvh"], axis=3),
        n("Mul", ["gvh", "validh"], ["vhr"]),
        n("Max", ["ic", "vr"], ["m1"]),
        n("Max", ["hr", "vhr"], ["m2"]),
        n("Max", ["m1", "m2"], ["closure"]),                          # (1,10,H,W)
        # recolour S -> L
        n("Mul", ["closure", "Sch"], ["sc"]),
        n("ReduceSum", ["sc"], ["sMask"], axes=[1], keepdims=1),      # (1,1,H,W)
        n("Sub", ["Lch", "Sch"], ["LmS"]),
        n("Mul", ["LmS", "sMask"], ["recold"]),
        n("Add", ["closure", "recold"], ["recol"]),                   # (1,10,H,W)
        # drop divider row & column:  src row = i + (i>=rd)   (float -> int)
        n("Reshape", ["rd", "shp1"], ["rd_1"]), n("Sub", ["rd_1", "half"], ["rdm"]),
        n("Greater", ["ar_r_f", "rdm"], ["sk_rb"]), cf("sk_rb", "sk_rf"),
        n("Add", ["ar_r_f", "sk_rf"], ["sr_f0"]), n("Min", ["sr_f0", "hmax"], ["sr_f"]),
        n("Cast", ["sr_f"], ["sr_idx"], to=I),
        n("Gather", ["recol", "sr_idx"], ["res_r"], axis=2),
        n("Reshape", ["cd", "shp1"], ["cd_1"]), n("Sub", ["cd_1", "half"], ["cdm"]),
        n("Greater", ["ar_c_f", "cdm"], ["sk_cb"]), cf("sk_cb", "sk_cf"),
        n("Add", ["ar_c_f", "sk_cf"], ["sc_f0"]), n("Min", ["sc_f0", "wmax"], ["sc_f"]),
        n("Cast", ["sc_f"], ["sc_idx"], to=I),
        n("Gather", ["res_r", "sc_idx"], ["res_rc"], axis=3),
        # zero rows >= H-1 and cols >= W-1
        n("Sub", ["Hsz", "one"], ["Hm1"]),
        n("Sub", ["Wsz", "one"], ["Wm1"]),
        n("Less", ["rowf", "Hm1"], ["okr_b"]), cf("okr_b", "okr"),
        n("Less", ["colf", "Wm1"], ["okc_b"]), cf("okc_b", "okc"),
        n("Mul", ["res_rc", "okr"], ["o1"]),
        n("Mul", ["o1", "okc"], ["colors_g"]),
        # rebuild background channel 0: set where in-grid and no colour
        n("Mul", ["okr", "okc"], ["gridmask"]),
        n("ReduceSum", ["colors_g"], ["hasColor"], axes=[1], keepdims=1),
        n("Sub", ["one", "hasColor"], ["nocolor"]),
        n("Mul", ["gridmask", "nocolor"], ["ch0v"]),
        n("Mul", ["e0", "ch0v"], ["bg_add"]),
        n("Add", ["colors_g", "bg_add"], ["output"]),
    ]
    # the two divider-skip comparisons need >= rd, i.e. rowf > rd-0.5
    init.append(numpy_helper.from_array(np.array(-0.5, np.float32), "neghalf"))
    init.append(numpy_helper.from_array(np.array([1, 1, HEIGHT, 1], np.int64), "shp_r"))
    init.append(numpy_helper.from_array(np.array([1, 1, 1, WIDTH], np.int64), "shp_c"))
    init.append(numpy_helper.from_array(np.array([1, 1, HEIGHT, 1], np.int64), "shp_rH"))
    init.append(numpy_helper.from_array(np.array([1, 1, 1, WIDTH], np.int64), "shp_cW"))
    init.append(numpy_helper.from_array(np.array(float(HEIGHT) - 0.5, np.float32), "hmaxh"))
    init.append(numpy_helper.from_array(np.array(float(WIDTH) - 0.5, np.float32), "wmaxh"))

    graph = helper.make_graph(nodes, "divider_fold",
                              [helper.make_tensor_value_info("input", F, FULL)],
                              [helper.make_tensor_value_info("output", F, FULL)],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_divider_fold(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
