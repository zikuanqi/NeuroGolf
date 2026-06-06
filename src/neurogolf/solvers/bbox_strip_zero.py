"""Solver: crop to the bounding box of the non-majority colour, blanking it.

Like `bbox_strip`, the output is the input cropped to the tight rectangle around
every cell whose colour != bg, shifted to the top-left.  Here bg is the *majority*
colour and, unlike `bbox_strip`, those bg cells inside the crop are mapped to
background 0 (so they decode as colour 0, not the bg colour).  Task 259.

Build: the dynamic crop is identical to `bbox_strip` (ArgMax/ReduceMax bbox +
two-axis `Gather` translation + bbox mask).  A final remap zeroes the bg channel
of the cropped tensor and re-adds it on channel 0.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from ..grids import CHANNELS, HEIGHT, WIDTH, all_examples


OPSET = 11
IR_VERSION = 8


def _detect_bg(task: dict) -> Optional[int]:
    """Return bg in {0..9} s.t. output = bbox-of-non-bg with bg cells -> 0."""
    examples = list(all_examples(task))
    if not examples:
        return None
    best = None
    for bg in range(CHANNELS):
        ok = True
        saw = False
        for ex in examples:
            i, o = ex["input"], ex["output"]
            if not i or not i[0] or len(i) > HEIGHT or len(i[0]) > WIDTH:
                continue
            H, W = len(i), len(i[0])
            rs = [r for r in range(H) if any(i[r][c] != bg for c in range(W))]
            cs = [c for c in range(W) if any(i[r][c] != bg for r in range(H))]
            if not rs or not cs:
                ok = False
                break
            r0, r1, c0, c1 = rs[0], rs[-1], cs[0], cs[-1]
            h, w = r1 - r0 + 1, c1 - c0 + 1
            if len(o) != h or len(o[0]) != w:
                ok = False
                break
            for r in range(h):
                for c in range(w):
                    src = i[r0 + r][c0 + c]
                    want = 0 if src == bg else src
                    if o[r][c] != want:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
            saw = True
        if ok and saw:
            best = bg
            break
    return best


def _build(bg: int) -> onnx.ModelProto:
    F = TensorProto.FLOAT
    n = helper.make_node
    notbg = np.ones((1, CHANNELS, 1, 1), np.float32); notbg[0, bg] = 0.0
    e0 = np.zeros((1, CHANNELS, 1, 1), np.float32); e0[0, 0] = 1.0
    init = [
        numpy_helper.from_array(np.array([bg], np.int64), "bg_s"),
        numpy_helper.from_array(np.array([bg + 1], np.int64), "bg_e"),
        numpy_helper.from_array(np.array([1], np.int64), "bg_ax"),
        numpy_helper.from_array(np.arange(HEIGHT, dtype=np.float32).reshape(1, 1, HEIGHT, 1), "ar_h"),
        numpy_helper.from_array(np.arange(WIDTH, dtype=np.float32).reshape(1, 1, 1, WIDTH), "ar_w"),
        numpy_helper.from_array(np.array([HEIGHT], np.int64), "h_idx_shape"),
        numpy_helper.from_array(np.array([WIDTH], np.int64), "w_idx_shape"),
        numpy_helper.from_array(np.array(0.0, np.float32), "clip_min"),
        numpy_helper.from_array(np.array(float(HEIGHT - 1), np.float32), "clip_max"),
        numpy_helper.from_array(np.array(1.0, np.float32), "one_f"),
        numpy_helper.from_array(notbg, "notbg"),
        numpy_helper.from_array(e0, "e0"),
    ]
    nodes = [
        n("ReduceSum", ["input"], ["all_sum"], axes=[1], keepdims=1),
        n("Slice", ["input", "bg_s", "bg_e", "bg_ax"], ["bg_only"]),
        n("Sub", ["all_sum", "bg_only"], ["nonbg"]),                  # (1,1,H,W)
        n("ReduceMax", ["nonbg"], ["row_proj"], axes=[3], keepdims=1),
        n("ReduceMax", ["nonbg"], ["col_proj"], axes=[2], keepdims=1),
        n("Mul", ["row_proj", "ar_h"], ["row_pos"]),
        n("Mul", ["col_proj", "ar_w"], ["col_pos"]),
        n("ReduceMax", ["row_pos"], ["r_max"], axes=[2], keepdims=1),
        n("ReduceMax", ["col_pos"], ["c_max"], axes=[3], keepdims=1),
        n("ArgMax", ["row_proj"], ["r_min_i"], axis=2, keepdims=1),
        n("ArgMax", ["col_proj"], ["c_min_i"], axis=3, keepdims=1),
        n("Cast", ["r_min_i"], ["r_min"], to=F),
        n("Cast", ["c_min_i"], ["c_min"], to=F),
        n("Sub", ["r_max", "r_min"], ["h_m1"]),
        n("Sub", ["c_max", "c_min"], ["w_m1"]),
        n("Add", ["ar_h", "r_min"], ["sh_r_f"]),
        n("Clip", ["sh_r_f", "clip_min", "clip_max"], ["sh_r_c"]),
        n("Cast", ["sh_r_c"], ["sh_r_i"], to=TensorProto.INT64),
        n("Reshape", ["sh_r_i", "h_idx_shape"], ["sh_r_1d"]),
        n("Add", ["ar_w", "c_min"], ["sh_c_f"]),
        n("Clip", ["sh_c_f", "clip_min", "clip_max"], ["sh_c_c"]),
        n("Cast", ["sh_c_c"], ["sh_c_i"], to=TensorProto.INT64),
        n("Reshape", ["sh_c_i", "w_idx_shape"], ["sh_c_1d"]),
        n("Gather", ["input", "sh_r_1d"], ["sh_rows"], axis=2),
        n("Gather", ["sh_rows", "sh_c_1d"], ["shifted"], axis=3),
        n("Add", ["h_m1", "one_f"], ["h_f"]),
        n("Add", ["w_m1", "one_f"], ["w_f"]),
        n("Less", ["ar_h", "h_f"], ["h_le_b"]),
        n("Less", ["ar_w", "w_f"], ["w_le_b"]),
        n("Cast", ["h_le_b"], ["h_le"], to=F),
        n("Cast", ["w_le_b"], ["w_le"], to=F),
        n("Mul", ["h_le", "w_le"], ["bbox_mask"]),
        n("Mul", ["shifted", "bbox_mask"], ["cropped"]),             # bbox at top-left
        # remap the bg colour inside the crop to background 0
        n("Slice", ["cropped", "bg_s", "bg_e", "bg_ax"], ["crop_bg"]),  # (1,1,H,W)
        n("Mul", ["cropped", "notbg"], ["crop_z"]),                  # drop bg channel
        n("Mul", ["e0", "crop_bg"], ["add0"]),                       # channel0 where bg was
        n("Add", ["crop_z", "add0"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "bbox_strip_zero",
                              [helper.make_tensor_value_info("input", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              [helper.make_tensor_value_info("output", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_bbox_strip_zero(task: dict) -> Optional[onnx.ModelProto]:
    bg = _detect_bg(task)
    if bg is None:
        return None
    return _build(bg)
