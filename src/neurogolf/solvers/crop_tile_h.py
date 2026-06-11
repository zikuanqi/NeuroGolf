"""Solver: crop the shape's bounding box and tile it twice horizontally (task 57).

The single coloured shape is cropped to its tight bounding box (background 0)
and the crop is repeated side-by-side once, giving an ``H x 2W`` output::

    . X .            X .  . X .  .          X . X .
    X X X    bbox    X X  X  ->  tiled      X X X X X X
    X . X    ---->   X .  X            (the crop placed at cols [0,W) and [W,2W))

The crop-to-top-left reuses the bbox-strip pipeline (two-axis ``Gather`` plus a
bbox mask); the second copy is the cropped tensor shifted right by the bbox
width ``W`` via a runtime column-shift matrix ``S[j,c] = 1 iff c - j = W``.
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


def _ref(g: np.ndarray) -> Optional[np.ndarray]:
    ys, xs = np.where(g != 0)
    if len(ys) == 0:
        return None
    crop = g[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return np.concatenate([crop, crop], axis=1)


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
    def i64(name, vals): return numpy_helper.from_array(np.array(vals, np.int64), name)
    def f32(name, vals): return numpy_helper.from_array(np.array(vals, np.float32), name)

    idx = np.arange(WIDTH)
    D2 = (idx[None, :] - idx[:, None]).astype(np.float32)   # D2[j,c] = c - j
    init = [
        i64("bg_start", [0]), i64("bg_end", [1]),
        f32("arange_h", np.arange(HEIGHT)), f32("arange_w", np.arange(WIDTH)),
        i64("ah_shape", [1, 1, HEIGHT, 1]), i64("aw_shape", [1, 1, 1, WIDTH]),
        i64("h_idx_shape", [HEIGHT]), i64("w_idx_shape", [WIDTH]),
        f32("clip_min", 0.0), f32("clip_max", float(HEIGHT - 1)),
        f32("one_f", 1.0), f32("half", 0.5),
        numpy_helper.from_array(D2, "D2"),
    ]
    nodes = [
        n("ReduceSum", ["input"], ["all_sum"], axes=[1], keepdims=1),
        n("Slice", ["input", "bg_start", "bg_end", "bg_axes"], ["bg_only"]),
        n("Sub", ["all_sum", "bg_only"], ["nbg"]),
        n("ReduceMax", ["nbg"], ["row_proj"], axes=[3], keepdims=1),
        n("ReduceMax", ["nbg"], ["col_proj"], axes=[2], keepdims=1),
        n("Reshape", ["arange_h", "ah_shape"], ["ah4"]),
        n("Reshape", ["arange_w", "aw_shape"], ["aw4"]),
        n("Mul", ["row_proj", "ah4"], ["row_pos"]),
        n("Mul", ["col_proj", "aw4"], ["col_pos"]),
        n("ReduceMax", ["row_pos"], ["r_max"], axes=[2], keepdims=1),
        n("ReduceMax", ["col_pos"], ["c_max"], axes=[3], keepdims=1),
        n("ArgMax", ["row_proj"], ["r_min_i"], axis=2, keepdims=1),
        n("ArgMax", ["col_proj"], ["c_min_i"], axis=3, keepdims=1),
        n("Cast", ["r_min_i"], ["r_min"], to=F),
        n("Cast", ["c_min_i"], ["c_min"], to=F),
        n("Sub", ["r_max", "r_min"], ["h_m1"]),
        n("Sub", ["c_max", "c_min"], ["w_m1"]),
        n("Add", ["ah4", "r_min"], ["sr_f"]),
        n("Clip", ["sr_f", "clip_min", "clip_max"], ["sr_c"]),
        n("Cast", ["sr_c"], ["sr_i"], to=TensorProto.INT64),
        n("Reshape", ["sr_i", "h_idx_shape"], ["sr_1d"]),
        n("Add", ["aw4", "c_min"], ["sc_f"]),
        n("Clip", ["sc_f", "clip_min", "clip_max"], ["sc_c"]),
        n("Cast", ["sc_c"], ["sc_i"], to=TensorProto.INT64),
        n("Reshape", ["sc_i", "w_idx_shape"], ["sc_1d"]),
        n("Gather", ["input", "sr_1d"], ["shift_rows"], axis=2),
        n("Gather", ["shift_rows", "sc_1d"], ["shifted"], axis=3),
        n("Add", ["h_m1", "one_f"], ["h_f"]),
        n("Add", ["w_m1", "one_f"], ["w_f"]),
        n("Less", ["ah4", "h_f"], ["h_le"]), n("Cast", ["h_le"], ["h_le_f"], to=F),
        n("Less", ["aw4", "w_f"], ["w_le"]), n("Cast", ["w_le"], ["w_le_f"], to=F),
        n("Mul", ["h_le_f", "w_le_f"], ["bbox_mask"]),
        n("Mul", ["shifted", "bbox_mask"], ["cropped"]),
        # second copy shifted right by W via column-shift matrix S[j,c]=1 iff c-j=w
        n("Sub", ["D2", "w_f"], ["ddiff"]),
        n("Abs", ["ddiff"], ["dabs"]),
        n("Less", ["dabs", "half"], ["sb"]), n("Cast", ["sb"], ["Scol"], to=F),
        n("MatMul", ["cropped", "Scol"], ["copy2"]),
        n("Add", ["cropped", "copy2"], ["output"]),
    ]
    init.append(i64("bg_axes", [1]))
    graph = helper.make_graph(nodes, "crop_tile_h",
                              [helper.make_tensor_value_info("input", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              [helper.make_tensor_value_info("output", F, [1, CHANNELS, HEIGHT, WIDTH])],
                              initializer=init)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                             ir_version=IR_VERSION)


def solve_crop_tile_h(task: dict) -> Optional[onnx.ModelProto]:
    if not _detect(task):
        return None
    return _build()
