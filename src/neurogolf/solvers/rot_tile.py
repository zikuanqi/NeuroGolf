"""Solver: rotational four-quadrant tiling.

A square N x N input maps to a 2N x 2N output made of four rotated copies::

    +-----------+-----------+
    |     I     |   rot270  |
    +-----------+-----------+
    |   rot90   |   rot180  |
    +-----------+-----------+

where rot90 / rot180 / rot270 are counter-clockwise rotations by 90 / 180 /
270 degrees (tasks 106 and 194).

Rotations are built from a spatial Transpose plus index-reversing Gathers:
  * rot90  (ccw) = flipud(transpose) = Gather(T, reversed_rows, axis=2)
  * rot270 (cw)  = fliplr(transpose) = Gather(T, reversed_cols, axis=3)
  * rot180       = fliplr(flipud(input))
The four quadrants are then assembled with Concats and the canvas is
zero-padded so the scorer crops back to 2N x 2N.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from ..grids import CHANNELS, HEIGHT, WIDTH, all_examples

OPSET = 11
IR_VERSION = 8


def _rot_tile(grid):
    i = np.array(grid)
    tl = i
    tr = np.rot90(i, 3)   # clockwise 90 == rot270 ccw
    bl = np.rot90(i, 1)   # counter-clockwise 90
    br = np.rot90(i, 2)
    return np.vstack([np.hstack([tl, tr]), np.hstack([bl, br])]).tolist()


def _detect(task: dict) -> Optional[int]:
    examples = list(all_examples(task))
    if not examples:
        return None
    # Every example must follow the rotational tiling rule on a square grid.
    sizes = set()
    for ex in examples:
        inp = ex["input"]
        if not inp or len(inp) != len(inp[0]):   # rotations need a square grid
            return None
        if _rot_tile(inp) != ex["output"]:
            return None
        sizes.add(len(inp))
    # The graph bakes in a single N, so all examples must share one grid size.
    if len(sizes) != 1:
        return None
    n = sizes.pop()
    if n < 2 or 2 * n > min(HEIGHT, WIDTH):
        return None
    return n


def _build(n: int) -> onnx.ModelProto:
    def i64(name, data):
        return numpy_helper.from_array(np.array(data, dtype=np.int64), name)

    rev = list(range(n - 1, -1, -1))
    init = [
        i64("crop_starts", [0, 0, 0, 0]),
        i64("crop_ends", [1, CHANNELS, n, n]),
        i64("crop_axes", [0, 1, 2, 3]),
        i64("rev_idx", rev),
        i64("pads", [0, 0, 0, 0, 0, 0, HEIGHT - 2 * n, WIDTH - 2 * n]),
    ]

    nodes = [
        helper.make_node("Slice", ["input", "crop_starts", "crop_ends",
                                   "crop_axes"], ["crop"]),
        # transpose the spatial axes once; both 90-degree rotations reuse it
        helper.make_node("Transpose", ["crop"], ["t"], perm=[0, 1, 3, 2]),
        helper.make_node("Gather", ["t", "rev_idx"], ["rot90"], axis=2),
        helper.make_node("Gather", ["t", "rev_idx"], ["rot270"], axis=3),
        helper.make_node("Gather", ["crop", "rev_idx"], ["flipud"], axis=2),
        helper.make_node("Gather", ["flipud", "rev_idx"], ["rot180"], axis=3),
        # assemble quadrants: [[I, rot270], [rot90, rot180]]
        helper.make_node("Concat", ["crop", "rot270"], ["top"], axis=3),
        helper.make_node("Concat", ["rot90", "rot180"], ["bot"], axis=3),
        helper.make_node("Concat", ["top", "bot"], ["tiled"], axis=2),
        helper.make_node("Pad", ["tiled", "pads"], ["output"], mode="constant"),
    ]

    inputs = [helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, CHANNELS, HEIGHT, WIDTH])]
    outputs = [helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, CHANNELS, HEIGHT, WIDTH])]
    value_info = [
        helper.make_tensor_value_info("crop", TensorProto.FLOAT,
                                      [1, CHANNELS, n, n]),
        helper.make_tensor_value_info("t", TensorProto.FLOAT,
                                      [1, CHANNELS, n, n]),
        helper.make_tensor_value_info("rot90", TensorProto.FLOAT,
                                      [1, CHANNELS, n, n]),
        helper.make_tensor_value_info("rot270", TensorProto.FLOAT,
                                      [1, CHANNELS, n, n]),
        helper.make_tensor_value_info("flipud", TensorProto.FLOAT,
                                      [1, CHANNELS, n, n]),
        helper.make_tensor_value_info("rot180", TensorProto.FLOAT,
                                      [1, CHANNELS, n, n]),
        helper.make_tensor_value_info("top", TensorProto.FLOAT,
                                      [1, CHANNELS, n, 2 * n]),
        helper.make_tensor_value_info("bot", TensorProto.FLOAT,
                                      [1, CHANNELS, n, 2 * n]),
        helper.make_tensor_value_info("tiled", TensorProto.FLOAT,
                                      [1, CHANNELS, 2 * n, 2 * n]),
    ]
    graph = helper.make_graph(nodes, "rot_tile", inputs, outputs,
                              initializer=init, value_info=value_info)
    return helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
        ir_version=IR_VERSION)


def solve_rot_tile(task: dict) -> Optional[onnx.ModelProto]:
    n = _detect(task)
    if n is None:
        return None
    return _build(n)
