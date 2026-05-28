"""Solver: each cell of a constant-size input expands to an N x N block.

For tasks where every example has the same input shape (H, W) and an output
of shape (N*H, N*W) where `output[r, c] = input[r // N, c // N]`, we can
implement the upscale with two `Gather`s on the cropped input followed by a
`Pad`. No shape detection at runtime — the geometry is baked into the graph.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from ..grids import CHANNELS, HEIGHT, WIDTH, all_examples

OPSET = 11
IR_VERSION = 8


def _detect(task: dict) -> tuple[int, int, int] | None:
    """Return (input_h, input_w, n) if all examples are N-x scaling."""
    examples = list(all_examples(task))
    if not examples:
        return None

    ih_set = {len(ex["input"]) for ex in examples}
    iw_set = {len(ex["input"][0]) for ex in examples if ex["input"]}
    if len(ih_set) != 1 or len(iw_set) != 1:
        return None
    ih = ih_set.pop()
    iw = iw_set.pop()
    if ih == 0 or iw == 0:
        return None

    # Determine scale from the first example.
    oh = len(examples[0]["output"])
    if oh == 0 or oh % ih != 0:
        return None
    n = oh // ih
    if n < 2 or n > 10:
        return None
    if n * ih > HEIGHT or n * iw > WIDTH:
        return None

    for ex in examples:
        if (len(ex["output"]) != n * ih or
                len(ex["output"][0]) != n * iw):
            return None
        for r in range(n * ih):
            for c in range(n * iw):
                if ex["output"][r][c] != ex["input"][r // n][c // n]:
                    return None
    return ih, iw, n


def _build(ih: int, iw: int, n: int) -> onnx.ModelProto:
    def i64(name, data):
        return numpy_helper.from_array(np.array(data, dtype=np.int64), name)

    starts = i64("starts", [0, 0, 0, 0])
    ends = i64("ends", [1, CHANNELS, ih, iw])
    axes_all = i64("axes_all", [0, 1, 2, 3])
    # Repeat each row/col index n times.
    row_idx = i64("row_idx",
                  [r for r in range(ih) for _ in range(n)])
    col_idx = i64("col_idx",
                  [c for c in range(iw) for _ in range(n)])
    pads = i64("pads",
               [0, 0, 0, 0, 0, 0, HEIGHT - n * ih, WIDTH - n * iw])

    nodes = [
        helper.make_node(
            "Slice", ["input", "starts", "ends", "axes_all"], ["crop"],
            name="slice_input",
        ),
        helper.make_node(
            "Gather", ["crop", "row_idx"], ["rows_expanded"],
            axis=2, name="expand_rows",
        ),
        helper.make_node(
            "Gather", ["rows_expanded", "col_idx"], ["expanded"],
            axis=3, name="expand_cols",
        ),
        helper.make_node(
            "Pad", ["expanded", "pads"], ["output"], mode="constant",
            name="pad_to_canvas",
        ),
    ]

    inputs = [helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, CHANNELS, HEIGHT, WIDTH])]
    outputs = [helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, CHANNELS, HEIGHT, WIDTH])]
    value_info = [
        helper.make_tensor_value_info(
            "crop", TensorProto.FLOAT, [1, CHANNELS, ih, iw]),
        helper.make_tensor_value_info(
            "rows_expanded", TensorProto.FLOAT,
            [1, CHANNELS, n * ih, iw]),
        helper.make_tensor_value_info(
            "expanded", TensorProto.FLOAT,
            [1, CHANNELS, n * ih, n * iw]),
    ]
    graph = helper.make_graph(
        nodes, "kron_scale", inputs, outputs,
        initializer=[starts, ends, axes_all, row_idx, col_idx, pads],
        value_info=value_info,
    )
    return helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
        ir_version=IR_VERSION,
    )


def solve_kron_scale(task: dict) -> Optional[onnx.ModelProto]:
    spec = _detect(task)
    if spec is None:
        return None
    return _build(*spec)
