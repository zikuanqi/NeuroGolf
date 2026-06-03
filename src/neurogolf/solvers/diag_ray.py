"""Diagonal-ray extrusion solver (task 327).

Each non-zero cell of an N x N input emits a ray travelling down-right across a
2N x 2N output, painting every cell on its diagonal at or below-right of the
marker with the marker's colour. The input sits in the top-left N x N quadrant.

    in (3x3)      out (6x6)
    6 1 .         6 1 . . . .
    3 . .    ->   3 6 1 . . .
    . . .         . 3 6 1 . .
                  . . 3 6 1 .
                  . . . 3 6 1
                  . . . . 3 6

The transform is a diagonal prefix-max along the down-right direction, computed
per colour channel by log-doubling shift-and-max (shift the grid down-right by
1, 2, 4, ... and take the elementwise max). The background channel is then
recomputed as "every output cell with no ray", and the 2N x 2N block is padded
back out to the full 30 x 30 frame.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from ..grids import CHANNELS, HEIGHT, WIDTH, all_examples

OPSET = 11
IR_VERSION = 8


# ----- reference implementation (also used for detection) -------------------

def _solve_array(inp: list) -> Optional[list]:
    h, w = len(inp), len(inp[0])
    if h != w:
        return None
    n = h
    H = W = 2 * n
    if H > HEIGHT or W > WIDTH:
        return None
    out = [[0] * W for _ in range(H)]
    diag = {}
    for r in range(n):
        for c in range(n):
            v = inp[r][c]
            if v:
                if (c - r) in diag:      # two markers share a diagonal: ambiguous
                    return None
                diag[c - r] = (r, c, v)
    for d, (r0, c0, v) in diag.items():
        r, c = r0, c0
        while r < H and c < W:
            out[r][c] = v
            r += 1
            c += 1
    return out


def _detect(task: dict) -> Optional[int]:
    examples = list(all_examples(task))
    if not examples:
        return None
    n_seen = set()
    saw_nonzero = False
    for ex in examples:
        inp, out = ex["input"], ex["output"]
        if not inp or not inp[0] or len(inp) > HEIGHT or len(inp[0]) > WIDTH:
            return None
        ref = _solve_array(inp)
        if ref is None or ref != out:
            return None
        n_seen.add(len(inp))
        if any(v for row in inp for v in row):
            saw_nonzero = True
    if len(n_seen) != 1 or not saw_nonzero:
        return None  # require a single constant N to bake a static graph
    return n_seen.pop()


# ----- ONNX graph -----------------------------------------------------------

def _i64(name, data):
    return numpy_helper.from_array(np.array(data, dtype=np.int64), name)


def _build(n: int) -> onnx.ModelProto:
    side = 2 * n
    init = [
        _i64("mk_start", [0, 1, 0, 0]),
        _i64("mk_end", [1, CHANNELS, side, side]),
        _i64("mk_axes", [0, 1, 2, 3]),
        _i64("mk_step", [1, 1, 1, 1]),
        numpy_helper.from_array(np.array([0.0], dtype=np.float32), "zero_f"),
        numpy_helper.from_array(np.array([1.0], dtype=np.float32), "one_f"),
    ]
    nodes = [
        # markers = input channels 1..9, top-left side x side region
        helper.make_node(
            "Slice", ["input", "mk_start", "mk_end", "mk_axes", "mk_step"],
            ["cur0"], name="slice_markers"),
    ]

    # diagonal down-right prefix-max via log-doubling
    cur = "cur0"
    k = 1
    step = 0
    while k < side:
        pads_name = f"pad_{step}"
        sl_s = f"sl_s_{step}"
        sl_e = f"sl_e_{step}"
        init += [
            # Pad pads for NCHW: [b0,c0,h0,w0, b1,c1,h1,w1]; add k at h/w begin
            _i64(pads_name, [0, 0, k, k, 0, 0, 0, 0]),
            _i64(sl_s, [0, 0, 0, 0]),
            _i64(sl_e, [1, CHANNELS - 1, side, side]),
        ]
        nodes += [
            helper.make_node(
                "Pad", [cur, pads_name], [f"padded_{step}"],
                mode="constant", name=f"pad_op_{step}"),
            helper.make_node(
                "Slice",
                [f"padded_{step}", sl_s, sl_e, "mk_axes", "mk_step"],
                [f"shifted_{step}"], name=f"slice_op_{step}"),
            helper.make_node(
                "Max", [cur, f"shifted_{step}"], [f"cur{step + 1}"],
                name=f"max_op_{step}"),
        ]
        cur = f"cur{step + 1}"
        k *= 2
        step += 1

    rays = cur  # (1, 9, side, side)

    # background channel 0 = 1 where no ray
    nodes += [
        helper.make_node(
            "ReduceSum", [rays], ["ray_sum"], axes=[1], keepdims=1,
            name="ray_sum_op"),
        helper.make_node(
            "Greater", ["ray_sum", "zero_f"], ["has_ray"], name="has_ray_op"),
        helper.make_node(
            "Cast", ["has_ray"], ["has_ray_f"], to=TensorProto.FLOAT,
            name="has_ray_cast"),
        helper.make_node(
            "Sub", ["one_f", "has_ray_f"], ["bg"], name="bg_op"),
        helper.make_node(
            "Concat", ["bg", rays], ["block"], axis=1, name="concat_ch0"),
    ]

    # pad the side x side block back to 30 x 30
    init.append(_i64("frame_pads",
                     [0, 0, 0, 0, 0, 0, HEIGHT - side, WIDTH - side]))
    nodes.append(helper.make_node(
        "Pad", ["block", "frame_pads"], ["output"], mode="constant",
        name="frame_pad_op"))

    value_info = [
        helper.make_tensor_value_info(
            "cur0", TensorProto.FLOAT, [1, CHANNELS - 1, side, side]),
        helper.make_tensor_value_info(
            "block", TensorProto.FLOAT, [1, CHANNELS, side, side]),
    ]

    inputs = [helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, CHANNELS, HEIGHT, WIDTH])]
    outputs = [helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, CHANNELS, HEIGHT, WIDTH])]
    graph = helper.make_graph(
        nodes, "diag_ray", inputs, outputs,
        initializer=init, value_info=value_info)
    return helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", OPSET)],
        ir_version=IR_VERSION)


def solve_diag_ray(task: dict) -> Optional[onnx.ModelProto]:
    n = _detect(task)
    if n is None:
        return None
    return _build(n)
