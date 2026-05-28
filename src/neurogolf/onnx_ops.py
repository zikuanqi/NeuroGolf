"""Helpers for building small ONNX graphs in opset 10."""
from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper

from .grids import CHANNELS, HEIGHT, WIDTH

DATA_TYPE = TensorProto.FLOAT
GRID_SHAPE = [1, CHANNELS, HEIGHT, WIDTH]
IR_VERSION = 10
OPSET_IMPORTS = [helper.make_opsetid("", 10)]


def make_io():
    x = helper.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = helper.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    return x, y


def finalize(nodes, initializers, name="graph"):
    x, y = make_io()
    graph = helper.make_graph(nodes, name, [x], [y], initializers)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=OPSET_IMPORTS)


def identity_model() -> onnx.ModelProto:
    """Output equals input - zero parameters."""
    node = helper.make_node("Identity", ["input"], ["output"])
    return finalize([node], [])


def zero_model() -> onnx.ModelProto:
    """Output is all zeros. Built with Sub(input, input) to avoid params."""
    node = helper.make_node("Sub", ["input", "input"], ["output"])
    return finalize([node], [])


def conv1x1_model(weight: np.ndarray) -> onnx.ModelProto:
    """A single 1x1 Conv layer.

    `weight` has shape (10, 10) and applies a per-pixel linear combination
    across input channels: out[k, r, c] = sum_j weight[k, j] * in[j, r, c].
    Score is thresholded at > 0, so weight entries of 1 act as "pass through".
    """
    assert weight.shape == (CHANNELS, CHANNELS), weight.shape
    w = weight.astype(np.float32).reshape(CHANNELS, CHANNELS, 1, 1)
    w_init = helper.make_tensor("W", DATA_TYPE, list(w.shape), w.flatten())
    node = helper.make_node(
        "Conv", ["input", "W"], ["output"],
        kernel_shape=[1, 1], pads=[0, 0, 0, 0],
    )
    return finalize([node], [w_init])
