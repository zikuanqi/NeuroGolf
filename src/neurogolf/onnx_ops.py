"""Helpers for building small ONNX graphs in opset 10."""
from __future__ import annotations

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
