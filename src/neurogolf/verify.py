"""Clean-room re-implementation of the official scorer.

Loads an ONNX network, runs it on a task's examples, and reports pass / fail
plus the (memory, params) score components. Mirrors logic from
`neurogolf_utils.py` in the competition data drop.
"""
from __future__ import annotations

import json
import math
import pathlib
import traceback
from dataclasses import dataclass

import numpy as np
import onnx
import onnxruntime

from .grids import from_onehot, to_onehot

EXCLUDED_OP_TYPES = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION",
                     "COMPRESS"}
FILESIZE_LIMIT = int(1.44 * 1024 * 1024)


@dataclass
class Score:
    passed: bool
    train_right: int = 0
    train_wrong: int = 0
    test_right: int = 0
    test_wrong: int = 0
    arc_gen_right: int = 0
    arc_gen_wrong: int = 0
    memory: int | None = None
    params: int | None = None
    points: float = 0.0
    error: str = ""

    @property
    def all_right(self) -> int:
        return self.train_right + self.test_right + self.arc_gen_right

    @property
    def all_wrong(self) -> int:
        return self.train_wrong + self.test_wrong + self.arc_gen_wrong


def _sanitize(model: onnx.ModelProto) -> onnx.ModelProto | None:
    for node in model.graph.node:
        if not node.output or "kernel_time" in node.output[0]:
            return None
        node.name = node.output[0]
    name_map: dict[str, str] = {}
    counter = 0

    def safe(name: str) -> str:
        nonlocal counter
        if not name or name in ("input", "output"):
            return name
        if name not in name_map:
            name_map[name] = f"safe_{counter}"
            counter += 1
        return name_map[name]

    for inp in model.graph.input:
        inp.name = safe(inp.name)
    for init in model.graph.initializer:
        init.name = safe(init.name)
    for node in model.graph.node:
        node.input[:] = [safe(n) for n in node.input]
        node.output[:] = [safe(n) for n in node.output]
        if node.output:
            node.name = node.output[0]
    for out in model.graph.output:
        out.name = safe(out.name)
    for vi in model.graph.value_info:
        vi.name = safe(vi.name)
    for node in model.graph.node:
        node.name = node.output[0]
    return model


def _calculate_params(model: onnx.ModelProto) -> int | None:
    params = 0
    for init in model.graph.initializer:
        if any(d <= 0 for d in init.dims):
            return None
        params += int(np.prod(init.dims))
    for sparse in model.graph.sparse_initializer:
        if any(d <= 0 for d in sparse.values.dims):
            return None
        params += int(np.prod(sparse.values.dims))
    for node in model.graph.node:
        if node.op_type != "Constant":
            continue
        for attr in node.attribute:
            if attr.name == "value":
                if any(d <= 0 for d in attr.t.dims):
                    return None
                params += int(np.prod(attr.t.dims))
            elif attr.name == "sparse_value":
                if any(d <= 0 for d in attr.sparse_tensor.values.dims):
                    return None
                params += int(np.prod(attr.sparse_tensor.values.dims))
            elif attr.name == "value_floats":
                params += len(attr.floats)
            elif attr.name == "value_ints":
                params += len(attr.ints)
            elif attr.name == "value_strings":
                params += len(attr.strings)
    return params


def _calculate_memory(model: onnx.ModelProto, trace_path: str) -> int | None:
    onnx.checker.check_model(model, full_check=True)
    graph = onnx.shape_inference.infer_shapes(model, strict_mode=True).graph
    if len(graph.input) > 1 or len(graph.output) > 1:
        return None
    init_names = {i.name for i in graph.initializer}
    init_names.update(i.name for i in graph.sparse_initializer)
    io_names = {t.name for t in list(graph.input) + list(graph.output)}
    if io_names & init_names:
        return None
    if model.functions:
        return None
    for opset in model.opset_import:
        if opset.domain not in {"", "ai.onnx"}:
            return None

    node_outputs = {}
    tensor_names = set()
    for node in graph.node:
        for attr in node.attribute:
            if attr.type in (onnx.AttributeProto.GRAPH,
                             onnx.AttributeProto.GRAPHS):
                return None
        node_outputs[node.name] = list(node.output)
        for o in node.output:
            if o:
                tensor_names.add(o)

    tensor_map = {t.name: t for t in list(graph.input) + list(graph.value_info)
                  + list(graph.output)}
    tensor_names.update(tensor_map.keys())
    tensor_memory: dict[str, int] = {}
    tensor_dtypes: dict[str, np.dtype] = {}
    for name in tensor_names:
        item = tensor_map.get(name)
        if not item:
            return None
        if item.type.HasField("sequence_type"):
            return None
        if not item.type.HasField("tensor_type"):
            continue
        ttype = item.type.tensor_type
        if not ttype.HasField("shape"):
            return None
        num = 1
        for dim in ttype.shape.dim:
            if dim.HasField("dim_param"):
                return None
            if not dim.HasField("dim_value"):
                return None
            if dim.dim_value <= 0:
                return None
            num *= dim.dim_value
        if name in ("input", "output"):
            continue
        dtype = onnx.helper.tensor_dtype_to_np_dtype(ttype.elem_type)
        tensor_memory[name] = num * np.dtype(dtype).itemsize
        tensor_dtypes[name] = dtype

    seen: set[str] = set()
    for item in (list(graph.input) + list(graph.value_info)
                 + list(graph.output)):
        if item.name in seen:
            return None
        seen.add(item.name)
    for node in graph.node:
        for o in node.output:
            if o and o != "output":
                item = tensor_map.get(o)
                if item is None or not item.type.HasField("tensor_type"):
                    return None

    with open(trace_path) as f:
        trace = json.load(f)
    for event in trace:
        if event.get("cat") != "Node" or "args" not in event:
            continue
        if "output_type_shape" not in event["args"]:
            continue
        node_name = event.get("name", "").replace("_kernel_time", "")
        if node_name not in node_outputs:
            continue
        for i, shape_dict in enumerate(event["args"]["output_type_shape"]):
            if i >= len(node_outputs[node_name]):
                continue
            out_name = node_outputs[node_name][i]
            if out_name not in tensor_dtypes:
                continue
            itemsize = np.dtype(tensor_dtypes[out_name]).itemsize
            mem = itemsize * sum(int(np.prod(dims))
                                 for dims in shape_dict.values())
            tensor_memory[out_name] = max(tensor_memory[out_name], mem)
    return sum(tensor_memory.values())


def _run_subset(session, examples) -> tuple[int, int]:
    right = wrong = 0
    for ex in examples:
        benchmark_input = to_onehot(ex["input"])
        benchmark_output = to_onehot(ex["output"])
        if benchmark_input is None or benchmark_output is None:
            continue
        try:
            result = session.run(["output"], {"input": benchmark_input})
            predicted = (result[0] > 0.0).astype(np.float32)
            if np.array_equal(predicted, benchmark_output):
                right += 1
            else:
                wrong += 1
        except Exception:
            wrong += 1
    return right, wrong


def verify(network_path: str | pathlib.Path, task: dict,
           task_num: int = 0) -> Score:
    path = pathlib.Path(network_path)
    score = Score(passed=False)
    if not path.is_file():
        score.error = f"missing file {path}"
        return score
    if path.stat().st_size > FILESIZE_LIMIT:
        score.error = f"size {path.stat().st_size} > {FILESIZE_LIMIT}"
        return score

    try:
        model = onnx.load(str(path))
        for node in model.graph.node:
            if node.op_type.upper() in EXCLUDED_OP_TYPES:
                score.error = f"banned op {node.op_type}"
                return score
            if "Sequence" in node.op_type:
                score.error = f"banned op {node.op_type}"
                return score

        sanitized = _sanitize(onnx.load(str(path)))
        if sanitized is None:
            score.error = "sanitize failed"
            return score
        opts = onnxruntime.SessionOptions()
        opts.enable_profiling = True
        opts.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL)
        opts.profile_file_prefix = f"{task_num:03d}"
        session = onnxruntime.InferenceSession(
            sanitized.SerializeToString(), opts)
    except Exception:
        score.error = traceback.format_exc()
        return score

    score.train_right, score.train_wrong = _run_subset(
        session, task.get("train", []))
    score.test_right, score.test_wrong = _run_subset(
        session, task.get("test", []))
    score.arc_gen_right, score.arc_gen_wrong = _run_subset(
        session, task.get("arc-gen", []))

    trace = session.end_profiling()
    memory = _calculate_memory(sanitized, trace)
    params = _calculate_params(sanitized)
    try:
        pathlib.Path(trace).unlink(missing_ok=True)
    except OSError:
        pass
    score.memory = memory
    score.params = params
    if memory is None or params is None or memory < 0 or params < 0:
        score.error = "score components invalid"
        return score
    if score.all_wrong == 0 and score.all_right > 0:
        score.passed = True
        score.points = max(1.0, 25.0 - math.log(max(1.0, memory + params)))
    return score
