"""Debug: print shape inference for each intermediate tensor."""
import sys, json, onnx, numpy as np
from onnx import helper, numpy_helper, TensorProto, shape_inference
sys.path.insert(0, 'src')
from neurogolf.solvers.shape_classify import solve_shape_classify

task = json.load(open('data/task056.json'))
model = solve_shape_classify(task)

# Run shape inference
inferred = shape_inference.infer_shapes(model, strict_mode=False)

print("=== All value_info after shape inference ===")
for vi in inferred.graph.value_info:
    dims = [d.dim_value for d in vi.type.tensor_type.shape.dim]
    print(f"  {vi.name}: shape={dims}")

print("\n=== Graph inputs ===")
for inp in inferred.graph.input:
    dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    print(f"  {inp.name}: shape={dims}")

print("\n=== Graph outputs ===")
for outp in inferred.graph.output:
    dims = [d.dim_value for d in outp.type.tensor_type.shape.dim]
    print(f"  {outp.name}: shape={dims}")

# Also print all nodes
print("\n=== Nodes ===")
for n in model.graph.node:
    print(f"  {n.name or n.op_type}: inputs={list(n.input)}, outputs={list(n.output)}")
