"""Debug shape_classify model."""
import sys, json, onnx
sys.path.insert(0, 'src')
from neurogolf.solvers.shape_classify import solve_shape_classify

task = json.load(open('data/task056.json'))
model = solve_shape_classify(task)

# Check the depth initializer
for init in model.graph.initializer:
    if init.name == 'depth_10':
        import numpy as np
        arr = np.frombuffer(init.raw_data, dtype=np.int64)
        print(f'depth_10: shape={init.dims}, values={arr}')
    if 'onehot' in init.name.lower() or init.name == 'oh_values' or 'win_rs' in init.name:
        import numpy as np
        if init.data_type == 7:  # int64
            arr = np.frombuffer(init.raw_data, dtype=np.int64)
        else:
            arr = np.frombuffer(init.raw_data, dtype=np.float32)
        print(f'{init.name}: shape={init.dims}, values={arr}')

# Check OneHot node
for n in model.graph.node:
    if n.op_type == 'OneHot':
        print(f'OneHot: inputs={list(n.input)}, outputs={list(n.output)}, axis={n.attribute[0].i}')

# Check ArgMax
for n in model.graph.node:
    if n.op_type == 'ArgMax':
        print(f'ArgMax: inputs={list(n.input)}, outputs={list(n.output)}')

# Check the Pad node
for n in model.graph.node:
    if n.op_type == 'Pad':
        print(f'Pad: inputs={list(n.input)}, outputs={list(n.output)}')

# Check shape inference
from onnx import shape_inference
res = shape_inference.infer_shapes(model, strict_mode=True)
for vi in res.graph.value_info:
    if 'onehot' in vi.name.lower() or 'winner' in vi.name.lower() or 'oh' in vi.name.lower():
        shape = [d.dim_value for d in vi.type.tensor_type.shape.dim]
        print(f'ValueInfo {vi.name}: shape={shape}')
for vi in res.graph.value_info:
    if 'scor' in vi.name.lower():
        shape = [d.dim_value for d in vi.type.tensor_type.shape.dim]
        print(f'ValueInfo {vi.name}: shape={shape}')