"""Debug: check sanitized model Equal node inputs."""
import sys, json, onnx
sys.path.insert(0, 'src')
from neurogolf.solvers.shape_classify import solve_shape_classify
from neurogolf.verify import _sanitize

task = json.load(open('data/task056.json'))
model = solve_shape_classify(task)
print(f'Model created: {model is not None}')

sanitized = _sanitize(model)
if sanitized is None:
    print('Sanitize returned None')
else:
    print(f'Sanitized OK, {len(sanitized.graph.node)} nodes')
    for n in sanitized.graph.node:
        if n.op_type == 'Equal':
            inp0 = n.input[0]
            inp1 = n.input[1]
            # Find the type of these inputs
            inp0_type = None
            inp1_type = None
            for vi in sanitized.graph.value_info:
                if vi.name == inp0:
                    inp0_type = [d.dim_value for d in vi.type.tensor_type.shape.dim]
                    break
            for vi in sanitized.graph.value_info:
                if vi.name == inp1:
                    inp1_type = [d.dim_value for d in vi.type.tensor_type.shape.dim]
                    break
            print(f'  Equal node {n.name}: inputs=({inp0}, {inp1}), '
                  f'types inferred: inp0_shape={inp0_type}, inp1_shape={inp1_type}')
    
    # Try to run shape inference on sanitized
    try:
        inferred = onnx.shape_inference.infer_shapes(sanitized, strict_mode=False)
        print('Shape inference (lenient) OK')
        for vi in inferred.graph.value_info:
            for n in sanitized.graph.node:
                if n.output[0] == vi.name:
                    dims = [d.dim_value for d in vi.type.tensor_type.shape.dim]
                    print(f'  {vi.name} ({n.op_type}): shape={dims}')
                    break
    except Exception as e:
        print(f'Shape inference failed: {e}')