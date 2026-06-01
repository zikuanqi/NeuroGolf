"""Debug: check sanitized model Equal node input types by running onnxruntime check."""
import sys, json, onnx, numpy as np
sys.path.insert(0, 'src')
from neurogolf.solvers.shape_classify import solve_shape_classify
from neurogolf.verify import _sanitize

task = json.load(open('data/task056.json'))
model = solve_shape_classify(task)
sanitized = _sanitize(model)

# Try to load with onnxruntime
import onnxruntime as ort
try:
    sess = ort.InferenceSession(sanitized.SerializeToString())
    print('onnxruntime loaded OK')
except Exception as e:
    print(f'onnxruntime error: {e}')
    # Try to find which node causes the issue
    for n in sanitized.graph.node:
        if n.op_type == 'Equal':
            print(f'  Equal node: name={n.name}, inputs={list(n.input)}')
            # Check types of inputs
            for inp_name in n.input:
                # Find in value_info
                found = False
                for vi in sanitized.graph.value_info:
                    if vi.name == inp_name:
                        dtype = vi.type.tensor_type.elem_type
                        print(f'    {inp_name}: dtype={dtype} (FLOAT=1, INT64=7, BOOL=9)')
                        found = True
                        break
                if not found:
                    # Check initializers
                    for init in sanitized.graph.initializer:
                        if init.name == inp_name:
                            print(f'    {inp_name}: initializer dtype={init.data_type}')
                            found = True
                            break
                if not found:
                    print(f'    {inp_name}: NOT FOUND in value_info or initializers!')
