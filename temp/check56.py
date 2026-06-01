import onnx
from onnx import helper

model = onnx.load('/tmp/test56.onnx')

print('Graph nodes:')
for n in model.graph.node:
    print(f'  {n.op_type}: inputs={list(n.input)}, outputs={list(n.output)}')

print('\nInitializers:')
for init in model.graph.initializer:
    print(f'  {init.name}')

print('\nGraph inputs:')
for inp in model.graph.input:
    print(f'  {inp.name}')

print('\nValue info:')
for vi in model.graph.value_info:
    print(f'  {vi.name}')

# Check which inputs to nodes are NOT initializers, inputs, or previous outputs
all_valid = set()
for inp in model.graph.input:
    all_valid.add(inp.name)
for init in model.graph.initializer:
    all_valid.add(init.name)

for n in model.graph.node:
    for inp_name in n.input:
        if inp_name not in all_valid:
            print(f'ERROR: node {n.name}({n.op_type}) input "{inp_name}" not resolved')
    for out_name in n.output:
        all_valid.add(out_name)

print('\nAll valid names:', sorted(all_valid))
print('Done')