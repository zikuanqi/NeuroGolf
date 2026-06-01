import sys, json, numpy as np, onnx, onnxruntime
sys.path.insert(0, 'src')
from neurogolf.solvers.shape_classify import solve_shape_classify
from neurogolf.verify import _sanitize

task = json.load(open('data/task056.json'))
model = solve_shape_classify(task)

# Save and reload (as pipeline does)
onnx.save(model, '/tmp/test56_pipe.onnx')
loaded = onnx.load('/tmp/test56_pipe.onnx')

# Check node outputs
for i, node in enumerate(loaded.graph.node):
    if not node.output:
        print(f'Node {i} has no output!')
    if not node.output[0]:
        print(f'Node {i} has empty output name!')
    print(f'  node[{i}] op={node.op_type}, in={list(node.input)}, out={list(node.output)}')

# Sanitize
sani = _sanitize(loaded)
print(f'Sanitized: {sani is not None}')
if sani:
    for i, node in enumerate(sani.graph.node):
        print(f'  sani_node[{i}] op={node.op_type}, in={list(node.input)}, out={list(node.output)}')

    sess = onnxruntime.InferenceSession(sani.SerializeToString())
    # Test first train example
    ex = task['train'][0]
    inp_arr = np.zeros((1, 10, 30, 30), dtype=np.float32)
    for r, row in enumerate(ex['input']):
        for c, v in enumerate(row):
            if v != 0:
                inp_arr[0, v, r, c] = 1.0
    out = sess.run(None, {'input': inp_arr})
    pred = int(np.argmax(out[0][0, :, 0, 0]))
    expected = ex['output'][0][0]
    print(f'Predicted: {pred}, Expected: {expected}, Match: {pred == expected}')