import sys, json, numpy as np, onnx, onnxruntime
sys.path.insert(0, 'src')
from neurogolf.solvers.shape_classify import solve_shape_classify
from neurogolf.verify import _sanitize, _run_subset, _calculate_memory, _calculate_params
import pathlib, traceback

task = json.load(open('data/task056.json'))
model = solve_shape_classify(task)

# Save to tmp file (as pipeline does)
tmp_path = pathlib.Path('/tmp/test56_final.onnx')
onnx.save(model, str(tmp_path))

# Load and verify (as pipeline does)
path = pathlib.Path(tmp_path)
model2 = onnx.load(str(path))

# Check banned ops
EXCLUDED_OP_TYPES = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}
for node in model2.graph.node:
    if node.op_type.upper() in EXCLUDED_OP_TYPES:
        print(f'BANNED OP: {node.op_type}')
        sys.exit(1)

# Sanitize
sani = _sanitize(model2)
if sani is None:
    print('SANITIZE FAILED')
    sys.exit(1)
print('Sanitize OK')

# Create session with profiling (as verify does)
opts = onnxruntime.SessionOptions()
opts.enable_profiling = True
opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
opts.profile_file_prefix = "056"
sess = onnxruntime.InferenceSession(sani.SerializeToString(), opts)

# Run all examples
trace = sess.end_profiling()
pathlib.Path(trace).unlink(missing_ok=True)

def run_subset(subset):
    right = 0
    wrong = 0
    for i, ex in enumerate(subset):
        inp_arr = np.zeros((1, 10, 30, 30), dtype=np.float32)
        for r, row in enumerate(ex['input']):
            for c, v in enumerate(row):
                if v != 0:
                    inp_arr[0, v, r, c] = 1.0
        out = sess.run(None, {'input': inp_arr})
        pred = int(np.argmax(out[0][0, :, 0, 0]))
        expected = ex['output'][0][0]
        if pred == expected:
            right += 1
        else:
            wrong += 1
            print(f'  WRONG [{i}]: pred={pred}, expected={expected}')
    return right, wrong

tr, tw = run_subset(task['train'])
print(f'train: {tr}/{tr+tw}')
te, we = run_subset(task['test'])
print(f'test: {te}/{te+we}')
if 'arc-gen' in task:
    ar, aw = run_subset(task['arc-gen'])
    print(f'arc-gen: {ar}/{ar+aw}')