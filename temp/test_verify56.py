import sys, json, onnx, onnxruntime, numpy as np, pathlib, math
sys.path.insert(0, 'src')
from neurogolf.solvers.shape_classify import solve_shape_classify
from neurogolf.verify import _sanitize, _run_subset, _calculate_memory, _calculate_params
from neurogolf.grids import to_onehot

task = json.load(open('data/task056.json'))
model = solve_shape_classify(task)

# Save to tmp
tmp_path = pathlib.Path('/tmp/test56_v2.onnx')
onnx.save(model, str(tmp_path))

# Load and verify (exact pipeline steps)
path = pathlib.Path(tmp_path)
model2 = onnx.load(str(path))

# Check banned ops
EXCLUDED = {"LOOP","SCAN","NONZERO","UNIQUE","SCRIPT","FUNCTION","COMPRESS"}
for node in model2.graph.node:
    if node.op_type.upper() in EXCLUDED:
        print(f'BANNED: {node.op_type}'); break

# Sanitize
sani = _sanitize(onnx.load(str(path)))
if sani is None:
    print('SANITIZE FAILED'); sys.exit(1)

# Create session
opts = onnxruntime.SessionOptions()
opts.enable_profiling = True
opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
opts.profile_file_prefix = "056"
sess = onnxruntime.InferenceSession(sani.SerializeToString(), opts)

# Run subsets
tr, tw = _run_subset(sess, task.get("train", []))
te, we = _run_subset(sess, task.get("test", []))
ar, aw = _run_subset(sess, task.get("arc-gen", []))

print(f'train: {tr}/{tr+tw}')
print(f'test: {te}/{te+we}')
print(f'arc-gen: {ar}/{ar+aw}')

trace = sess.end_profiling()
pathlib.Path(trace).unlink(missing_ok=True)

memory = _calculate_memory(sani, trace)
params = _calculate_params(sani)
print(f'memory: {memory}, params: {params}')

score = type('Score', (), {'passed': False, 'train_right': tr, 'train_wrong': tw,
    'test_right': te, 'test_wrong': we, 'arc_gen_right': ar, 'arc_gen_wrong': aw,
    'memory': memory, 'params': params, 'points': 0.0, 'error': '', 'all_right': tr+te+ar, 'all_wrong': tw+we+aw})()

if memory is None or params is None or memory < 0 or params < 0:
    print('SCORE INVALID: memory/params')
elif score.all_wrong == 0 and score.all_right > 0:
    score.passed = True
    score.points = max(1.0, 25.0 - math.log(max(1.0, memory + params)))
    print(f'PASSED! points={score.points:.3f}')
else:
    print(f'FAILED: all_wrong={score.all_wrong}, all_right={score.all_right}')