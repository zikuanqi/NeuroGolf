import sys, json, numpy as np, onnx, onnxruntime, pathlib
sys.path.insert(0, 'src')
from neurogolf.solvers.shape_classify import solve_shape_classify
from neurogolf.verify import _sanitize
from neurogolf.grids import to_onehot

task = json.load(open('data/task056.json'))
model = solve_shape_classify(task)

tmp_path = pathlib.Path('/tmp/test56_v3.onnx')
onnx.save(model, str(tmp_path))

sani = _sanitize(onnx.load(str(tmp_path)))
opts = onnxruntime.SessionOptions()
opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
sess = onnxruntime.InferenceSession(sani.SerializeToString(), opts)

for i, ex in enumerate(task['train']):
    inp = ex['input']
    out = ex['output']
    
    # Manual input
    inp_arr = np.zeros((1, 10, 30, 30), dtype=np.float32)
    for r, row in enumerate(inp):
        for c, v in enumerate(row):
            if v != 0:
                inp_arr[0, v, r, c] = 1.0
    
    # to_onehot input
    bench_inp = to_onehot(inp)
    bench_out = to_onehot(out)
    
    print(f'\nExample {i}:')
    print(f'  input shape: {inp_arr.shape} vs to_onehot shape: {bench_inp.shape if bench_inp is not None else None}')
    print(f'  input equal: {np.array_equal(inp_arr, bench_inp) if bench_inp is not None else False}')
    print(f'  output shape: {bench_out.shape if bench_out is not None else None}')
    
    # Run with manual input
    result1 = sess.run(['output'], {'input': inp_arr})
    print(f'  manual run OK, shape={result1[0].shape}')
    print(f'  manual (0,:,0,0): {result1[0][0,:,0,0]}')
    
    # Run with benchmark input
    try:
        result2 = sess.run(['output'], {'input': bench_inp})
        print(f'  benchmark run OK, shape={result2[0].shape}')
        print(f'  benchmark (0,:,0,0): {result2[0][0,:,0,0]}')
        predicted = (result2[0] > 0.0).astype(np.float32)
        match = np.array_equal(predicted, bench_out)
        print(f'  match: {match}')
    except Exception as e:
        print(f'  benchmark run FAILED: {e}')