import sys, json, numpy as np, onnx, onnxruntime
sys.path.insert(0, 'src')
from neurogolf.solvers.shape_classify import solve_shape_classify
from neurogolf.verify import _sanitize, _run_subset

task = json.load(open('data/task056.json'))
model = solve_shape_classify(task)
onnx.save(model, '/tmp/test56_pipe2.onnx')
loaded = onnx.load('/tmp/test56_pipe2.onnx')
sani = _sanitize(loaded)

# Run all examples through sanitized model
sess = onnxruntime.InferenceSession(sani.SerializeToString())

for subset_name, subset in [('train', task['train']), ('test', task['test']), ('arc-gen', task.get('arc-gen', []))]:
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
            print(f'  WRONG [{subset_name}#{i}]: pred={pred}, expected={expected}')
    print(f'{subset_name}: {right}/{right+wrong}')