import json, sys
sys.path.insert(0, 'src')
t = json.load(open('data/task056.json'))
print('train:', len(t['train']))
print('test:', len(t['test']))
print('arc-gen:', len(t.get('arc-gen', [])))

# Also run the full test/examples
from neurogolf.solvers.shape_classify import solve_shape_classify, _check_task, _verify_rule
print('_check_task:', _check_task(t))
print('_verify_rule:', _verify_rule(t))
model = solve_shape_classify(t)
print('model:', model is not None)

import onnx, numpy as np, onnxruntime as ort

# Save and reload
onnx.save(model, '/tmp/test56.onnx')
m2 = onnx.load('/tmp/test56.onnx')
sess = ort.InferenceSession(m2.SerializeToString())

for subset_name, subset in [('train', t['train']), ('test', t['test']), ('arc-gen', t.get('arc-gen', []))]:
    right = 0
    wrong = 0
    for ex in subset:
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
            print(f'  WRONG: {subset_name} example, got {pred}, expected {expected}')
    print(f'{subset_name}: {right}/{right+wrong}')