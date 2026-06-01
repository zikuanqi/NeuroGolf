"""Debug: test UNSANITIZED model with onnxruntime."""
import sys, json, onnx, numpy as np
sys.path.insert(0, 'src')
from neurogolf.solvers.shape_classify import solve_shape_classify
import onnxruntime as ort

task = json.load(open('data/task056.json'))
model = solve_shape_classify(task)

# Test UNSANITIZED model
try:
    sess = ort.InferenceSession(model.SerializeToString())
    print('Unsanitized model: OK')
    # Test with train example
    ex = task['train'][0]
    inp_arr = np.zeros((1, 10, 30, 30), dtype=np.float32)
    for r, row in enumerate(ex['input']):
        for c, v in enumerate(row):
            if v != 0:
                inp_arr[0, v, r, c] = 1.0
    out = sess.run(None, {'input': inp_arr})
    print(f'  Output at (0,:,0,0): {out[0][0,:,0,0]}')
    print(f'  Expected: {ex["output"][0][0]}')
except Exception as e:
    print(f'Unsanitized model ERROR: {e}')

# Now let's check what opset the model uses
for opset in model.opset_import:
    print(f'Opset: domain={opset.domain}, version={opset.version}')