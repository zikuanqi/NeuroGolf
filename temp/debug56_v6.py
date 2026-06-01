import sys, json, onnx, numpy as np
sys.path.insert(0, 'src')
from neurogolf.solvers.shape_classify import solve_shape_classify, _build_pattern_map, HASH_WEIGHTS
import onnxruntime as ort

task = json.load(open('data/task056.json'))
pmap = _build_pattern_map(task)
print(f'Pattern map: {pmap}')

model = solve_shape_classify(task)

for ex in task['train']:
    inp, exp_out = ex['input'], ex['output']
    # Compute hash
    h = sum(HASH_WEIGHTS[r * 3 + c]
            for r, row in enumerate(inp)
            for c, v in enumerate(row) if v != 0)
    print(f'\nTrain: hash={h}, expected color={exp_out[0][0]}')
    
    # Build input
    inp_arr = np.zeros((1, 10, 30, 30), dtype=np.float32)
    for r, row in enumerate(inp):
        for c, v in enumerate(row):
            if v != 0:
                inp_arr[0, v, r, c] = 1.0
    
    sess = ort.InferenceSession(model.SerializeToString())
    out = sess.run(None, {'input': inp_arr})
    result = out[0][0, :, 0, 0]
    pred_color = int(np.argmax(result))
    print(f'  Output at (0,:,0,0): {result}, predicted color={pred_color}')
    print(f'  Match: {pred_color == exp_out[0][0]}')