import sys
import numpy as np
import onnxruntime as ort

print('Starting...', flush=True)
session = ort.InferenceSession('/tmp/test56.onnx')
print('Session created OK', flush=True)

# L-shape pattern: (0,1)(0,2)(1,1)(1,2)(2,0) -> output 3
inp = np.zeros((1,10,30,30), dtype=np.float32)
inp[0,3,0,1]=1.0; inp[0,3,0,2]=1.0
inp[0,3,1,1]=1.0; inp[0,3,1,2]=1.0
inp[0,3,2,0]=1.0
out = session.run(None, {'input': inp})[0]
vals = out[0,:,0,0]
print('L-shape (expect 3):', [round(v,4) for v in vals.tolist()], 'sum=', out.sum(), flush=True)

# All zeros test
inp0 = np.zeros((1,10,30,30), dtype=np.float32)
out0 = session.run(None, {'input': inp0})[0]
print('All zeros:', out0[0,:,0,0].tolist(), flush=True)

print('Done', flush=True)
sys.exit(0)