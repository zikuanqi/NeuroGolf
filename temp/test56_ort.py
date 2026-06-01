import numpy as np
import onnxruntime as ort

session = ort.InferenceSession('/tmp/test56.onnx')
print('Session created OK')

# L-shape pattern: (0,1)(0,2)(1,1)(1,2)(2,0) -> output 3
inp = np.zeros((1,10,30,30), dtype=np.float32)
inp[0,3,0,1]=1.0; inp[0,3,0,2]=1.0
inp[0,3,1,1]=1.0; inp[0,3,1,2]=1.0
inp[0,3,2,0]=1.0
out = session.run(None, {'input': inp})[0]
print('L-shape output[0,:,0,0]:', out[0,:,0,0])

# T-shape: (0,1)(1,0)(1,1)(1,2)(2,1) -> output 6
inp2 = np.zeros((1,10,30,30), dtype=np.float32)
inp2[0,2,0,1]=1.0; inp2[0,2,1,0]=1.0
inp2[0,2,1,1]=1.0; inp2[0,2,1,2]=1.0
inp2[0,2,2,1]=1.0
out2 = session.run(None, {'input': inp2})[0]
print('T-shape output[0,:,0,0]:', out2[0,:,0,0])

# C-shape: (0,0)(0,1)(1,0)(1,2)(2,1) -> output 1
inp3 = np.zeros((1,10,30,30), dtype=np.float32)
inp3[0,5,0,0]=1.0; inp3[0,5,0,1]=1.0
inp3[0,5,1,0]=1.0; inp3[0,5,1,2]=1.0
inp3[0,5,2,1]=1.0
out3 = session.run(None, {'input': inp3})[0]
print('C-shape output[0,:,0,0]:', out3[0,:,0,0])

# Dot-shape: (0,0)(0,2)(1,1)(2,0)(2,2) -> output 2
inp4 = np.zeros((1,10,30,30), dtype=np.float32)
inp4[0,1,0,0]=1.0; inp4[0,1,0,2]=1.0
inp4[0,1,1,1]=1.0; inp4[0,1,2,0]=1.0
inp4[0,1,2,2]=1.0
out4 = session.run(None, {'input': inp4})[0]
print('Dot-shape output[0,:,0,0]:', out4[0,:,0,0])