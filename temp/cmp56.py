import json, numpy as np, sys
sys.path.insert(0, 'src')
from neurogolf.grids import to_onehot

t = json.load(open('data/task056.json'))
ex = t['train'][0]

onehot = to_onehot(ex['input'])

manual = np.zeros((1, 10, 30, 30), dtype=np.float32)
for r, row in enumerate(ex['input']):
    for c, v in enumerate(row):
        if v != 0:
            manual[0, v, r, c] = 1.0

print(f'shapes: onehot={onehot.shape}, manual={manual.shape}')
print(f'equal: {np.array_equal(onehot, manual)}')

# Find differences
diff = np.where(onehot != manual)
if diff[0].size > 0:
    print(f'diff count: {diff[0].size}')
    for i in range(min(20, diff[0].size)):
        idx = tuple(d[i] for d in diff)
        print(f'  pos {idx}: onehot={onehot[idx]}, manual={manual[idx]}')

# Check non-zero in onehot
nz = np.where(onehot > 0)
print(f'onehot non-zero count: {nz[0].size}')
for i in range(min(20, nz[0].size)):
    idx = tuple(d[i] for d in nz)
    print(f'  onehot[{idx}]={onehot[idx]}')

# Check non-zero in manual
nz2 = np.where(manual > 0)
print(f'manual non-zero count: {nz2[0].size}')
for i in range(min(20, nz2[0].size)):
    idx = tuple(d[i] for d in nz2)
    print(f'  manual[{idx}]={manual[idx]}')