import sys, json
sys.path.insert(0, 'src')
from neurogolf.solvers.spatial_classify import _check_task, _verify_rule, _build_pattern_map

for tid in [48, 291, 346, 355]:
    task = json.load(open(f'data/task{tid:03d}.json'))
    c = _check_task(task)
    v = _verify_rule(task) if c else False
    print(f'task {tid}: check={c}, verify={v}')
    if c and not v:
        pmap = _build_pattern_map(task)
        print(f'  hash map: {sorted(pmap.items())}')
        # Show each example's hash
        for i, ex in enumerate(task['train']):
            inp, out = ex['input'], ex['output']
            h = sum((1 << (r*3+c)) for r, row in enumerate(inp) for c, v in enumerate(row) if v != 0)
            print(f'  train[{i}]: hash={h}, expected={out[0][0]}, mapped={pmap.get(h)}')
    elif c:
        print(f'  rule verified!')
    print()