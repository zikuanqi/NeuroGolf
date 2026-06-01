import sys, json
sys.path.insert(0, 'src')
from neurogolf.solvers.spatial_classify import _check_task

for tid in [48, 291, 346, 355]:
    task = json.load(open(f'data/task{tid:03d}.json'))
    c = _check_task(task)
    print(f'task {tid}: check={c}')
    if not c:
        # Show why it fails
        for split in ['train', 'test']:
            for i, ex in enumerate(task.get(split, [])):
                inp = ex['input']
                out = ex['output']
                rows = len(inp)
                cols = len(inp[0]) if rows > 0 else 0
                colors = set()
                nz = 0
                for r in inp:
                    for v in r:
                        if v != 0:
                            colors.add(v)
                            nz += 1
                if rows != 3 or cols != 3:
                    print(f'  {split}[{i}]: size={rows}x{cols}, FAIL')
                    break
                if len(out) != 1 or len(out[0]) != 1:
                    print(f'  {split}[{i}]: output size={len(out)}x{len(out[0]) if out else 0}, FAIL')
                    break
                if len(colors) != 1:
                    print(f'  {split}[{i}]: {nz} non-zero, colors={colors}, FAIL')
                    break
            else:
                continue
            break
    print()