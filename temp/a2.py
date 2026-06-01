import sys, json
sys.path.insert(0, 'src')

for tid in [48, 291, 346, 355]:
    task = json.load(open(f'data/task{tid:03d}.json'))
    print(f'=== TASK {tid} ===')
    
    for split in ['train', 'test']:
        for i, ex in enumerate(task.get(split, [])):
            inp = ex['input']
            out = ex['output']
            
            rows = len(inp)
            cols = max(len(r) for r in inp) if rows > 0 else 0
            padded = [list(r) + [0] * (cols - len(r)) for r in inp]
            
            colors = set()
            positions = []
            for r, row in enumerate(padded):
                for c, v in enumerate(row):
                    if v != 0:
                        colors.add(v)
                        positions.append((r, c))
            
            if positions:
                min_r = min(p[0] for p in positions)
                max_r = max(p[0] for p in positions)
                min_c = min(p[1] for p in positions)
                max_c = max(p[1] for p in positions)
                bb_area = (max_r - min_r + 1) * (max_c - min_c + 1)
            else:
                min_r = max_r = min_c = max_c = bb_area = 0
            
            print(f'  {split}[{i}]: size={rows}x{cols}, output={out[0][0]}')
            print(f'    nz={len(positions)}, colors={colors}, bb={min_r}:{max_r},{min_c}:{max_c}, bb_area={bb_area}, nz/bb={len(positions)/bb_area:.2f}' if positions else '    empty')
            
            print('    Grid:')
            for r, row in enumerate(padded):
                line = '      ' + ''.join(str(v) if v != 0 else '.' for v in row)
                print(line)
        print()
    print()