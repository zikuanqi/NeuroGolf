import json
from collections import Counter

def load(n):
    return json.load(open(f'data/task{n:03d}.json'))

def count_colors(g):
    c = Counter()
    for row in g:
        for v in row:
            if v != 0: c[v] += 1
    return dict(c)

def connected_components_all(g):
    h, w = len(g), len(g[0])
    visited = set()
    result = Counter()
    for r in range(h):
        for c in range(w):
            if g[r][c] != 0 and (r,c) not in visited:
                color = g[r][c]
                result[color] += 1
                stack = [(r,c)]
                visited.add((r,c))
                while stack:
                    cr, cc = stack.pop()
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if 0<=nr<h and 0<=nc<w and g[nr][nc]==color and (nr,nc) not in visited:
                            visited.add((nr,nc))
                            stack.append((nr,nc))
    return dict(result)

for tn in [346, 56]:
    t = load(tn)
    ok = 0
    fail = 0
    for split in ['train','test','arc-gen']:
        for idx, ex in enumerate(t.get(split,[])):
            inp = ex['input']
            out = ex['output'][0][0]
            colors = count_colors(inp)
            cc = connected_components_all(inp)
            
            if tn == 346:
                ratios = {c: colors[c]/cc.get(c,1) for c in colors}
                guess = min(ratios, key=ratios.get)
            elif tn == 56:
                # Build normalized shape key
                positions = tuple(sorted((r,c) for r,row in enumerate(inp) for c,v in enumerate(row) if v!=0))
                # Lookup from train data
                shape_map = {}
                for tex in t['train']:
                    tpos = tuple(sorted((r,c) for r,row in enumerate(tex['input']) for c,v in enumerate(row) if v!=0))
                    shape_map[tpos] = tex['output'][0][0]
                guess = shape_map.get(positions, -1)
            
            if guess == out:
                ok += 1
            else:
                fail += 1
                print(f'  FAIL {split}[{idx}]: guess={guess} actual={out}')
    print(f'Task {tn}: {ok}/{ok+fail} correct')