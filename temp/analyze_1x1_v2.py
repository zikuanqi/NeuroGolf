"""Deep-dive predicate exploration for 1x1 output tasks."""
import json
import os
from collections import Counter, deque

PROJECT = '/home/ansel/NeuroGolf'
DATA_DIR = os.path.join(PROJECT, 'data')

def load(n):
    return json.load(open(os.path.join(DATA_DIR, f'task{n:03d}.json')))

def bbox(g):
    rows, cols = [], []
    for r, row in enumerate(g):
        for c, val in enumerate(row):
            if val != 0:
                rows.append(r); cols.append(c)
    if not rows: return (0,0,0,0)
    return (min(rows), min(cols), max(rows), max(cols))

def grid_str(g):
    return '\n'.join(''.join(str(c) if c else '.' for c in row) for row in g)

def pixel_count(g):
    return sum(1 for row in g for v in row if v != 0)

def count_colors(g):
    c = Counter()
    for row in g:
        for v in row:
            if v != 0: c[v] += 1
    return dict(c)

def connected_components_all(g):
    """Return dict color -> number of connected components for that color."""
    if not g: return {}
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

def has_hole_for_color(g, color):
    """Check if a specific color region has internal holes."""
    h, w = len(g), len(g[0])
    # BFS from border zeros that touch this color
    visited = [[False]*w for _ in range(h)]
    q = deque()
    for r in range(h):
        for c in [0, w-1]:
            if g[r][c] == 0 and not visited[r][c]:
                visited[r][c] = True
                q.append((r,c))
    for c in range(w):
        for r in [0, h-1]:
            if g[r][c] == 0 and not visited[r][c]:
                visited[r][c] = True
                q.append((r,c))
    while q:
        cr, cc = q.popleft()
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = cr+dr, cc+dc
            if 0<=nr<h and 0<=nc<w and g[nr][nc]==0 and not visited[nr][nc]:
                visited[nr][nc] = True
                q.append((nr,nc))
    # Check if there's a zero cell surrounded by 'color' cells
    color_present = any(g[r][c]==color for r in range(h) for c in range(w))
    if not color_present: return False
    for r in range(1,h-1):
        for c in range(1,w-1):
            if g[r][c] == 0 and not visited[r][c]:
                # Check if all 4 neighbors are 'color' or out of bounds
                all_color = True
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr, c+dc
                    if 0<=nr<h and 0<=nc<w:
                        if g[nr][nc] != color:
                            all_color = False
                            break
                if all_color:
                    return True
    return False

def is_solid_rect_color(g, color):
    """Check if color forms a solid rectangle."""
    rows, cols = [], []
    for r, row in enumerate(g):
        for c, v in enumerate(row):
            if v == color:
                rows.append(r); cols.append(c)
    if not rows: return False
    rmin, rmax = min(rows), max(rows)
    cmin, cmax = min(cols), max(cols)
    for r in range(rmin, rmax+1):
        for c in range(cmin, cmax+1):
            if g[r][c] != color:
                return False
    for r, row in enumerate(g):
        for c, v in enumerate(row):
            if v == color and (r < rmin or r > rmax or c < cmin or c > cmax):
                return False
    return True

def get_colored_regions(g):
    """Return list of (color, pixel_count, is_solid, has_hole) for each distinct
    contiguous region/block, trying to identify rectangular chunks."""
    h, w = len(g), len(g[0])
    colors = count_colors(g)
    result = []
    for color in colors:
        solid = is_solid_rect_color(g, color)
        hole = has_hole_for_color(g, color)
        result.append((color, colors[color], solid, hole))
    return result

# ============================================================
# TASK 48: Output 0 or 8. Hypothesis tests.
# ============================================================
print("="*70)
print("TASK 48 - Deeper Analysis")
print("="*70)
t48 = load(48)

for idx, ex in enumerate(t48['train']):
    inp = ex['input']
    out = ex['output'][0][0]
    colors = count_colors(inp)
    cc = connected_components_all(inp)
    total_nonzero = pixel_count(inp)
    
    # Color 2 vs 8 properties
    c2_cc = cc.get(2, 0)
    c8_cc = cc.get(8, 0)
    c2_px = colors.get(2, 0)
    c8_px = colors.get(8, 0)
    
    # Check: is the answer the color with fewer components? (more clustered)
    # Or is the answer about connectivity between colors?
    
    # Check if colors 2 and 8 form interlocking shapes
    # Count adjacency between 2 and 8
    adj_2_8 = 0
    h, w = len(inp), len(inp[0])
    for r in range(h):
        for c in range(w):
            if inp[r][c] == 2:
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr, c+dc
                    if 0<=nr<h and 0<=nc<w and inp[nr][nc] == 8:
                        adj_2_8 += 1
    
    print(f"Ex {idx} -> {out}: 2({c2_px}px,{c2_cc}cc) 8({c8_px}px,{c8_cc}cc) adj={adj_2_8}")
    if c2_cc > c8_cc:
        guess = 2
    elif c8_cc > c2_cc:
        guess = 8
    else:
        guess = 'tie'
    print(f"  More components rule -> {guess}, actual={out}, match={guess==out}")

# Check arc-gen for task 48 too
print("\nTask 48 arc-gen:")
for idx, ex in enumerate(t48.get('arc-gen', [])):
    inp = ex['input']
    out = ex['output'][0][0]
    colors = count_colors(inp)
    cc = connected_components_all(inp)
    c2_cc = cc.get(2, 0)
    c8_cc = cc.get(8, 0)
    guess = 2 if c2_cc > c8_cc else (8 if c8_cc > c2_cc else 'tie')
    print(f"  Arc {idx} -> {out}: 2({cc.get(2,0)}cc) 8({cc.get(8,0)}cc) guess={guess} match={guess==out}")

# ============================================================
# TASK 56: 3x3 shape classification
# ============================================================
print("\n" + "="*70)
print("TASK 56 - Shape Classification Analysis")
print("="*70)
t56 = load(56)

# Define reference patterns and their outputs
# Build a pattern -> output map from train data
shape_to_output = {}
for ex in t56['train']:
    inp = ex['input']
    out = ex['output'][0][0]
    # Create a normalized key: positions of non-zero pixels (ignoring color)
    positions = []
    for r, row in enumerate(inp):
        for c, v in enumerate(row):
            if v != 0:
                positions.append((r, c))
    key = tuple(sorted(positions))
    shape_to_output[key] = out
    print(f"Shape {tuple(sorted(positions))} -> {out}")
    print(grid_str(inp))
    print()

print(f"Shape map: {len(shape_to_output)} unique shapes -> outputs {set(shape_to_output.values())}")

# ============================================================
# TASK 291: Broken rectangle detection
# ============================================================
print("="*70)
print("TASK 291 - Broken Rectangle Analysis")
print("="*70)
t291 = load(291)

for idx, ex in enumerate(t291['train']):
    inp = ex['input']
    out = ex['output'][0][0]
    colors = count_colors(inp)
    print(f"\nEx {idx} -> {out}:")
    
    # For each color, check if it forms a solid rectangle
    for color, count in sorted(colors.items(), key=lambda x: -x[1]):
        solid = is_solid_rect_color(inp, color)
        hole = has_hole_for_color(inp, color)
        cc = connected_components_all(inp).get(color, 0)
        print(f"  Color {color}: {count}px solid={solid} hole={hole} components={cc}")
    
    # Find the non-solid color
    broken_colors = [c for c in colors if not is_solid_rect_color(inp, c)]
    print(f"  Broken colors: {broken_colors}, output: {out}, match: {out in broken_colors}")

# Check arc-gen
print("\nTask 291 arc-gen:")
for idx, ex in enumerate(t291.get('arc-gen', [])):
    inp = ex['input']
    out = ex['output'][0][0]
    colors = count_colors(inp)
    broken = [c for c in colors if not is_solid_rect_color(inp, c)]
    print(f"  Arc {idx} -> {out}: broken={broken}, match={out in broken}")

# ============================================================
# TASK 346: Scattered vs clustered color
# ============================================================
print("\n" + "="*70)
print("TASK 346 - Scattered Color Analysis")
print("="*70)
t346 = load(346)

for idx, ex in enumerate(t346['train']):
    inp = ex['input']
    out = ex['output'][0][0]
    colors = count_colors(inp)
    cc = connected_components_all(inp)
    print(f"\nEx {idx} -> {out}:")
    for color in colors:
        print(f"  Color {color}: {colors[color]}px, {cc.get(color,0)} components, ratio={colors[color]/cc.get(color,1):.1f}")
    
    # Hypothesis: output = color with most components (most scattered)
    max_cc_color = max(cc, key=cc.get) if cc else None
    # Hypothesis: output = color with fewest pixels per component
    min_ratio_color = min(colors, key=lambda c: colors[c]/cc.get(c,1))
    max_ratio_color = max(colors, key=lambda c: colors[c]/cc.get(c,1))
    print(f"  Max cc -> {max_cc_color}, Min ratio -> {min_ratio_color}, Max ratio -> {max_ratio_color}, actual={out}")

# Check if output = color with LEAST pixels per component (most scattered)
print("346: Least pixels-per-component rule:")
for idx, ex in enumerate(t346['train']):
    inp = ex['input']
    out = ex['output'][0][0]
    colors = count_colors(inp)
    cc = connected_components_all(inp)
    ratios = {c: colors[c]/cc.get(c,1) for c in colors}
    min_color = min(ratios, key=ratios.get)
    print(f"  Ex {idx}: ratios={ratios}, min={min_color}, actual={out}, match={min_color==out}")

# ============================================================
# TASK 355: Multi-region analysis
# ============================================================
print("\n" + "="*70)
print("TASK 355 - Multi-Region Analysis")
print("="*70)
t355 = load(355)

for idx, ex in enumerate(t355['train']):
    inp = ex['input']
    out = ex['output'][0][0]
    print(f"\nEx {idx} -> {out}:")
    print(grid_str(inp))
    print()
    
    colors = count_colors(inp)
    cc = connected_components_all(inp)
    
    # Find the "main" colors (those with large connected regions)
    # Noise colors appear as isolated pixels
    main_colors = {}
    noise_colors = {}
    for color, count in colors.items():
        comps = cc.get(color, 0)
        if comps <= 2 and count <= 15:
            noise_colors[color] = count
        else:
            main_colors[color] = count
    
    print(f"  Main colors: {main_colors}")
    print(f"  Noise colors: {noise_colors}")
    
    # Hypothesis: output is the main color that shares the border with another main color
    # Or output is the least common main color
    if main_colors:
        least_main = min(main_colors, key=main_colors.get)
        most_main = max(main_colors, key=main_colors.get)
        print(f"  Least main: {least_main}, Most main: {most_main}, actual={out}")

# Check hypothesis: output is the color that appears only in ONE large contiguous region
# while excluding noise colors
print("\n355: Single-region color hypothesis:")
for idx, ex in enumerate(t355['train']):
    inp = ex['input']
    out = ex['output'][0][0]
    colors = count_colors(inp)
    
    # For each color, check if it appears in one contiguous block
    color_regions = {}
    for color in colors:
        # Find all pixels of this color
        # Then count contiguous blocks
        h, w = len(inp), len(inp[0])
        visited = set()
        blocks = 0
        for r in range(h):
            for c in range(w):
                if inp[r][c] == color and (r,c) not in visited:
                    blocks += 1
                    stack = [(r,c)]
                    visited.add((r,c))
                    while stack:
                        cr, cc = stack.pop()
                        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                            nr, nc = cr+dr, cc+dc
                            if 0<=nr<h and 0<=nc<w and inp[nr][nc]==color and (nr,nc) not in visited:
                                visited.add((nr,nc))
                                stack.append((nr,nc))
        color_regions[color] = blocks
    
    # "Main" colors: those with count >= 20 and blocks == 1
    main_colors = {c: colors[c] for c in colors if colors[c] >= 30 and color_regions[c] == 1}
    print(f"  Ex {idx}: main colors={main_colors}, noise={[(c,colors[c]) for c in colors if c not in main_colors]}, actual={out}")

# Hypothesis: output = the color whose region has the LEAST noise contamination
# Noise = minority color pixels within a majority-color region
print("\n355: Noise analysis per region:")
for idx, ex in enumerate(t355['train']):
    inp = ex['input']
    out = ex['output'][0][0]
    h, w = len(inp), len(inp[0])
    colors = count_colors(inp)
    
    # Identify region boundaries: rows/cols where the dominant color changes
    # For each row, find the dominant color
    row_colors = []
    for r in range(h):
        rc = Counter()
        for c in range(w):
            if inp[r][c] != 0:
                rc[inp[r][c]] += 1
        if rc:
            row_colors.append(rc.most_common(1)[0][0])
        else:
            row_colors.append(0)
    
    # Find region splits
    regions = []
    cur_color = None
    cur_start = 0
    for r, c in enumerate(row_colors):
        if c != cur_color:
            if cur_color is not None and cur_color != 0:
                regions.append((cur_start, r-1, cur_color))
            cur_color = c
            cur_start = r
    if cur_color is not None and cur_color != 0:
        regions.append((cur_start, h-1, cur_color))
    
    print(f"  Ex {idx}: regions by row={regions}, actual={out}")

print("\nDone.")