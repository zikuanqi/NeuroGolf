"""Analyze 1x1 output tasks: 48, 56, 291, 346, 355.
Find spatial predicates that determine the output color."""
import json
import os
import sys
from collections import Counter

PROJECT = '/home/ansel/NeuroGolf'
DATA_DIR = os.path.join(PROJECT, 'data')

def load(n):
    path = os.path.join(DATA_DIR, f'task{n:03d}.json')
    if not os.path.exists(path):
        path = os.path.join(DATA_DIR, f'task{n}.json')
    return json.load(open(path))

def grid_str(g, highlight_nonzero=True):
    """Compact string representation of grid."""
    if not g:
        return '(empty)'
    lines = []
    for row in g:
        if highlight_nonzero:
            line = ''.join(str(c) if c != 0 else '.' for c in row)
        else:
            line = ''.join(str(c) for c in row)
        lines.append(line)
    return '\n'.join(lines)

def bbox(g):
    """Return (min_r, min_c, max_r, max_c) of non-zero pixels."""
    rows, cols = [], []
    for r, row in enumerate(g):
        for c, val in enumerate(row):
            if val != 0:
                rows.append(r)
                cols.append(c)
    if not rows:
        return (0, 0, len(g)-1, len(g[0])-1 if g else 0)
    return (min(rows), min(cols), max(rows), max(cols))

def count_colors(g):
    """Count non-zero colors."""
    c = Counter()
    for row in g:
        for v in row:
            if v != 0:
                c[v] += 1
    return dict(c)

def connected_components(g):
    """Count connected components (non-zero pixels, 4-connectivity)."""
    if not g:
        return 0
    h, w = len(g), len(g[0])
    visited = set()
    count = 0
    for r in range(h):
        for c in range(w):
            if g[r][c] != 0 and (r, c) not in visited:
                count += 1
                # BFS
                stack = [(r, c)]
                visited.add((r, c))
                while stack:
                    cr, cc = stack.pop()
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if 0 <= nr < h and 0 <= nc < w:
                            if g[nr][nc] != 0 and (nr, nc) not in visited:
                                visited.add((nr, nc))
                                stack.append((nr, nc))
    return count

def is_solid_rect(g):
    """Check if non-zero pixels form a solid axis-aligned rectangle."""
    rmin, cmin, rmax, cmax = bbox(g)
    if rmin > rmax or cmin > cmax:
        return False
    for r in range(rmin, rmax+1):
        for c in range(cmin, cmax+1):
            if g[r][c] == 0:
                return False
    # Also check no non-zero outside bbox
    for r, row in enumerate(g):
        for c, val in enumerate(row):
            if val != 0 and (r < rmin or r > rmax or c < cmin or c > cmax):
                return False
    return True

def count_corners(g):
    """Count corner pixels in non-zero shape. A corner is a non-zero pixel
    with exactly 2 orthogonal neighbors that form a 90-degree turn."""
    if not g:
        return 0
    h, w = len(g), len(g[0])
    corners = 0
    for r in range(h):
        for c in range(w):
            if g[r][c] == 0:
                continue
            # 4 neighbors
            nbrs = []
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < h and 0 <= nc < w and g[nr][nc] != 0:
                    nbrs.append((dr, dc))
            # Corner: exactly 2 neighbors that are perpendicular
            if len(nbrs) == 2:
                d1, d2 = nbrs
                if d1[0] * d2[0] + d1[1] * d2[1] == 0:  # perpendicular
                    corners += 1
            # Endpoint: exactly 1 neighbor
            elif len(nbrs) == 1:
                corners += 1
    return corners

def count_holes(g):
    """Count holes in non-zero shape using flood fill on zero cells."""
    if not g:
        return 0
    h, w = len(g), len(g[0])
    # Mark all zero cells reachable from border
    visited = [[False]*w for _ in range(h)]
    # BFS from border zeros
    from collections import deque
    q = deque()
    for r in range(h):
        for c in [0, w-1]:
            if g[r][c] == 0 and not visited[r][c]:
                visited[r][c] = True
                q.append((r, c))
    for c in range(w):
        for r in [0, h-1]:
            if g[r][c] == 0 and not visited[r][c]:
                visited[r][c] = True
                q.append((r, c))
    while q:
        cr, cc = q.popleft()
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = cr+dr, cc+dc
            if 0 <= nr < h and 0 <= nc < w and g[nr][nc] == 0 and not visited[nr][nc]:
                visited[nr][nc] = True
                q.append((nr, nc))
    # Count unvisited zeros = holes
    holes = 0
    for r in range(h):
        for c in range(w):
            if g[r][c] == 0 and not visited[r][c]:
                holes += 1
    # Return number of distinct hole regions by BFS
    hole_regions = 0
    hole_visited = [[False]*w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            if g[r][c] == 0 and not visited[r][c] and not hole_visited[r][c]:
                hole_regions += 1
                q2 = deque([(r, c)])
                hole_visited[r][c] = True
                while q2:
                    cr2, cc2 = q2.popleft()
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr2+dr, cc2+dc
                        if 0 <= nr < h and 0 <= nc < w and g[nr][nc] == 0 and not visited[nr][nc] and not hole_visited[nr][nc]:
                            hole_visited[nr][nc] = True
                            q2.append((nr, nc))
    return hole_regions

def pixel_count(g):
    """Count non-zero pixels."""
    return sum(1 for row in g for v in row if v != 0)

def aspect_ratio(g):
    """Aspect ratio (w/h) of the bounding box."""
    rmin, cmin, rmax, cmax = bbox(g)
    h = rmax - rmin + 1
    w = cmax - cmin + 1
    if h == 0:
        return None
    return w / h

def shape_type(g):
    """Try to classify shape type: 'L', 'T', '+', 'line', 'rect', 'other'."""
    rmin, cmin, rmax, cmax = bbox(g)
    h = rmax - rmin + 1
    w = cmax - cmin + 1
    px = pixel_count(g)
    cc = connected_components(g)
    corners = count_corners(g)
    holes = count_holes(g)
    
    if px == 0:
        return 'empty'
    if cc > 1:
        return f'multi({cc})'
    if holes > 0:
        return f'hole({holes})'
    
    # Solid rectangle check
    if is_solid_rect(g):
        if h == 1 or w == 1:
            return f'line_{max(h,w)}'
        return f'rect_{w}x{h}'
    
    # Check L shape: exactly one "missing" corner
    expected_px = h * w
    missing = expected_px - px
    if corners == 6:  # L shape has 6 corners
        return f'L_{w}x{h}'
    if corners == 8:  # +/T shape
        # T shape vs + shape
        # + shape: cross in center
        center_r = (rmin + rmax) // 2
        center_c = (cmin + cmin) // 2
        return f'corners_{corners}'
    
    return f'shape_{w}x{h}_p{px}_c{corners}'

def analyze_task(tn):
    """Full analysis of one task."""
    t = load(tn)
    print(f"\n{'='*70}")
    print(f"TASK {tn}")
    print(f"{'='*70}")
    
    # Collect train examples
    train_examples = t.get('train', [])
    test_examples = t.get('test', [])
    
    # Output distribution
    outputs = []
    for ex in train_examples:
        o = ex['output']
        if o and o[0]:
            outputs.append(o[0][0])
    
    print(f"\nTrain examples: {len(train_examples)}")
    print(f"Outputs: {Counter(outputs)}")
    print(f"Output colors: {sorted(set(outputs))}")
    
    # Detailed analysis per example
    for idx, ex in enumerate(train_examples):
        inp = ex['input']
        out = ex['output']
        out_color = out[0][0] if out and out[0] else '?'
        
        bb = bbox(inp)
        h = bb[2] - bb[0] + 1
        w = bb[3] - bb[1] + 1
        px = pixel_count(inp)
        cc = connected_components(inp)
        corners = count_corners(inp)
        holes = count_holes(inp)
        solid = is_solid_rect(inp)
        ar = aspect_ratio(inp)
        colors = count_colors(inp)
        
        print(f"\n--- Example {idx} | Output: {out_color} ---")
        print(f"Input ({len(inp)}x{len(inp[0])}):")
        print(grid_str(inp))
        print(f"  BBox: r{bb[0]}-{bb[2]}, c{bb[1]}-{bb[3]} ({h}x{w})")
        print(f"  Pixels: {px}, SolidRect: {solid}, Holes: {holes}")
        print(f"  Components: {cc}, Corners: {corners}, AR: {ar}")
        print(f"  Colors: {colors}")
        print(f"  Shape: {shape_type(inp)}")
    
    # Try predicates
    print(f"\n--- Predicate Analysis ---")
    for ex in train_examples:
        inp = ex['input']
        o = ex['output']
        out_color = o[0][0] if o and o[0] else '?'
        
        bb = bbox(inp)
        h = bb[2] - bb[0] + 1
        w = bb[3] - bb[1] + 1
        px = pixel_count(inp)
        cc = connected_components(inp)
        corners = count_corners(inp)
        holes = count_holes(inp)
        solid = is_solid_rect(inp)
        
        # Additional predicates
        colors = count_colors(inp)
        unique_colors = len(colors)
        has_color = {c: c in colors for c in range(1, 10)}
        is_line = (h == 1 or w == 1) and solid
        
        print(f"  Ex {idx} -> {out_color}: solid={solid} line={is_line} holes={holes} "
              f"cc={cc} corners={corners} px={px} {h}x{w} colors={unique_colors}")
    
    return t

# Run analysis for each task
for tn in [48, 56, 291, 346, 355]:
    try:
        analyze_task(tn)
    except Exception as e:
        print(f"\nERROR analyzing task {tn}: {e}")
        import traceback
        traceback.print_exc()

print("\n\nDone.")