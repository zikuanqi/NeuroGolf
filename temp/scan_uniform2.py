import json, os

BASE = "/home/ansel/NeuroGolf"
DATA_DIR = os.path.join(BASE, "data")
NETWORKS_DIR = os.path.join(BASE, "networks")

onnx_solved = set()
for f in os.listdir(NETWORKS_DIR):
    if f.endswith('.onnx'):
        try:
            num = int(f.replace('task', '').replace('.onnx', ''))
            onnx_solved.add(num)
        except ValueError:
            pass

def grid_size(grid):
    if not grid: return (0,0)
    return len(grid), len(grid[0]) if grid else 0

def is_single_color(grid):
    colors = set()
    for row in grid:
        for c in row:
            if c != 0: colors.add(c)
    return len(colors) <= 1, list(colors)[0] if len(colors) == 1 else None

def is_uniform(grid):
    if not grid: return True, 0
    first = grid[0][0]
    for row in grid:
        for c in row:
            if c != first: return False, None
    return True, first

uniform_tasks = []
single_color_1x1 = []
single_color_multi = []

for fname in sorted(os.listdir(DATA_DIR)):
    if not fname.endswith('.json'): continue
    task_id = int(fname.replace('.json','').replace('task',''))
    if task_id in onnx_solved: continue
    
    with open(os.path.join(DATA_DIR, fname)) as f:
        td = json.load(f)
    
    all_outputs = [p["output"] for p in td["train"]]
    
    out_colors = [is_single_color(o) for o in all_outputs]
    if not all(sc[0] for sc in out_colors): continue
    
    uniforms = [is_uniform(o) for o in all_outputs]
    all_uniform = all(u[0] for u in uniforms)
    uniform_color = uniforms[0][1] if all_uniform else None
    
    out_sizes = [grid_size(o) for o in all_outputs]
    all_1x1 = all(s == (1,1) for s in out_sizes)
    
    if all_uniform:
        uniform_tasks.append((task_id, uniform_color, out_sizes[0]))
    elif all_1x1:
        single_color_1x1.append((task_id, out_colors[0][1]))
    else:
        single_color_multi.append((task_id, out_colors[0][1], out_sizes))

print("=== TRULY UNIFORM OUTPUT ===")
print(f"Count: {len(uniform_tasks)}")
for tid, color, sz in uniform_tasks:
    print(f"  task {tid}: color={color}, size={sz}")

print(f"\n=== 1x1 SINGLE COLOR ===")
print(f"Count: {len(single_color_1x1)}")
for tid, color in single_color_1x1:
    print(f"  task {tid}: color={color}")

print(f"\n=== MULTI-CELL SINGLE COLOR ===")
print(f"Count: {len(single_color_multi)}")
for tid, color, sizes in single_color_multi[:30]:
    print(f"  task {tid}: color={color}, sizes={sizes}")

print(f"\n=== TOTAL ===")
print(f"Uniform: {len(uniform_tasks)}, 1x1: {len(single_color_1x1)}, Multi: {len(single_color_multi)}")
