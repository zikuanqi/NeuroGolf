
import json, os, sys

BASE = "/home/ansel/NeuroGolf"
DATA_DIR = os.path.join(BASE, "data")
NETWORKS_DIR = os.path.join(BASE, "networks")

# Get list of all task data files
data_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.json')])
print(f"Found {len(data_files)} task data files")

# Get list of solved ONNX files
onnx_files = set()
for f in os.listdir(NETWORKS_DIR):
    if f.endswith('.onnx'):
        # Extract task number
        onnx_files.add(f.replace('.onnx', ''))

print(f"Found {len(onnx_files)} ONNX model files")

def grid_size(grid):
    if not grid:
        return (0, 0)
    return len(grid), len(grid[0]) if grid else 0

def is_single_color(grid):
    colors = set()
    for row in grid:
        for c in row:
            if c != 0:
                colors.add(c)
    return len(colors) <= 1, list(colors)[0] if len(colors) == 1 else None

def check_identity(task_data):
    for pair in task_data.get("train", []):
        if pair["input"] != pair["output"]:
            return False
    return True

# Scan all tasks
identity_tasks = []
single_color_tasks = []
small_tasks = []
same_size_tasks = []

for fname in data_files:
    task_id_str = fname.replace('.json', '').replace('task', '')
    task_id = int(task_id_str)
    
    # Check if solved
    solved = task_id_str in onnx_files
    
    if solved:
        continue
    
    # Load and analyze unsolved task
    fpath = os.path.join(DATA_DIR, fname)
    with open(fpath) as f:
        td = json.load(f)
    
    score = td.get("score", 0)
    
    # Identity check
    if check_identity(td):
        identity_tasks.append((task_id, score, len(td.get("train", []))))
        continue
    
    # Output analysis
    all_outputs = [p["output"] for p in td["train"]]
    all_inputs = [p["input"] for p in td["train"]]
    
    # Single color output
    out_colors = [is_single_color(o) for o in all_outputs]
    if all(sc[0] for sc in out_colors):
        single_color_tasks.append((task_id, score, out_colors[0][1]))
        continue
    
    # Size analysis
    input_sizes = [grid_size(i) for i in all_inputs]
    output_sizes = [grid_size(o) for o in all_outputs]
    max_in = max(max(s) for s in input_sizes)
    max_out = max(max(s) for s in output_sizes)
    same_size = all(gs == go for gs, go in zip(input_sizes, output_sizes))
    
    if max_in <= 3 and max_out <= 3:
        small_tasks.append((task_id, score, max_in, max_out))
    elif same_size and max_in <= 10:
        same_size_tasks.append((task_id, score, max_in, max_out))

# Sort by score descending
identity_tasks.sort(key=lambda x: x[1], reverse=True)
single_color_tasks.sort(key=lambda x: x[1], reverse=True)
small_tasks.sort(key=lambda x: x[1], reverse=True)
same_size_tasks.sort(key=lambda x: x[1], reverse=True)

print(f"\n=== RESULTS ===")
print(f"\nIDENTITY TASKS (input==output): {len(identity_tasks)}")
for tid, score, ntrain in identity_tasks:
    print(f"  task {tid}: score={score:.2f}, train_pairs={ntrain}")

print(f"\nSINGLE COLOR OUTPUT: {len(single_color_tasks)}")
for tid, score, color in single_color_tasks[:30]:
    print(f"  task {tid}: score={score:.2f}, color={color}")

print(f"\nSMALL TASKS (<=3x3): {len(small_tasks)}")
for tid, score, mi, mo in small_tasks[:30]:
    print(f"  task {tid}: score={score:.2f}, in_dim<={mi}, out_dim<={mo}")

print(f"\nSAME SIZE (<=10): {len(same_size_tasks)}")
for tid, score, mi, mo in same_size_tasks[:20]:
    print(f"  task {tid}: score={score:.2f}, in_dim<={mi}, out_dim<={mo}")

# Grand total stats
total_unsolved = len(data_files) - len(onnx_files)
total_potential = sum(s for _, s, *_ in identity_tasks) + sum(s for _, s, *_ in single_color_tasks)
print(f"\n=== SUMMARY ===")
print(f"Total unsolved: {total_unsolved}")
print(f"Identity: {len(identity_tasks)} (score: {sum(s for _,s,*_ in identity_tasks):.2f})")
print(f"Single color: {len(single_color_tasks)} (score: {sum(s for _,s,*_ in single_color_tasks):.2f})")
print(f"Small (<=3x3): {len(small_tasks)}")
print(f"Same-size (<=10): {len(same_size_tasks)}")
