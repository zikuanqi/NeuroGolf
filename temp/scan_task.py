
import json, os, sys
from collections import Counter, defaultdict

BASE = "/home/ansel/NeuroGolf"
DATA_DIR = os.path.join(BASE, "data")
BUILD_SUMMARY_PATH = os.path.join(BASE, "networks", "build_summary.json")

with open(BUILD_SUMMARY_PATH) as f:
    bs = json.load(f)

solved = bs.get("solved_count", 0)
total = bs.get("total_tasks", 400)
total_score = bs.get("total_score", 0)
print(f"=== CURRENT PROGRESS ===")
print(f"Solved: {solved}/{total}, Score: {total_score:.2f}")

tasks = bs.get("tasks", [])
unsolved = [t for t in tasks if t.get("solver") == "none" or t.get("solver") is None]
solved_tasks = [t for t in tasks if t.get("solver") != "none" and t.get("solver") is not None]

print(f"\nUnsolved tasks: {len(unsolved)}")
print(f"Solved tasks: {len(solved_tasks)}")

def load_task(task_id):
    # Try with zero-padding
    paths = [
        os.path.join(DATA_DIR, f"task{task_id}.json"),
        os.path.join(DATA_DIR, f"{int(task_id):03d}.json"),
    ]
    for p in paths:
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f)
    return None

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

def analyze_task(task_id, task_data):
    info = {"task_id": task_id, "train_pairs": len(task_data.get("train", []))}
    
    if not task_data or "train" not in task_data:
        info["error"] = "no data"
        return info
    
    info["identity"] = check_identity(task_data)
    
    all_outputs = [p["output"] for p in task_data["train"]]
    all_inputs = [p["input"] for p in task_data["train"]]
    
    out_single_colors = [is_single_color(o) for o in all_outputs]
    info["output_single_color"] = all(sc[0] for sc in out_single_colors)
    if info["output_single_color"]:
        info["output_color"] = out_single_colors[0][1]
    
    input_sizes = [grid_size(i) for i in all_inputs]
    output_sizes = [grid_size(o) for o in all_outputs]
    
    max_input_dim = max(max(s) for s in input_sizes)
    max_output_dim = max(max(s) for s in output_sizes)
    info["max_input_dim"] = max_input_dim
    info["max_output_dim"] = max_output_dim
    info["small_task"] = max_input_dim <= 3 and max_output_dim <= 3
    
    # Check if input == output size (same size tasks may be identity or simple transforms)
    same_size = all(gs == go for gs, go in zip(input_sizes, output_sizes))
    info["same_size"] = same_size
    
    # Score
    info["score"] = task_data.get("score", 0)
    
    return info

# Analyze all unsolved tasks
results = []
for t in unsolved:
    tid = t.get("task_id", 0)
    td = load_task(tid)
    if td:
        r = analyze_task(tid, td)
        results.append(r)
    else:
        results.append({"task_id": tid, "error": "file not found"})

identity_tasks = [r for r in results if r.get("identity")]
single_color_outputs = [r for r in results if r.get("output_single_color") and not r.get("identity")]
small_tasks = [r for r in results if r.get("small_task")]
same_size_tasks = [r for r in results if r.get("same_size") and not r.get("identity")]

print(f"\n=== LOW-HANGING FRUITS ===")
print(f"\nIdentity tasks (input==output, use identity_model): {len(identity_tasks)}")
for r in sorted(identity_tasks, key=lambda x: x.get("score", 0), reverse=True):
    print(f"  task {r['task_id']}: score={r['score']:.2f}, pairs={r['train_pairs']}")

print(f"\nSingle-color output tasks: {len(single_color_outputs)}")
for r in sorted(single_color_outputs, key=lambda x: x.get("score", 0), reverse=True):
    print(f"  task {r['task_id']}: score={r['score']:.2f}, color={r.get('output_color')}, sizes={r.get('output_size_range', '?')}")

print(f"\nSmall tasks (max 3x3): {len(small_tasks)}")
for r in sorted(small_tasks, key=lambda x: x.get("score", 0), reverse=True):
    tags = []
    if r.get("identity"): tags.append("IDENTITY")
    if r.get("output_single_color"): tags.append("SINGLE_COLOR")
    tag_str = f" [{','.join(tags)}]" if tags else ""
    print(f"  task {r['task_id']}: score={r['score']:.2f}, in_dim<={r['max_input_dim']}, out_dim<={r['max_output_dim']}{tag_str}")

print(f"\nSame-size tasks (non-identity): {len(same_size_tasks)}")
for r in sorted(same_size_tasks, key=lambda x: x.get("score", 0), reverse=True)[:20]:
    print(f"  task {r['task_id']}: score={r['score']:.2f}, dim={r['max_input_dim']}")

print(f"\n=== SUMMARY ===")
print(f"Total unsolved analyzed: {len(results)}")
print(f"Identity: {len(identity_tasks)} (total score: {sum(r['score'] for r in identity_tasks):.2f})")
print(f"Single color: {len(single_color_outputs)} (total score: {sum(r['score'] for r in single_color_outputs):.2f})")
print(f"Small: {len(small_tasks)}")
print(f"Same size: {len(same_size_tasks)}")

output_path = os.path.join(BASE, "temp", "scan_results.json")
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nDetailed results saved to: {output_path}")
