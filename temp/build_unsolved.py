import sys, os, json, time
sys.path.insert(0, "/home/ansel/NeuroGolf/src")
from neurogolf.grids import load_task
from neurogolf.pipeline import build_one
from pathlib import Path

NETWORKS = Path("/home/ansel/NeuroGolf/networks")
DATA_DIR = "/home/ansel/NeuroGolf/data"

# Find unsolved tasks
solved = set()
for f in os.listdir(str(NETWORKS)):
    if f.endswith('.onnx'):
        try: solved.add(int(f.replace('task','').replace('.onnx','')))
        except: pass

# Find all task IDs
all_tasks = []
for f in os.listdir(DATA_DIR):
    if f.endswith('.json'):
        try: all_tasks.append(int(f.replace('task','').replace('.json','')))
        except: pass

unsolved = sorted([t for t in all_tasks if t not in solved])
print(f"Total: {len(all_tasks)}, Solved: {len(solved)}, Unsolved: {len(unsolved)}")

# Try building only unsolved tasks
results = []
started = time.time()
new_solved = 0
new_points = 0.0

for tid in unsolved:
    try:
        task = load_task(tid, DATA_DIR)
        result = build_one(tid, task, NETWORKS)
        if result.saved and result.score:
            new_solved += 1
            new_points += result.score.points
            print(f"task{tid:03d}: SOLVED by {result.solver:20s} pts={result.score.points:.3f} params={result.score.params} mem={result.score.memory}")
            results.append({"task": tid, "solver": result.solver, "points": result.score.points})
        else:
            if tid % 20 == 0:
                print(f"  ... scanned up to task {tid} ...")
    except Exception as e:
        print(f"task{tid:03d}: ERROR {type(e).__name__}: {e}")

elapsed = time.time() - started
print(f"\nDone: {new_solved}/{len(unsolved)} newly solved, {new_points:.2f} pts, {elapsed:.1f}s")
