import sys, os, json, time
sys.path.insert(0, "/home/ansel/NeuroGolf/src")
from neurogolf.grids import load_task
from neurogolf.pipeline import build_one
from pathlib import Path

NETWORKS = Path("/home/ansel/NeuroGolf/networks")
DATA_DIR = "/home/ansel/NeuroGolf/data"

# Single-color tasks to try (first batch)
task_ids = [21, 34, 36, 38, 57, 58, 71, 79, 88, 93, 104, 109, 121, 124, 134, 137, 141, 149, 161, 168]

results = []
for tid in task_ids:
    try:
        task = load_task(tid, DATA_DIR)
        result = build_one(tid, task, NETWORKS)
        if result.saved and result.score:
            print(f"task{tid:03d}: SOLVED by {result.solver:20s} pts={result.score.points:.3f}")
            results.append({"task": tid, "solver": result.solver, "points": result.score.points})
        else:
            print(f"task{tid:03d}: SKIP ({result.notes})")
    except FileNotFoundError:
        print(f"task{tid:03d}: NOT FOUND")
    except Exception as e:
        print(f"task{tid:03d}: ERROR {type(e).__name__}: {e}")

print(f"\nSolved: {len(results)}/{len(task_ids)}")
for r in results:
    print(f"  task {r['task']}: {r['solver']} ({r['points']:.2f} pts)")
