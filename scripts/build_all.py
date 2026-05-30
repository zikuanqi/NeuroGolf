"""Run the solver pipeline over every task and write `networks/taskNNN.onnx`."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from neurogolf.grids import load_task  # noqa: E402
from neurogolf.pipeline import build_one  # noqa: E402

NETWORKS = ROOT / "networks"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="start", type=int, default=1)
    parser.add_argument("--to", dest="end", type=int, default=400)
    parser.add_argument("--data", default=str(ROOT / "data"))
    args = parser.parse_args()

    NETWORKS.mkdir(exist_ok=True)
    summary = []
    started = time.time()
    solved = 0
    total_points = 0.0
    for task_num in range(args.start, args.end + 1):
        try:
            task = load_task(task_num, args.data)
        except FileNotFoundError:
            continue
        try:
            result = build_one(task_num, task, NETWORKS)
        except Exception as exc:  # noqa: BLE001 - one bad task must not abort the run
            print(f"task{task_num:03d}: ERROR ({type(exc).__name__}: {exc})")
            summary.append({
                "task": task_num, "solver": "error", "saved": False,
                "points": 0.0, "memory": None, "params": None,
            })
            continue
        if result.saved and result.score:
            solved += 1
            total_points += result.score.points
            print(f"task{task_num:03d}: {result.solver:20s} "
                  f"mem={result.score.memory} params={result.score.params} "
                  f"pts={result.score.points:.3f}")
        else:
            print(f"task{task_num:03d}: SKIP ({result.notes})")
        summary.append({
            "task": task_num,
            "solver": result.solver,
            "saved": result.saved,
            "points": result.score.points if result.score else 0.0,
            "memory": result.score.memory if result.score else None,
            "params": result.score.params if result.score else None,
        })
    elapsed = time.time() - started
    print(f"\nDone: {solved}/{args.end - args.start + 1} solved, "
          f"{total_points:.2f} pts, {elapsed:.1f}s")
    (NETWORKS / "build_summary.json").write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
