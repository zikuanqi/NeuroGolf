"""Top-level pipeline: pick the best solver for each task, verify, save."""
from __future__ import annotations

import pathlib
from dataclasses import dataclass

import onnx

from .solvers import ALL_SOLVERS
from .verify import Score, verify


@dataclass
class BuildResult:
    task_num: int
    solver: str
    saved: bool
    score: Score | None
    notes: str = ""


def build_one(task_num: int, task: dict, out_dir: pathlib.Path) -> BuildResult:
    out_path = out_dir / f"task{task_num:03d}.onnx"
    best: tuple[float, str, onnx.ModelProto, Score] | None = None
    last_err = ""

    tmp_path = out_dir / f"_tmp_{task_num:03d}.onnx"
    for solver in ALL_SOLVERS:
        # A crash in one solver (build, save, or ONNX Runtime verify) must not
        # abort the remaining solvers for this task, and the temp file must
        # always be cleaned up regardless of how this iteration exits.
        try:
            candidate = solver(task)
            if candidate is None:
                continue
            onnx.save(candidate, str(tmp_path))
            score = verify(tmp_path, task, task_num)
        except Exception as exc:  # noqa: BLE001 - isolate per-solver failures
            last_err = f"{solver.__name__}: {type(exc).__name__}: {exc}"
            continue
        finally:
            tmp_path.unlink(missing_ok=True)
        if not score.passed:
            last_err = f"{solver.__name__}: {score.error or 'examples wrong'}"
            continue
        if best is None or score.points > best[0]:
            best = (score.points, solver.__name__, candidate, score)

    if best is None:
        return BuildResult(task_num, "none", False, None, last_err)

    points, name, model, score = best
    onnx.save(model, str(out_path))
    return BuildResult(task_num, name, True, score)
