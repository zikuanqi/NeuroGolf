"""Pipeline-level tests: a crashing solver must not abort a task or leak temp files."""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from neurogolf import pipeline  # noqa: E402
from neurogolf.solvers.identity import solve_identity  # noqa: E402


def _identity_task():
    grid = [[1, 2], [3, 4]]
    return {"train": [{"input": grid, "output": grid}], "test": [], "arc-gen": []}


def _boom(_task):
    raise RuntimeError("solver blew up")


def test_build_one_isolates_crashing_solver(tmp_path, monkeypatch):
    """A solver that raises must be skipped, letting a good solver still win."""
    monkeypatch.setattr(pipeline, "ALL_SOLVERS", [_boom, solve_identity])
    result = pipeline.build_one(1, _identity_task(), tmp_path)
    assert result.saved
    assert result.solver == "solve_identity"
    assert (tmp_path / "task001.onnx").is_file()


def test_build_one_cleans_temp_file_on_crash(tmp_path, monkeypatch):
    """No `_tmp_*.onnx` may survive even when every solver crashes."""
    monkeypatch.setattr(pipeline, "ALL_SOLVERS", [_boom])
    result = pipeline.build_one(7, _identity_task(), tmp_path)
    assert not result.saved
    assert "solver blew up" in result.notes
    assert list(tmp_path.glob("_tmp_*.onnx")) == []
