"""Smoke tests that the registered solvers handle their target patterns."""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from neurogolf.solvers.identity import solve_identity  # noqa: E402
from neurogolf.solvers.remap import solve_remap  # noqa: E402
from neurogolf.solvers.spatial import solve_transpose  # noqa: E402


def _make_task(pairs):
    train = [{"input": i, "output": o} for i, o in pairs]
    return {"train": train, "test": [], "arc-gen": []}


def test_identity_picks_only_identity():
    task = _make_task([([[1, 2], [3, 4]], [[1, 2], [3, 4]])])
    model = solve_identity(task)
    assert model is not None
    assert model.graph.node[0].op_type == "Identity"


def test_identity_rejects_remap():
    task = _make_task([([[1, 2], [3, 4]], [[2, 1], [4, 3]])])
    assert solve_identity(task) is None


def test_remap_simple_swap():
    task = _make_task([
        ([[1, 0], [0, 1]], [[2, 0], [0, 2]]),
        ([[1, 1], [0, 0]], [[2, 2], [0, 0]]),
    ])
    model = solve_remap(task)
    assert model is not None
    assert model.graph.node[0].op_type == "Conv"
    assert len(model.graph.initializer) == 1
    w = model.graph.initializer[0]
    assert list(w.dims) == [10, 10, 1, 1]


def test_remap_rejects_position_dependent():
    task = _make_task([([[1, 1], [1, 1]], [[2, 3], [4, 5]])])
    assert solve_remap(task) is None


def test_transpose():
    task = _make_task([([[1, 2, 3], [4, 5, 6]], [[1, 4], [2, 5], [3, 6]])])
    model = solve_transpose(task)
    assert model is not None
    assert model.graph.node[0].op_type == "Transpose"
