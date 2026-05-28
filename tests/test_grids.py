"""Round-trip tests for the one-hot grid encoding."""
from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from neurogolf.grids import from_onehot, to_onehot  # noqa: E402


def test_roundtrip_small():
    grid = [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 0],
    ]
    encoded = to_onehot(grid)
    assert encoded is not None
    assert encoded.shape == (1, 10, 30, 30)
    decoded = from_onehot(encoded)
    assert decoded == grid


def test_roundtrip_max_size():
    grid = [[(r * 7 + c) % 10 for c in range(30)] for r in range(30)]
    encoded = to_onehot(grid)
    decoded = from_onehot(encoded)
    assert decoded == grid


def test_oversized_rejected():
    big = [[0] * 31] * 31
    assert to_onehot(big) is None


def test_padding_decodes_to_empty():
    encoded = np.zeros((1, 10, 30, 30), dtype=np.float32)
    assert from_onehot(encoded) == []
