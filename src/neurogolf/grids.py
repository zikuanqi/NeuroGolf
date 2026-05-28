"""Grid <-> one-hot tensor conversions matching the official scorer."""
from __future__ import annotations

import json
import pathlib
from typing import Iterable

import numpy as np

BATCH, CHANNELS, HEIGHT, WIDTH = 1, 10, 30, 30
TENSOR_SHAPE = (BATCH, CHANNELS, HEIGHT, WIDTH)


def to_onehot(grid: list[list[int]]) -> np.ndarray | None:
    if max(len(grid), len(grid[0])) > 30:
        return None
    out = np.zeros(TENSOR_SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            out[0, color, r, c] = 1.0
    return out


def from_onehot(tensor: np.ndarray) -> list[list[int]]:
    _, channels, height, width = tensor.shape
    rows: list[list[int]] = []
    for r in range(height):
        row: list[int] = []
        for c in range(width):
            colors = [k for k in range(channels) if tensor[0, k, r, c] == 1]
            if len(colors) == 1:
                row.append(colors[0])
            elif colors:
                row.append(11)
            else:
                row.append(10)
        while row and row[-1] == 10:
            row.pop()
        rows.append(row)
    while rows and not rows[-1]:
        rows.pop()
    return rows


def load_task(task_num: int, data_dir: str | pathlib.Path = "data") -> dict:
    path = pathlib.Path(data_dir) / f"task{task_num:03d}.json"
    with path.open() as f:
        return json.load(f)


def all_examples(task: dict) -> Iterable[dict]:
    for split in ("train", "test", "arc-gen"):
        for ex in task.get(split, []):
            yield ex


def is_identity(task: dict) -> bool:
    """True if every (input, output) pair has input == output."""
    for ex in all_examples(task):
        if ex["input"] != ex["output"]:
            return False
    return True
