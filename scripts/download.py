"""Fetch competition data via the Kaggle CLI into ./data."""
from __future__ import annotations

import pathlib
import subprocess
import sys
import zipfile

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"


def main() -> int:
    DATA_DIR.mkdir(exist_ok=True)
    zip_path = DATA_DIR / "neurogolf-2026.zip"
    if not zip_path.exists():
        result = subprocess.run(
            ["kaggle", "competitions", "download",
             "-c", "neurogolf-2026", "-p", str(DATA_DIR)],
            check=False,
        )
        if result.returncode != 0:
            print("kaggle download failed", file=sys.stderr)
            return result.returncode
    if not any(DATA_DIR.glob("task001.json")):
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(DATA_DIR)
    tasks = sorted(DATA_DIR.glob("task*.json"))
    print(f"data/: {len(tasks)} task files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
