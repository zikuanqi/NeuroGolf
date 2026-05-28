"""Submit a packaged zip to the Kaggle competition."""
from __future__ import annotations

import subprocess
import sys


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: submit.py <path-to-zip> <message>")
        return 2
    zip_path, message = sys.argv[1], sys.argv[2]
    return subprocess.call([
        "kaggle", "competitions", "submit",
        "-c", "neurogolf-2026",
        "-f", zip_path,
        "-m", message,
    ])


if __name__ == "__main__":
    raise SystemExit(main())
