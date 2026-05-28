"""Zip all `task*.onnx` under networks/ into submissions/submission_<ts>.zip."""
from __future__ import annotations

import pathlib
import time
import zipfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
NETWORKS = ROOT / "networks"
SUBMISSIONS = ROOT / "submissions"


def main() -> int:
    SUBMISSIONS.mkdir(exist_ok=True)
    networks = sorted(NETWORKS.glob("task*.onnx"))
    if not networks:
        print("no networks found")
        return 1
    # Kaggle requires the submission file be named exactly `submission.zip`.
    ts = time.strftime("%Y%m%d_%H%M%S")
    versioned = SUBMISSIONS / f"submission_{ts}.zip"
    out = SUBMISSIONS / "submission.zip"
    with zipfile.ZipFile(versioned, "w", zipfile.ZIP_DEFLATED) as z:
        for n in networks:
            z.write(n, arcname=n.name)
    out.write_bytes(versioned.read_bytes())
    print(f"wrote {out} ({out.stat().st_size} bytes, {len(networks)} files); "
          f"snapshot at {versioned.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
