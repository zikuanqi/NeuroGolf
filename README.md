# NeuroGolf 2026

Solutions for the [2026 NeuroGolf Championship](https://www.kaggle.com/competitions/neurogolf-2026) on Kaggle.

The competition asks for ONNX networks that solve ARC-AGI tasks while minimizing
`memory_bytes + parameter_count`. Each task that passes all train / test /
arc-gen examples earns `max(1, 25 - ln(memory + params))` points.

## Layout

- `src/neurogolf/` — Python package
  - `grids.py` — grid <-> one-hot tensor conversions and example helpers
  - `onnx_ops.py` — small helpers for building ONNX graphs in opset 10
  - `solvers/` — per-task or per-family solvers, each returning an `onnx.ModelProto`
  - `pipeline.py` — orchestrates: try each solver, verify, write the network
  - `verify.py` — clean-room reimplementation of the official scorer
- `scripts/`
  - `download.py` — pulls competition data via Kaggle CLI into `data/`
  - `build_all.py` — runs the pipeline over all 400 tasks
  - `package_submission.py` — zips the produced `.onnx` files
  - `submit.py` — posts the submission zip to Kaggle
- `networks/` — generated `task001.onnx` ... `task400.onnx`
- `submissions/` — packaged zip files ready for upload

## Quick start

```
pip install -r requirements.txt
python scripts/download.py            # populates data/
python scripts/build_all.py           # produces networks/
python scripts/package_submission.py  # produces submissions/submission_<ts>.zip
python scripts/submit.py submissions/submission_<ts>.zip "baseline run"
```

## Scoring notes

- Networks must be ONNX opset 10, statically shaped, input `(1,10,30,30)` named
  `input`, output `(1,10,30,30)` named `output`. The output is thresholded at
  `> 0.0` to recover the predicted one-hot encoding.
- File size limit: 1.44 MB per `.onnx`.
- Banned ops: LOOP, SCAN, NONZERO, UNIQUE, SCRIPT, FUNCTION, COMPRESS, any
  `Sequence*` op, graph attributes, sub-graphs, functions, custom domains.
- arc-gen examples larger than 30x30 are skipped by the official scorer.
