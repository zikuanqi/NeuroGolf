import sys, json, onnx, onnxruntime, numpy as np, pathlib
sys.path.insert(0, 'src')
from neurogolf.solvers import ALL_SOLVERS
from neurogolf.verify import verify
from neurogolf.grids import load_task, to_onehot

task = load_task(56, 'data')
out_dir = pathlib.Path('/tmp/test56_out')
out_dir.mkdir(exist_ok=True)
out_path = out_dir / 'task056.onnx'

# List all solvers
for s in ALL_SOLVERS:
    print(f'Solver: {s.__name__}')
    try:
        candidate = s(task)
        if candidate is None:
            print(f'  -> None (skipped)')
            continue
        print(f'  -> Model returned')
        
        # Save to temp
        tmp_path = out_dir / '_tmp_056.onnx'
        onnx.save(candidate, str(tmp_path))
        print(f'  -> Saved to {tmp_path}')
        
        # Verify
        score = verify(tmp_path, task, 56)
        print(f'  -> score.passed={score.passed}, error={score.error}')
        if score.error:
            print(f'  -> ERROR: {score.error[:500]}')
        if score.passed:
            print(f'  -> train: {score.train_right}/{score.train_right+score.train_wrong}')
            print(f'  -> test:  {score.test_right}/{score.test_right+score.test_wrong}')
            print(f'  -> arc-gen: {score.arc_gen_right}/{score.arc_gen_right+score.arc_gen_wrong}')
            print(f'  -> points: {score.points}')
        
        tmp_path.unlink(missing_ok=True)
    except Exception as exc:
        import traceback
        print(f'  -> EXCEPTION: {type(exc).__name__}: {exc}')
        traceback.print_exc()