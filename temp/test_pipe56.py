import sys
sys.path.insert(0, 'src')
from neurogolf.pipeline import build_one
from neurogolf.grids import load_task
import pathlib

task = load_task(56, 'data')
print(f'Task loaded: {len(task["train"])} train, {len(task["test"])} test')

result = build_one(56, task, pathlib.Path('networks'))
print(f'solver: {result.solver}')
print(f'saved: {result.saved}')
print(f'notes: {result.notes}')
if result.score:
    print(f'score.passed: {result.score.passed}')
    print(f'score.error: {result.score.error}')
    print(f'train: {result.score.train_right}/{result.score.train_right + result.score.train_wrong}')
    print(f'test: {result.score.test_right}/{result.score.test_right + result.score.test_wrong}')
    if result.score.arc_gen_right + result.score.arc_gen_wrong > 0:
        print(f'arc-gen: {result.score.arc_gen_right}/{result.score.arc_gen_right + result.score.arc_gen_wrong}')
    print(f'memory: {result.score.memory}')
    print(f'params: {result.score.params}')
    print(f'points: {result.score.points}')