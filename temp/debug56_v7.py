import sys, json
sys.path.insert(0, 'src')
from neurogolf.verify import verify
from neurogolf.solvers.shape_classify import solve_shape_classify

task = json.load(open('data/task056.json'))
model = solve_shape_classify(task)
result = verify(model, task)
print(f'Verify result: {result}')