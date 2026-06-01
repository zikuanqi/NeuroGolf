import sys, json
sys.path.insert(0, 'src')

from neurogolf.solvers.shape_classify import solve_shape_classify
from neurogolf.verify import _sanitize

task = json.load(open('data/task056.json'))
model = solve_shape_classify(task)
print(f'Model: {model is not None}')

if model:
    sanitized = _sanitize(model)
    if sanitized:
        print(f'Sanitized OK, {len(sanitized.graph.node)} nodes')
        for n in sanitized.graph.node:
            inputs = list(n.input)
            outputs = list(n.output)
            print(f'  {n.name}({n.op_type}): {inputs} -> {outputs}')
        # Check for type issues
        all_defined = {inp.name: inp.type for inp in sanitized.graph.input}
        all_defined.update({init.name: init.data_type for init in sanitized.graph.initializer})
        for n in sanitized.graph.node:
            for inp_name in n.input:
                if inp_name not in all_defined:
                    print(f'  ERROR: {inp_name} used by {n.name} but not defined')
            for outp_name in n.output:
                all_defined[outp_name] = n.op_type  # approximate
    else:
        print('Sanitize returned None')