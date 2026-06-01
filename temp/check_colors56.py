import json
t = json.load(open('data/task056.json'))
for i, ex in enumerate(t['train']):
    print(f'Example {i}:')
    print(f'  input: {ex["input"]}')
    colors = set()
    for r in ex['input']:
        for c in r:
            if c != 0:
                colors.add(c)
    print(f'  colors used: {colors}')
    print(f'  output: {ex["output"]}')
    print()