"""Verify task 103 = (h-sym AND v-sym -> colour A, else colour B), and scan
other 1x1 tasks for any 'symmetry predicate -> colour' classifier."""
import json

def load(n):
    return json.load(open(f'data/task{n:03d}.json'))

def all_pairs(t):
    for split in ('train', 'test', 'arc-gen'):
        for ex in t.get(split, []):
            yield ex['input'], ex['output']

def hsym(g):
    return g == [list(reversed(r)) for r in g]

def vsym(g):
    return g == [list(r) for r in reversed(g)]

# predicates of the grid -> bool
PREDS = {
    'h&v': lambda g: hsym(g) and vsym(g),
    'h|v': lambda g: hsym(g) or vsym(g),
    'h':   lambda g: hsym(g),
    'v':   lambda g: vsym(g),
}

for tn in (48, 56, 103, 291, 346, 355):
    t = load(tn)
    best = None
    for pname, pred in PREDS.items():
        m = {}
        ok = True
        for i, o in all_pairs(t):
            if len(i) > 30 or len(i[0]) > 30:
                continue
            key = pred(i)
            out = o[0][0]
            if m.get(key, out) != out:
                ok = False
                break
            m[key] = out
        if ok and len(m) == 2:        # predicate actually splits the data
            best = (pname, m)
            break
    print(f'task {tn}: {best}')
