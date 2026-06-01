
import json
d = json.load(open("/home/ansel/NeuroGolf/networks/build_summary.json"))
print("type:", type(d).__name__)
if isinstance(d, list):
    print("len:", len(d))
    if d:
        print("first keys:", list(d[0].keys()))
        solved = sum(1 for t in d if t.get("solver") != "none" and t.get("solver") is not None)
        total = len(d)
        total_score = sum(t.get("score", 0) for t in d if t.get("solver") != "none" and t.get("solver") is not None)
        print(f"solved: {solved}/{total}, score: {total_score}")
        for t in d[:3]: 
            print(t)
elif isinstance(d, dict):
    print("keys:", list(d.keys())[:10])
