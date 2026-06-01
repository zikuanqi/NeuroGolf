import json
import sys

s = json.load(open("networks/build_summary.json"))
solved = [e["task"] for e in s if e.get("passed")]
unsolved = [e["task"] for e in s if not e.get("passed")]

print("Total entries:", len(s))
print("Solved:", len(solved))
print("Unsolved:", len(unsolved))

print("\n--- First 30 unsolved tasks ---")
print(unsolved[:30])

print("\n--- Last 15 entries ---")
for e in s[-15:]:
    print(f"  task {e['task']}: passed={e.get('passed')}, solver={e.get('solver','?')}, points={e.get('points','?')}")

# Group unsolved by solver attempt
from collections import Counter
attempts = Counter()
for e in s:
    if not e.get("passed"):
        attempts[e.get("solver", "none")] += 1
print("\n--- Unsolved by last attempted solver ---")
for name, count in attempts.most_common():
    print(f"  {name}: {count}")