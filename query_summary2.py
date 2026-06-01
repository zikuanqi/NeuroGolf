import json

s = json.load(open("networks/build_summary.json"))

# Check the structure of first few entries
print("=== First 3 entries full ===")
for e in s[:3]:
    print(json.dumps(e, indent=2))

# Check passed field values
passed_vals = set()
for e in s:
    passed_vals.add(str(e.get("passed")))
print(f"\nUnique passed values: {passed_vals}")

# Count entries where solver is not 'none' and has points > 0
solved = [e for e in s if e.get("solver") != "none" and e.get("points", 0) > 0]
print(f"\nEntries with solver and points > 0: {len(solved)}")

# Check: entries where solver is 'none'
none_count = sum(1 for e in s if e.get("solver") == "none")
print(f"Entries with solver='none': {none_count}")

# Show all solved tasks
print(f"\n=== Solved tasks ({len(solved)}) ===")
for e in solved:
    print(f"  task {e['task']:03d}: {e['solver']:30s} points={e['points']:.2f}")

# Show unsolved with solver != none (attempted but failed or different structure)
attempted_not_solved = [e for e in s if e.get("solver") != "none" and e.get("points", 0) == 0]
print(f"\n=== Attempted but zero points ===")
for e in attempted_not_solved:
    print(f"  task {e['task']:03d}: {e.get('solver')} err={e.get('error','')[:80]}")

# Show total points
total = sum(e.get("points", 0) for e in s)
print(f"\nTotal points from build_summary: {total:.2f}")

# Check if there's a separate 'passed' key
# Check all keys in the entries
all_keys = set()
for e in s:
    all_keys.update(e.keys())
print(f"\nAll keys in entries: {all_keys}")