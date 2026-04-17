"""
Count Total Targets Reached
Standalone script to count how many target specifications have at least one solution that reached the target.
"""
import json
import sys
from pathlib import Path

# Configuration
BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"
DATA_DIR = BASE_DIR / "data"

DATA_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("COUNTING TOTAL TARGETS REACHED")
print("="*80)

# Load data
data_path = RESULTS_DIR / "all_input_output_values_complete.json"
if not data_path.exists():
    print(f"[ERROR] Data file not found: {data_path}")
    sys.exit(1)

print(f"\n[1] Loading data from: {data_path}")
with open(data_path, 'r') as f:
    data = json.load(f)

print(f"[INFO] Loaded {data['total_solutions']} solutions from {data['total_specifications']} specifications")

# Count targets reached
print("\n[2] Counting targets reached...")

# Count total solutions reached
total_solutions_reached = 0
total_solutions_unreached = 0

# Count specs with reached solutions
specs_with_reached = set()
specs_with_unreached = set()

for sol in data['solutions']:
    spec_num = sol['spec']
    if sol.get('target_reached') == 'Yes':
        total_solutions_reached += 1
        specs_with_reached.add(spec_num)
    else:
        total_solutions_unreached += 1
        # Track unreached specs (but don't add if already in reached)
        if spec_num not in specs_with_reached:
            specs_with_unreached.add(spec_num)

# Specification-level counts
total_reached_specs = len(specs_with_reached)
total_specs = data['total_specifications']
total_unreached_specs = total_specs - total_reached_specs
reached_specs_percentage = (total_reached_specs / total_specs) * 100

# Solution-level counts
total_solutions = data['total_solutions']
reached_solutions_percentage = (total_solutions_reached / total_solutions) * 100

# Display results
print("\n" + "="*80)
print("RESULTS")
print("="*80)
print("\n[Solution-Level Counts]")
print(f"Total solutions: {total_solutions}")
print(f"Solutions that reached target: {total_solutions_reached}")
print(f"Solutions that did not reach target: {total_solutions_unreached}")
print(f"Reached percentage: {reached_solutions_percentage:.1f}%")

print("\n[Specification-Level Counts]")
print(f"Total specifications: {total_specs}")
print(f"Specifications with at least one reached solution: {total_reached_specs}")
print(f"Specifications with no reached solutions: {total_unreached_specs}")
print(f"Reached percentage: {reached_specs_percentage:.1f}%")
print("="*80)

# Get list of unreached specs
unreached_specs = sorted([s for s in range(1, total_specs + 1) if s not in specs_with_reached])

if unreached_specs:
    print(f"\nUnreached specifications ({len(unreached_specs)}):")
    print(f"  {unreached_specs}")
else:
    print("\nAll specifications have at least one reached solution!")

# Save results
print("\n[3] Saving results...")

results = {
    'solution_level': {
        'total_solutions': total_solutions,
        'total_solutions_reached': total_solutions_reached,
        'total_solutions_unreached': total_solutions_unreached,
        'reached_percentage': reached_solutions_percentage
    },
    'specification_level': {
        'total_specifications': total_specs,
        'total_reached_specs': total_reached_specs,
        'total_unreached_specs': total_unreached_specs,
        'reached_percentage': reached_specs_percentage,
        'reached_specifications': sorted(list(specs_with_reached)),
        'unreached_specifications': unreached_specs
    }
}

output_file = DATA_DIR / "target_count_results.json"
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"  Saved: {output_file}")

# Also save as text summary
summary_file = DATA_DIR / "target_count_summary.txt"
with open(summary_file, 'w') as f:
    f.write("="*80 + "\n")
    f.write("TARGET COUNT SUMMARY\n")
    f.write("="*80 + "\n\n")
    f.write("[Solution-Level Counts]\n")
    f.write(f"Total solutions: {total_solutions}\n")
    f.write(f"Solutions that reached target: {total_solutions_reached}\n")
    f.write(f"Solutions that did not reach target: {total_solutions_unreached}\n")
    f.write(f"Reached percentage: {reached_solutions_percentage:.1f}%\n")
    f.write("\n[Specification-Level Counts]\n")
    f.write(f"Total specifications: {total_specs}\n")
    f.write(f"Specifications with at least one reached solution: {total_reached_specs}\n")
    f.write(f"Specifications with no reached solutions: {total_unreached_specs}\n")
    f.write(f"Reached percentage: {reached_specs_percentage:.1f}%\n")
    f.write("="*80 + "\n\n")
    
    if unreached_specs:
        f.write(f"Unreached specifications ({len(unreached_specs)}):\n")
        f.write(f"  {unreached_specs}\n")
    else:
        f.write("All specifications have at least one reached solution!\n")

print(f"  Saved: {summary_file}")

print("\n" + "="*80)
print("COUNTING COMPLETE")
print("="*80)

