"""
Count Total Targets Reached
Standalone script to count how many target specifications have at least one solution that reached the target.
Reads from CSV file and counts both "Target Reached" and "Paper_Target_Reached".
"""
import json
import pandas as pd
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

# Load data from CSV
csv_path = RESULTS_DIR / "evaluation_on_1000.csv"
if not csv_path.exists():
    print(f"[ERROR] CSV file not found: {csv_path}")
    sys.exit(1)

print(f"\n[1] Loading data from: {csv_path}")
df = pd.read_csv(csv_path)

print(f"[INFO] Loaded {len(df)} solutions from CSV file")

# Count targets reached
print("\n[2] Counting targets reached...")

# Count total solutions reached (Target Reached column)
total_solutions_reached = len(df[df['Target Reached'] == 'Yes'])
total_solutions_unreached = len(df[df['Target Reached'] == 'No'])

# Count Paper_Target_Reached
total_paper_reached = len(df[df['Paper_Target_Reached'] == True])
total_paper_unreached = len(df[df['Paper_Target_Reached'] == False])

# Count specs with reached solutions (Target Reached)
specs_with_reached = set(df[df['Target Reached'] == 'Yes']['Spec'].unique())
specs_with_unreached = set(df[df['Target Reached'] == 'No']['Spec'].unique())

# Count specs with Paper_Target_Reached
specs_with_paper_reached = set(df[df['Paper_Target_Reached'] == True]['Spec'].unique())
specs_with_paper_unreached = set(df[df['Paper_Target_Reached'] == False]['Spec'].unique())

# Specification-level counts (Target Reached)
total_reached_specs = len(specs_with_reached)
total_specs = df['Spec'].nunique()
total_unreached_specs = total_specs - total_reached_specs
reached_specs_percentage = (total_reached_specs / total_specs) * 100

# Specification-level counts (Paper_Target_Reached)
total_paper_reached_specs = len(specs_with_paper_reached)
total_paper_unreached_specs = total_specs - total_paper_reached_specs
paper_reached_specs_percentage = (total_paper_reached_specs / total_specs) * 100

# Solution-level counts
total_solutions = len(df)
reached_solutions_percentage = (total_solutions_reached / total_solutions) * 100
paper_reached_solutions_percentage = (total_paper_reached / total_solutions) * 100

# Display results
print("\n" + "="*80)
print("RESULTS")
print("="*80)

print("\n[Solution-Level Counts - Target Reached]")
print(f"Total solutions: {total_solutions}")
print(f"Solutions that reached target: {total_solutions_reached}")
print(f"Solutions that did not reach target: {total_solutions_unreached}")
print(f"Reached percentage: {reached_solutions_percentage:.1f}%")

print("\n[Specification-Level Counts - Target Reached]")
print(f"Total specifications: {total_specs}")
print(f"Specifications with at least one reached solution: {total_reached_specs}")
print(f"Specifications with no reached solutions: {total_unreached_specs}")
print(f"Reached percentage: {reached_specs_percentage:.1f}%")

print("\n" + "-"*80)
print("\n[Solution-Level Counts - Paper_Target_Reached]")
print(f"Total solutions: {total_solutions}")
print(f"Solutions that reached paper target: {total_paper_reached}")
print(f"Solutions that did not reach paper target: {total_paper_unreached}")
print(f"Paper reached percentage: {paper_reached_solutions_percentage:.1f}%")

print("\n[Specification-Level Counts - Paper_Target_Reached]")
print(f"Total specifications: {total_specs}")
print(f"Specifications with at least one paper reached solution: {total_paper_reached_specs}")
print(f"Specifications with no paper reached solutions: {total_paper_unreached_specs}")
print(f"Paper reached percentage: {paper_reached_specs_percentage:.1f}%")
print("="*80)

# Get list of unreached specs
all_specs = set(df['Spec'].unique())
unreached_specs = sorted(list(all_specs - specs_with_reached))
paper_unreached_specs = sorted(list(all_specs - specs_with_paper_reached))

if unreached_specs:
    print(f"\nUnreached specifications - Target Reached ({len(unreached_specs)}):")
    print(f"  {unreached_specs}")
else:
    print("\nAll specifications have at least one reached solution!")

if paper_unreached_specs:
    print(f"\nUnreached specifications - Paper_Target_Reached ({len(paper_unreached_specs)}):")
    print(f"  {paper_unreached_specs}")
else:
    print("\nAll specifications have at least one paper reached solution!")

# Save results
print("\n[3] Saving results...")

results = {
    'target_reached': {
        'solution_level': {
            'total_solutions': int(total_solutions),
            'total_solutions_reached': int(total_solutions_reached),
            'total_solutions_unreached': int(total_solutions_unreached),
            'reached_percentage': float(reached_solutions_percentage)
        },
        'specification_level': {
            'total_specifications': int(total_specs),
            'total_reached_specs': int(total_reached_specs),
            'total_unreached_specs': int(total_unreached_specs),
            'reached_percentage': float(reached_specs_percentage),
            'reached_specifications': [int(x) for x in sorted(list(specs_with_reached))],
            'unreached_specifications': [int(x) for x in unreached_specs]
        }
    },
    'paper_target_reached': {
        'solution_level': {
            'total_solutions': int(total_solutions),
            'total_solutions_reached': int(total_paper_reached),
            'total_solutions_unreached': int(total_paper_unreached),
            'reached_percentage': float(paper_reached_solutions_percentage)
        },
        'specification_level': {
            'total_specifications': int(total_specs),
            'total_reached_specs': int(total_paper_reached_specs),
            'total_unreached_specs': int(total_paper_unreached_specs),
            'reached_percentage': float(paper_reached_specs_percentage),
            'reached_specifications': [int(x) for x in sorted(list(specs_with_paper_reached))],
            'unreached_specifications': [int(x) for x in paper_unreached_specs]
        }
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
    
    f.write("[Target Reached]\n")
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
    if unreached_specs:
        f.write(f"\nUnreached specifications ({len(unreached_specs)}):\n")
        f.write(f"  {unreached_specs}\n")
    else:
        f.write("\nAll specifications have at least one reached solution!\n")
    
    f.write("\n" + "-"*80 + "\n\n")
    f.write("[Paper_Target_Reached]\n")
    f.write("[Solution-Level Counts]\n")
    f.write(f"Total solutions: {total_solutions}\n")
    f.write(f"Solutions that reached paper target: {total_paper_reached}\n")
    f.write(f"Solutions that did not reach paper target: {total_paper_unreached}\n")
    f.write(f"Paper reached percentage: {paper_reached_solutions_percentage:.1f}%\n")
    f.write("\n[Specification-Level Counts]\n")
    f.write(f"Total specifications: {total_specs}\n")
    f.write(f"Specifications with at least one paper reached solution: {total_paper_reached_specs}\n")
    f.write(f"Specifications with no paper reached solutions: {total_paper_unreached_specs}\n")
    f.write(f"Paper reached percentage: {paper_reached_specs_percentage:.1f}%\n")
    if paper_unreached_specs:
        f.write(f"\nUnreached specifications ({len(paper_unreached_specs)}):\n")
        f.write(f"  {paper_unreached_specs}\n")
    else:
        f.write("\nAll specifications have at least one paper reached solution!\n")
    
    f.write("="*80 + "\n")

print(f"  Saved: {summary_file}")

print("\n" + "="*80)
print("COUNTING COMPLETE")
print("="*80)

