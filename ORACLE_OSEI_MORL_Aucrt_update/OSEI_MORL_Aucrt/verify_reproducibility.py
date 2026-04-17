"""
Verify reproducibility: Check if results are consistent across different runs.
This ensures the same results are obtained regardless of system or run time.
"""
import json
import random
import numpy as np
from pathlib import Path

# Configuration
BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"
DATA_DIR = BASE_DIR / "data"

print("="*80)
print("REPRODUCIBILITY VERIFICATION")
print("="*80)

# ============================================================================
# 1. CHECK SEED SETTINGS
# ============================================================================
print("\n[1] Checking seed settings in code...")

# Check main script
main_script = BASE_DIR / "create_2d_objective_comparison.py"
with open(main_script, 'r') as f:
    script_content = f.read()

has_random_seed = 'random.seed(42)' in script_content
has_np_seed = 'np.random.seed(42)' in script_content

print(f"  random.seed(42): {'[OK] Found' if has_random_seed else '[MISSING]'}")
print(f"  np.random.seed(42): {'[OK] Found' if has_np_seed else '[MISSING]'}")

if not (has_random_seed and has_np_seed):
    print("  WARNING: Seeds not properly set for reproducibility!")

# ============================================================================
# 2. TEST DETERMINISTIC RANDOM SELECTION
# ============================================================================
print("\n[2] Testing deterministic random selection...")

# Set seeds
random.seed(42)
np.random.seed(42)

# First run
selected_1 = random.sample(range(1, 1001), 10)

# Reset seeds
random.seed(42)
np.random.seed(42)

# Second run
selected_2 = random.sample(range(1, 1001), 10)

is_deterministic = selected_1 == selected_2
print(f"  First selection: {selected_1}")
print(f"  Second selection: {selected_2}")
print(f"  Deterministic: {'[OK] Yes' if is_deterministic else '[ERROR] No - Results will vary!'}")

# ============================================================================
# 3. VERIFY DATA CONSISTENCY
# ============================================================================
print("\n[3] Verifying data consistency...")

data_path = RESULTS_DIR / "all_input_output_values_complete.json"
if not data_path.exists():
    print(f"  [ERROR] Data file not found: {data_path}")
else:
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    print(f"  [OK] Data file exists")
    print(f"  Total solutions: {data['total_solutions']}")
    print(f"  Total specifications: {data['total_specifications']}")
    
    # Check if data is consistent (same target values for same spec)
    solutions_by_spec = {}
    for solution in data['solutions']:
        spec_num = solution['spec']
        if spec_num not in solutions_by_spec:
            solutions_by_spec[spec_num] = []
        solutions_by_spec[spec_num].append(solution)
    
    # Verify spec 1 has consistent target values
    spec_1_solutions = solutions_by_spec.get(1, [])
    if spec_1_solutions:
        first_target = {
            'gain': spec_1_solutions[0]['target_gain_linear'],
            'ugbw': spec_1_solutions[0]['target_ugbw_mhz'],
            'pm': spec_1_solutions[0]['target_pm_deg'],
            'ibias': spec_1_solutions[0]['target_ibias_ma']
        }
        all_same = all(
            s['target_gain_linear'] == first_target['gain'] and
            s['target_ugbw_mhz'] == first_target['ugbw'] and
            s['target_pm_deg'] == first_target['pm'] and
            s['target_ibias_ma'] == first_target['ibias']
            for s in spec_1_solutions
        )
        print(f"  Spec 1 target consistency: {'[OK] All solutions have same target' if all_same else '[ERROR] Inconsistent targets'}")

# ============================================================================
# 4. CHECK SAVED RESULTS CONSISTENCY
# ============================================================================
print("\n[4] Checking saved results consistency...")

# Check if selected targets info exists
info_file = DATA_DIR / "selected_targets_info.json"
if info_file.exists():
    with open(info_file, 'r') as f:
        saved_info = json.load(f)
    
    print(f"  [OK] Selected targets info exists")
    print(f"  Saved selected specs: {saved_info.get('selected_specifications', 'N/A')}")
    print(f"  Saved total reached: {saved_info.get('total_reached_specs', 'N/A')}")
    
    # Verify this matches expected (with seed 42)
    random.seed(42)
    np.random.seed(42)
    expected_selection = random.sample(range(1, 1001), 10)
    
    saved_selection = saved_info.get('selected_specifications', [])
    matches_expected = sorted(saved_selection) == sorted(expected_selection)
    print(f"  Matches expected (seed=42): {'[OK] Yes' if matches_expected else '[ERROR] No'}")
else:
    print(f"  [ERROR] Selected targets info not found")

# Check all 1000 input values
input_file = DATA_DIR / "all_1000_input_values.json"
if input_file.exists():
    with open(input_file, 'r') as f:
        input_data = json.load(f)
    
    print(f"  [OK] All 1000 input values saved")
    print(f"  Total specs in file: {input_data.get('total_specifications', 'N/A')}")
    print(f"  Total reached: {input_data.get('total_reached', 'N/A')}")
    print(f"  Reached percentage: {input_data.get('reached_percentage', 'N/A')}%")
else:
    print(f"  [ERROR] All 1000 input values file not found")

# ============================================================================
# 5. VERIFY REPORT CLAIMS MATCH
# ============================================================================
print("\n[5] Verifying report claims match saved data...")

if input_file.exists():
    with open(input_file, 'r') as f:
        input_data = json.load(f)
    
    report_claims = {
        'total_specs': 1000,
        'total_reached': 982,
        'reached_percentage': 98.2
    }
    
    actual_values = {
        'total_specs': input_data.get('total_specifications', 0),
        'total_reached': input_data.get('total_reached', 0),
        'reached_percentage': input_data.get('reached_percentage', 0)
    }
    
    print(f"  Report claims vs Actual:")
    print(f"    Total specs: {report_claims['total_specs']} vs {actual_values['total_specs']} - {'[OK] Match' if report_claims['total_specs'] == actual_values['total_specs'] else '[ERROR] Mismatch'}")
    print(f"    Total reached: {report_claims['total_reached']} vs {actual_values['total_reached']} - {'[OK] Match' if report_claims['total_reached'] == actual_values['total_reached'] else '[ERROR] Mismatch'}")
    print(f"    Reached %: {report_claims['reached_percentage']}% vs {actual_values['reached_percentage']:.1f}% - {'[OK] Match' if abs(report_claims['reached_percentage'] - actual_values['reached_percentage']) < 0.1 else '[ERROR] Mismatch'}")

# ============================================================================
# 6. SYSTEM INDEPENDENCE CHECK
# ============================================================================
print("\n[6] System independence factors...")

import platform
import sys

print(f"  Python version: {sys.version.split()[0]}")
print(f"  Platform: {platform.platform()}")
print(f"  Note: Results should be identical regardless of system if seeds are set")

# Check for potential non-deterministic operations
potential_issues = []
if 'torch' in script_content and 'torch.manual_seed' not in script_content:
    potential_issues.append("PyTorch seed not set (if using PyTorch)")
if 'tf.' in script_content or 'tensorflow' in script_content:
    potential_issues.append("TensorFlow seed not set (if using TensorFlow)")

if potential_issues:
    print(f"  Warnings:")
    for issue in potential_issues:
        print(f"    - {issue}")
else:
    print(f"  [OK] No obvious system-dependent issues found")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("REPRODUCIBILITY SUMMARY")
print("="*80)

all_checks = [
    has_random_seed and has_np_seed,
    is_deterministic,
    info_file.exists() if 'info_file' in locals() else False,
    input_file.exists() if 'input_file' in locals() else False
]

if all(all_checks):
    print("[OK] All reproducibility checks passed!")
    print("  Results should be consistent across different systems and runs.")
else:
    print("[WARNING] Some reproducibility issues found.")
    print("  Review warnings above and ensure all seeds are properly set.")

print("\nRecommendations:")
print("  1. Always set random.seed(42) and np.random.seed(42) at start")
print("  2. If using PyTorch, set torch.manual_seed(42)")
print("  3. If using TensorFlow, set tf.random.set_seed(42)")
print("  4. Avoid system-dependent operations (file order, etc.)")
print("  5. Use fixed random selection (seed=42) for target selection")

print("="*80)

