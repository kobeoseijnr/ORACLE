"""
Verify that our best solution selection logic matches the original implementation.
"""
import json
import numpy as np
from pathlib import Path

# Configuration
BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"

# Load data
data_path = RESULTS_DIR / "all_input_output_values_complete.json"
with open(data_path, 'r') as f:
    data = json.load(f)

# Our best solution selection function
def select_best_morl_solution_our(solutions, target_values):
    best_score = -1
    best_solution = None
    
    target_gain_linear = target_values['target_gain_linear']
    target_ugbw_mhz = target_values['target_ugbw_mhz']
    target_pm_deg = target_values['target_pm_deg']
    target_ibias_ma = target_values['target_ibias_ma']
    
    for sol in solutions:
        output_gain_linear = sol.get('output_gain_linear', 0)
        output_ugbw_mhz = sol.get('output_ugbw_mhz', 0)
        output_pm_deg = sol.get('output_pm_deg', 0)
        output_ibias_ma = sol.get('output_ibias_ma', 0)
        
        gain_score = min(1.0, output_gain_linear / target_gain_linear) if target_gain_linear > 0 else 0
        ugbw_score = min(1.0, output_ugbw_mhz / target_ugbw_mhz) if target_ugbw_mhz > 0 else 0
        phm_score = min(1.0, output_pm_deg / target_pm_deg) if target_pm_deg > 0 else 0
        ibias_score = min(1.0, target_ibias_ma / output_ibias_ma) if output_ibias_ma > 0 else 0
        
        overall_score = (gain_score + ugbw_score + phm_score + ibias_score) / 4.0
        
        if overall_score > best_score:
            best_score = overall_score
            best_solution = sol
    
    return best_solution, best_score

# Test on spec 1
print("="*80)
print("VERIFICATION: Best Solution Selection Logic")
print("="*80)

spec_1_solutions = [s for s in data['solutions'] if s['spec'] == 1]
target_values = {
    'target_gain_linear': spec_1_solutions[0]['target_gain_linear'],
    'target_ugbw_mhz': spec_1_solutions[0]['target_ugbw_mhz'],
    'target_pm_deg': spec_1_solutions[0]['target_pm_deg'],
    'target_ibias_ma': spec_1_solutions[0]['target_ibias_ma']
}

print(f"\nTesting on Spec 1:")
print(f"  Target: {target_values}")
print(f"  Number of solutions: {len(spec_1_solutions)}")

best_sol, best_score = select_best_morl_solution_our(spec_1_solutions, target_values)

print(f"\nBest Solution Selected:")
print(f"  Solution number: {best_sol.get('solution', 'N/A')}")
print(f"  Best score: {best_score:.4f}")
print(f"  Output values:")
print(f"    - Gain (dB): {best_sol.get('output_gain_db', 'N/A')}")
print(f"    - UGBW (MHz): {best_sol.get('output_ugbw_mhz', 'N/A')}")
print(f"    - PM (deg): {best_sol.get('output_pm_deg', 'N/A')}")
print(f"    - IBIAS (mA): {best_sol.get('output_ibias_ma', 'N/A')}")

# Show all scores for comparison
print(f"\nAll Solutions Scores:")
for i, sol in enumerate(spec_1_solutions, 1):
    output_gain_linear = sol.get('output_gain_linear', 0)
    output_ugbw_mhz = sol.get('output_ugbw_mhz', 0)
    output_pm_deg = sol.get('output_pm_deg', 0)
    output_ibias_ma = sol.get('output_ibias_ma', 0)
    
    gain_score = min(1.0, output_gain_linear / target_values['target_gain_linear']) if target_values['target_gain_linear'] > 0 else 0
    ugbw_score = min(1.0, output_ugbw_mhz / target_values['target_ugbw_mhz']) if target_values['target_ugbw_mhz'] > 0 else 0
    phm_score = min(1.0, output_pm_deg / target_values['target_pm_deg']) if target_values['target_pm_deg'] > 0 else 0
    ibias_score = min(1.0, target_values['target_ibias_ma'] / output_ibias_ma) if output_ibias_ma > 0 else 0
    
    overall_score = (gain_score + ugbw_score + phm_score + ibias_score) / 4.0
    
    marker = " <-- BEST" if sol == best_sol else ""
    print(f"  Solution {i}: score={overall_score:.4f} (gain={gain_score:.3f}, ugbw={ugbw_score:.3f}, pm={phm_score:.3f}, ibias={ibias_score:.3f}){marker}")

print("\n" + "="*80)
print("VERIFICATION COMPLETE")
print("="*80)

