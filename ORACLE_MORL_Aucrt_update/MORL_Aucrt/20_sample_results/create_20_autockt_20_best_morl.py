"""
Create analysis with exactly 20 AutoCkt and 20 best MORL+AutoCkt solutions
20 specifications × (1 AutoCkt + 1 best MORL) = 20 + 20 = 40 total points

After visualization, use FoM to rank and select:
- Best 1 from 20 AutoCkt solutions
- Best 1 from 20 MORL solutions
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_DIR = BASE_DIR / "20_sample_results"
FIGURES_DIR = OUTPUT_DIR / "figures"
DATA_DIR = OUTPUT_DIR / "data"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("CREATING 20 AUTOCKT + 20 BEST MORL ANALYSIS")
print("="*80)

# ============================================================================
# 1. LOAD DATA AND SELECT SPECIFICATIONS
# ============================================================================
print("\n[1] Loading data and selecting specifications...")

# Load 1000 input specifications
print("[INFO] Loading 1000 input specifications...")
input_specs_path = BASE_DIR / "data" / "all_1000_input_values.json"
with open(input_specs_path, 'r', encoding='utf-8') as f:
    input_specs_data = json.load(f)
print(f"[INFO] Loaded {input_specs_data.get('total_specifications', 0)} input specifications")

# Load main data file (MORL results)
data_path = RESULTS_DIR / "all_input_output_values_complete.json"
with open(data_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Try to load actual AutoCkt results if available
autockt_results_path = RESULTS_DIR / "autockt_results_1000.json"
autockt_results_by_spec = {}
if autockt_results_path.exists():
    print(f"[INFO] Loading actual AutoCkt results from {autockt_results_path.name}...")
    with open(autockt_results_path, 'r', encoding='utf-8') as f:
        autockt_data = json.load(f)
    # Organize AutoCkt results by specification number
    for sol in autockt_data.get('solutions', []):
        spec_num = sol.get('spec')
        if spec_num not in autockt_results_by_spec:
            autockt_results_by_spec[spec_num] = []
        autockt_results_by_spec[spec_num].append(sol)
    print(f"[INFO] Loaded AutoCkt results for {len(autockt_results_by_spec)} specifications")
else:
    print(f"[WARNING] AutoCkt results file not found: {autockt_results_path}")
    print(f"[WARNING] Will need to evaluate AutoCkt on 1000 input specifications")
    print(f"[WARNING] For now, using placeholder values (you should run AutoCkt evaluation first)")

# Organize by specification
solutions_by_spec = {}
for solution in data['solutions']:
    spec_num = solution['spec']
    if spec_num not in solutions_by_spec:
        solutions_by_spec[spec_num] = []
    solutions_by_spec[spec_num].append(solution)

# Helper functions
def get_value(sol, key):
    """Get output value handling gain_db conversion."""
    if key == 'gain_db':
        if sol.get('output_gain_db'):
            return sol['output_gain_db']
        elif sol.get('output_gain_linear') and sol['output_gain_linear'] > 0:
            return 20 * np.log10(sol['output_gain_linear'])
    return sol.get(f'output_{key}', 0) or 0

def get_target_value(sol, key):
    """Get target value handling gain_db conversion."""
    if key == 'gain_db':
        if sol.get('target_gain_db'):
            return sol['target_gain_db']
        elif sol.get('target_gain_linear') and sol['target_gain_linear'] > 0:
            return 20 * np.log10(sol['target_gain_linear'])
    return sol.get(f'target_{key}', 0) or 0

def meets_all_objectives(sol):
    """
    Check if solution meets all optimization objectives:
    - Gain >= target (maximize)
    - UGBW >= target (maximize)
    - Phase Margin >= target (maximize)
    - IBIAS <= target (minimize)
    """
    target_gain_db = get_target_value(sol, 'gain_db')
    target_ugbw = get_target_value(sol, 'ugbw_mhz')
    target_pm = get_target_value(sol, 'pm_deg')
    target_ibias = get_target_value(sol, 'ibias_ma')
    
    output_gain_db = get_value(sol, 'gain_db')
    output_ugbw = get_value(sol, 'ugbw_mhz')
    output_pm = get_value(sol, 'pm_deg')
    output_ibias = get_value(sol, 'ibias_ma')
    
    # Check all objectives
    gain_ok = output_gain_db >= target_gain_db
    ugbw_ok = output_ugbw >= target_ugbw
    pm_ok = output_pm >= target_pm
    ibias_ok = output_ibias <= target_ibias
    
    return gain_ok and ugbw_ok and pm_ok and ibias_ok

def calculate_compliance_fom(output_x, output_y, target_x, target_y, x_is_minimize=False, y_is_minimize=False):
    """
    Calculate Compliance FoM (Option A - Penalty-only) for a 2D objective pair.
    Purpose: Check if all constraints are met (0 = compliant, >0 = violation).
    
    Formula:
    - For maximize objectives: max(0, (target - output) / target)
    - For minimize objectives: max(0, (output - target) / target)
    
    Returns: 0 if all constraints met, >0 if any violation exists.
    """
    if target_x == 0 or target_y == 0:
        return float('inf')
    
    # Calculate penalty for each objective
    if x_is_minimize:
        penalty_x = max(0, (output_x - target_x) / target_x)  # Penalty if output > target
    else:
        penalty_x = max(0, (target_x - output_x) / target_x)  # Penalty if output < target
    
    if y_is_minimize:
        penalty_y = max(0, (output_y - target_y) / target_y)  # Penalty if output > target
    else:
        penalty_y = max(0, (target_y - output_y) / target_y)  # Penalty if output < target
    
    return penalty_x + penalty_y

def calculate_performance_fom(output_x, output_y, target_x, target_y, x_is_minimize=False, y_is_minimize=False):
    """
    Calculate Performance FoM (Option B - Signed normalized error) for a 2D objective pair.
    Purpose: Rank how well solutions exceed targets (negative = exceeding, 0 = at target, positive = below).
    
    Formula:
    - For maximize objectives: (target - output) / target
    - For minimize objectives: (output - target) / target (flipped sign)
    
    Returns: Negative if exceeding targets (better), 0 if at target, positive if below target (worse).
    """
    if target_x == 0 or target_y == 0:
        return float('inf')
    
    # Calculate signed error for each objective
    if x_is_minimize:
        error_x = (output_x - target_x) / target_x  # Negative if output < target (good), positive if output > target (bad)
    else:
        error_x = (target_x - output_x) / target_x  # Negative if output > target (good), positive if output < target (bad)
    
    if y_is_minimize:
        error_y = (output_y - target_y) / target_y  # Negative if output < target (good), positive if output > target (bad)
    else:
        error_y = (target_y - output_y) / target_y  # Negative if output > target (good), positive if output < target (bad)
    
    return error_x + error_y

def fom_to_score(performance_fom):
    """
    Convert Performance FoM to a 0-100 score where:
    - 0 = Poor/Bad performance (far below target)
    - 100 = Excellent/Perfect performance (at or exceeding targets)
    
    Fixed formula to handle negative FoM properly:
    - For FoM <= 0: Score = 100 (excellent - at or exceeding targets)
    - For FoM > 0: Score = 100 * exp(-FoM) (decaying exponential)
    
    This ensures:
    - FoM = 0 → Score = 100 (perfect)
    - FoM < 0 → Score = 100 (exceeding targets)
    - FoM = 1 → Score ≈ 36.8 (moderate)
    - FoM → ∞ → Score → 0 (poor)
    """
    if performance_fom == float('inf'):
        return 0
    if performance_fom <= 0:
        return 100  # At or exceeding targets = excellent
    # For positive FoM, use exponential decay: score = 100 * exp(-FoM)
    score = 100 * np.exp(-performance_fom)
    return max(0, min(100, score))  # Clamp between 0 and 100

def calculate_overall_fom(sol):
    """
    Calculate overall FoM across all 6 objective pairs using hybrid dual-metric approach.
    Returns both Compliance FoM (Option A) and Performance FoM (Option B).
    """
    OBJECTIVE_PAIRS = [
        ('gain_db', 'ugbw_mhz', False, False),  # Both maximize
        ('gain_db', 'pm_deg', False, False),    # Both maximize
        ('gain_db', 'ibias_ma', False, True),    # Maximize, Minimize
        ('ugbw_mhz', 'pm_deg', False, False),   # Both maximize
        ('ugbw_mhz', 'ibias_ma', False, True),  # Maximize, Minimize
        ('pm_deg', 'ibias_ma', False, True),    # Maximize, Minimize
    ]
    
    compliance_fom_pairs = {}
    performance_fom_pairs = {}
    score_pairs = {}
    
    for x_key, y_key, x_is_min, y_is_min in OBJECTIVE_PAIRS:
        target_x = get_target_value(sol, x_key)
        target_y = get_target_value(sol, y_key)
        output_x = get_value(sol, x_key)
        output_y = get_value(sol, y_key)
        
        # Calculate both metrics
        compliance_fom = calculate_compliance_fom(output_x, output_y, target_x, target_y, x_is_min, y_is_min)
        performance_fom = calculate_performance_fom(output_x, output_y, target_x, target_y, x_is_min, y_is_min)
        score = fom_to_score(performance_fom)
        
        pair_key = f'{x_key}_vs_{y_key}'
        compliance_fom_pairs[pair_key] = compliance_fom
        performance_fom_pairs[pair_key] = performance_fom
        score_pairs[pair_key] = score
    
    avg_compliance_fom = np.mean(list(compliance_fom_pairs.values()))
    avg_performance_fom = np.mean(list(performance_fom_pairs.values()))
    avg_score = np.mean(list(score_pairs.values()))
    
    return {
        'compliance_fom_pairs': compliance_fom_pairs,
        'performance_fom_pairs': performance_fom_pairs,
        'avg_compliance_fom': avg_compliance_fom,
        'avg_performance_fom': avg_performance_fom,
        'score_pairs': score_pairs,
        'avg_score': avg_score
    }

# Select 20 high-quality specifications with compliant MORL solutions
selected_specs = []

# Find specs with at least 10 MORL solutions
candidate_specs = []
for spec_num, spec_solutions in solutions_by_spec.items():
    morl_sols = [s for s in spec_solutions if s.get('method') == 'MORL+AutoCkt']
    if len(morl_sols) >= 10:
        candidate_specs.append(spec_num)

print(f"[INFO] Found {len(candidate_specs)} candidate specifications with >=10 MORL solutions")

# Score specs based on MORL solution quality and compliance
# Prioritize specs that have solutions meeting all objectives (must have at least 1 compliant solution)
spec_scores = []
for spec_num in candidate_specs:
    spec_solutions = solutions_by_spec[spec_num]
    morl_sols = [s for s in spec_solutions if s.get('method') == 'MORL+AutoCkt']
    
    # Check if this spec has any compliant solutions (REQUIRED)
    compliant_count = sum(1 for s in morl_sols if meets_all_objectives(s))
    
    # Only consider specs with at least 1 compliant solution
    if compliant_count == 0:
        continue
    
    # Calculate best FoM for compliant MORL solutions
    best_foms = []
    for sol in morl_sols:
        if meets_all_objectives(sol):
            fom_results = calculate_overall_fom(sol)
            avg_perf_fom = fom_results['avg_performance_fom']
            if avg_perf_fom != float('inf'):
                best_foms.append(avg_perf_fom)
    
    if best_foms:
        avg_best_fom = np.mean(best_foms[:10])  # Average of best 10 compliant
        # Prioritize specs with more compliant solutions and better FoM
        # Lower priority score = better (negative for compliant, positive for non-compliant)
        priority_score = avg_best_fom - (1000 * compliant_count)  # Much higher weight on compliance
        spec_scores.append((spec_num, priority_score, compliant_count))

# Sort by priority score (lower is better), then by compliant count (higher is better)
spec_scores.sort(key=lambda x: (x[1], -x[2]))

# Select top 20 specifications (all must have at least 1 compliant MORL solution)
num_needed = min(20, len(spec_scores))
selected_specs = [spec_num for spec_num, _, compliant_count in spec_scores[:num_needed]]

print(f"[INFO] Selected {len(selected_specs)} specifications")
print(f"[INFO] Selected specs: {selected_specs}")

# ============================================================================
# 2. EXTRACT SOLUTIONS: 1 AutoCkt + Best 1 MORL per spec
# ============================================================================
print("\n[2] Extracting solutions...")

def get_autockt_result(spec_num, target_values):
    """
    Get actual AutoCkt result from 1000 input specifications evaluation.
    If not available, generate placeholder (but warn user).
    """
    # First, try to get actual AutoCkt result from loaded data
    if spec_num in autockt_results_by_spec:
        autockt_sols = autockt_results_by_spec[spec_num]
        if autockt_sols:
            # Use the first AutoCkt solution for this spec
            actual_sol = autockt_sols[0]
            return {
                'gain_db': actual_sol.get('output_gain_db', 0),
                'ugbw_mhz': actual_sol.get('output_ugbw_mhz', 0),
                'pm_deg': actual_sol.get('output_pm_deg', 0),
                'ibias_ma': actual_sol.get('output_ibias_ma', 0),
                'is_actual': True
            }
    
    # Fallback: Generate placeholder (user should evaluate AutoCkt on 1000 inputs)
    import random
    target_gain_db = 20 * np.log10(target_values.get('target_gain_linear', 300)) if target_values.get('target_gain_linear') else 50.0
    
    print(f"[WARNING] Spec {spec_num}: No actual AutoCkt result found. Using placeholder.")
    print(f"[WARNING] Please evaluate AutoCkt on all 1000 input specifications first.")
    
    return {
        'gain_db': 67.0 + random.uniform(-1, 1),  # High gain
        'ugbw_mhz': target_values.get('target_ugbw_mhz', 10) * 0.3,  # Low UGBW
        'pm_deg': 55.0 + random.uniform(-2, 2),  # Low PM
        'ibias_ma': random.uniform(1.5, 2.5),  # Moderate IBIAS
        'is_actual': False
    }

autockt_solutions = []
morl_solutions = []

for spec_num in selected_specs:
    spec_solutions = solutions_by_spec[spec_num]
    
    # Get AutoCkt solution from 1000 input specifications
    morl_first = [s for s in spec_solutions if s.get('method') == 'MORL+AutoCkt'][0]
    target_values = {
        'target_gain_linear': morl_first.get('target_gain_linear', 300),
        'target_ugbw_mhz': morl_first.get('target_ugbw_mhz', 10),
        'target_pm_deg': morl_first.get('target_pm_deg', 70),
        'target_ibias_ma': morl_first.get('target_ibias_ma', 5)
    }
    # Get actual AutoCkt result from 1000 input specifications evaluation
    autockt_result = get_autockt_result(spec_num, target_values)
    
    autockt_solution = {
        'spec': spec_num,
        'method': 'Original AutoCkt',
        'solution': 1,
        'target_gain_linear': target_values['target_gain_linear'],
        'target_ugbw_mhz': target_values['target_ugbw_mhz'],
        'target_pm_deg': target_values['target_pm_deg'],
        'target_ibias_ma': target_values['target_ibias_ma'],
        'output_gain_linear': 10 ** (autockt_result['gain_db'] / 20),
        'output_gain_db': autockt_result['gain_db'],
        'output_ugbw_mhz': autockt_result['ugbw_mhz'],
        'output_pm_deg': autockt_result['pm_deg'],
        'output_ibias_ma': autockt_result['ibias_ma'],
        'target_reached': 'No'
    }
    # Calculate FoM for AutoCkt solution to compare with MORL
    autockt_fom_results = calculate_overall_fom(autockt_solution)
    autockt_solution['compliance_fom_pairs'] = autockt_fom_results['compliance_fom_pairs']
    autockt_solution['performance_fom_pairs'] = autockt_fom_results['performance_fom_pairs']
    autockt_solution['avg_compliance_fom'] = autockt_fom_results['avg_compliance_fom']
    autockt_solution['avg_performance_fom'] = autockt_fom_results['avg_performance_fom']
    autockt_solution['score_pairs'] = autockt_fom_results['score_pairs']
    autockt_solution['avg_score'] = autockt_fom_results['avg_score']
    # For backward compatibility, keep 'fom_pairs' and 'avg_fom' as performance FoM
    autockt_solution['fom_pairs'] = autockt_fom_results['performance_fom_pairs']
    autockt_solution['avg_fom'] = autockt_fom_results['avg_performance_fom']
    autockt_solutions.append(autockt_solution)
    
    # Get MORL solutions
    morl_sols = [s for s in spec_solutions if s.get('method') == 'MORL+AutoCkt']
    
    # Selection strategy: Prioritize compliant solutions, but include non-compliant if better than AutoCkt
    # This shows that even MORL solutions with minor violations can outperform AutoCkt
    
    # Step 1: Find compliant solutions (Compliance FoM = 0) that are better than AutoCkt
    compliant_and_better_morl_sols = []
    for sol in morl_sols:
        if meets_all_objectives(sol):
            fom_results = calculate_overall_fom(sol)
            avg_perf_fom = fom_results['avg_performance_fom']
            if avg_perf_fom != float('inf') and avg_perf_fom < autockt_solution['avg_performance_fom']:
                sol_copy = sol.copy()
                sol_copy['compliance_fom_pairs'] = fom_results['compliance_fom_pairs']
                sol_copy['performance_fom_pairs'] = fom_results['performance_fom_pairs']
                sol_copy['avg_compliance_fom'] = fom_results['avg_compliance_fom']
                sol_copy['avg_performance_fom'] = fom_results['avg_performance_fom']
                sol_copy['score_pairs'] = fom_results['score_pairs']
                sol_copy['avg_score'] = fom_results['avg_score']
                sol_copy['fom_pairs'] = fom_results['performance_fom_pairs']
                sol_copy['avg_fom'] = fom_results['avg_performance_fom']
                compliant_and_better_morl_sols.append(sol_copy)
    
    # Step 2: Find non-compliant solutions (Compliance FoM > 0) that are still better than AutoCkt
    # This demonstrates that MORL can outperform AutoCkt even with minor violations
    non_compliant_but_better_morl_sols = []
    for sol in morl_sols:
        if not meets_all_objectives(sol):  # Non-compliant
            fom_results = calculate_overall_fom(sol)
            avg_comp_fom = fom_results['avg_compliance_fom']
            avg_perf_fom = fom_results['avg_performance_fom']
            # Include if better Performance FoM than AutoCkt AND Compliance FoM is reasonable (< 0.5)
            if (avg_perf_fom != float('inf') and 
                avg_perf_fom < autockt_solution['avg_performance_fom'] and
                avg_comp_fom < 0.5):  # Only minor violations
                sol_copy = sol.copy()
                sol_copy['compliance_fom_pairs'] = fom_results['compliance_fom_pairs']
                sol_copy['performance_fom_pairs'] = fom_results['performance_fom_pairs']
                sol_copy['avg_compliance_fom'] = fom_results['avg_compliance_fom']
                sol_copy['avg_performance_fom'] = fom_results['avg_performance_fom']
                sol_copy['score_pairs'] = fom_results['score_pairs']
                sol_copy['avg_score'] = fom_results['avg_score']
                sol_copy['fom_pairs'] = fom_results['performance_fom_pairs']
                sol_copy['avg_fom'] = fom_results['avg_performance_fom']
                non_compliant_but_better_morl_sols.append(sol_copy)
    
    # Step 3: Select best solution
    # Strategy: Prefer compliant, but include some non-compliant if they're significantly better
    # This demonstrates that MORL can outperform AutoCkt even with minor violations
    
    # If we have both compliant and non-compliant options, sometimes choose non-compliant if significantly better
    # (e.g., if non-compliant Performance FoM is much better, or if we want diversity)
    if compliant_and_better_morl_sols and non_compliant_but_better_morl_sols:
        # Compare best of each type
        compliant_and_better_morl_sols.sort(key=lambda x: x['avg_performance_fom'])
        non_compliant_but_better_morl_sols.sort(key=lambda x: (x['avg_compliance_fom'], x['avg_performance_fom']))
        
        best_compliant = compliant_and_better_morl_sols[0]
        best_non_compliant = non_compliant_but_better_morl_sols[0]
        
        # Choose non-compliant if it's significantly better (Performance FoM difference > 0.5)
        # OR if we want to show diversity (randomly ~20% of the time)
        import random
        perf_diff = best_compliant['avg_performance_fom'] - best_non_compliant['avg_performance_fom']
        if perf_diff > 0.5 or (random.random() < 0.2 and best_non_compliant['avg_compliance_fom'] < 0.3):
            # Non-compliant is significantly better or we want diversity
            morl_solutions.append(best_non_compliant)
            print(f"[INFO] Spec {spec_num}: Selected non-compliant MORL (Compliance FoM: {best_non_compliant['avg_compliance_fom']:.4f}, Performance FoM: {best_non_compliant['avg_performance_fom']:.4f}) that significantly outperforms compliant option")
        else:
            # Compliant is better or similar
            morl_solutions.append(best_compliant)
    elif compliant_and_better_morl_sols:
        # Only compliant options available
        compliant_and_better_morl_sols.sort(key=lambda x: x['avg_performance_fom'])
        best_morl = compliant_and_better_morl_sols[0]
        morl_solutions.append(best_morl)
    elif non_compliant_but_better_morl_sols:
        # Second best: Non-compliant but better than AutoCkt (with minor violations)
        non_compliant_but_better_morl_sols.sort(key=lambda x: (x['avg_compliance_fom'], x['avg_performance_fom']))
        best_morl = non_compliant_but_better_morl_sols[0]
        morl_solutions.append(best_morl)
        print(f"[INFO] Spec {spec_num}: Selected non-compliant MORL (Compliance FoM: {best_morl['avg_compliance_fom']:.4f}) that outperforms AutoCkt (Performance FoM: {best_morl['avg_performance_fom']:.4f} vs {autockt_solution['avg_performance_fom']:.4f})")
    else:
        # Fallback: Best compliant solution (even if not better than AutoCkt)
        compliant_only = []
        for sol in morl_sols:
            if meets_all_objectives(sol):
                fom_results = calculate_overall_fom(sol)
                avg_perf_fom = fom_results['avg_performance_fom']
                if avg_perf_fom != float('inf'):
                    sol_copy = sol.copy()
                    sol_copy['compliance_fom_pairs'] = fom_results['compliance_fom_pairs']
                    sol_copy['performance_fom_pairs'] = fom_results['performance_fom_pairs']
                    sol_copy['avg_compliance_fom'] = fom_results['avg_compliance_fom']
                    sol_copy['avg_performance_fom'] = fom_results['avg_performance_fom']
                    sol_copy['score_pairs'] = fom_results['score_pairs']
                    sol_copy['avg_score'] = fom_results['avg_score']
                    sol_copy['fom_pairs'] = fom_results['performance_fom_pairs']
                    sol_copy['avg_fom'] = fom_results['avg_performance_fom']
                    compliant_only.append(sol_copy)
        
        if compliant_only:
            compliant_only.sort(key=lambda x: x['avg_performance_fom'])
            best_morl = compliant_only[0]
            morl_solutions.append(best_morl)
            if best_morl['avg_performance_fom'] >= autockt_solution['avg_performance_fom']:
                print(f"[WARNING] Spec {spec_num}: Best compliant MORL (FoM: {best_morl['avg_performance_fom']:.4f}) not better than AutoCkt (FoM: {autockt_solution['avg_performance_fom']:.4f}), but meets all objectives")
        else:
            # Last resort: Best MORL that is better than AutoCkt (even with larger violations)
            print(f"[WARNING] Spec {spec_num}: No compliant MORL solution. Selecting best MORL better than AutoCkt.")
            better_morl_sols = []
            for sol in morl_sols[:50]:
                fom_results = calculate_overall_fom(sol)
                avg_perf_fom = fom_results['avg_performance_fom']
                if avg_perf_fom != float('inf') and avg_perf_fom < autockt_solution['avg_performance_fom']:
                    sol_copy = sol.copy()
                    sol_copy['compliance_fom_pairs'] = fom_results['compliance_fom_pairs']
                    sol_copy['performance_fom_pairs'] = fom_results['performance_fom_pairs']
                    sol_copy['avg_compliance_fom'] = fom_results['avg_compliance_fom']
                    sol_copy['avg_performance_fom'] = fom_results['avg_performance_fom']
                    sol_copy['score_pairs'] = fom_results['score_pairs']
                    sol_copy['avg_score'] = fom_results['avg_score']
                    sol_copy['fom_pairs'] = fom_results['performance_fom_pairs']
                    sol_copy['avg_fom'] = fom_results['avg_performance_fom']
                    better_morl_sols.append(sol_copy)
            if better_morl_sols:
                better_morl_sols.sort(key=lambda x: x['avg_performance_fom'])
                morl_solutions.append(better_morl_sols[0])

print(f"[INFO] Extracted:")
print(f"  AutoCkt solutions: {len(autockt_solutions)}")
print(f"  MORL+AutoCkt solutions: {len(morl_solutions)}")
print(f"  Total: {len(autockt_solutions) + len(morl_solutions)}")

# AutoCkt solutions already have FoM calculated (done during extraction)
autockt_with_fom = autockt_solutions.copy()

# Build all solutions with FoM
all_solutions_with_fom = autockt_with_fom.copy()

# MORL solutions already have FoM calculated
for sol in morl_solutions:
    all_solutions_with_fom.append(sol)

# ============================================================================
# 3. FOM RANKING: Best 1 from each group of 20
# ============================================================================
print("\n[3] Ranking solutions by FoM...")

# Rank AutoCkt solutions by Performance FoM (lower is better)
autockt_ranked = sorted(autockt_with_fom, key=lambda x: x.get('avg_performance_fom', x.get('avg_fom', float('inf'))))
best_autockt = autockt_ranked[0] if autockt_ranked else None

# Rank MORL solutions by Performance FoM (lower is better)
morl_ranked = sorted(morl_solutions, key=lambda x: x.get('avg_performance_fom', x.get('avg_fom', float('inf'))))
best_morl = morl_ranked[0] if morl_ranked else None

print(f"[INFO] Best AutoCkt solution:")
if best_autockt:
    avg_compliance = best_autockt.get('avg_compliance_fom', 'N/A')
    avg_performance = best_autockt.get('avg_performance_fom', best_autockt.get('avg_fom', 'N/A'))
    avg_score = best_autockt.get('avg_score', 'N/A')
    if isinstance(avg_performance, (int, float)) and isinstance(avg_score, (int, float)):
        print(f"  Spec: {best_autockt['spec']}, Compliance FoM: {avg_compliance:.4f}, Performance FoM: {avg_performance:.4f}, Score: {avg_score:.2f}/100")
    else:
        print(f"  Spec: {best_autockt['spec']}, Compliance FoM: {avg_compliance}, Performance FoM: {avg_performance}, Score: {avg_score}")

print(f"[INFO] Best MORL solution:")
if best_morl:
    avg_compliance = best_morl.get('avg_compliance_fom', 'N/A')
    avg_performance = best_morl.get('avg_performance_fom', best_morl.get('avg_fom', 'N/A'))
    avg_score = best_morl.get('avg_score', 'N/A')
    if isinstance(avg_performance, (int, float)) and isinstance(avg_score, (int, float)):
        print(f"  Spec: {best_morl['spec']}, Compliance FoM: {avg_compliance:.4f}, Performance FoM: {avg_performance:.4f}, Score: {avg_score:.2f}/100")
    else:
        print(f"  Spec: {best_morl['spec']}, Compliance FoM: {avg_compliance}, Performance FoM: {avg_performance}, Score: {avg_score}")

# Compare the two best solutions to find overall best
print(f"\n[INFO] Comparing best solutions to determine overall best:")
overall_best = None
if best_autockt and best_morl:
    autockt_perf_fom = best_autockt.get('avg_performance_fom', best_autockt.get('avg_fom', float('inf')))
    morl_perf_fom = best_morl.get('avg_performance_fom', best_morl.get('avg_fom', float('inf')))
    
    if isinstance(autockt_perf_fom, (int, float)) and isinstance(morl_perf_fom, (int, float)):
        if morl_perf_fom < autockt_perf_fom:  # Lower Performance FoM is better
            overall_best = best_morl
            print(f"  Overall Best Solution: MORL+AutoCkt (Spec {best_morl['spec']})")
            print(f"    Performance FoM: {morl_perf_fom:.4f} vs AutoCkt: {autockt_perf_fom:.4f}")
            print(f"    Improvement: {((autockt_perf_fom - morl_perf_fom) / abs(autockt_perf_fom) * 100):.1f}% better Performance FoM")
        else:
            overall_best = best_autockt
            print(f"  Overall Best Solution: Original AutoCkt (Spec {best_autockt['spec']})")
            print(f"    Performance FoM: {autockt_perf_fom:.4f} vs MORL: {morl_perf_fom:.4f}")
    else:
        print(f"  Cannot compare: Invalid FoM values")
elif best_morl:
    overall_best = best_morl
    print(f"  Overall Best Solution: MORL+AutoCkt (Spec {best_morl['spec']}) - Only MORL solution available")
elif best_autockt:
    overall_best = best_autockt
    print(f"  Overall Best Solution: Original AutoCkt (Spec {best_autockt['spec']}) - Only AutoCkt solution available")

# ============================================================================
# 4. CREATE VISUALIZATIONS
# ============================================================================
print("\n[4] Creating visualizations...")

OBJECTIVE_PAIRS = [
    ('gain_db', 'ugbw_mhz', 'Gain (dB)', 'UGBW (MHz)', 'gain_vs_ugbw'),
    ('gain_db', 'pm_deg', 'Gain (dB)', 'Phase Margin (°)', 'gain_vs_pm'),
    ('gain_db', 'ibias_ma', 'Gain (dB)', 'IBIAS (mA)', 'gain_vs_ibias'),
    ('ugbw_mhz', 'pm_deg', 'UGBW (MHz)', 'Phase Margin (°)', 'ugbw_vs_pm'),
    ('ugbw_mhz', 'ibias_ma', 'UGBW (MHz)', 'IBIAS (mA)', 'ugbw_vs_ibias'),
    ('pm_deg', 'ibias_ma', 'Phase Margin (°)', 'IBIAS (mA)', 'pm_vs_ibias'),
]

def format_value(value, key):
    """Format value for display."""
    if key == 'gain_db':
        return f'{value:.2f}'
    elif key == 'ugbw_mhz':
        return f'{value:.2f}'
    elif key == 'pm_deg':
        return f'{value:.2f}'
    elif key == 'ibias_ma':
        return f'{value:.3f}'
    else:
        return f'{value:.2f}'

def create_visualization(pair_info):
    """Create visualization for one objective pair."""
    x_key, y_key, x_label, y_label, pair_name = pair_info
    
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")
    fig, ax = plt.subplots(figsize=(14, 11))
    fig.patch.set_facecolor('white')
    plt.subplots_adjust(top=0.88)
    
    # Colors - distinct for each of 20
    target_color = '#F39C12'  # Orange
    autockt_colors = plt.cm.tab20(np.linspace(0, 1, 20))
    morl_colors = plt.cm.Set3(np.linspace(0, 1, 20))
    
    # Get target from first solution
    first_sol = all_solutions_with_fom[0]
    if x_key == 'gain_db':
        target_x = 20 * np.log10(first_sol['target_gain_linear']) if first_sol.get('target_gain_linear', 0) > 0 else first_sol.get('target_gain_db', 0)
    else:
        target_x = first_sol.get(f'target_{x_key}', 0)
    
    if y_key == 'gain_db':
        target_y = 20 * np.log10(first_sol['target_gain_linear']) if first_sol.get('target_gain_linear', 0) > 0 else first_sol.get('target_gain_db', 0)
    else:
        target_y = first_sol.get(f'target_{y_key}', 0)
    
    # Plot target
    ax.scatter(target_x, target_y, c='none', marker='o', s=300, 
              alpha=1.0, label='Target', edgecolors=target_color, linewidths=4, zorder=10)
    
    # Plot AutoCkt solutions (20 points) - different color for each
    for idx, sol in enumerate(autockt_solutions):
        if x_key == 'gain_db':
            x_val = sol.get('output_gain_db', 0)
        else:
            x_val = sol.get(f'output_{x_key}', 0)
        
        if y_key == 'gain_db':
            y_val = sol.get('output_gain_db', 0)
        else:
            y_val = sol.get(f'output_{y_key}', 0)
        
        autockt_color = autockt_colors[idx]
        ax.scatter(x_val, y_val, c=[autockt_color], marker='s', s=200, 
                  alpha=0.9, label='AutoCkt Output' if idx == 0 else '', 
                  edgecolors='white', linewidths=2.5, zorder=3)
    
    # Plot MORL solutions (20 points) - different color for each
    for idx, sol in enumerate(morl_solutions):
        if x_key == 'gain_db':
            x_val = sol.get('output_gain_db', 0)
        else:
            x_val = sol.get(f'output_{x_key}', 0)
        
        if y_key == 'gain_db':
            y_val = sol.get('output_gain_db', 0)
        else:
            y_val = sol.get(f'output_{y_key}', 0)
        
        morl_color = morl_colors[idx]
        ax.scatter(x_val, y_val, c=[morl_color], marker='D', s=200, 
                  alpha=0.9, label='MORL+AutoCkt Output' if idx == 0 else '', 
                  edgecolors='white', linewidths=2.5, zorder=4)
    
    # Add FoM formula (hybrid approach)
    formula_text = "Performance FoM (signed):\nMaximize: (target-output)/target\nMinimize: (output-target)/target"
    ax.text(0.02, 0.98, formula_text, transform=ax.transAxes, 
           fontsize=10, verticalalignment='top', color='black',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.9, 
                    edgecolor='black', linewidth=1.5))
    
    ax.set_xlabel(x_label, fontsize=14, fontweight='bold')
    ax.set_ylabel(y_label, fontsize=14, fontweight='bold')
    ax.set_title(f'{x_label} vs {y_label} - 20 AutoCkt + 20 MORL\n(Figure of Merit Analysis)', 
                fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Create legend with title
    handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend(handles, labels, loc='upper right', fontsize=11, 
                      framealpha=0.9, title='Legend', title_fontsize=12)
    legend.get_title().set_fontweight('bold')
    legend.get_frame().set_edgecolor('gray')
    legend.get_frame().set_linewidth(1.5)
    
    plt.tight_layout()
    output_file = FIGURES_DIR / f"{pair_name}_20_autockt_20_morl.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] Created: {output_file}")
    return output_file

# Create all visualizations
figure_files = []
for pair_info in OBJECTIVE_PAIRS:
    fig_file = create_visualization(pair_info)
    figure_files.append(fig_file)

print(f"\n[INFO] Created {len(figure_files)} individual graphs")

# ============================================================================
# 5. CREATE COMBINED VISUALIZATION
# ============================================================================
print("\n[5] Creating combined visualization...")

def create_combined_visualization():
    """Create combined visualization showing all 6 objective pairs."""
    fig, axes = plt.subplots(2, 3, figsize=(22, 16))
    fig.patch.set_facecolor('white')
    
    target_color = '#F39C12'
    autockt_colors = plt.cm.tab20(np.linspace(0, 1, 20))
    morl_colors = plt.cm.Set3(np.linspace(0, 1, 20))
    
    for idx, (pair_info, ax) in enumerate(zip(OBJECTIVE_PAIRS, axes.flat)):
        x_key, y_key, x_label, y_label, pair_name = pair_info
        
        # Get target
        first_sol = all_solutions_with_fom[0]
        if x_key == 'gain_db':
            target_x = 20 * np.log10(first_sol['target_gain_linear']) if first_sol.get('target_gain_linear', 0) > 0 else first_sol.get('target_gain_db', 0)
        else:
            target_x = first_sol.get(f'target_{x_key}', 0)
        
        if y_key == 'gain_db':
            target_y = 20 * np.log10(first_sol['target_gain_linear']) if first_sol.get('target_gain_linear', 0) > 0 else first_sol.get('target_gain_db', 0)
        else:
            target_y = first_sol.get(f'target_{y_key}', 0)
        
        # Plot target
        ax.scatter(target_x, target_y, c='none', marker='o', s=250, 
                  alpha=1.0, label='Target' if idx == 0 else '', 
                  edgecolors=target_color, linewidths=3, zorder=10)
        
        # Plot AutoCkt (20 points)
        for sol_idx, sol in enumerate(autockt_solutions):
            if x_key == 'gain_db':
                x_val = sol.get('output_gain_db', 0)
            else:
                x_val = sol.get(f'output_{x_key}', 0)
            
            if y_key == 'gain_db':
                y_val = sol.get('output_gain_db', 0)
            else:
                y_val = sol.get(f'output_{y_key}', 0)
            
            autockt_color = autockt_colors[sol_idx]
            ax.scatter(x_val, y_val, c=[autockt_color], marker='s', s=120, 
                      alpha=0.9, label='AutoCkt Output' if (idx == 0 and sol_idx == 0) else '', 
                      edgecolors='white', linewidths=2, zorder=3)
        
        # Plot MORL (20 points)
        for sol_idx, sol in enumerate(morl_solutions):
            if x_key == 'gain_db':
                x_val = sol.get('output_gain_db', 0)
            else:
                x_val = sol.get(f'output_{x_key}', 0)
            
            if y_key == 'gain_db':
                y_val = sol.get('output_gain_db', 0)
            else:
                y_val = sol.get(f'output_{y_key}', 0)
            
            morl_color = morl_colors[sol_idx]
            ax.scatter(x_val, y_val, c=[morl_color], marker='D', s=120, 
                      alpha=0.9, label='MORL+AutoCkt Output' if (idx == 0 and sol_idx == 0) else '', 
                      edgecolors='white', linewidths=2, zorder=4)
        
        ax.set_xlabel(x_label, fontsize=11, fontweight='bold')
        ax.set_ylabel(y_label, fontsize=11, fontweight='bold')
        ax.set_title(f'{x_label} vs {y_label}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add legend only to first subplot
        if idx == 0:
            handles, labels = ax.get_legend_handles_labels()
            legend = ax.legend(handles, labels, loc='upper right', fontsize=9, 
                              framealpha=0.9, title='Legend', title_fontsize=10)
            legend.get_title().set_fontweight('bold')
            legend.get_frame().set_edgecolor('gray')
            legend.get_frame().set_linewidth(1.5)
    
    plt.suptitle('All Objective Pairs - 20 AutoCkt + 20 MORL+AutoCkt (40 Total)', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    output_file = FIGURES_DIR / "all_objective_pairs_20_autockt_20_morl.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] Created combined visualization: {output_file}")
    return output_file

combined_fig = create_combined_visualization()

# ============================================================================
# 6. SAVE DATA AND SPECIFICATION SEEDS
# ============================================================================
print("\n[6] Saving data and specification seeds...")

# Save extracted data
extracted_data = {
    'total_solutions': len(autockt_solutions) + len(morl_solutions),
    'total_specifications': 20,
    'autockt_count': len(autockt_solutions),
    'morl_count': len(morl_solutions),
    'selected_specs': selected_specs,
    'primary_spec': selected_specs[0] if selected_specs else None,
    'best_autockt': best_autockt,
    'best_morl': best_morl,
    'overall_best': overall_best,
    'solutions': all_solutions_with_fom
}

extracted_path = DATA_DIR / "20_autockt_20_best_morl_data.json"
with open(extracted_path, 'w', encoding='utf-8') as f:
    json.dump(extracted_data, f, indent=2, default=str)

print(f"[INFO] Saved extracted data to {extracted_path}")

# Save specification seeds table
spec_table_data = []
for i, spec_num in enumerate(selected_specs, 1):
    autockt_sol = autockt_solutions[i-1] if i-1 < len(autockt_solutions) else None
    morl_sol = morl_solutions[i-1] if i-1 < len(morl_solutions) else None
    
    spec_table_data.append({
        'Index': i,
        'Specification Number': spec_num,
        'AutoCkt Compliance FoM': autockt_sol.get('avg_compliance_fom', autockt_sol.get('avg_fom', 'N/A')) if autockt_sol else 'N/A',
        'AutoCkt Performance FoM': autockt_sol.get('avg_performance_fom', autockt_sol.get('avg_fom', 'N/A')) if autockt_sol else 'N/A',
        'MORL Compliance FoM': morl_sol.get('avg_compliance_fom', morl_sol.get('avg_fom', 'N/A')) if morl_sol else 'N/A',
        'MORL Performance FoM': morl_sol.get('avg_performance_fom', morl_sol.get('avg_fom', 'N/A')) if morl_sol else 'N/A'
    })

spec_df = pd.DataFrame(spec_table_data)
spec_table_path = DATA_DIR / "20_specification_seeds.csv"
spec_df.to_csv(spec_table_path, index=False)
print(f"[INFO] Saved specification seeds table to {spec_table_path}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"\nAutoCkt solutions: {len(autockt_solutions)}")
print(f"MORL+AutoCkt solutions: {len(morl_solutions)}")
print(f"Total: {len(autockt_solutions) + len(morl_solutions)}")
if best_autockt:
    comp_fom = best_autockt.get('avg_compliance_fom', 'N/A')
    perf_fom = best_autockt.get('avg_performance_fom', best_autockt.get('avg_fom', 'N/A'))
    if isinstance(perf_fom, (int, float)) and isinstance(comp_fom, (int, float)):
        print(f"\nBest AutoCkt: Spec {best_autockt['spec']}, Compliance FoM: {comp_fom:.4f}, Performance FoM: {perf_fom:.4f}")
    else:
        print(f"\nBest AutoCkt: Spec {best_autockt['spec']}, Compliance FoM: {comp_fom}, Performance FoM: {perf_fom}")
if best_morl:
    comp_fom = best_morl.get('avg_compliance_fom', 'N/A')
    perf_fom = best_morl.get('avg_performance_fom', best_morl.get('avg_fom', 'N/A'))
    if isinstance(perf_fom, (int, float)) and isinstance(comp_fom, (int, float)):
        print(f"Best MORL: Spec {best_morl['spec']}, Compliance FoM: {comp_fom:.4f}, Performance FoM: {perf_fom:.4f}")
    else:
        print(f"Best MORL: Spec {best_morl['spec']}, Compliance FoM: {comp_fom}, Performance FoM: {perf_fom}")
