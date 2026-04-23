"""
Main script for Original AutoCkt results processing, visualization, and comparison
Processes pickle files, generates graphs, and compares results
"""

import pickle
import json
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os

def lookup(spec, goal_spec):
    """Calculate normalized error (from original code)"""
    spec = np.array([float(e) for e in spec])
    goal_spec = np.array([float(e) for e in goal_spec])
    norm_spec = (spec - goal_spec) / (goal_spec + spec)
    return norm_spec

def reward(spec, goal_spec, specs_id):
    """Calculate reward (from original code)"""
    rel_specs = lookup(spec, goal_spec)
    reward_val = 0.0
    for i, rel_spec in enumerate(rel_specs):
        if specs_id[i] == 'ibias_max':
            rel_spec = rel_spec * -1.0
        if rel_spec < 0:
            reward_val += rel_spec
    return reward_val if reward_val < -0.02 else 10

def check_target_reached_tolerance(actual_specs, target_specs, specs_id):
    """Check if solution meets target using tolerance-based evaluation"""
    actual_dict = dict(zip(specs_id, actual_specs))
    target_dict = dict(zip(specs_id, target_specs))
    
    gain_actual = actual_dict.get('gain_min', actual_specs[0] if len(actual_specs) > 0 else 0)
    ugbw_actual = actual_dict.get('ugbw_min', actual_specs[3] if len(actual_specs) > 3 else 0)
    phm_actual = actual_dict.get('phm_min', actual_specs[2] if len(actual_specs) > 2 else 0)
    ibias_actual = actual_dict.get('ibias_max', actual_specs[1] if len(actual_specs) > 1 else 0)
    
    gain_target = target_dict.get('gain_min', target_specs[0] if len(target_specs) > 0 else 0)
    ugbw_target = target_dict.get('ugbw_min', target_specs[3] if len(target_specs) > 3 else 0)
    phm_target = target_dict.get('phm_min', target_specs[2] if len(target_specs) > 2 else 0)
    ibias_target = target_dict.get('ibias_max', target_specs[1] if len(target_specs) > 1 else 0)
    
    if gain_actual < 100:
        gain_actual_linear = 10 ** (gain_actual / 20)
    else:
        gain_actual_linear = gain_actual
    
    if gain_target < 100:
        gain_target_linear = 10 ** (gain_target / 20)
    else:
        gain_target_linear = gain_target
    
    tolerance = 0.15
    gain_ok = gain_actual_linear >= gain_target_linear * (1 - tolerance)
    ugbw_ok = ugbw_actual >= ugbw_target * (1 - tolerance)
    phm_ok = phm_actual >= phm_target * (1 - tolerance)
    ibias_ok = ibias_actual <= ibias_target * (1 + tolerance)
    
    if not (gain_ok and ugbw_ok and phm_ok and ibias_ok):
        close_tolerance = 0.05
        gain_close = gain_actual_linear >= gain_target_linear * (1 - tolerance - close_tolerance)
        ugbw_close = ugbw_actual >= ugbw_target * (1 - tolerance - close_tolerance)
        phm_close = phm_actual >= phm_target * (1 - tolerance - close_tolerance)
        ibias_close = ibias_actual <= ibias_target * (1 + tolerance + close_tolerance)
        
        met_count = sum([gain_ok, ugbw_ok, phm_ok, ibias_ok])
        close_count = sum([gain_close, ugbw_close, phm_close, ibias_close])
        if met_count >= 3 and close_count == 4:
            gain_ok = gain_close
            ugbw_ok = ugbw_close
            phm_ok = phm_close
            ibias_ok = ibias_close
    
    return gain_ok and ugbw_ok and phm_ok and ibias_ok

def process_results(pickle_reached_path, pickle_nreached_path, target_specs_path, output_dir, eval_method, target_type):
    """Process AutoCkt results"""
    print(f"\n{'='*80}")
    print(f"PROCESSING: {target_type.upper()} TARGETS - {eval_method.upper()} EVALUATION")
    print(f"{'='*80}")
    
    with open(target_specs_path, 'r') as f:
        target_specs = json.load(f)
    
    if os.path.exists(pickle_reached_path):
        with open(pickle_reached_path, 'rb') as f:
            obs_reached = pickle.load(f)
    else:
        obs_reached = []
    
    if os.path.exists(pickle_nreached_path):
        with open(pickle_nreached_path, 'rb') as f:
            obs_nreached = pickle.load(f)
    else:
        obs_nreached = []
    
    results = []
    specs_id = ['gain_min', 'ibias_max', 'phm_min', 'ugbw_min']
    
    # Process reached specs
    for idx, reached_spec in enumerate(obs_reached):
        if isinstance(reached_spec, (list, np.ndarray)) and len(reached_spec) >= 4:
            spec_num = idx
            target_key = str(spec_num)
            if target_key in target_specs:
                target = target_specs[target_key]
                gain_linear = float(reached_spec[0])
                ibias_A = float(reached_spec[1])
                pm_deg = float(reached_spec[2])
                ugbw_Hz = float(reached_spec[3])
                
                actual_specs = [gain_linear, ibias_A, pm_deg, ugbw_Hz]
                target_array = [
                    target.get('target_gain_linear', 0),
                    target.get('target_ibias_ma', 0) / 1000.0,
                    target.get('target_pm_deg', 0),
                    target.get('target_ugbw_mhz', 0) * 1e6
                ]
                
                if eval_method == 'strict':
                    # Strict: already passed (reward >= 10)
                    target_reached = 'Yes'
                else:
                    # Tolerance: re-evaluate
                    target_reached = 'Yes' if check_target_reached_tolerance(actual_specs, target_array, specs_id) else 'No'
                
                result = {
                    'spec': spec_num,
                    'evaluation_method': eval_method,
                    'target_reached': target_reached,
                    'target_gain_linear': target.get('target_gain_linear'),
                    'target_ugbw_mhz': target.get('target_ugbw_mhz'),
                    'target_pm_deg': target.get('target_pm_deg'),
                    'target_ibias_ma': target.get('target_ibias_ma'),
                    'output_gain_linear': gain_linear,
                    'output_gain_db': 20 * np.log10(gain_linear) if gain_linear > 0 else None,
                    'output_ibias_ma': ibias_A * 1000.0,
                    'output_pm_deg': pm_deg,
                    'output_ugbw_mhz': ugbw_Hz / 1e6,
                }
                results.append(result)
    
    # Process unreached specs
    for idx, nreached_spec in enumerate(obs_nreached):
        if isinstance(nreached_spec, (list, np.ndarray)) and len(nreached_spec) >= 4:
            spec_num = len(obs_reached) + idx
            target_key = str(spec_num)
            if target_key in target_specs:
                target = target_specs[target_key]
                gain_linear = float(nreached_spec[0])
                ibias_A = float(nreached_spec[1])
                pm_deg = float(nreached_spec[2])
                ugbw_Hz = float(nreached_spec[3])
                
                actual_specs = [gain_linear, ibias_A, pm_deg, ugbw_Hz]
                target_array = [
                    target.get('target_gain_linear', 0),
                    target.get('target_ibias_ma', 0) / 1000.0,
                    target.get('target_pm_deg', 0),
                    target.get('target_ugbw_mhz', 0) * 1e6
                ]
                
                if eval_method == 'strict':
                    # Strict: already failed (reward < 10)
                    target_reached = 'No'
                else:
                    # Tolerance: re-evaluate
                    target_reached = 'Yes' if check_target_reached_tolerance(actual_specs, target_array, specs_id) else 'No'
                
                result = {
                    'spec': spec_num,
                    'evaluation_method': eval_method,
                    'target_reached': target_reached,
                    'target_gain_linear': target.get('target_gain_linear'),
                    'target_ugbw_mhz': target.get('target_ugbw_mhz'),
                    'target_pm_deg': target.get('target_pm_deg'),
                    'target_ibias_ma': target.get('target_ibias_ma'),
                    'output_gain_linear': gain_linear,
                    'output_gain_db': 20 * np.log10(gain_linear) if gain_linear > 0 else None,
                    'output_ibias_ma': ibias_A * 1000.0,
                    'output_pm_deg': pm_deg,
                    'output_ugbw_mhz': ugbw_Hz / 1e6,
                }
                results.append(result)
    
    # Save results
    scenario_name = f"{target_type}_{eval_method}"
    csv_path = output_dir / f"original_autockt_results_{scenario_name}.csv"
    json_path = output_dir / f"original_autockt_results_{scenario_name}.json"
    
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
    
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {json_path}")
    
    passed = sum(1 for r in results if r['target_reached'] == 'Yes')
    total = len(results)
    print(f"Summary: {passed}/{total} ({passed/total*100:.2f}%) passed")
    
    return csv_path

def generate_graphs(csv_path, output_path, target_type, eval_method):
    """Generate 4 subplots"""
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['target_gain_linear', 'output_gain_linear', 
                           'target_ugbw_mhz', 'output_ugbw_mhz',
                           'target_pm_deg', 'output_pm_deg',
                           'target_ibias_ma', 'output_ibias_ma'])
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    target_desc = "15% Increased Targets" if "15percent" in target_type else "Original Targets"
    eval_desc = "Strict Evaluation - No Tolerance (reward >= 10)" if eval_method == "strict" else "Tolerance-Based Evaluation - 15% Tolerance (85%/80% rules)"
    title = f'Original AutoCkt Results ({target_desc})\n{eval_desc}\nTarget vs Output Comparison'
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    reached = df[df['target_reached'] == 'Yes']
    unreached = df[df['target_reached'] == 'No']
    
    # Gain
    ax = axes[0, 0]
    if len(reached) > 0:
        ax.scatter(reached['target_gain_linear'], reached['output_gain_linear'], 
                  alpha=0.6, s=30, c='green', label=f'Reached (n={len(reached)})', edgecolors='darkgreen')
    if len(unreached) > 0:
        ax.scatter(unreached['target_gain_linear'], unreached['output_gain_linear'], 
                  alpha=0.6, s=30, c='red', label=f'Unreached (n={len(unreached)})', edgecolors='darkred')
    min_val = min(df['target_gain_linear'].min(), df['output_gain_linear'].min())
    max_val = max(df['target_gain_linear'].max(), df['output_gain_linear'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3, label='Target = Output')
    ax.set_xlabel('Target Gain (linear)', fontsize=12)
    ax.set_ylabel('Output Gain (linear)', fontsize=12)
    ax.set_title('Gain: Target vs Output', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # UGBW
    ax = axes[0, 1]
    if len(reached) > 0:
        ax.scatter(reached['target_ugbw_mhz'], reached['output_ugbw_mhz'], 
                  alpha=0.6, s=30, c='green', label=f'Reached (n={len(reached)})', edgecolors='darkgreen')
    if len(unreached) > 0:
        ax.scatter(unreached['target_ugbw_mhz'], unreached['output_ugbw_mhz'], 
                  alpha=0.6, s=30, c='red', label=f'Unreached (n={len(unreached)})', edgecolors='darkred')
    min_val = min(df['target_ugbw_mhz'].min(), df['output_ugbw_mhz'].min())
    max_val = max(df['target_ugbw_mhz'].max(), df['output_ugbw_mhz'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3, label='Target = Output')
    ax.set_xlabel('Target UGBW (MHz)', fontsize=12)
    ax.set_ylabel('Output UGBW (MHz)', fontsize=12)
    ax.set_title('UGBW: Target vs Output', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # PM
    ax = axes[1, 0]
    target_pm = df['target_pm_deg'].values
    output_pm = df['output_pm_deg'].values
    if len(np.unique(target_pm)) == 1 and len(np.unique(output_pm)) == 1:
        jitter_scale = 0.5
        target_pm_jittered = target_pm + np.random.normal(0, jitter_scale, size=len(target_pm))
        output_pm_jittered = output_pm + np.random.normal(0, jitter_scale, size=len(output_pm))
        if len(reached) > 0:
            passed_pm_t = target_pm_jittered[df['target_reached'] == 'Yes']
            passed_pm_o = output_pm_jittered[df['target_reached'] == 'Yes']
            ax.scatter(passed_pm_t, passed_pm_o, alpha=0.6, s=30, c='green', label=f'Reached (n={len(reached)})', edgecolors='darkgreen')
        if len(unreached) > 0:
            failed_pm_t = target_pm_jittered[df['target_reached'] == 'No']
            failed_pm_o = output_pm_jittered[df['target_reached'] == 'No']
            ax.scatter(failed_pm_t, failed_pm_o, alpha=0.6, s=30, c='red', label=f'Unreached (n={len(unreached)})', edgecolors='darkred')
        ax.text(0.5, 0.95, 'Note: Jitter added for visibility\n(all points overlap)', 
                transform=ax.transAxes, fontsize=9, verticalalignment='top', horizontalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        if len(reached) > 0:
            ax.scatter(reached['target_pm_deg'], reached['output_pm_deg'], 
                      alpha=0.6, s=30, c='green', label=f'Reached (n={len(reached)})', edgecolors='darkgreen')
        if len(unreached) > 0:
            ax.scatter(unreached['target_pm_deg'], unreached['output_pm_deg'], 
                      alpha=0.6, s=30, c='red', label=f'Unreached (n={len(unreached)})', edgecolors='darkred')
    min_val = min(df['target_pm_deg'].min(), df['output_pm_deg'].min())
    max_val = max(df['target_pm_deg'].max(), df['output_pm_deg'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3, label='Target = Output')
    ax.set_xlabel('Target PM (Degrees)', fontsize=12)
    ax.set_ylabel('Output PM (Degrees)', fontsize=12)
    ax.set_title('PM: Target vs Output', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # IBIAS
    ax = axes[1, 1]
    if len(reached) > 0:
        ax.scatter(reached['target_ibias_ma'], reached['output_ibias_ma'], 
                  alpha=0.6, s=30, c='green', label=f'Reached (n={len(reached)})', edgecolors='darkgreen')
    if len(unreached) > 0:
        ax.scatter(unreached['target_ibias_ma'], unreached['output_ibias_ma'], 
                  alpha=0.6, s=30, c='red', label=f'Unreached (n={len(unreached)})', edgecolors='darkred')
    min_val = min(df['target_ibias_ma'].min(), df['output_ibias_ma'].min())
    max_val = max(df['target_ibias_ma'].max(), df['output_ibias_ma'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3, label='Target = Output')
    ax.set_xlabel('Target IBIAS (mA)', fontsize=12)
    ax.set_ylabel('Output IBIAS (mA)', fontsize=12)
    ax.set_title('IBIAS: Target vs Output', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def main():
    """Main function"""
    BASE_DIR = Path(__file__).parent
    PICKLE_REACHED = BASE_DIR / "opamp_obs_reached_test"
    PICKLE_NREACHED = BASE_DIR / "opamp_obs_nreached_test"
    OUTPUT_DIR = BASE_DIR / "results"
    GRAPHS_DIR = BASE_DIR / "graphs"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    
    if not PICKLE_REACHED.exists() and not PICKLE_NREACHED.exists():
        print(f"ERROR: Pickle files not found")
        print("Please run evaluation first: python evaluate.py <checkpoint_path>")
        sys.exit(1)
    
    # Process 15% increased targets
    TARGETS_15PERCENT = BASE_DIR / "data" / "target_specs_15percent.json"
    if TARGETS_15PERCENT.exists():
        csv_strict = process_results(PICKLE_REACHED, PICKLE_NREACHED, TARGETS_15PERCENT, 
                                    OUTPUT_DIR, 'strict', '15percent')
        csv_tolerance = process_results(PICKLE_REACHED, PICKLE_NREACHED, TARGETS_15PERCENT, 
                                       OUTPUT_DIR, 'tolerance', '15percent')
        
        if csv_strict:
            graph_path = GRAPHS_DIR / "original_autockt_4_subplots_15percent_strict.png"
            generate_graphs(csv_strict, graph_path, '15percent', 'strict')
        
        if csv_tolerance:
            graph_path = GRAPHS_DIR / "original_autockt_4_subplots_15percent_tolerance.png"
            generate_graphs(csv_tolerance, graph_path, '15percent', 'tolerance')
    
    # Process original targets
    TARGETS_ORIGINAL = BASE_DIR / "data" / "target_specs_original.json"
    if TARGETS_ORIGINAL.exists():
        csv_strict = process_results(PICKLE_REACHED, PICKLE_NREACHED, TARGETS_ORIGINAL, 
                                    OUTPUT_DIR, 'strict', 'original')
        csv_tolerance = process_results(PICKLE_REACHED, PICKLE_NREACHED, TARGETS_ORIGINAL, 
                                       OUTPUT_DIR, 'tolerance', 'original')
        
        if csv_strict:
            graph_path = GRAPHS_DIR / "original_autockt_4_subplots_original_strict.png"
            generate_graphs(csv_strict, graph_path, 'original', 'strict')
        
        if csv_tolerance:
            graph_path = GRAPHS_DIR / "original_autockt_4_subplots_original_tolerance.png"
            generate_graphs(csv_tolerance, graph_path, 'original', 'tolerance')
    
    print("\n" + "="*80)
    print("ALL PROCESSING COMPLETED")
    print("="*80)

if __name__ == "__main__":
    main()
