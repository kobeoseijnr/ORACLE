"""
Generate CONSISTENT comparison CSVs for Standard DDQN Cosine vs LLM DDQN Cosine.
Uses the exact same FOM formula and data pipeline for both.
Also regenerates the per-agent CSVs in the same format as morl_autockt_results_original_morl_cosine.csv.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

RESULTS = Path("results")

# ---------------------------------------------------------------------------
# FOM formula (same as main.py _fom)
# ---------------------------------------------------------------------------
def fom(g_out, u_out, p_out, i_out, g_tgt, u_tgt, p_tgt, i_tgt):
    """FoM = (G-Gt)/Gt + (U-Ut)/Ut + (P-Pt)/Pt - (I-It)/It. Higher is better."""
    def s(x): return max(float(x), 1e-9) if x is not None else 1e-9
    gt, ut, pt, it = s(g_tgt), s(u_tgt), s(p_tgt), s(i_tgt)
    go, uo, po, io = float(g_out), float(u_out), float(p_out), float(i_out)
    return (go - gt)/gt + (uo - ut)/ut + (po - pt)/pt - (io - it)/it


# ---------------------------------------------------------------------------
# Process raw JSON into per-agent CSV (same format as original CSV)
# ---------------------------------------------------------------------------
def raw_to_csv(raw_path, out_csv, label):
    """Convert raw JSON to CSV in same format as morl_autockt_results_original_morl_cosine.csv."""
    with open(raw_path) as f:
        data = json.load(f)

    rows = []
    for sol in data['all_solutions']:
        g = sol['output_gain_linear']
        u = sol['output_ugbw_mhz']
        p = sol['output_pm_deg']
        i = sol['output_ibias_ma']
        gt = sol['target_gain_linear']
        ut = sol['target_ugbw_mhz']
        pt = sol['target_pm_deg']
        it = sol['target_ibias_ma']

        f_val = fom(g, u, p, i, gt, ut, pt, it)

        g_ok = g >= gt
        u_ok = u >= ut
        p_ok = p >= pt
        i_ok = i <= it

        rows.append({
            'spec': sol['spec'],
            'solution': sol['solution'],
            'target_gain_linear': gt,
            'target_ugbw_mhz': ut,
            'target_pm_deg': pt,
            'target_ibias_ma': it,
            'output_gain_linear': g,
            'output_gain_db': sol.get('output_gain_db', 20*np.log10(g) if g > 0 else 0),
            'output_ugbw_mhz': u,
            'output_pm_deg': p,
            'output_ibias_ma': i,
            'fom': round(f_val, 6),
            'gain_pass': 'Yes' if g_ok else 'No',
            'ugbw_pass': 'Yes' if u_ok else 'No',
            'pm_pass': 'Yes' if p_ok else 'No',
            'ibias_pass': 'Yes' if i_ok else 'No',
            'complete_pass': 'Yes' if (g_ok and u_ok and p_ok and i_ok) else 'No',
        })

    df = pd.DataFrame(rows)
    num_solutions = len(df)
    num_specs = df['spec'].nunique()
    yes = (df['complete_pass'] == 'Yes').sum()
    avg_fom_all = df['fom'].mean()
    best_per_spec = df.groupby('spec')['fom'].max()
    avg_best_fom = best_per_spec.mean()
    top20_best = best_per_spec.nlargest(20).mean()

    # Add summary rows
    summaries = pd.DataFrame([
        {'spec': 'summary', 'complete_pass': f'{yes}/{num_solutions} solutions; {num_specs}/{num_specs} specs'},
        {'spec': f'summary_avg_fom_{num_solutions}_solutions', 'fom': round(avg_fom_all, 6)},
        {'spec': f'summary_avg_best_fom_{num_specs}_specs', 'fom': round(avg_best_fom, 6)},
        {'spec': 'summary_avg_top20_best_fom', 'fom': round(top20_best, 6)},
    ])
    df = pd.concat([df, summaries], ignore_index=True)
    df.to_csv(out_csv, index=False)
    print(f"  {label}: {yes}/{num_solutions} pass, avg FOM (all)={avg_fom_all:.4f}, avg best FOM={avg_best_fom:.4f}, top20={top20_best:.4f}")
    return avg_fom_all, avg_best_fom


# ---------------------------------------------------------------------------
# Per-spec best comparison
# ---------------------------------------------------------------------------
def best_per_spec(solutions):
    by_spec = {}
    for sol in solutions:
        spec = sol['spec']
        f = fom(sol['output_gain_linear'], sol['output_ugbw_mhz'],
                sol['output_pm_deg'], sol['output_ibias_ma'],
                sol['target_gain_linear'], sol['target_ugbw_mhz'],
                sol['target_pm_deg'], sol['target_ibias_ma'])
        sol['_fom'] = f
        g_ok = sol['output_gain_linear'] >= sol['target_gain_linear']
        u_ok = sol['output_ugbw_mhz'] >= sol['target_ugbw_mhz']
        p_ok = sol['output_pm_deg'] >= sol['target_pm_deg']
        i_ok = sol['output_ibias_ma'] <= sol['target_ibias_ma']
        sol['_pass'] = 'Yes' if (g_ok and u_ok and p_ok and i_ok) else 'No'
        if spec not in by_spec or f > by_spec[spec]['_fom']:
            by_spec[spec] = sol
    return by_spec


# ===========================================================================
# MAIN
# ===========================================================================
print("=" * 70)
print("CONSISTENT COMPARISON: Standard DDQN Cosine vs LLM DDQN Cosine")
print("=" * 70)

# 1. Generate per-agent CSVs in same format as original
print("\n--- Per-Agent CSVs ---")
std_avg, std_best = raw_to_csv(
    RESULTS / "morl_raw_cosine_agent.json",
    RESULTS / "morl_autockt_results_standard_ddqn_cosine.csv",
    "Standard DDQN Cosine"
)
llm_avg, llm_best = raw_to_csv(
    RESULTS / "morl_raw_llm_cosine_agent.json",
    RESULTS / "morl_autockt_results_llm_ddqn_cosine.csv",
    "LLM DDQN Cosine"
)

# 2. Per-spec comparison
print("\n--- Per-Spec Comparison ---")
with open(RESULTS / "morl_raw_cosine_agent.json") as f:
    std_data = json.load(f)
with open(RESULTS / "morl_raw_llm_cosine_agent.json") as f:
    llm_data = json.load(f)

std_best_map = best_per_spec(std_data['all_solutions'])
llm_best_map = best_per_spec(llm_data['all_solutions'])

all_specs = sorted(set(std_best_map.keys()) & set(llm_best_map.keys()))
rows = []
wins_std = wins_llm = ties = 0
std_foms = []
llm_foms = []

for spec in all_specs:
    ss = std_best_map[spec]
    ls = llm_best_map[spec]
    std_foms.append(ss['_fom'])
    llm_foms.append(ls['_fom'])

    if abs(ss['_fom'] - ls['_fom']) < 1e-9:
        w = 'tie'; ties += 1
    elif ls['_fom'] > ss['_fom']:
        w = 'llm_ddqn'; wins_llm += 1
    else:
        w = 'standard_ddqn'; wins_std += 1

    rows.append({
        'spec': spec,
        'solution_standard_ddqn': ss['solution'],
        'fom_standard_ddqn': round(ss['_fom'], 6),
        'complete_pass_standard_ddqn': ss['_pass'],
        'preference_standard_ddqn': str(ss.get('preference', '')),
        'solution_llm_ddqn': ls['solution'],
        'fom_llm_ddqn': round(ls['_fom'], 6),
        'complete_pass_llm_ddqn': ls['_pass'],
        'preference_llm_ddqn': str(ls.get('preference', '')),
        'fom_winner': w,
    })

df = pd.DataFrame(rows)

# Summary rows
summary_rows = pd.DataFrame([
    {'spec': 'summary_specs', 'solution_standard_ddqn': len(rows)},
    {'spec': 'summary_wins_standard_ddqn', 'solution_standard_ddqn': wins_std},
    {'spec': 'summary_wins_llm_ddqn', 'solution_standard_ddqn': wins_llm},
    {'spec': 'summary_ties', 'solution_standard_ddqn': ties},
    {'spec': 'summary_avg_fom_standard_ddqn', 'fom_standard_ddqn': round(np.mean(std_foms), 6)},
    {'spec': 'summary_avg_fom_llm_ddqn', 'fom_llm_ddqn': round(np.mean(llm_foms), 6)},
    {'spec': 'summary_avg_best_fom_standard_ddqn', 'fom_standard_ddqn': round(std_best, 6)},
    {'spec': 'summary_avg_best_fom_llm_ddqn', 'fom_llm_ddqn': round(llm_best, 6)},
])
df = pd.concat([df, summary_rows], ignore_index=True)
df.to_csv(RESULTS / "morl_compare_cosine_llm_vs_standard.csv", index=False)

print(f"\n  Standard DDQN Cosine: avg best FOM = {np.mean(std_foms):.4f}")
print(f"  LLM DDQN Cosine:     avg best FOM = {np.mean(llm_foms):.4f}")
print(f"  Improvement:          {np.mean(llm_foms) - np.mean(std_foms):.4f}")
print(f"  LLM wins: {wins_llm}, Standard wins: {wins_std}, Ties: {ties}")
print(f"\n  Saved: morl_compare_cosine_llm_vs_standard.csv")
print(f"  Saved: morl_autockt_results_standard_ddqn_cosine.csv")
print(f"  Saved: morl_autockt_results_llm_ddqn_cosine.csv")
