"""Generate detailed CSVs for LLM-guided DDQN agents and comparison."""
import json
import numpy as np
import pandas as pd
from scipy import stats

RESULTS = "results"

def fom(s):
    gt = max(float(s.get('target_gain_linear', 1e-9)), 1e-9)
    ut = max(float(s.get('target_ugbw_mhz', 1e-9)), 1e-9)
    pt = max(float(s.get('target_pm_deg', 1e-9)), 1e-9)
    it = max(float(s.get('target_ibias_ma', 1e-9)), 1e-9)
    g, u, p, i = float(s['output_gain_linear']), float(s['output_ugbw_mhz']), float(s['output_pm_deg']), float(s['output_ibias_ma'])
    return (g-gt)/gt + (u-ut)/ut + (p-pt)/pt - (i-it)/it

def gen_agent_csv(raw_path, out_csv, label):
    with open(raw_path) as f:
        data = json.load(f)
    rows = []
    for sol in data['all_solutions']:
        g, u, p, i = sol['output_gain_linear'], sol['output_ugbw_mhz'], sol['output_pm_deg'], sol['output_ibias_ma']
        gt, ut, pt, it = sol['target_gain_linear'], sol['target_ugbw_mhz'], sol['target_pm_deg'], sol['target_ibias_ma']
        f_val = fom(sol)
        g_ok = g >= gt if gt else True
        u_ok = u >= ut if ut else True
        p_ok = p >= pt if pt else True
        i_ok = i <= it if it else True
        rows.append({
            'spec': sol['spec'], 'solution': sol['solution'],
            'target_gain_linear': gt, 'target_ugbw_mhz': ut, 'target_pm_deg': pt, 'target_ibias_ma': it,
            'output_gain_linear': g, 'output_gain_db': sol.get('output_gain_db', 20*np.log10(g) if g > 0 else 0),
            'output_ugbw_mhz': u, 'output_pm_deg': p, 'output_ibias_ma': i,
            'fom': round(f_val, 6),
            'gain_pass': 'Yes' if g_ok else 'No', 'ugbw_pass': 'Yes' if u_ok else 'No',
            'pm_pass': 'Yes' if p_ok else 'No', 'ibias_pass': 'Yes' if i_ok else 'No',
            'complete_pass': 'Yes' if (g_ok and u_ok and p_ok and i_ok) else 'No',
            'scalarized_value': '', 'preference': str(sol.get('preference', '')), 'scalarization': label,
        })
    df = pd.DataFrame(rows)
    yes = (df['complete_pass'] == 'Yes').sum()
    no = (df['complete_pass'] == 'No').sum()
    mean_fom = df['fom'].mean()
    summary = {c: np.nan for c in df.columns}
    summary['spec'] = 'summary'; summary['fom'] = round(mean_fom, 6)
    summary['complete_pass'] = f"{yes}/{len(df)} solutions"; summary['scalarization'] = label
    df = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)
    df.to_csv(out_csv, index=False)
    print(f"  {label}: {yes}/{len(rows)} pass, {no} fail, mean FOM={mean_fom:.4f}")
    return no

def best_per_spec(data):
    by_spec = {}
    for sol in data:
        spec = sol['spec']
        f = fom(sol)
        sol['fom'] = f
        g_ok = sol['output_gain_linear'] >= sol['target_gain_linear']
        u_ok = sol['output_ugbw_mhz'] >= sol['target_ugbw_mhz']
        p_ok = sol['output_pm_deg'] >= sol['target_pm_deg']
        i_ok = sol['output_ibias_ma'] <= sol['target_ibias_ma']
        sol['complete_pass'] = 'Yes' if (g_ok and u_ok and p_ok and i_ok) else 'No'
        if spec not in by_spec or f > by_spec[spec]['fom']:
            by_spec[spec] = sol
    return by_spec

# 1. Per-agent CSVs
print("=" * 60)
print("LLM-GUIDED DDQN AGENT CSVs")
print("=" * 60)
fail_cos = gen_agent_csv(f"{RESULTS}/morl_raw_llm_cosine_agent.json",
                         f"{RESULTS}/morl_autockt_results_llm_cosine_agent.csv", "llm_ddqn_cosine")
fail_nw = gen_agent_csv(f"{RESULTS}/morl_raw_llm_nw_agent.json",
                        f"{RESULTS}/morl_autockt_results_llm_nw_agent.csv", "llm_ddqn_nw")

# 2. LLM NW vs Cosine comparison
print("\n" + "=" * 60)
print("LLM DDQN: NW vs COSINE")
print("=" * 60)
with open(f"{RESULTS}/morl_raw_llm_nw_agent.json") as f: nw_data = json.load(f)
with open(f"{RESULTS}/morl_raw_llm_cosine_agent.json") as f: cos_data = json.load(f)
nw_best = best_per_spec(nw_data['all_solutions'])
cos_best = best_per_spec(cos_data['all_solutions'])
all_specs = sorted(set(nw_best.keys()) | set(cos_best.keys()))
rows = []; wins_nw = wins_cos = ties = 0
for spec in all_specs:
    ns, cs = nw_best.get(spec), cos_best.get(spec)
    if ns is None or cs is None: continue
    if abs(ns['fom'] - cs['fom']) < 1e-9: w = 'tie'; ties += 1
    elif ns['fom'] > cs['fom']: w = 'nw_agent'; wins_nw += 1
    else: w = 'cosine_agent'; wins_cos += 1
    rows.append({'spec': spec, 'fom_nw': ns['fom'], 'fom_cosine': cs['fom'], 'winner': w})
df = pd.DataFrame(rows)
df.to_csv(f"{RESULTS}/morl_compare_llm_nw_vs_cosine.csv", index=False)
print(f"  NW wins: {wins_nw}, Cosine wins: {wins_cos}, Ties: {ties}")
print(f"  NW mean FOM: {df['fom_nw'].mean():.4f}, Cosine mean FOM: {df['fom_cosine'].mean():.4f}")

# 3. Detailed LLM vs Standard DDQN comparison (matching morl_compare_nw_agent_vs_cosine_agent.csv format)
print("\n" + "=" * 60)
print("LLM-GUIDED vs STANDARD DDQN (per-spec)")
print("=" * 60)
with open(f"{RESULTS}/morl_raw_nw_agent.json") as f: std_nw = json.load(f)
with open(f"{RESULTS}/morl_raw_cosine_agent.json") as f: std_cos = json.load(f)
llm_all = nw_data['all_solutions'] + cos_data['all_solutions']
std_all = std_nw['all_solutions'] + std_cos['all_solutions']
llm_best = best_per_spec(llm_all)
std_best = best_per_spec(std_all)
all_specs = sorted(set(llm_best.keys()) & set(std_best.keys()))
rows = []; wins_llm = wins_std = ties = 0
llm_foms = []; std_foms = []
for spec in all_specs:
    ls, ss = llm_best[spec], std_best[spec]
    llm_foms.append(ls['fom']); std_foms.append(ss['fom'])
    if abs(ls['fom'] - ss['fom']) < 1e-9: w = 'tie'; ties += 1
    elif ls['fom'] > ss['fom']: w = 'llm_ddqn'; wins_llm += 1
    else: w = 'standard_ddqn'; wins_std += 1
    rows.append({
        'spec': spec,
        'solution_llm_ddqn': ls['solution'],
        'fom_llm_ddqn': ls['fom'],
        'complete_pass_llm_ddqn': ls['complete_pass'],
        'preference_llm_ddqn': str(ls.get('preference', '')),
        'solution_standard_ddqn': ss['solution'],
        'fom_standard_ddqn': ss['fom'],
        'complete_pass_standard_ddqn': ss['complete_pass'],
        'preference_standard_ddqn': str(ss.get('preference', '')),
        'fom_winner': w,
    })
df = pd.DataFrame(rows)
# Add summary rows
summary_rows = [
    {'spec': 'summary_specs', 'solution_llm_ddqn': len(rows)},
    {'spec': 'summary_wins_llm_ddqn', 'solution_llm_ddqn': wins_llm},
    {'spec': 'summary_wins_standard_ddqn', 'solution_llm_ddqn': wins_std},
    {'spec': 'summary_ties', 'solution_llm_ddqn': ties},
    {'spec': 'summary_avg_fom_llm_ddqn', 'fom_llm_ddqn': np.mean(llm_foms)},
    {'spec': 'summary_avg_fom_standard_ddqn', 'fom_standard_ddqn': np.mean(std_foms)},
]
df = pd.concat([df, pd.DataFrame(summary_rows)], ignore_index=True)
df.to_csv(f"{RESULTS}/morl_compare_llm_vs_ddqn.csv", index=False)
print(f"  Standard DDQN wins: {wins_std} | Mean FOM: {np.mean(std_foms):.4f}")
print(f"  LLM-guided wins:    {wins_llm} | Mean FOM: {np.mean(llm_foms):.4f}")
print(f"  Ties:               {ties}")

# 4. Statistical Tests
print("\n" + "=" * 60)
print("STATISTICAL SIGNIFICANCE TESTS")
print("=" * 60)
llm_arr = np.array(llm_foms)
std_arr = np.array(std_foms)
diff = llm_arr - std_arr

# Wilcoxon signed-rank test
try:
    w_stat, w_pval = stats.wilcoxon(llm_arr, std_arr)
except Exception as e:
    w_stat, w_pval = None, None
    print(f"  Wilcoxon error: {e}")

# Paired t-test
t_stat, t_pval = stats.ttest_rel(llm_arr, std_arr)

# Cohen's d
d = diff.mean() / diff.std() if diff.std() > 0 else 0

results_txt = []
results_txt.append("LLM-Guided DDQN vs Standard DDQN")
results_txt.append(f"  N = {len(llm_foms)} specs")
results_txt.append(f"  LLM mean FOM:      {llm_arr.mean():.6f}")
results_txt.append(f"  Standard mean FOM: {std_arr.mean():.6f}")
results_txt.append(f"  Mean difference:   {diff.mean():.6f}")
results_txt.append(f"  LLM wins: {wins_llm}, Standard wins: {wins_std}, Ties: {ties}")
results_txt.append(f"")
results_txt.append(f"  Wilcoxon signed-rank: W={w_stat}, p={w_pval:.6f}" if w_pval else "  Wilcoxon: N/A")
results_txt.append(f"  Paired t-test:        t={t_stat:.4f}, p={t_pval:.6f}")
results_txt.append(f"  Cohen's d:            {d:.4f}")
results_txt.append(f"")
if t_pval < 0.05:
    results_txt.append(f"  ** Statistically significant at p<0.05 **")
else:
    results_txt.append(f"  Not statistically significant at p<0.05")

for line in results_txt:
    print(line)

with open(f"{RESULTS}/llm_statistical_tests.txt", 'w') as f:
    f.write('\n'.join(results_txt))

print(f"\nPass rates: LLM NW={10000-fail_nw}/10000, LLM Cosine={10000-fail_cos}/10000")
