import json, numpy as np, pandas as pd

raw_path = r"C:\Users\kobeo\OneDrive\Desktop\trail\with_15%_with_20\with_15%\morl_autockt\results\morl_autockt_results_raw.json"
with open(raw_path) as f:
    cos_data = json.load(f)

nw_path = r"C:\Users\kobeo\OneDrive\Desktop\trail\OSEI_MORL_Aucrt (1)\OSEI_MORL_Aucrt\results\morl_autockt_results_nw_agent.csv"
df_nw = pd.read_csv(nw_path)
df_nw = df_nw[~df_nw["spec"].astype(str).str.startswith("summary")]

def fom(g, u, p, i, gt, ut, pt, it):
    gt = max(float(gt), 1e-9)
    ut = max(float(ut), 1e-9)
    pt = max(float(pt), 1e-9)
    it = max(float(it), 1e-9)
    return (float(g)-gt)/gt + (float(u)-ut)/ut + (float(p)-pt)/pt - (float(i)-it)/it

cos_best = {}
for sol in cos_data["all_solutions"]:
    sp = sol["spec"]
    f = fom(sol["output_gain_linear"], sol["output_ugbw_mhz"], sol["output_pm_deg"],
            sol["output_ibias_ma"], sol["target_gain_linear"], sol["target_ugbw_mhz"],
            sol["target_pm_deg"], sol["target_ibias_ma"])
    if sp not in cos_best or f > cos_best[sp]:
        cos_best[sp] = f

df_nw["spec"] = df_nw["spec"].astype(int)
nw_best = dict(df_nw.groupby("spec")["fom"].max())

common = sorted(set(cos_best.keys()) & set(nw_best.keys()))
wins_cos = wins_nw = ties = 0
rows = []
for sp in common:
    cf = cos_best[sp]
    nf = nw_best[sp]
    if abs(cf - nf) < 1e-9:
        w = "tie"; ties += 1
    elif cf > nf:
        w = "cosine"; wins_cos += 1
    else:
        w = "nw"; wins_nw += 1
    rows.append({"spec": sp, "fom_cosine": round(cf, 6), "fom_nw": round(nf, 6), "winner": w})

cos_avg = np.mean([cos_best[s] for s in common])
nw_avg = np.mean([nw_best[s] for s in common])

print("=== NW (trained in OSEI env) vs COSINE (1.488) ===")
print(f"Specs compared: {len(common)}")
print(f"Cosine avg best FOM: {cos_avg:.6f}")
print(f"NW avg best FOM:     {nw_avg:.6f}")
print(f"Cosine wins: {wins_cos}")
print(f"NW wins:     {wins_nw}")
print(f"Ties:        {ties}")
if nw_avg > cos_avg:
    print(f"WINNER: NW by {nw_avg - cos_avg:.6f}")
else:
    print(f"WINNER: COSINE by {cos_avg - nw_avg:.6f}")

print(f"\nNW gain range: {df_nw['output_gain_linear'].min():.1f} - {df_nw['output_gain_linear'].max():.1f}")
print(f"NW gain mean:  {df_nw['output_gain_linear'].mean():.1f}")

# Save comparison
cmp = pd.DataFrame(rows)
sums = pd.DataFrame([
    {"spec": "summary_specs", "fom_cosine": len(common)},
    {"spec": "summary_wins_cosine", "fom_cosine": wins_cos},
    {"spec": "summary_wins_nw", "fom_nw": wins_nw},
    {"spec": "summary_ties", "fom_cosine": ties},
    {"spec": "summary_avg_best_fom_cosine", "fom_cosine": round(cos_avg, 6)},
    {"spec": "summary_avg_best_fom_nw", "fom_nw": round(nw_avg, 6)},
])
out = pd.concat([cmp, sums], ignore_index=True)
out_path = r"C:\Users\kobeo\OneDrive\Desktop\trail\OSEI_MORL_Aucrt (1)\OSEI_MORL_Aucrt\results\morl_compare_nw_vs_cosine_1488.csv"
out.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")
