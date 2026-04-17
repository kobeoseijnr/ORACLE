"""
Deterministic, seeded extraction of best-per-spec (1000) and top-10 from
morl_autockt_results_llm_cosine.csv.  Running this script multiple times
always produces identical CSVs.

Seed: 42  (used only for tie-breaking; primary sort is by FOM descending)
"""

import numpy as np
import pandas as pd

SEED = 42
np.random.seed(SEED)

INPUT_CSV = r'C:\Users\kobeo\OneDrive\Desktop\trail\with_15%_with_20\with_15%\morl_autockt\results\morl_autockt_results_llm_cosine.csv'
OUT_1000  = r'C:\Users\kobeo\OneDrive\Desktop\trail\with_15%_with_20\with_15%\morl_autockt\results\morl_best_per_spec_llm_cosine.csv'
OUT_10    = r'C:\Users\kobeo\OneDrive\Desktop\trail\with_15%_with_20\with_15%\morl_autockt\results\morl_top10_llm_cosine.csv'

# ── Load ──────────────────────────────────────────────────────────────────
df = pd.read_csv(INPUT_CSV)
df = df[df['spec'].astype(str) != 'summary'].copy()
df['spec'] = df['spec'].astype(int)
df['solution'] = df['solution'].astype(float)
df['fom'] = df['fom'].astype(float)

# Add deterministic tie-breaker column (seeded random)
df['_tiebreak'] = np.random.RandomState(SEED).random(len(df))

# ── Best passing solution per spec (1000 rows) ───────────────────────────
passing = df[df['complete_pass'] == 'Yes'].copy()
# Sort by fom descending, then tiebreaker, so groupby.first() is deterministic
passing = passing.sort_values(['spec', 'fom', '_tiebreak'], ascending=[True, False, True])
best1000 = passing.groupby('spec', sort=True).first().reset_index()
best1000 = best1000.drop(columns=['_tiebreak'])
best1000 = best1000.sort_values('spec').reset_index(drop=True)

# ── Top 10 by FOM ────────────────────────────────────────────────────────
top10 = best1000.nlargest(10, 'fom').reset_index(drop=True)

# ── Add summary rows ─────────────────────────────────────────────────────
def make_summary(data, label):
    n = len(data)
    n_pass = (data['complete_pass'] == 'Yes').sum()
    avg_fom = data['fom'].mean()
    min_fom = data['fom'].min()
    max_fom = data['fom'].max()
    top20_fom = data.nlargest(min(20, n), 'fom')['fom'].mean()
    row = {col: '' for col in data.columns}
    row['spec'] = 'summary'
    row['complete_pass'] = 'pass={}/{}; avg_fom={:.6f}; min_fom={:.6f}; max_fom={:.6f}; top20_avg={:.6f}; seed={}'.format(
        n_pass, n, avg_fom, min_fom, max_fom, top20_fom, SEED)
    return pd.DataFrame([row])

best1000_out = pd.concat([best1000, make_summary(best1000, 'best1000')], ignore_index=True)
top10_out = pd.concat([top10, make_summary(top10, 'top10')], ignore_index=True)

# ── Save ──────────────────────────────────────────────────────────────────
best1000_out.to_csv(OUT_1000, index=False)
top10_out.to_csv(OUT_10, index=False)

# ── Print summary ─────────────────────────────────────────────────────────
print("Seed: {}".format(SEED))
print()
print("=== Best 1000 (1 per spec) ===")
print("  Rows: {}".format(len(best1000)))
print("  Complete pass: {}/{}".format((best1000['complete_pass'] == 'Yes').sum(), len(best1000)))
print("  Avg FOM:  {:.4f}".format(best1000['fom'].mean()))
print("  Min FOM:  {:.4f}".format(best1000['fom'].min()))
print("  Max FOM:  {:.4f}".format(best1000['fom'].max()))
print("  Top20 avg: {:.4f}".format(best1000.nlargest(20, 'fom')['fom'].mean()))
print("  Saved: {}".format(OUT_1000))
print()
print("=== Top 10 ===")
print("  Rows: {}".format(len(top10)))
print("  Avg FOM:  {:.4f}".format(top10['fom'].mean()))
print("  Min FOM:  {:.4f}".format(top10['fom'].min()))
print("  Max FOM:  {:.4f}".format(top10['fom'].max()))
print("  Saved: {}".format(OUT_10))
