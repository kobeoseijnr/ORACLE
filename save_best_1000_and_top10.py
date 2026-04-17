import pandas as pd

RESULTS = r'C:\Users\kobeo\OneDrive\Desktop\trail\with_15%_with_20\with_15%\morl_autockt\results'

# Load full 10,000
df = pd.read_csv(RESULTS + r'\morl_autockt_results_llm_cosine.csv')
df = df[df['spec'].astype(str) != 'summary']
df['spec'] = df['spec'].astype(int)
df['fom'] = df['fom'].astype(float)

# Best passing solution per spec
passing = df[df['complete_pass'] == 'Yes']
best1000 = passing.loc[passing.groupby('spec')['fom'].idxmax()].reset_index(drop=True)
best1000 = best1000.sort_values('spec').reset_index(drop=True)

# Top 10 by FOM
top10 = best1000.nlargest(10, 'fom').reset_index(drop=True)

# Save both
out_1000 = RESULTS + r'\morl_best_per_spec_llm_cosine.csv'
out_10 = RESULTS + r'\morl_top10_llm_cosine.csv'

best1000.to_csv(out_1000, index=False)
top10.to_csv(out_10, index=False)

print("=== Best 1000 (1 per spec) ===")
print("Rows:", len(best1000))
print("Complete pass: {}/{}".format((best1000['complete_pass'] == 'Yes').sum(), len(best1000)))
print("Avg FOM: {:.4f}".format(best1000['fom'].mean()))
print("Saved:", out_1000)

print("\n=== Top 10 by FOM ===")
print("Rows:", len(top10))
print("Avg FOM: {:.4f}".format(top10['fom'].mean()))
print("Saved:", out_10)
