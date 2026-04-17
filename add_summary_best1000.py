import pandas as pd

path = r'C:\Users\kobeo\OneDrive\Desktop\trail\with_15%_with_20\with_15%\morl_autockt\results\morl_best_per_spec_llm_cosine.csv'
df = pd.read_csv(path)

n = len(df)
n_pass = (df['complete_pass'] == 'Yes').sum()
avg_fom = df['fom'].mean()
avg_best = df['fom'].mean()  # already best per spec
top20 = df.nlargest(20, 'fom')['fom'].mean()
top10 = df.nlargest(10, 'fom')['fom'].mean()

summary = pd.DataFrame([{
    'spec': 'summary',
    'solution': '',
    'target_gain_linear': '',
    'target_ugbw_mhz': '',
    'target_pm_deg': '',
    'target_ibias_ma': '',
    'output_gain_linear': '',
    'output_gain_db': '',
    'output_ugbw_mhz': '',
    'output_pm_deg': '',
    'output_ibias_ma': '',
    'fom': '',
    'gain_pass': '',
    'ugbw_pass': '',
    'pm_pass': '',
    'ibias_pass': '',
    'complete_pass': 'pass={}/{}; avg_fom={:.6f}; avg_best_fom={:.6f}; top20={:.6f}; top10={:.6f}'.format(
        n_pass, n, avg_fom, avg_best, top20, top10),
    'scalarization': '',
}])

df = pd.concat([df, summary], ignore_index=True)
df.to_csv(path, index=False)
print("Summary added:")
print("  pass={}/{}".format(n_pass, n))
print("  avg_fom={:.4f}".format(avg_fom))
print("  top20_avg={:.4f}".format(top20))
print("  top10_avg={:.4f}".format(top10))
print("Saved:", path)
