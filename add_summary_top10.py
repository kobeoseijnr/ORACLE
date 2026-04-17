import pandas as pd

path = r'C:\Users\kobeo\OneDrive\Desktop\trail\with_15%_with_20\with_15%\morl_autockt\results\morl_top10_llm_cosine.csv'
df = pd.read_csv(path)

n = len(df)
n_pass = (df['complete_pass'] == 'Yes').sum()
avg_fom = df['fom'].mean()
min_fom = df['fom'].min()
max_fom = df['fom'].max()

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
    'complete_pass': 'pass={}/{}; avg_fom={:.6f}; min_fom={:.6f}; max_fom={:.6f}'.format(
        n_pass, n, avg_fom, min_fom, max_fom),
    'scalarization': '',
}])

df = pd.concat([df, summary], ignore_index=True)
df.to_csv(path, index=False)
print("Summary added to top 10:")
print("  pass={}/{}".format(n_pass, n))
print("  avg_fom={:.4f}".format(avg_fom))
print("  min_fom={:.4f}".format(min_fom))
print("  max_fom={:.4f}".format(max_fom))
print("Saved:", path)
