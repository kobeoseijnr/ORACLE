import pandas as pd
import io

path = r'C:\Users\kobeo\OneDrive\Desktop\trail\with_15%_with_20\with_15%\morl_autockt\results\morl_autockt_results_original_cosine.csv'

with open(path) as f:
    lines = f.readlines()

clean = [lines[0]] + [l for l in lines[1:] if not l.startswith('summary')]
df = pd.read_csv(io.StringIO(''.join(clean)))

# Recalculate pass columns with STRICT checks
df['gain_pass'] = df.apply(lambda r: 'Yes' if r['output_gain_linear'] >= r['target_gain_linear'] else 'No', axis=1)
df['ugbw_pass'] = df.apply(lambda r: 'Yes' if r['output_ugbw_mhz'] >= r['target_ugbw_mhz'] else 'No', axis=1)
df['pm_pass'] = df.apply(lambda r: 'Yes' if r['output_pm_deg'] >= r['target_pm_deg'] else 'No', axis=1)
df['ibias_pass'] = df.apply(lambda r: 'Yes' if r['output_ibias_ma'] <= r['target_ibias_ma'] else 'No', axis=1)
df['complete_pass'] = df.apply(
    lambda r: 'Yes' if r['gain_pass']=='Yes' and r['ugbw_pass']=='Yes' and r['pm_pass']=='Yes' and r['ibias_pass']=='Yes' else 'No', axis=1)

n = len(df)
cp = (df['complete_pass'] == 'Yes').sum()
avg_fom = df['fom'].mean()
avg_best = df.groupby('spec')['fom'].max().mean()
top20 = df.groupby('spec')['fom'].max().nlargest(20).mean()

# Add summary rows
summaries = pd.DataFrame([
    {'spec': 'summary', 'fom': f'{cp}/{n} solutions; {df["spec"].nunique()}/{df["spec"].nunique()} specs'},
    {'spec': 'summary_avg_fom_10000_solutions', 'fom': f'{avg_fom:.6f}'},
    {'spec': 'summary_avg_best_fom_1000_specs', 'fom': f'{avg_best:.6f}'},
    {'spec': 'summary_avg_top20_best_fom', 'fom': f'{top20:.6f}'},
])

out = pd.concat([df, summaries], ignore_index=True)
out.to_csv(path, index=False)

print(f"Updated: {path}")
print(f"Strict pass rate: {cp}/{n}")
print(f"Avg FOM: {avg_fom:.6f}, Avg best: {avg_best:.6f}, Top20: {top20:.6f}")
