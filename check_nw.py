import pandas as pd, io

path = r'C:\Users\kobeo\OneDrive\Desktop\trail\with_15%_with_20\with_15%\morl_autockt\results\morl_autockt_results_nw.csv'
with open(path) as f:
    lines = f.readlines()
clean = [lines[0]] + [l for l in lines[1:] if not l.startswith('summary')]
df = pd.read_csv(io.StringIO(''.join(clean)))

fails = df[df['complete_pass'] == 'No']
print(f"Total failures: {len(fails)}")
print(f"gain_No: {(fails['gain_pass']=='No').sum()}")
print(f"ugbw_No: {(fails['ugbw_pass']=='No').sum()}")
print(f"pm_No: {(fails['pm_pass']=='No').sum()}")
print(f"ibias_No: {(fails['ibias_pass']=='No').sum()}")
print()
print("Sample fails:")
print(fails[['spec','solution','output_gain_linear','output_pm_deg','target_pm_deg','output_ibias_ma','target_ibias_ma']].head(15).to_string())
print()
print("Output ranges:")
print(f"  gain: {df['output_gain_linear'].min():.1f} - {df['output_gain_linear'].max():.1f}")
print(f"  ugbw: {df['output_ugbw_mhz'].min():.1f} - {df['output_ugbw_mhz'].max():.1f}")
print(f"  pm:   {df['output_pm_deg'].min():.1f} - {df['output_pm_deg'].max():.1f}")
print(f"  ibias:{df['output_ibias_ma'].min():.3f} - {df['output_ibias_ma'].max():.3f}")
