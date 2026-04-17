import pandas as pd, io

def load_csv(path):
    with open(path) as f:
        lines = f.readlines()
    clean = [lines[0]] + [l for l in lines[1:] if not l.startswith('summary')]
    return pd.read_csv(io.StringIO(''.join(clean)))

strict = load_csv(r'C:\Users\kobeo\OneDrive\Desktop\trail\with_15%_with_20\with_15%\morl_autockt\results\morl_autockt_results_original_strict.csv')
cosine = load_csv(r'C:\Users\kobeo\OneDrive\Desktop\trail\with_15%_with_20\with_15%\morl_autockt\results\morl_autockt_results_original_cosine.csv')

print("=== STRICT CSV ===")
print(f"Columns: {list(strict.columns)}")
print(f"Rows: {len(strict)}")

# Recalculate strict pass for strict CSV
gp = (strict['output_gain_linear'] >= strict['target_gain_linear']).sum()
up = (strict['output_ugbw_mhz'] >= strict['target_ugbw_mhz']).sum()
pp = (strict['output_pm_deg'] >= strict['target_pm_deg']).sum()
ip = (strict['output_ibias_ma'] <= strict['target_ibias_ma']).sum()
cp = ((strict['output_gain_linear'] >= strict['target_gain_linear']) &
      (strict['output_ugbw_mhz'] >= strict['target_ugbw_mhz']) &
      (strict['output_pm_deg'] >= strict['target_pm_deg']) &
      (strict['output_ibias_ma'] <= strict['target_ibias_ma'])).sum()
print(f"Recalculated: gain={gp} ugbw={up} pm={pp} ibias={ip} complete={cp}")

print("\n=== COSINE CSV ===")
print(f"Columns: {list(cosine.columns)}")
print(f"Rows: {len(cosine)}")

gp2 = (cosine['output_gain_linear'] >= cosine['target_gain_linear']).sum()
up2 = (cosine['output_ugbw_mhz'] >= cosine['target_ugbw_mhz']).sum()
pp2 = (cosine['output_pm_deg'] >= cosine['target_pm_deg']).sum()
ip2 = (cosine['output_ibias_ma'] <= cosine['target_ibias_ma']).sum()
cp2 = ((cosine['output_gain_linear'] >= cosine['target_gain_linear']) &
       (cosine['output_ugbw_mhz'] >= cosine['target_ugbw_mhz']) &
       (cosine['output_pm_deg'] >= cosine['target_pm_deg']) &
       (cosine['output_ibias_ma'] <= cosine['target_ibias_ma'])).sum()
print(f"Recalculated: gain={gp2} ugbw={up2} pm={pp2} ibias={ip2} complete={cp2}")

# Compare outputs for same spec
print("\n=== SAMPLE COMPARISON (spec 1, sol 1) ===")
s1s = strict[(strict['spec']==1) & (strict['solution']==1)]
s1c = cosine[(cosine['spec']==1) & (cosine['solution']==1)]
for col in ['output_gain_linear','output_ugbw_mhz','output_pm_deg','output_ibias_ma']:
    v1 = s1s[col].values[0]; v2 = s1c[col].values[0]
    match = "SAME" if abs(v1-v2)<1e-6 else "DIFF"
    print(f"  {col}: strict={v1}, cosine={v2} [{match}]")

# Check which rows in cosine fail
fails = ~((cosine['output_gain_linear'] >= cosine['target_gain_linear']) &
          (cosine['output_ugbw_mhz'] >= cosine['target_ugbw_mhz']) &
          (cosine['output_pm_deg'] >= cosine['target_pm_deg']) &
          (cosine['output_ibias_ma'] <= cosine['target_ibias_ma']))
fdf = cosine[fails]
if len(fdf) > 0:
    print(f"\nCosine failing rows: {len(fdf)}")
    pm_fail = (fdf['output_pm_deg'] < fdf['target_pm_deg']).sum()
    ib_fail = (fdf['output_ibias_ma'] > fdf['target_ibias_ma']).sum()
    print(f"  PM failures: {pm_fail}, ibias failures: {ib_fail}")
    for _, r in fdf.head(5).iterrows():
        print(f"  spec={int(r.spec)} sol={int(r.solution)} pm={r.output_pm_deg}/{r.target_pm_deg} ib={r.output_ibias_ma}/{r.target_ibias_ma}")
