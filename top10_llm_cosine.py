import pandas as pd

best = pd.read_csv(r'C:\Users\kobeo\OneDrive\Desktop\trail\with_15%_with_20\with_15%\morl_autockt\results\morl_best_per_spec_llm_cosine.csv')
top10 = best.nlargest(10, 'fom').reset_index(drop=True)

print("Top 10 Specs by FOM (from 1000 best-per-spec):")
print("-" * 130)
header = "{:>5} {:>12} {:>12} {:>12} {:>12} {:>8} {:>8} {:>12} {:>12} {:>10}".format(
    "Spec", "Gain_tgt", "Gain_out", "UGBW_tgt", "UGBW_out", "PM_tgt", "PM_out", "IBias_tgt", "IBias_out", "FOM")
print(header)
print("-" * 130)
for _, r in top10.iterrows():
    print("{:>5} {:>12.2f} {:>12.2f} {:>12.4f} {:>12.4f} {:>8.1f} {:>8.1f} {:>12.4f} {:>12.4f} {:>10.4f}".format(
        int(r['spec']),
        r['target_gain_linear'], r['output_gain_linear'],
        r['target_ugbw_mhz'], r['output_ugbw_mhz'],
        r['target_pm_deg'], r['output_pm_deg'],
        r['target_ibias_ma'], r['output_ibias_ma'],
        r['fom']))
print("-" * 130)
print("Avg FOM of top 10: {:.4f}".format(top10['fom'].mean()))
print("Min FOM of top 10: {:.4f}".format(top10['fom'].min()))
print("Max FOM of top 10: {:.4f}".format(top10['fom'].max()))
