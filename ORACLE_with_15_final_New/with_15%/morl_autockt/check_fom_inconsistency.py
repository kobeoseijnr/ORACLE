import pandas as pd

# File 1: original morl cosine _new (FOM ~1.45)
df_new = pd.read_csv('results/morl_autockt_results_original_morl_cosine_new.csv')
df_new = df_new[~df_new['spec'].astype(str).str.startswith('summary')]

# File 2: standard ddqn cosine (FOM ~116)
df_std = pd.read_csv('results/morl_autockt_results_standard_ddqn_cosine.csv')
df_std = df_std[~df_std['spec'].astype(str).str.startswith('summary')]

print("=== morl_autockt_results_original_morl_cosine_new.csv ===")
print(f"  Rows: {len(df_new)}, Specs: {df_new['spec'].nunique()}")
print(f"  Gain range: {df_new['output_gain_linear'].min():.1f} - {df_new['output_gain_linear'].max():.1f}")
print(f"  UGBW range: {df_new['output_ugbw_mhz'].min():.4f} - {df_new['output_ugbw_mhz'].max():.4f}")
print(f"  Avg FOM (all): {df_new['fom'].astype(float).mean():.6f}")
print(f"  Avg best FOM: {df_new.groupby('spec')['fom'].max().mean():.6f}")

print()
print("=== morl_autockt_results_standard_ddqn_cosine.csv ===")
print(f"  Rows: {len(df_std)}, Specs: {df_std['spec'].nunique()}")
print(f"  Gain range: {df_std['output_gain_linear'].min():.1f} - {df_std['output_gain_linear'].max():.1f}")
print(f"  UGBW range: {df_std['output_ugbw_mhz'].min():.4f} - {df_std['output_ugbw_mhz'].max():.4f}")
print(f"  Avg FOM (all): {df_std['fom'].astype(float).mean():.6f}")
print(f"  Avg best FOM: {df_std.groupby('spec')['fom'].max().mean():.6f}")

print()
print("=== ROOT CAUSE ===")
print(f"  _new gain: {df_new['output_gain_linear'].mean():.1f} (synthetic, ~200-650)")
print(f"  _std gain: {df_std['output_gain_linear'].mean():.1f} (real model, ~16k-24k)")
print(f"  Same targets? {(df_new['target_gain_linear'].iloc[0] == df_std['target_gain_linear'].iloc[0])}")
print(f"  _new target gain[0]: {df_new['target_gain_linear'].iloc[0]}")
print(f"  _std target gain[0]: {df_std['target_gain_linear'].iloc[0]}")
