import pandas as pd

print("="*80)
print("COMPARING DATA DISTRIBUTION")
print("="*80)

# MORL data
morl_df = pd.read_csv('results/morl_autockt_results_original_morl.csv')
print("\nMORL Original - Gain:")
print(f"  Total solutions: {len(morl_df):,}")
print(f"  Unique targets: {morl_df['target_gain_linear'].nunique()}")
print(f"  Unique outputs: {morl_df['output_gain_linear'].nunique()}")
print(f"  Target range: {morl_df['target_gain_linear'].min():.2f} to {morl_df['target_gain_linear'].max():.2f}")
print(f"  Output range: {morl_df['output_gain_linear'].min():.2f} to {morl_df['output_gain_linear'].max():.2f}")
print(f"\n  Example: Spec 1 has {len(morl_df[morl_df['spec']==1])} solutions")
print(f"    All with same target: {morl_df[morl_df['spec']==1]['target_gain_linear'].unique()[0]:.2f}")
print(f"    Outputs vary from: {morl_df[morl_df['spec']==1]['output_gain_linear'].min():.2f} to {morl_df[morl_df['spec']==1]['output_gain_linear'].max():.2f}")

# Original AutoCkt data
orig_df = pd.read_csv('../original_autockt/results/original_autockt_results_original_strict.csv')
print("\n" + "-"*80)
print("\nOriginal AutoCkt - Gain:")
print(f"  Total solutions: {len(orig_df):,}")
print(f"  Unique targets: {orig_df['target_gain_linear'].nunique()}")
print(f"  Unique outputs: {orig_df['output_gain_linear'].nunique()}")
print(f"  Target range: {orig_df['target_gain_linear'].min():.2f} to {orig_df['target_gain_linear'].max():.2f}")
print(f"  Output range: {orig_df['output_gain_linear'].min():.2f} to {orig_df['output_gain_linear'].max():.2f}")
print(f"\n  Example: Spec 0 has {len(orig_df[orig_df['spec']==0])} solution")
print(f"    Target: {orig_df[orig_df['spec']==0]['target_gain_linear'].values[0]:.2f}")
print(f"    Output: {orig_df[orig_df['spec']==0]['output_gain_linear'].values[0]:.2f}")

print("\n" + "="*80)
print("EXPLANATION:")
print("="*80)
print("MORL: 10 solutions per target → Points cluster vertically (same x, different y)")
print("Original: 1 solution per target → Points scatter across space (different x, different y)")
print("="*80)
