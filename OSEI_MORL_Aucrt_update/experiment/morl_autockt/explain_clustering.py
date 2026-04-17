import pandas as pd

print("="*80)
print("EXPLAINING THE DIFFERENCE IN GRAPH PATTERNS")
print("="*80)

# MORL data
morl_df = pd.read_csv('results/morl_autockt_results_original_morl.csv')
print("\n[MORL]")
print(f"Total solutions: {len(morl_df):,}")
print(f"Unique targets (specs): {morl_df['spec'].nunique()}")
print(f"Solutions per target: {len(morl_df) / morl_df['spec'].nunique():.0f}")

spec1 = morl_df[morl_df['spec']==1]
print(f"\nExample - Spec 1 (target specification):")
print(f"  Number of solutions: {len(spec1)}")
print(f"  Target gain (same for all): {spec1['target_gain_linear'].iloc[0]:.2f}")
print(f"  Output gains (vary): {spec1['output_gain_linear'].min():.2f} to {spec1['output_gain_linear'].max():.2f}")
print(f"  -> Creates VERTICAL CLUSTERING (same x, different y)")

# Original AutoCkt data
orig_df = pd.read_csv('../original_autockt/results/original_autockt_results_original_strict.csv')
print("\n" + "-"*80)
print("\n[Original AutoCkt]")
print(f"Total solutions: {len(orig_df):,}")
print(f"Unique targets (specs): {orig_df['spec'].nunique()}")
print(f"Solutions per target: {len(orig_df) / orig_df['spec'].nunique():.0f}")

spec0 = orig_df[orig_df['spec']==0]
print(f"\nExample - Spec 0 (target specification):")
print(f"  Number of solutions: {len(spec0)}")
print(f"  Target gain: {spec0['target_gain_linear'].iloc[0]:.2f}")
print(f"  Output gain: {spec0['output_gain_linear'].iloc[0]:.2f}")
print(f"  -> Creates SCATTERED PATTERN (different x, different y)")

print("\n" + "="*80)
print("KEY DIFFERENCE:")
print("="*80)
print("MORL: 10 solutions per target -> Multiple points share same target (x-axis)")
print("      This creates VERTICAL CLUSTERS in the graphs")
print("")
print("Original: 1 solution per target -> Each point has unique target (x-axis)")
print("          This creates SCATTERED DISTRIBUTION in the graphs")
print("="*80)
