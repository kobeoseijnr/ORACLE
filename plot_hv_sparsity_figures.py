"""
Generate clear, publication-quality figures for HV & Sparsity comparison:
  Original AutoCkt (single-objective) vs MORL Cosine (multi-objective)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
import io

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ORIG_CSV = Path(r"C:\Users\kobeo\OneDrive\Desktop\trail\with_15%_final_New\with_15%\original_autockt\results\original_autockt_results_original.csv")
MORL_CSV = Path(r"C:\Users\kobeo\OneDrive\Desktop\trail\with_15%_final_New\with_15%\morl_autockt\results\morl_autockt_results_original_cosine.csv")
HV_CSV   = Path(r"C:\Users\kobeo\OneDrive\Desktop\trail\with_15%_final_New\with_15%\morl_autockt\results\hypervolume_sparsity_comparison.csv")
FIG_DIR  = Path(r"C:\Users\kobeo\OneDrive\Desktop\trail\with_15%_final_New\with_15%\morl_autockt\results\figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Colors
C_ORIG = "#e74c3c"   # red
C_MORL = "#2980b9"   # blue
C_NW   = "#27ae60"   # green

plt.rcParams.update({
    'font.size': 13,
    'axes.titlesize': 15,
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


# ---------------------------------------------------------------------------
# Load HV/Sparsity CSV
# ---------------------------------------------------------------------------
df = pd.read_csv(HV_CSV)
df_data = df[~df['spec'].isin(['MEAN', 'MEDIAN', 'STD'])].copy()
df_data['spec'] = df_data['spec'].astype(int)

orig_hvs = df_data['hv_original'].values
morl_hvs = df_data['hv_morl_cosine'].values
if 'hv_improvement' in df_data.columns:
    hv_improvements = df_data['hv_improvement'].values
else:
    hv_improvements = morl_hvs - orig_hvs
morl_sps = df_data['sparsity_morl_cosine'].values
orig_fronts = df_data['front_size_original'].values
morl_fronts = df_data['front_size_morl_cosine'].values
nw_hvs = df_data['hv_morl_nw'].values
nw_sps = df_data['sparsity_morl_nw'].values
nw_fronts = df_data['front_size_morl_nw'].values


# ===========================================================================
# FIGURE 1: Bar chart — Mean HV comparison (simple, high-level)
# ===========================================================================
fig, ax = plt.subplots(figsize=(6, 5))
labels = ['Original\nAutoCkt', 'MORL\nCosine', 'MORL\nNW']
means = [orig_hvs.mean(), morl_hvs.mean(), nw_hvs.mean()]
stds = [orig_hvs.std(), morl_hvs.std(), nw_hvs.std()]
colors = [C_ORIG, C_MORL, C_NW]

bars = ax.bar(labels, means, yerr=stds, capsize=8, color=colors, edgecolor='white',
              linewidth=1.5, width=0.55, alpha=0.9, error_kw={'linewidth': 1.5})

# Add value labels on bars
for bar, m, s in zip(bars, means, stds):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(s, 1.0) * 0.15,
            f'{m/1e6:.2f}M', ha='center', va='bottom', fontweight='bold', fontsize=13)

ax.set_ylabel('Mean Hypervolume')
ax.set_title('Hypervolume: Original AutoCkt vs MORL')
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M'))
ax.set_ylim(0, max(means) * 1.4)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add annotation for improvement
pct = (means[1] - means[0]) / means[0] * 100
ax.annotate(f'+{pct:.0f}%', xy=(1, means[1]), xytext=(1.35, means[1]*0.85),
            fontsize=16, fontweight='bold', color=C_MORL,
            arrowprops=dict(arrowstyle='->', color=C_MORL, lw=2))

plt.tight_layout()
fig.savefig(FIG_DIR / 'fig1_hv_bar_comparison.png')
print(f"  Saved fig1_hv_bar_comparison.png")
plt.close()


# ===========================================================================
# FIGURE 2: Per-spec HV scatter — MORL vs Original (each dot = 1 spec)
# ===========================================================================
fig, ax = plt.subplots(figsize=(7, 6))

ax.scatter(orig_hvs / 1e6, morl_hvs / 1e6, alpha=0.28, s=14, color=C_MORL, zorder=2, label='Cosine')
ax.scatter(orig_hvs / 1e6, nw_hvs / 1e6, alpha=0.28, s=14, color=C_NW, zorder=2, label='NW')

# Diagonal line (equal HV)
lim_max = max(orig_hvs.max(), morl_hvs.max()) / 1e6 * 1.1
ax.plot([0, lim_max], [0, lim_max], 'k--', alpha=0.4, linewidth=1, label='Equal HV')

ax.set_xlabel('Original AutoCkt — Hypervolume (millions)')
ax.set_ylabel('MORL — Hypervolume (millions)')
ax.set_title('Per-Spec Hypervolume Comparison\n(each dot = 1 of 1000 specs)')
ax.legend(loc='lower right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Annotate how many above diagonal
above_cos = (morl_hvs > orig_hvs).sum()
above_nw = (nw_hvs > orig_hvs).sum()
ax.text(0.05, 0.92, f'Cos wins: {above_cos}/1000\nNW wins: {above_nw}/1000',
        transform=ax.transAxes, fontsize=12, fontweight='bold',
        color='black', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

plt.tight_layout()
fig.savefig(FIG_DIR / 'fig2_hv_scatter_per_spec.png')
print(f"  Saved fig2_hv_scatter_per_spec.png")
plt.close()


# ===========================================================================
# FIGURE 3: HV improvement distribution (histogram)
# ===========================================================================
fig, ax = plt.subplots(figsize=(7, 4.5))

ax.hist(hv_improvements / 1e6, bins=40, color=C_MORL, alpha=0.75, edgecolor='white')
ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax.axvline(hv_improvements.mean() / 1e6, color=C_MORL, linestyle='-', linewidth=2,
           label=f'Mean: +{hv_improvements.mean()/1e6:.2f}M')

ax.set_xlabel('HV Improvement (MORL - Original, millions)')
ax.set_ylabel('Number of Specs')
ax.set_title('Distribution of Hypervolume Improvement per Spec')
ax.legend(fontsize=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
fig.savefig(FIG_DIR / 'fig3_hv_improvement_hist.png')
print(f"  Saved fig3_hv_improvement_hist.png")
plt.close()


# ===========================================================================
# FIGURE 4: Pareto front size comparison (box plot)
# ===========================================================================
fig, ax = plt.subplots(figsize=(6.5, 5))

bp = ax.boxplot([orig_fronts, morl_fronts, nw_fronts],
                tick_labels=['Original\nAutoCkt', 'MORL\nCosine', 'MORL\nNW'],
                patch_artist=True, widths=0.45,
                medianprops=dict(color='black', linewidth=2))

bp['boxes'][0].set_facecolor(C_ORIG)
bp['boxes'][0].set_alpha(0.7)
bp['boxes'][1].set_facecolor(C_MORL)
bp['boxes'][1].set_alpha(0.7)
bp['boxes'][2].set_facecolor(C_NW)
bp['boxes'][2].set_alpha(0.7)

ax.set_ylabel('Pareto Front Size (solutions per spec)')
ax.set_title('Pareto Front Size:\nOriginal AutoCkt vs MORL')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add mean annotations
for i, data in enumerate([orig_fronts, morl_fronts, nw_fronts]):
    ax.text(i + 1, np.median(data) + 0.3, f'median={np.median(data):.0f}',
            ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
fig.savefig(FIG_DIR / 'fig4_pareto_front_size.png')
print(f"  Saved fig4_pareto_front_size.png")
plt.close()


# ===========================================================================
# FIGURE 5: Sparsity distribution for MORL (Original is always 0)
# ===========================================================================
fig, ax = plt.subplots(figsize=(7, 4.5))

ax.hist(morl_sps, bins=40, color=C_MORL, alpha=0.55, edgecolor='white', label='Cosine')
ax.hist(nw_sps, bins=40, color=C_NW, alpha=0.45, edgecolor='white', label='NW')
ax.axvline(morl_sps.mean(), color=C_MORL, linestyle='-', linewidth=2,
           label=f'Cos mean: {morl_sps.mean():.1f}')
ax.axvline(nw_sps.mean(), color=C_NW, linestyle='-', linewidth=2,
           label=f'NW mean: {nw_sps.mean():.1f}')

ax.set_xlabel('Sparsity (lower = more uniform spread)')
ax.set_ylabel('Number of Specs')
ax.set_title('MORL Cosine — Sparsity Distribution\n(Original AutoCkt = 0 for all specs, single solution)')
ax.legend(fontsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
fig.savefig(FIG_DIR / 'fig5_sparsity_distribution.png')
print(f"  Saved fig5_sparsity_distribution.png")
plt.close()


# ===========================================================================
# FIGURE 6: Combined 2x2 summary panel
# ===========================================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Panel A: Bar chart HV
ax = axes[0, 0]
bars = ax.bar(['Original\nAutoCkt', 'MORL\nCosine'],
              [orig_hvs.mean()/1e6, morl_hvs.mean()/1e6],
              color=[C_ORIG, C_MORL], edgecolor='white', width=0.5, alpha=0.9)
for bar, m in zip(bars, [orig_hvs.mean()/1e6, morl_hvs.mean()/1e6]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.02,
            f'{m:.2f}M', ha='center', va='bottom', fontweight='bold', fontsize=11)
ax.set_ylabel('Mean Hypervolume (millions)')
ax.set_title('(a) Mean Hypervolume', fontweight='bold')
ax.set_ylim(0, morl_hvs.mean()/1e6 * 1.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Panel B: Scatter per-spec
ax = axes[0, 1]
ax.scatter(orig_hvs/1e6, morl_hvs/1e6, alpha=0.3, s=10, color=C_MORL)
lim = max(orig_hvs.max(), morl_hvs.max())/1e6 * 1.1
ax.plot([0, lim], [0, lim], 'k--', alpha=0.4, linewidth=1)
ax.set_xlabel('Original AutoCkt HV (M)')
ax.set_ylabel('MORL Cosine HV (M)')
ax.set_title('(b) Per-Spec HV Comparison', fontweight='bold')
ax.text(0.05, 0.9, f'MORL wins:\n{(morl_hvs > orig_hvs).sum()}/1000',
        transform=ax.transAxes, fontsize=10, fontweight='bold', color=C_MORL,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Panel C: Front size box
ax = axes[1, 0]
bp = ax.boxplot([orig_fronts, morl_fronts],
                tick_labels=['Original', 'MORL Cosine'],
                patch_artist=True, widths=0.45,
                medianprops=dict(color='black', linewidth=2))
bp['boxes'][0].set_facecolor(C_ORIG); bp['boxes'][0].set_alpha(0.7)
bp['boxes'][1].set_facecolor(C_MORL); bp['boxes'][1].set_alpha(0.7)
ax.set_ylabel('Pareto Front Size')
ax.set_title('(c) Pareto Front Size', fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Panel D: Sparsity histogram
ax = axes[1, 1]
ax.hist(morl_sps, bins=30, color=C_MORL, alpha=0.75, edgecolor='white')
ax.axvline(morl_sps.mean(), color=C_MORL, linewidth=2,
           label=f'Mean: {morl_sps.mean():.1f}')
ax.set_xlabel('Sparsity')
ax.set_ylabel('Number of Specs')
ax.set_title('(d) MORL Sparsity Distribution', fontweight='bold')
ax.legend(fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.suptitle('Hypervolume & Sparsity: Original AutoCkt vs MORL Cosine',
             fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
fig.savefig(FIG_DIR / 'fig6_combined_summary_panel.png')
print(f"  Saved fig6_combined_summary_panel.png")
plt.close()


# ===========================================================================
# FIGURE 7: Paper-ready summary of the table (Mean HV / Sparsity / PF Size)
# ===========================================================================
methods = ['AutoCKT', 'ORACLE\n(Cosine)', 'ORACLE\n(NW)']
hv_means = [orig_hvs.mean(), morl_hvs.mean(), nw_hvs.mean()]
hv_stds = [orig_hvs.std(), morl_hvs.std(), nw_hvs.std()]
sp_means = [0.0, morl_sps.mean(), nw_sps.mean()]
sp_stds = [0.0, morl_sps.std(), nw_sps.std()]
pf_means = [orig_fronts.mean(), morl_fronts.mean(), nw_fronts.mean()]
pf_stds = [orig_fronts.std(), morl_fronts.std(), nw_fronts.std()]

fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6))

# Panel (a): Hypervolume (log scale)
ax = axes[0]
bars = ax.bar(methods, hv_means, yerr=hv_stds, capsize=5,
              color=[C_ORIG, C_MORL, C_NW], edgecolor='white', linewidth=1.2, alpha=0.9)
ax.set_yscale('log')
ax.set_ylabel('Mean Hypervolume (log scale)')
ax.set_title('(a) Hypervolume', fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
for bar, v in zip(bars, hv_means):
    ax.text(bar.get_x() + bar.get_width()/2, v * 1.15, f'{v:.2e}',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

# Panel (b): Sparsity (log scale)
ax = axes[1]
bars = ax.bar(methods, sp_means, yerr=sp_stds, capsize=5,
              color=[C_ORIG, C_MORL, C_NW], edgecolor='white', linewidth=1.2, alpha=0.9)
ax.set_yscale('log')
ax.set_ylabel('Mean Sparsity (log scale)')
ax.set_title('(b) Sparsity', fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Avoid log(0) label placement for AutoCKT
for bar, v in zip(bars, sp_means):
    if v <= 0:
        ax.text(bar.get_x() + bar.get_width()/2, ax.get_ylim()[0] * 1.2, '0.00',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    else:
        ax.text(bar.get_x() + bar.get_width()/2, v * 1.15, f'{v:.2e}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

# Panel (c): Pareto front size
ax = axes[2]
bars = ax.bar(methods, pf_means, yerr=pf_stds, capsize=5,
              color=[C_ORIG, C_MORL, C_NW], edgecolor='white', linewidth=1.2, alpha=0.9)
ax.set_ylabel('Mean PF Size')
ax.set_title('(c) Pareto Front Size', fontweight='bold')
ax.set_ylim(0, max(pf_means) * 1.35)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
for bar, v in zip(bars, pf_means):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.15, f'{v:.2f}',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

fig.suptitle('AutoCKT vs MORL ORACLE (Cosine / NW): Summary Metrics',
             fontsize=14, fontweight='bold', y=1.05)
plt.tight_layout()
fig.savefig(FIG_DIR / 'fig7_oracle_table_summary.png')
fig.savefig(FIG_DIR / 'fig7_oracle_table_summary.pdf')
print(f"  Saved fig7_oracle_table_summary.png")
print(f"  Saved fig7_oracle_table_summary.pdf")
plt.close()


print(f"\nAll figures saved to: {FIG_DIR}")
print("DONE")
