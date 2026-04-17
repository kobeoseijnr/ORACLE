"""
Generate Analysis-style graphs: Standard MORL (cosine) vs LLM MORL (cosine+llm) and vs NW.
Implements EXACTLY the same steps as Analysis 1 in generate_all_report_graphs.py.
Seed: 42
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

SEED = 42
np.random.seed(SEED)

BASE = Path(__file__).parent
RESULTS = BASE.parent / "morl_autockt" / "results"

STD_1000_CSV = RESULTS / "morl_best_per_spec_1000.csv"
LLM_1000_CSV = RESULTS / "morl_best_per_spec_llm_cosine.csv"
LLM_TOP10_CSV = RESULTS / "morl_top10_llm_cosine.csv"
NW_1000_CSV = RESULTS / "morl_best_per_spec_nw.csv"1
NW_TOP10_CSV = RESULTS / "morl_top10_nw.csv"

OUT_DIR = BASE / "std_vs_llm_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def gain_linear_to_db(g):
    g = np.asarray(g, dtype=float)
    g = np.maximum(g, 1e-12)
    return 20 * np.log10(g)


def _plot_6subplots_merged(sample, title, filename, use_shuffled_outputs=False,
                           target_suffix="_orig", triangle_label="LLM MORL (cosine+llm)",
                           jitter_scale=0.08, group_spread=0.06):
    """Plot 6 objective-pair subplots from merged data. One color per spec."""
    n = len(sample)
    sample = sample.copy()
    sample["tg_db"] = gain_linear_to_db(sample["target_gain_linear" + target_suffix].values)
    sample["tu"] = sample["target_ugbw_mhz" + target_suffix].astype(float)
    sample["tp"] = sample["target_pm_deg" + target_suffix].astype(float)
    sample["ti"] = sample["target_ibias_ma" + target_suffix].astype(float)

    if use_shuffled_outputs:
        orig_out = sample[["output_gain_db_orig", "output_ugbw_mhz_orig", "output_pm_deg_orig", "output_ibias_ma_orig"]].copy()
        morl_out = sample[["output_gain_db_morl", "output_ugbw_mhz_morl", "output_pm_deg_morl", "output_ibias_ma_morl"]].copy()
        perm = np.random.RandomState(SEED + 99).permutation(n)
        sample["output_gain_db_orig"] = orig_out.values[perm, 0]
        sample["output_ugbw_mhz_orig"] = orig_out.values[perm, 1]
        sample["output_pm_deg_orig"] = orig_out.values[perm, 2]
        sample["output_ibias_ma_orig"] = orig_out.values[perm, 3]
        perm2 = np.random.RandomState(SEED + 100).permutation(n)
        sample["output_gain_db_morl"] = morl_out.values[perm2, 0]
        sample["output_ugbw_mhz_morl"] = morl_out.values[perm2, 1]
        sample["output_pm_deg_morl"] = morl_out.values[perm2, 2]
        sample["output_ibias_ma_morl"] = morl_out.values[perm2, 3]

    spec_colors = plt.cm.tab20(np.linspace(0, 1, max(20, n)))[:n]

    # Grid jitter — deterministic, evenly spread
    def jitter_grid(n_pts, scale):
        side = int(np.ceil(np.sqrt(n_pts)))
        jx = np.repeat(np.arange(side), side)[:n_pts]
        jy = np.tile(np.arange(side), side)[:n_pts]
        center = (side - 1) / 2
        return (jx - center) * scale, (jy - center) * scale

    # Grid jitter — different shuffle per group so markers don't overlap
    jx_t, jy_t = jitter_grid(n, jitter_scale)
    jx_o, jy_o = jitter_grid(n, jitter_scale)
    jx_m, jy_m = jitter_grid(n, jitter_scale)
    # Shuffle each group differently
    perm_o = np.random.RandomState(SEED + 1).permutation(n)
    jx_o, jy_o = jx_o[perm_o], jy_o[perm_o]
    perm_m = np.random.RandomState(SEED + 2).permutation(n)
    jx_m, jy_m = jx_m[perm_m], jy_m[perm_m]
    GS = group_spread

    def plot_pair(ax, x_tgt, y_tgt, x_out, y_out, x_lbl, y_lbl, subplot_title):
        all_x = np.concatenate([sample[x_tgt].values, sample[x_out].values,
                                sample[x_out.replace("_orig", "_morl")].values])
        all_y = np.concatenate([sample[y_tgt].values, sample[y_out].values,
                                sample[y_out.replace("_orig", "_morl")].values])
        x_range = max(float(np.nanmax(all_x) - np.nanmin(all_x)), 1e-6)
        y_range = max(float(np.nanmax(all_y) - np.nanmin(all_y)), 1e-6)
        gx_t, gy_t = -GS * x_range, -GS * y_range
        gx_o, gy_o =  GS * x_range, -GS * y_range
        gx_m, gy_m =  0.0,           GS * y_range
        all_px, all_py = [], []
        for i in range(n):
            c = spec_colors[i]
            px_t = sample.iloc[i][x_tgt] + jx_t[i] * x_range + gx_t
            py_t = sample.iloc[i][y_tgt] + jy_t[i] * y_range + gy_t
            px_o = sample.iloc[i][x_out] + jx_o[i] * x_range + gx_o
            py_o = sample.iloc[i][y_out] + jy_o[i] * y_range + gy_o
            px_m = sample.iloc[i][x_out.replace("_orig", "_morl")] + jx_m[i] * x_range + gx_m
            py_m = sample.iloc[i][y_out.replace("_orig", "_morl")] + jy_m[i] * y_range + gy_m
            ax.scatter(px_t, py_t, c=[c], s=70, marker="o", alpha=0.95, edgecolors="black", linewidths=0.7)
            ax.scatter(px_o, py_o, c=[c], s=70, marker="s", alpha=0.95, edgecolors="black", linewidths=0.7)
            ax.scatter(px_m, py_m, c=[c], s=70, marker="^", alpha=0.95, edgecolors="black", linewidths=0.7)
            all_px.extend([px_t, px_o, px_m])
            all_py.extend([py_t, py_o, py_m])
        xlo, xhi = min(all_px), max(all_px)
        ylo, yhi = min(all_py), max(all_py)
        xpad = (xhi - xlo) * 0.10 + 1
        ypad = (yhi - ylo) * 0.10 + 1
        ax.set_xlim(max(0, xlo - xpad) if ("UGBW" in x_lbl or "Bias" in x_lbl) else xlo - xpad, xhi + xpad)
        ax.set_ylim(max(0, ylo - ypad) if ("UGBW" in y_lbl or "Bias" in y_lbl) else ylo - ypad, yhi + ypad)
        ax.set_xlabel(x_lbl)
        ax.set_ylabel(y_lbl)
        ax.set_title(subplot_title)
        ax.grid(True, alpha=0.3)

    pairs = [
        ("tu", "tp", "output_ugbw_mhz_orig", "output_pm_deg_orig", "UGBW (MHz)", "PM (deg)", "UGBW vs PM"),
        ("tu", "ti", "output_ugbw_mhz_orig", "output_ibias_ma_orig", "UGBW (MHz)", "I-Bias (mA)", "UGBW vs I-Bias"),
        ("tp", "ti", "output_pm_deg_orig", "output_ibias_ma_orig", "PM (deg)", "I-Bias (mA)", "PM vs I-Bias"),
        ("tg_db", "tp", "output_gain_db_orig", "output_pm_deg_orig", "Gain (dB)", "PM (deg)", "PM vs Gain"),
        ("tg_db", "tu", "output_gain_db_orig", "output_ugbw_mhz_orig", "Gain (dB)", "UGBW (MHz)", "Gain vs UGBW"),
        ("tg_db", "ti", "output_gain_db_orig", "output_ibias_ma_orig", "Gain (dB)", "I-Bias (mA)", "Gain vs I-Bias"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(title, fontsize=12, fontweight="bold")
    for ax, (xt, yt, xo, yo, xlbl, ylbl, st) in zip(axes.flat, pairs):
        plot_pair(ax, xt, yt, xo, yo, xlbl, ylbl, st)

    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markeredgecolor="black", markersize=8, label="Target"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="gray", markeredgecolor="black", markersize=8, label="Std MORL (cosine)"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="gray", markeredgecolor="black", markersize=8, label=triangle_label),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=3, frameon=True)
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    fig.savefig(OUT_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: {}".format(OUT_DIR / filename))


# ══════════════════════════════════════════════════════════════════════════
# Load data
# ══════════════════════════════════════════════════════════════════════════
print("Loading data...")

df_orig = pd.read_csv(STD_1000_CSV)
df_orig = df_orig[df_orig["spec"].astype(str) != "summary"]
df_orig["spec_key"] = df_orig["spec"].astype(int)

df_morl = pd.read_csv(LLM_1000_CSV)
df_morl = df_morl[df_morl["spec"].astype(str) != "summary"]
df_morl["spec_key"] = df_morl["spec"].astype(int)

merged = df_orig.merge(df_morl, on="spec_key", suffixes=("_orig", "_morl"))

llm_top10 = pd.read_csv(LLM_TOP10_CSV)
llm_top10 = llm_top10[llm_top10["spec"].astype(str) != "summary"]
top10_specs = llm_top10["spec"].astype(int).values

sample = merged[merged["spec_key"].isin(top10_specs)].reset_index(drop=True)
print("LLM top 10 sample: {} rows".format(len(sample)))

# LLM Cosine Analysis 1
_plot_6subplots_merged(sample,
    "Top 10 LLM Cosine Specs: Std MORL vs LLM Cosine Outputs\n"
    "Same color = same spec | o Target  s Std MORL  ^ LLM Cosine | seed={}".format(SEED),
    "top10_6subplots_std_vs_llm_FINAL.png",
    jitter_scale=0.08, group_spread=0.06)

# NW data
df_nw = pd.read_csv(NW_1000_CSV)
df_nw = df_nw[df_nw["spec"].astype(str) != "summary"]
df_nw["spec_key"] = df_nw["spec"].astype(int)

merged_nw = df_orig.merge(df_nw, on="spec_key", suffixes=("_orig", "_morl"))

nw_top10 = pd.read_csv(NW_TOP10_CSV)
nw_top10 = nw_top10[nw_top10["spec"].astype(str) != "summary"]
nw_top10_specs = nw_top10["spec"].astype(int).values

sample_nw = merged_nw[merged_nw["spec_key"].isin(nw_top10_specs)].reset_index(drop=True)
print("NW top 10 sample: {} rows".format(len(sample_nw)))

# NW Analysis 1 — tighter spacing (jitter=0.05, group_spread=0.04)
_plot_6subplots_merged(sample_nw,
    "Top 10 NW Specs: Std MORL vs NW Outputs\n"
    "Same color = same spec | o Target  s Std MORL (cosine)  ^ NW | seed={}".format(SEED),
    "top10_6subplots_std_vs_nw_FINAL.png",
    triangle_label="NW",
    jitter_scale=0.08, group_spread=0.06)

print("\nAll done. Output: {}".format(OUT_DIR))
