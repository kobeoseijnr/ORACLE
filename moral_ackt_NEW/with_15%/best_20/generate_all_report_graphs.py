"""
Generate all graphs used in Comprehensive_Comparison_Report.md.
Single entry point. Uses seed=20 for reproducibility.
Outputs: best20_pca_scatter.png, best20_both_4subplots.png, best1000_original_morl_4subplots.png, random20_pca_scatter.png
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# Default seed=20 for reproducibility. Use --random to pick a random seed each run.
DEFAULT_SEED = 20
SEED = DEFAULT_SEED  # Set by main(); used for sampling in graph_3 and graph_4

BASE = Path(__file__).parent
BEST_20_CSV = BASE / "best_20_both.csv"
ORIGINAL_CSV = BASE.parent / "original_autockt" / "results" / "original_autockt_results_original.csv"
MORL_BEST_1000 = BASE.parent / "morl_autockt" / "results" / "morl_best_per_spec_1000.csv"
OUT_DIR = BASE


def gain_linear_to_db(g):
    """Convert linear gain to dB."""
    g = np.asarray(g, dtype=float)
    g = np.maximum(g, 1e-12)
    return 20 * np.log10(g)


def parse_val(val):
    """Extract numeric from '(orig)val' or plain number."""
    if pd.isna(val) or val == "":
        return np.nan
    s = str(val).strip()
    m = re.match(r"\([^)]+\)(.+)", s)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def build_feature_matrix(df, use_target=True):
    """Build [N x 4] matrix: Gain(dB), UGBW(MHz), PM(deg), IBIAS(mA)."""
    if use_target:
        gain = gain_linear_to_db(df["target_gain_linear"].values)
        ugbw = df["target_ugbw_mhz"].values.astype(float)
        pm = df["target_pm_deg"].values.astype(float)
        ibias = df["target_ibias_ma"].values.astype(float)
    else:
        if "output_gain_db" in df.columns and df["output_gain_db"].notna().all():
            gain = df["output_gain_db"].values.astype(float)
        else:
            gain = gain_linear_to_db(df["output_gain_linear"].values)
        ugbw = df["output_ugbw_mhz"].values.astype(float)
        pm = df["output_pm_deg"].values.astype(float)
        ibias = df["output_ibias_ma"].values.astype(float)
    return np.column_stack([gain, ugbw, pm, ibias])


def _draw_pca_scatter(ax, t_x, t_y, o_x, o_y, m_x, m_y, title, obj_offset, obj_colors, obj_names):
    """Shared PCA scatter drawing logic."""
    src_offset = 2.5
    offset_t, offset_o, offset_m = (0, 0), (src_offset, 0), (-src_offset * 0.5, src_offset)
    for obj_idx in range(4):
        tx = t_x + obj_offset[obj_idx, 0] + offset_t[0]
        ty = t_y + obj_offset[obj_idx, 1] + offset_t[1]
        ax.scatter(tx, ty, c=obj_colors[obj_idx], s=40, marker="o", edgecolors="black", linewidths=0.5, alpha=0.9)
    for obj_idx in range(4):
        ox = o_x + obj_offset[obj_idx, 0] + offset_o[0]
        oy = o_y + obj_offset[obj_idx, 1] + offset_o[1]
        ax.scatter(ox, oy, c=obj_colors[obj_idx], s=40, marker="s", edgecolors="black", linewidths=0.5, alpha=0.9)
    for obj_idx in range(4):
        mx = m_x + obj_offset[obj_idx, 0] + offset_m[0]
        my = m_y + obj_offset[obj_idx, 1] + offset_m[1]
        ax.scatter(mx, my, c=obj_colors[obj_idx], s=40, marker="^", edgecolors="black", linewidths=0.5, alpha=0.9)
    ax.set_title(title)
    shape_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markeredgecolor="black", markersize=10, label="Target"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="gray", markeredgecolor="black", markersize=10, label="Original AutoCkt"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="gray", markeredgecolor="black", markersize=10, label="MORL+AutoCkt"),
    ]
    color_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=obj_colors[0], markersize=8, label=obj_names[0]),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=obj_colors[1], markersize=8, label=obj_names[1]),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=obj_colors[2], markersize=8, label=obj_names[2]),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=obj_colors[3], markersize=8, label=obj_names[3]),
    ]
    ax.legend(handles=shape_handles + color_handles, loc="upper left", bbox_to_anchor=(1.02, 1), ncol=1, title="Shape = source | Color = objective", frameon=True)


def graph_1_best20_pca():
    """Best 20 PCA scatter."""
    df = pd.read_csv(BEST_20_CSV)
    df = df[df["spec"].astype(str) != "summary"]
    if len(df) == 0:
        print("No data in best_20_both.csv")
        return
    orig = df[df["method"] == "original"]
    morl = df[df["method"] == "morl"]
    target_mat = build_feature_matrix(orig, use_target=True)
    orig_mat = build_feature_matrix(orig, use_target=False)
    morl_mat = build_feature_matrix(morl, use_target=False)
    all_mat = np.vstack([target_mat, orig_mat, morl_mat])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(all_mat)
    X_pca = PCA(n_components=2).fit_transform(X_scaled)
    pc1, pc2 = X_pca[:, 0], X_pca[:, 1]
    pc1_s = 100 * (pc1 - pc1.min()) / (pc1.max() - pc1.min() + 1e-10) if pc1.max() - pc1.min() > 1e-10 else np.full_like(pc1, 50)
    pc2_s = 100 * (pc2 - pc2.min()) / (pc2.max() - pc2.min() + 1e-10) if pc2.max() - pc2.min() > 1e-10 else np.full_like(pc2, 50)
    n = 20
    t_x, t_y = pc1_s[:n].copy(), pc2_s[:n].copy()
    o_x, o_y = pc1_s[n : 2 * n].copy(), pc2_s[n : 2 * n].copy()
    m_x, m_y = pc1_s[2 * n :].copy(), pc2_s[2 * n :].copy()
    if np.mean(t_x) < np.mean(o_x):
        t_x, o_x, m_x = 100 - t_x, 100 - o_x, 100 - m_x
    if np.mean(o_y) < np.mean(m_y):
        t_y, o_y, m_y = 100 - t_y, 100 - o_y, 100 - m_y
    obj_offset = np.array([[0, 1.5], [1.5, 0], [0, -1.5], [-1.5, 0]])
    obj_colors = ["#E69F00", "#9966CC", "#D55E00", "#009E73"]
    obj_names = ["Gain (dB)", "UGBW (MHz)", "PM (°)", "IBIAS (mA)"]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.set_xticks(range(0, 101, 10))
    ax.set_yticks(range(0, 101, 10))
    ax.set_aspect("equal")
    ax.set_xlabel("Value")
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.3)
    _draw_pca_scatter(ax, t_x, t_y, o_x, o_y, m_x, m_y,
        "Best 20 – 20 Target, 20 Original AutoCkt, 20 MORL+AutoCkt – positions from real objectives (PCA, scaled 0-100)",
        obj_offset, obj_colors, obj_names)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    fig.savefig(OUT_DIR / "best20_pca_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUT_DIR / 'best20_pca_scatter.png'}")


def graph_1b_best20_pca_colored():
    """Same data as best20_pca_scatter.png (best_20_both.csv), but one color per row instead of per objective."""
    df = pd.read_csv(BEST_20_CSV)
    df = df[df["spec"].astype(str) != "summary"]
    if len(df) == 0:
        print("SKIP colored PCA: no data in best_20_both.csv")
        return
    orig = df[df["method"] == "original"]
    morl = df[df["method"] == "morl"]
    target_mat = build_feature_matrix(orig, use_target=True)
    orig_mat = build_feature_matrix(orig, use_target=False)
    morl_mat = build_feature_matrix(morl, use_target=False)
    all_mat = np.vstack([target_mat, orig_mat, morl_mat])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(all_mat)
    X_pca = PCA(n_components=2).fit_transform(X_scaled)
    pc1, pc2 = X_pca[:, 0], X_pca[:, 1]
    pc1_s = 100 * (pc1 - pc1.min()) / (pc1.max() - pc1.min() + 1e-10) if pc1.max() - pc1.min() > 1e-10 else np.full_like(pc1, 50)
    pc2_s = 100 * (pc2 - pc2.min()) / (pc2.max() - pc2.min() + 1e-10) if pc2.max() - pc2.min() > 1e-10 else np.full_like(pc2, 50)
    n = 20
    t_x, t_y = pc1_s[:n].copy(), pc2_s[:n].copy()
    o_x, o_y = pc1_s[n : 2 * n].copy(), pc2_s[n : 2 * n].copy()
    m_x, m_y = pc1_s[2 * n :].copy(), pc2_s[2 * n :].copy()
    if np.mean(t_x) < np.mean(o_x):
        t_x, o_x, m_x = 100 - t_x, 100 - o_x, 100 - m_x
    if np.mean(o_y) < np.mean(m_y):
        t_y, o_y, m_y = 100 - t_y, 100 - o_y, 100 - m_y

    # Same layout as best20_pca_scatter: 4 obj_offset positions per point. Only change: color by row.
    obj_offset = np.array([[0, 1.5], [1.5, 0], [0, -1.5], [-1.5, 0]])
    spec_colors = plt.cm.tab20(np.linspace(0, 1, 20))

    src_offset = 2.5
    offset_t, offset_o, offset_m = (0, 0), (src_offset, 0), (-src_offset * 0.5, src_offset)

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.set_xticks(range(0, 101, 10))
    ax.set_yticks(range(0, 101, 10))
    ax.set_aspect("equal")
    ax.set_xlabel("Value")
    ax.set_ylabel("Value")
    ax.set_title("Best 20 – same data and layout as best20_pca_scatter.png; one color per row")
    ax.grid(True, alpha=0.3)

    for spec_idx in range(n):
        c = spec_colors[spec_idx]
        for obj_idx in range(4):
            tx = t_x[spec_idx] + obj_offset[obj_idx, 0] + offset_t[0]
            ty = t_y[spec_idx] + obj_offset[obj_idx, 1] + offset_t[1]
            ax.scatter(tx, ty, c=[c], s=40, marker="o", edgecolors="black", linewidths=0.5, alpha=0.9)
            ox = o_x[spec_idx] + obj_offset[obj_idx, 0] + offset_o[0]
            oy = o_y[spec_idx] + obj_offset[obj_idx, 1] + offset_o[1]
            ax.scatter(ox, oy, c=[c], s=40, marker="s", edgecolors="black", linewidths=0.5, alpha=0.9)
            mx = m_x[spec_idx] + obj_offset[obj_idx, 0] + offset_m[0]
            my = m_y[spec_idx] + obj_offset[obj_idx, 1] + offset_m[1]
            ax.scatter(mx, my, c=[c], s=40, marker="^", edgecolors="black", linewidths=0.5, alpha=0.9)

    shape_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markeredgecolor="black", markersize=10, label="Target"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="gray", markeredgecolor="black", markersize=10, label="Original AutoCkt"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="gray", markeredgecolor="black", markersize=10, label="MORL+AutoCkt"),
    ]
    ax.legend(handles=shape_handles, loc="upper left", bbox_to_anchor=(1.02, 1), ncol=1, title="Shape = source", frameon=True)
    plt.tight_layout(rect=[0, 0, 0.75, 1])
    fig.savefig(OUT_DIR / "best20_pca_scatter_colored.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUT_DIR / 'best20_pca_scatter_colored.png'}")


def graph_2_best20_4subplots():
    """Best 20 four subplots."""
    df20 = pd.read_csv(BEST_20_CSV)
    df20 = df20[df20["spec"].astype(str) != "summary"]
    if len(df20) == 0:
        print("No data in best_20_both.csv")
        return
    for c in ["target_gain_linear", "target_ugbw_mhz", "target_pm_deg", "target_ibias_ma"]:
        df20[c + "_num"] = df20[c].apply(parse_val)
    df20["tg"] = df20["target_gain_linear_num"]
    df20["tu"] = df20["target_ugbw_mhz_num"]
    df20["tp"] = df20["target_pm_deg_num"]
    df20["ti"] = df20["target_ibias_ma_num"]
    orig20 = df20[df20["method"] == "original"]
    morl20 = df20[df20["method"] == "morl"]

    def plot_best20(ax, x_tgt, y_tgt, x_out, y_out, x_lbl, y_lbl, title):
        if len(orig20) > 0:
            ax.scatter(orig20[x_tgt], orig20[y_tgt], c="gray", s=80, marker="o", label="Target", alpha=0.8, edgecolors="black")
            ax.scatter(orig20[x_out], orig20[y_out], c="blue", s=80, marker="s", label="Original", alpha=0.9, edgecolors="black")
        if len(morl20) > 0:
            if len(orig20) == 0:
                ax.scatter(morl20[x_tgt], morl20[y_tgt], c="gray", s=80, marker="o", label="Target", alpha=0.8, edgecolors="black")
            ax.scatter(morl20[x_out], morl20[y_out], c="green", s=80, marker="^", label="MORL", alpha=0.9, edgecolors="black")
        ax.set_xlabel(x_lbl)
        ax.set_ylabel(y_lbl)
        ax.set_title(title)
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Best 20 Both: Target (circle) vs Output\nSquare=Original, Triangle=MORL", fontsize=14, fontweight="bold")
    plot_best20(axes[0, 0], "tg", "tp", "output_gain_linear", "output_pm_deg", "Gain (V/V)", "PM (deg)", "PM vs GAIN")
    plot_best20(axes[0, 1], "tu", "ti", "output_ugbw_mhz", "output_ibias_ma", "UGBW (MHz)", "I-Bias (mA)", "UGBW vs I-Bias")
    plot_best20(axes[1, 0], "tg", "tu", "output_gain_linear", "output_ugbw_mhz", "Gain (V/V)", "UGBW (MHz)", "Gain vs UGBW")
    plot_best20(axes[1, 1], "tp", "ti", "output_pm_deg", "output_ibias_ma", "PM (deg)", "I-Bias (mA)", "PM vs I-Bias")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "best20_both_4subplots.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUT_DIR / 'best20_both_4subplots.png'}")


def graph_2b_best20_6subplots():
    """Best 20 with 6 subplots using actual values. One color per spec (40 specs: 20 Original + 20 MORL)."""
    df = pd.read_csv(BEST_20_CSV)
    df = df[df["spec"].astype(str) != "summary"]
    if len(df) == 0:
        print("SKIP 6 subplots: no data in best_20_both.csv")
        return
    for c in ["target_gain_linear", "target_ugbw_mhz", "target_pm_deg", "target_ibias_ma"]:
        df[c + "_num"] = df[c].apply(parse_val)
    df["tg_db"] = gain_linear_to_db(df["target_gain_linear_num"].values)
    df["tu"] = df["target_ugbw_mhz_num"]
    df["tp"] = df["target_pm_deg_num"]
    df["ti"] = df["target_ibias_ma_num"]
    orig20 = df[df["method"] == "original"].reset_index(drop=True)
    morl20 = df[df["method"] == "morl"].reset_index(drop=True)

    # 40 colors: one per spec (20 Original + 20 MORL)
    spec_colors = np.vstack([
        plt.cm.tab20(np.linspace(0, 1, 20)),
        plt.cm.tab20b(np.linspace(0, 1, 20)),
    ])

    # Jitter to separate overlapping points: grid pattern so each point gets unique offset
    def jitter_grid(n, scale=0.018):
        """Deterministic grid offsets for n points, scale as fraction of range."""
        side = int(np.ceil(np.sqrt(n)))
        jx = np.repeat(np.arange(side), side)[:n]
        jy = np.tile(np.arange(side), side)[:n]
        center = (side - 1) / 2
        return (jx - center) * scale, (jy - center) * scale

    jx_orig, jy_orig = jitter_grid(len(orig20) * 2)  # tgt+out per spec
    jx_morl, jy_morl = jitter_grid(len(morl20) * 2)

    def plot_pair(ax, x_tgt, y_tgt, x_out, y_out, x_lbl, y_lbl, title):
        all_x = np.concatenate([orig20[x_tgt].values, orig20[x_out].values, morl20[x_tgt].values, morl20[x_out].values])
        all_y = np.concatenate([orig20[y_tgt].values, orig20[y_out].values, morl20[y_tgt].values, morl20[y_out].values])
        x_range = max(all_x.max() - all_x.min(), 1e-6)
        y_range = max(all_y.max() - all_y.min(), 1e-6)

        idx = 0
        for i in range(len(orig20)):
            c = spec_colors[i]
            jx_t, jy_t = jx_orig[idx] * x_range, jy_orig[idx] * y_range
            jx_o, jy_o = jx_orig[idx + 1] * x_range, jy_orig[idx + 1] * y_range
            idx += 2
            ax.scatter(orig20.iloc[i][x_tgt] + jx_t, orig20.iloc[i][y_tgt] + jy_t,
                      c=[c], s=45, marker="D", alpha=0.95, edgecolors="black", linewidths=0.5)
            ax.scatter(orig20.iloc[i][x_out] + jx_o, orig20.iloc[i][y_out] + jy_o,
                      c=[c], s=45, marker="s", alpha=0.95, edgecolors="black", linewidths=0.5)
        idx = 0
        for i in range(len(morl20)):
            c = spec_colors[20 + i]
            jx_t, jy_t = jx_morl[idx] * x_range, jy_morl[idx] * y_range
            jx_o, jy_o = jx_morl[idx + 1] * x_range, jy_morl[idx + 1] * y_range
            idx += 2
            ax.scatter(morl20.iloc[i][x_tgt] + jx_t, morl20.iloc[i][y_tgt] + jy_t,
                      c=[c], s=45, marker="o", alpha=0.95, edgecolors="black", linewidths=0.5)
            ax.scatter(morl20.iloc[i][x_out] + jx_o, morl20.iloc[i][y_out] + jy_o,
                      c=[c], s=45, marker="^", alpha=0.95, edgecolors="black", linewidths=0.5)
        ax.set_xlabel(x_lbl)
        ax.set_ylabel(y_lbl)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle("Best 20: Target vs Output — Same color = Target + its Output pair\n◇ Target (AutoCkt) + ■ Original  |  ○ Target (MORL) + ▲ MORL", fontsize=12, fontweight="bold")

    plot_pair(axes[0, 0], "tg_db", "tp", "output_gain_db", "output_pm_deg", "Gain (dB)", "PM (°)", "PM vs Gain")
    plot_pair(axes[0, 1], "tg_db", "tu", "output_gain_db", "output_ugbw_mhz", "Gain (dB)", "UGBW (MHz)", "Gain vs UGBW")
    plot_pair(axes[0, 2], "tg_db", "ti", "output_gain_db", "output_ibias_ma", "Gain (dB)", "I-Bias (mA)", "Gain vs I-Bias")
    plot_pair(axes[1, 0], "tu", "tp", "output_ugbw_mhz", "output_pm_deg", "UGBW (MHz)", "PM (°)", "UGBW vs PM")
    plot_pair(axes[1, 1], "tu", "ti", "output_ugbw_mhz", "output_ibias_ma", "UGBW (MHz)", "I-Bias (mA)", "UGBW vs I-Bias")
    plot_pair(axes[1, 2], "tp", "ti", "output_pm_deg", "output_ibias_ma", "PM (°)", "I-Bias (mA)", "PM vs I-Bias")

    # Single shared legend for the figure
    handles = [
        Line2D([0], [0], marker="D", color="w", markerfacecolor="gray", markeredgecolor="black", markersize=8, label="Target (AutoCkt)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markeredgecolor="black", markersize=8, label="Target (MORL)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="gray", markeredgecolor="black", markersize=8, label="Original"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="gray", markeredgecolor="black", markersize=8, label="MORL"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=4, frameon=True)

    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    fig.savefig(OUT_DIR / "best20_6subplots_actual.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUT_DIR / 'best20_6subplots_actual.png'}")


def graph_3_best1000_4subplots():
    """Best 1000 four subplots (subsampled for readability)."""
    if not ORIGINAL_CSV.exists() or not MORL_BEST_1000.exists():
        print("SKIP best 1000 graph: missing CSV")
        return
    df_orig = pd.read_csv(ORIGINAL_CSV)
    df_orig = df_orig[df_orig["spec"].astype(str) != "summary"]
    df_morl = pd.read_csv(MORL_BEST_1000)
    df_orig["spec_key"] = df_orig["spec"].astype(int)
    df_morl["spec_key"] = df_morl["spec"].astype(int) - 1
    merged = df_orig.merge(df_morl, on="spec_key", suffixes=("_orig", "_morl"))
    merged["tg"] = merged["target_gain_linear_orig"].apply(parse_val)
    merged["tu"] = merged["target_ugbw_mhz_orig"].apply(parse_val)
    merged["tp"] = merged["target_pm_deg_orig"].apply(parse_val)
    merged["ti"] = merged["target_ibias_ma_orig"].apply(parse_val)
    n_show = min(500, len(merged))
    m = merged.sample(n=n_show, random_state=SEED) if len(merged) > n_show else merged

    def plot_best1000(ax, x_tgt, y_tgt, x_o, y_o, x_m, y_m, x_lbl, y_lbl, title):
        ax.scatter(m[x_tgt], m[y_tgt], c="gray", s=25, marker="o", label="Target", alpha=0.6)
        ax.scatter(m[x_o], m[y_o], c="blue", s=20, marker="s", label="Original", alpha=0.6)
        ax.scatter(m[x_m], m[y_m], c="green", s=20, marker="^", label="MORL", alpha=0.6)
        ax.set_xlabel(x_lbl)
        ax.set_ylabel(y_lbl)
        ax.set_title(title)
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Best 1000: Target (circle) vs Outputs\nSquare=Original, Triangle=MORL", fontsize=14, fontweight="bold")
    plot_best1000(axes[0, 0], "tg", "tp", "output_gain_linear_orig", "output_pm_deg_orig", "output_gain_linear_morl", "output_pm_deg_morl",
        "Gain (V/V)", "PM (deg)", "PM vs GAIN")
    plot_best1000(axes[0, 1], "tu", "ti", "output_ugbw_mhz_orig", "output_ibias_ma_orig", "output_ugbw_mhz_morl", "output_ibias_ma_morl",
        "UGBW (MHz)", "I-Bias (mA)", "UGBW vs I-Bias")
    plot_best1000(axes[1, 0], "tg", "tu", "output_gain_linear_orig", "output_ugbw_mhz_orig", "output_gain_linear_morl", "output_ugbw_mhz_morl",
        "Gain (V/V)", "UGBW (MHz)", "Gain vs UGBW")
    plot_best1000(axes[1, 1], "tp", "ti", "output_pm_deg_orig", "output_ibias_ma_orig", "output_pm_deg_morl", "output_ibias_ma_morl",
        "PM (deg)", "I-Bias (mA)", "PM vs I-Bias")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "best1000_original_morl_4subplots.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUT_DIR / 'best1000_original_morl_4subplots.png'}")


def graph_4_random20_pca():
    """Random 20 PCA scatter (20 random specs from merged best 1000)."""
    if not ORIGINAL_CSV.exists() or not MORL_BEST_1000.exists():
        print("SKIP random20 PCA: missing CSV")
        return
    df_orig = pd.read_csv(ORIGINAL_CSV)
    df_orig = df_orig[df_orig["spec"].astype(str) != "summary"]
    df_morl = pd.read_csv(MORL_BEST_1000)
    df_morl = df_morl[df_morl["spec"].astype(str) != "summary"]
    df_orig["spec_key"] = df_orig["spec"].astype(int)
    df_morl["spec_key"] = df_morl["spec"].astype(int) - 1
    merged = df_orig.merge(df_morl, on="spec_key", suffixes=("_orig", "_morl"))
    n_sample = 20
    sample = merged.sample(n=min(n_sample, len(merged)), random_state=SEED)
    # Save random 20 sample for report table
    export_cols = ["spec_key", "target_gain_linear_orig", "target_ugbw_mhz_orig", "target_pm_deg_orig", "target_ibias_ma_orig",
                   "output_gain_linear_orig", "output_ugbw_mhz_orig", "output_pm_deg_orig", "output_ibias_ma_orig",
                   "output_gain_linear_morl", "output_ugbw_mhz_morl", "output_pm_deg_morl", "output_ibias_ma_morl",
                   "fom_orig", "fom_morl"]
    export_cols = [c for c in export_cols if c in merged.columns]
    if "fom_orig" not in merged.columns:
        merged["fom_orig"] = merged.apply(lambda r: (r["output_gain_linear_orig"] - r["target_gain_linear_orig"]) / max(r["target_gain_linear_orig"], 1e-9)
        + (r["output_ugbw_mhz_orig"] - r["target_ugbw_mhz_orig"]) / max(r["target_ugbw_mhz_orig"], 1e-9)
        + (r["output_pm_deg_orig"] - r["target_pm_deg_orig"]) / max(r["target_pm_deg_orig"], 1e-9)
        - (r["output_ibias_ma_orig"] - r["target_ibias_ma_orig"]) / max(r["target_ibias_ma_orig"], 1e-9), axis=1)
    if "fom_morl" not in merged.columns:
        merged["fom_morl"] = merged.apply(lambda r: (r["output_gain_linear_morl"] - r["target_gain_linear_orig"]) / max(r["target_gain_linear_orig"], 1e-9)
        + (r["output_ugbw_mhz_morl"] - r["target_ugbw_mhz_orig"]) / max(r["target_ugbw_mhz_orig"], 1e-9)
        + (r["output_pm_deg_morl"] - r["target_pm_deg_orig"]) / max(r["target_pm_deg_orig"], 1e-9)
        - (r["output_ibias_ma_morl"] - r["target_ibias_ma_orig"]) / max(r["target_ibias_ma_orig"], 1e-9), axis=1)
    sample_exp = sample[["spec_key", "target_gain_linear_orig", "target_ugbw_mhz_orig", "target_pm_deg_orig", "target_ibias_ma_orig",
                          "output_gain_linear_orig", "output_ugbw_mhz_orig", "output_pm_deg_orig", "output_ibias_ma_orig",
                          "output_gain_linear_morl", "output_ugbw_mhz_morl", "output_pm_deg_morl", "output_ibias_ma_morl"]].copy()
    if "fom" in df_orig.columns:
        fom_map = df_orig.set_index("spec_key")["fom"]
        sample_exp["fom_orig"] = sample_exp["spec_key"].map(fom_map)
    if "fom" in df_morl.columns:
        fom_map = df_morl.set_index("spec_key")["fom"]
        sample_exp["fom_morl"] = sample_exp["spec_key"].map(fom_map)
    sample_exp.to_csv(OUT_DIR / "random20_sample.csv", index=False)

    target_mat = np.column_stack([
        gain_linear_to_db(sample["target_gain_linear_orig"].values),
        sample["target_ugbw_mhz_orig"].values.astype(float),
        sample["target_pm_deg_orig"].values.astype(float),
        sample["target_ibias_ma_orig"].values.astype(float),
    ])
    orig_mat = np.column_stack([
        sample["output_gain_db_orig"].values.astype(float),
        sample["output_ugbw_mhz_orig"].values.astype(float),
        sample["output_pm_deg_orig"].values.astype(float),
        sample["output_ibias_ma_orig"].values.astype(float),
    ])
    morl_mat = np.column_stack([
        sample["output_gain_db_morl"].values.astype(float),
        sample["output_ugbw_mhz_morl"].values.astype(float),
        sample["output_pm_deg_morl"].values.astype(float),
        sample["output_ibias_ma_morl"].values.astype(float),
    ])
    all_mat = np.vstack([target_mat, orig_mat, morl_mat])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(all_mat)
    X_pca = PCA(n_components=2).fit_transform(X_scaled)
    pc1, pc2 = X_pca[:, 0], X_pca[:, 1]
    pc1_s = 100 * (pc1 - pc1.min()) / (pc1.max() - pc1.min() + 1e-10) if pc1.max() - pc1.min() > 1e-10 else np.full_like(pc1, 50)
    pc2_s = 100 * (pc2 - pc2.min()) / (pc2.max() - pc2.min() + 1e-10) if pc2.max() - pc2.min() > 1e-10 else np.full_like(pc2, 50)
    n = n_sample
    t_x, t_y = pc1_s[:n].copy(), pc2_s[:n].copy()
    o_x, o_y = pc1_s[n : 2 * n].copy(), pc2_s[n : 2 * n].copy()
    m_x, m_y = pc1_s[2 * n :].copy(), pc2_s[2 * n :].copy()
    if np.mean(t_x) < np.mean(o_x):
        t_x, o_x, m_x = 100 - t_x, 100 - o_x, 100 - m_x
    if np.mean(o_y) < np.mean(m_y):
        t_y, o_y, m_y = 100 - t_y, 100 - o_y, 100 - m_y
    obj_offset = np.array([[0, 1.5], [1.5, 0], [0, -1.5], [-1.5, 0]])
    obj_colors = ["#E69F00", "#9966CC", "#D55E00", "#009E73"]
    obj_names = ["Gain (dB)", "UGBW (MHz)", "PM (°)", "IBIAS (mA)"]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.set_xticks(range(0, 101, 10))
    ax.set_yticks(range(0, 101, 10))
    ax.set_aspect("equal")
    ax.set_xlabel("Value")
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.3)
    _draw_pca_scatter(ax, t_x, t_y, o_x, o_y, m_x, m_y,
        "Random 20 – 20 Target, 20 Original AutoCkt, 20 MORL+AutoCkt – positions from real objectives (PCA, scaled 0-100)",
        obj_offset, obj_colors, obj_names)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    fig.savefig(OUT_DIR / "random20_pca_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUT_DIR / 'random20_pca_scatter.png'}")


def graph_4b_random20_pca_colored():
    """Random 20 PCA scatter with updated colors: one color per spec, Target (diamond), Original (square), MORL (triangle)."""
    if not ORIGINAL_CSV.exists() or not MORL_BEST_1000.exists():
        print("SKIP random20 PCA colored: missing CSV")
        return
    df_orig = pd.read_csv(ORIGINAL_CSV)
    df_orig = df_orig[df_orig["spec"].astype(str) != "summary"]
    df_morl = pd.read_csv(MORL_BEST_1000)
    df_morl = df_morl[df_morl["spec"].astype(str) != "summary"]
    df_orig["spec_key"] = df_orig["spec"].astype(int)
    df_morl["spec_key"] = df_morl["spec"].astype(int) - 1
    merged = df_orig.merge(df_morl, on="spec_key", suffixes=("_orig", "_morl"))
    n_sample = 20
    sample = merged.sample(n=min(n_sample, len(merged)), random_state=SEED).reset_index(drop=True)

    target_mat = np.column_stack([
        gain_linear_to_db(sample["target_gain_linear_orig"].values),
        sample["target_ugbw_mhz_orig"].values.astype(float),
        sample["target_pm_deg_orig"].values.astype(float),
        sample["target_ibias_ma_orig"].values.astype(float),
    ])
    orig_mat = np.column_stack([
        sample["output_gain_db_orig"].values.astype(float),
        sample["output_ugbw_mhz_orig"].values.astype(float),
        sample["output_pm_deg_orig"].values.astype(float),
        sample["output_ibias_ma_orig"].values.astype(float),
    ])
    morl_mat = np.column_stack([
        sample["output_gain_db_morl"].values.astype(float),
        sample["output_ugbw_mhz_morl"].values.astype(float),
        sample["output_pm_deg_morl"].values.astype(float),
        sample["output_ibias_ma_morl"].values.astype(float),
    ])
    all_mat = np.vstack([target_mat, orig_mat, morl_mat])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(all_mat)
    X_pca = PCA(n_components=2).fit_transform(X_scaled)
    pc1, pc2 = X_pca[:, 0], X_pca[:, 1]
    pc1_s = 100 * (pc1 - pc1.min()) / (pc1.max() - pc1.min() + 1e-10) if pc1.max() - pc1.min() > 1e-10 else np.full_like(pc1, 50)
    pc2_s = 100 * (pc2 - pc2.min()) / (pc2.max() - pc2.min() + 1e-10) if pc2.max() - pc2.min() > 1e-10 else np.full_like(pc2, 50)
    n = n_sample
    t_x, t_y = pc1_s[:n].copy(), pc2_s[:n].copy()
    o_x, o_y = pc1_s[n : 2 * n].copy(), pc2_s[n : 2 * n].copy()
    m_x, m_y = pc1_s[2 * n :].copy(), pc2_s[2 * n :].copy()
    if np.mean(t_x) < np.mean(o_x):
        t_x, o_x, m_x = 100 - t_x, 100 - o_x, 100 - m_x
    if np.mean(o_y) < np.mean(m_y):
        t_y, o_y, m_y = 100 - t_y, 100 - o_y, 100 - m_y

    obj_offset = np.array([[0, 1.5], [1.5, 0], [0, -1.5], [-1.5, 0]])
    spec_colors = plt.cm.tab20(np.linspace(0, 1, 20))

    src_offset = 2.5
    offset_t, offset_o, offset_m = (0, 0), (src_offset, 0), (-src_offset * 0.5, src_offset)

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.set_xticks(range(0, 101, 10))
    ax.set_yticks(range(0, 101, 10))
    ax.set_aspect("equal")
    ax.set_xlabel("Value")
    ax.set_ylabel("Value")
    ax.set_title("Random 20 – Same color = Target + Original + MORL per spec\n○ Target  ■ Original  ▲ MORL")
    ax.grid(True, alpha=0.3)

    for spec_idx in range(n):
        c = spec_colors[spec_idx]
        for obj_idx in range(4):
            tx = t_x[spec_idx] + obj_offset[obj_idx, 0] + offset_t[0]
            ty = t_y[spec_idx] + obj_offset[obj_idx, 1] + offset_t[1]
            ax.scatter(tx, ty, c=[c], s=40, marker="o", alpha=0.95, edgecolors="black", linewidths=0.5)
            ox = o_x[spec_idx] + obj_offset[obj_idx, 0] + offset_o[0]
            oy = o_y[spec_idx] + obj_offset[obj_idx, 1] + offset_o[1]
            ax.scatter(ox, oy, c=[c], s=40, marker="s", alpha=0.95, edgecolors="black", linewidths=0.5)
            mx = m_x[spec_idx] + obj_offset[obj_idx, 0] + offset_m[0]
            my = m_y[spec_idx] + obj_offset[obj_idx, 1] + offset_m[1]
            ax.scatter(mx, my, c=[c], s=40, marker="^", alpha=0.95, edgecolors="black", linewidths=0.5)

    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markeredgecolor="black", markersize=10, label="Target"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="gray", markeredgecolor="black", markersize=10, label="Original AutoCkt"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="gray", markeredgecolor="black", markersize=10, label="MORL+AutoCkt"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1), ncol=1, title="Shape = source", frameon=True)
    plt.tight_layout(rect=[0, 0, 0.75, 1])
    fig.savefig(OUT_DIR / "random20_pca_scatter_colored.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUT_DIR / 'random20_pca_scatter_colored.png'}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate all report graphs")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for reproducibility (default: 20)")
    parser.add_argument("--random", action="store_true", help="Use random seed (ignores --seed)")
    args = parser.parse_args()
    global SEED
    SEED = np.random.randint(0, 2**31) if args.random else args.seed
    if SEED is not None:
        np.random.seed(SEED)
    print("=" * 60)
    print("GENERATING ALL REPORT GRAPHS (seed={})".format(SEED if SEED is not None else "random"))
    print("=" * 60)
    graph_1_best20_pca()
    graph_1b_best20_pca_colored()
    graph_2_best20_4subplots()
    graph_2b_best20_6subplots()
    graph_3_best1000_4subplots()
    graph_4_random20_pca()
    graph_4b_random20_pca_colored()
    print("=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
