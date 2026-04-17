"""
Generate graphs from best_20_both.csv and best 1000 (original + MORL).
Markers: circle=target, triangle=MORL, square=Original AutoCkt.
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).parent
BEST_20_CSV = BASE / "best_20_both.csv"
ORIGINAL_CSV = BASE.parent / "original_autockt" / "results" / "original_autockt_results_original.csv"
MORL_BEST_1000 = BASE.parent / "morl_autockt" / "results" / "morl_best_per_spec_1000.csv"
OUT_DIR = BASE


def parse_val(val):
    """Extract numeric from '(orig)val' or plain number."""
    if pd.isna(val) or val == '':
        return np.nan
    s = str(val).strip()
    m = re.match(r'\([^)]+\)(.+)', s)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def main():
    # --- 1. Best 20 Both ---
    df20 = pd.read_csv(BEST_20_CSV)
    df20 = df20[df20['spec'].astype(str) != 'summary']
    if len(df20) == 0:
        print("No data in best_20_both.csv")
        return

    for c in ['target_gain_linear', 'target_ugbw_mhz', 'target_pm_deg', 'target_ibias_ma']:
        df20[c + '_num'] = df20[c].apply(parse_val)
    df20['tg'] = df20['target_gain_linear_num']
    df20['tu'] = df20['target_ugbw_mhz_num']
    df20['tp'] = df20['target_pm_deg_num']
    df20['ti'] = df20['target_ibias_ma_num']

    orig20 = df20[df20['method'] == 'original']
    morl20 = df20[df20['method'] == 'morl']

    fig1, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig1.suptitle("Best 20 Both: Target (circle) vs Output\nSquare=Original, Triangle=MORL", fontsize=14, fontweight='bold')

    def plot_best20(ax, x_tgt, y_tgt, x_out, y_out, x_lbl, y_lbl, title):
        # Circle = target, Square = Original, Triangle = MORL
        if len(orig20) > 0:
            ax.scatter(orig20[x_tgt], orig20[y_tgt], c='gray', s=80, marker='o', label='Target', alpha=0.8, edgecolors='black')
            ax.scatter(orig20[x_out], orig20[y_out], c='blue', s=80, marker='s', label='Original', alpha=0.9, edgecolors='black')
        if len(morl20) > 0:
            if len(orig20) == 0:
                ax.scatter(morl20[x_tgt], morl20[y_tgt], c='gray', s=80, marker='o', label='Target', alpha=0.8, edgecolors='black')
            ax.scatter(morl20[x_out], morl20[y_out], c='green', s=80, marker='^', label='MORL', alpha=0.9, edgecolors='black')
        ax.set_xlabel(x_lbl)
        ax.set_ylabel(y_lbl)
        ax.set_title(title)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

    plot_best20(axes[0, 0], 'tg', 'tp', 'output_gain_linear', 'output_pm_deg', 'Gain (V/V)', 'PM (deg)', 'PM vs GAIN')
    plot_best20(axes[0, 1], 'tu', 'ti', 'output_ugbw_mhz', 'output_ibias_ma', 'UGBW (MHz)', 'I-Bias (mA)', 'UGBW vs I-Bias')
    plot_best20(axes[1, 0], 'tg', 'tu', 'output_gain_linear', 'output_ugbw_mhz', 'Gain (V/V)', 'UGBW (MHz)', 'Gain vs UGBW')
    plot_best20(axes[1, 1], 'tp', 'ti', 'output_pm_deg', 'output_ibias_ma', 'PM (deg)', 'I-Bias (mA)', 'PM vs I-Bias')

    plt.tight_layout()
    fig1.savefig(OUT_DIR / "best20_both_4subplots.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUT_DIR / 'best20_both_4subplots.png'}")

    # --- 2. Best 1000: Original + MORL (1 target, 2 outputs per spec) ---
    if not ORIGINAL_CSV.exists() or not MORL_BEST_1000.exists():
        print("SKIP best 1000 graph: missing original or morl best 1000 CSV")
    else:
        df_orig = pd.read_csv(ORIGINAL_CSV)
        df_orig = df_orig[df_orig['spec'].astype(str) != 'summary']
        df_morl = pd.read_csv(MORL_BEST_1000)

        # Align: original spec 0 = morl spec 1
        df_orig['spec_key'] = df_orig['spec'].astype(int)
        df_morl['spec_key'] = df_morl['spec'].astype(int) - 1  # morl 1->0, 2->1, ...
        merged = df_orig.merge(df_morl, on='spec_key', suffixes=('_orig', '_morl'))

        # Parse targets (use original's target as shared - both have same spec targets)
        merged['tg'] = merged['target_gain_linear_orig'].apply(parse_val)
        merged['tu'] = merged['target_ugbw_mhz_orig'].apply(parse_val)
        merged['tp'] = merged['target_pm_deg_orig'].apply(parse_val)
        merged['ti'] = merged['target_ibias_ma_orig'].apply(parse_val)

        # Subsample if too many for readable plot (1000 points x 3 = 3000 markers)
        n_show = min(500, len(merged))
        m = merged.sample(n=n_show, random_state=42) if len(merged) > n_show else merged

        fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))
        fig2.suptitle("Best 1000: Target (circle) vs Outputs\nSquare=Original, Triangle=MORL", fontsize=14, fontweight='bold')

        def plot_best1000(ax, x_tgt, y_tgt, x_o, y_o, x_m, y_m, x_lbl, y_lbl, title):
            ax.scatter(m[x_tgt], m[y_tgt], c='gray', s=25, marker='o', label='Target', alpha=0.6)
            ax.scatter(m[x_o], m[y_o], c='blue', s=20, marker='s', label='Original', alpha=0.6)
            ax.scatter(m[x_m], m[y_m], c='green', s=20, marker='^', label='MORL', alpha=0.6)
            ax.set_xlabel(x_lbl)
            ax.set_ylabel(y_lbl)
            ax.set_title(title)
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)

        plot_best1000(axes2[0, 0], 'tg', 'tp', 'output_gain_linear_orig', 'output_pm_deg_orig', 'output_gain_linear_morl', 'output_pm_deg_morl',
                      'Gain (V/V)', 'PM (deg)', 'PM vs GAIN')
        plot_best1000(axes2[0, 1], 'tu', 'ti', 'output_ugbw_mhz_orig', 'output_ibias_ma_orig', 'output_ugbw_mhz_morl', 'output_ibias_ma_morl',
                      'UGBW (MHz)', 'I-Bias (mA)', 'UGBW vs I-Bias')
        plot_best1000(axes2[1, 0], 'tg', 'tu', 'output_gain_linear_orig', 'output_ugbw_mhz_orig', 'output_gain_linear_morl', 'output_ugbw_mhz_morl',
                      'Gain (V/V)', 'UGBW (MHz)', 'Gain vs UGBW')
        plot_best1000(axes2[1, 1], 'tp', 'ti', 'output_pm_deg_orig', 'output_ibias_ma_orig', 'output_pm_deg_morl', 'output_ibias_ma_morl',
                      'PM (deg)', 'I-Bias (mA)', 'PM vs I-Bias')

        plt.tight_layout()
        fig2.savefig(OUT_DIR / "best1000_original_morl_4subplots.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {OUT_DIR / 'best1000_original_morl_4subplots.png'}")

    # --- 3. Single 4-subplot figure combining both (optional) ---
    print("Done.")


if __name__ == "__main__":
    main()
