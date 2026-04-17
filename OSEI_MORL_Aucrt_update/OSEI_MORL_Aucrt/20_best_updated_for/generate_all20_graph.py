"""
Generate the Best-20 all-objectives scatter (axes 0–100) from local CSVs.

Data and script live in 20_best_updated_for:
- original_autockt_results.csv
- original_morl_autockt_results.csv

Output: figures/all20_sketch_axes_0_100.png
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

BASE = Path(__file__).resolve().parent
FIG_DIR = BASE / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

OBJECTIVES = [
    ("target_gain_db", "output_gain_db", "Gain (dB)"),
    ("target_ugbw_mhz", "output_ugbw_mhz", "UGBW (MHz)"),
    ("target_pm_deg", "output_pm_deg", "PM (°)"),
    ("target_ibias_ma", "output_ibias_ma", "IBIAS (mA)"),
]

OBJECTIVE_COLORS_2D = ["#9B59B6", "#E67E22", "#E74C3C", "#27AE60"]  # purple, orange, red, green


def load_csv(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def safe_float(v):
    if v is None or (isinstance(v, str) and v.strip() == ""):
        return np.nan
    try:
        return float(v)
    except (ValueError, TypeError):
        return np.nan


def get_common_20_specs_from_morl(morl_rows, autockt_rows):
    """20 spec IDs where MORL performs best (target_reached=Yes) and that exist in Original."""
    autockt_specs = {str(r.get("spec", "")) for r in autockt_rows}
    for r in morl_rows:
        try:
            r["_fom"] = float((r.get("fom") or "").strip())
        except (ValueError, TypeError):
            r["_fom"] = -np.inf
        r["_yes"] = (r.get("target_reached") or "").strip().lower() == "yes"
    by_spec_best = {}
    for r in morl_rows:
        s = str(r.get("spec", ""))
        if s not in autockt_specs:
            continue
        f = r["_fom"] if r["_fom"] is not None else -np.inf
        if not r["_yes"]:
            continue
        if s not in by_spec_best or f > (by_spec_best[s]["_fom"] or -np.inf):
            by_spec_best[s] = r
    best_list = sorted(by_spec_best.values(), key=lambda r: r["_fom"] or -np.inf, reverse=True)
    return [str(r.get("spec", "")) for r in best_list[:20]]


def get_top20_original(rows):
    """Original AutoCkt top 20 by FOM (target_reached=Yes)."""
    yes = [r for r in rows if (r.get("target_reached") or "").strip().lower() == "yes"]
    for r in yes:
        try:
            r["_fom"] = float((r.get("fom") or "").strip())
        except (ValueError, TypeError):
            r["_fom"] = None
    yes = [r for r in yes if r["_fom"] is not None]
    yes.sort(key=lambda r: r["_fom"], reverse=True)
    return yes[:20]


def get_common_20_specs(autockt_rows):
    """Return 20 spec IDs: Original's top 20 (target_reached=Yes)."""
    top = get_top20_original(autockt_rows)
    return [str(r.get("spec", "")) for r in top]


def get_original_rows_for_specs(autockt_rows, spec_ids):
    """One Original row per spec in spec_ids order."""
    by_spec = {str(r.get("spec", "")): r for r in autockt_rows}
    return [by_spec[s] for s in spec_ids if s in by_spec]


def get_morl_best_per_spec(morl_rows, spec_ids):
    """For each spec, MORL row with highest FOM (best per spec). Order matches spec_ids."""
    for r in morl_rows:
        try:
            r["_fom"] = float((r.get("fom") or "").strip())
        except (ValueError, TypeError):
            r["_fom"] = -np.inf
        r["_yes"] = (r.get("target_reached") or "").strip().lower() == "yes"
    by_spec_best_yes = {}
    by_spec_best_any = {}
    for r in morl_rows:
        s = str(r.get("spec", ""))
        f = r["_fom"] if r["_fom"] is not None else -np.inf
        if r["_yes"] and (s not in by_spec_best_yes or f > (by_spec_best_yes[s]["_fom"] or -np.inf)):
            by_spec_best_yes[s] = r
        if s not in by_spec_best_any or f > (by_spec_best_any[s]["_fom"] or -np.inf):
            by_spec_best_any[s] = r
    out = []
    for s in spec_ids:
        out.append(by_spec_best_yes.get(s) or by_spec_best_any.get(s))
    return [r for r in out if r is not None]


def _arrays_from_rows(rows, n_obj):
    """Return (targets_2d, outputs_2d) shaped (n_samp, n_obj)."""
    n_samp = len(rows)
    t = np.full((n_samp, n_obj), np.nan)
    o = np.full((n_samp, n_obj), np.nan)
    for j, (tcol, ocol, _) in enumerate(OBJECTIVES):
        t[:, j] = [safe_float(r.get(tcol)) for r in rows]
        o[:, j] = [safe_float(r.get(ocol)) for r in rows]
    return t, o


def _pca_2d(X):
    """Project X (n, 4) to 2D using PCA (numpy only)."""
    mean = np.nanmean(X, axis=0)
    Xc = np.where(np.isfinite(X), X, mean) - mean
    cov = (Xc.T @ Xc) / max(Xc.shape[0] - 1, 1)
    w, v = np.linalg.eigh(cov)
    idx = np.argsort(w)[::-1][:2]
    return Xc @ v[:, idx]


def _scale_to_0_100(xy):
    """Scale 2D array so both columns lie in [0, 100]. Handles NaNs via finite min/max."""
    xy = np.asarray(xy, dtype=float)
    out = np.full_like(xy, np.nan)
    for col in (0, 1):
        v = xy[:, col]
        finite = v[np.isfinite(v)]
        lo, hi = np.min(finite), np.max(finite) if len(finite) else 0.0
        if hi <= lo:
            hi = lo + 1.0
        out[:, col] = np.clip(100.0 * (np.where(np.isfinite(v), v, lo) - lo) / (hi - lo), 0, 100)
    return out


def plot_all20_sketch_0_100(orig_20, morl_20, out_path):
    """
    Scatter with axes 0–100: all 20 targets, 20 original, 20 MORL.
    (x,y) from real objectives: PCA on (Gain, UGBW, PM, IBIAS) per solution, scaled to [0,100]².
    Shape = source (○ Target, □ Original, △ MORL), color = objective. Legend outside.
    """
    n_obj = len(OBJECTIVES)
    t_common, o_orig = _arrays_from_rows(orig_20, n_obj)
    _, o_morl = _arrays_from_rows(morl_20, n_obj)

    n = min(20, len(orig_20), len(morl_20))
    all_4 = np.vstack([t_common[:n], o_orig[:n], o_morl[:n]])
    xy = _pca_2d(all_4)
    xy = _scale_to_0_100(xy)

    spread = 1.2
    offsets = [
        (-spread, -spread), (-spread, spread), (spread, -spread), (spread, spread),
    ]

    fig, ax = plt.subplots(figsize=(12, 10))
    fig.patch.set_facecolor("white")

    markers = ["o"] * n + ["s"] * n + ["^"] * n
    for i in range(60):
        cx, cy = xy[i, 0], xy[i, 1]
        if not (np.isfinite(cx) and np.isfinite(cy)):
            continue
        mk = markers[i]
        for j in range(n_obj):
            dx, dy = offsets[j]
            x, y = cx + dx, cy + dy
            ax.scatter(
                x, y,
                c=OBJECTIVE_COLORS_2D[j],
                marker=mk,
                s=52,
                edgecolors="black",
                linewidths=0.6,
                alpha=0.9,
                zorder=5,
                label="_nolegend",
            )

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xticks(np.arange(0, 101, 10))
    ax.set_yticks(np.arange(0, 101, 10))
    ax.set_xlabel("Value", fontsize=11, fontweight="bold")
    ax.set_ylabel("Value", fontsize=11, fontweight="bold")
    ax.set_title(
        "Best 20 — 20 Target, 20 Original AutoCkt, 20 MORL+AutoCkt — positions from real objectives (PCA, scaled 0–100)\n"
        "Shape: ○ Target   □ Original AutoCkt   △ MORL+AutoCkt  |  Color: Gain, UGBW, PM, IBIAS",
        fontsize=10, fontweight="bold", pad=10,
    )
    shape_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markeredgecolor="black",
               markersize=10, label="Target"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="coral", markeredgecolor="black",
               markersize=10, label="Original AutoCkt"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="green", markeredgecolor="black",
               markersize=10, label="MORL+AutoCkt"),
    ]
    obj_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=OBJECTIVE_COLORS_2D[k],
               markersize=8, label=OBJECTIVES[k][2])
        for k in range(n_obj)
    ]
    ax.legend(
        handles=shape_handles + obj_handles,
        title="Shape = source  |  Color = objective",
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        fontsize=9,
        frameon=True,
    )
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout(rect=[0, 0, 0.82, 1])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    autockt_path = BASE / "original_autockt_results.csv"
    morl_path = BASE / "moral_autockt_results.csv"

    if not autockt_path.exists():
        print(f"Missing: {autockt_path}")
        return
    if not morl_path.exists():
        print(f"Missing: {morl_path}")
        return

    autockt_rows = load_csv(autockt_path)
    morl_rows = load_csv(morl_path)

    spec_ids = get_common_20_specs_from_morl(morl_rows, autockt_rows)
    if len(spec_ids) < 20:
        spec_ids = get_common_20_specs(autockt_rows)
    orig_20 = get_original_rows_for_specs(autockt_rows, spec_ids)
    morl_20 = get_morl_best_per_spec(morl_rows, spec_ids)

    if len(orig_20) != 20 or len(morl_20) != 20:
        print(f"[WARN] Common specs: {len(orig_20)} Original, {len(morl_20)} MORL (expected 20 each)")

    plot_all20_sketch_0_100(orig_20, morl_20, FIG_DIR / "all20_sketch_axes_0_100.png")


if __name__ == "__main__":
    main()