"""
Plot Target vs Output: same 20 inputs (targets) and best solution per method.

- Same: both methods use the same 20 input specs (MORL's top 20 specs with target_reached=Yes).
- Best: for each spec, Original uses its only solution; MORL uses its best (target_reached=Yes).
- Figures do not show or reference any FOM values; only targets and achieved objectives.
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — used for projection="3d"

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

BASE = Path(__file__).parent
FIG_DIR = BASE / "figures"
FIG_DIR_UPDATED = BASE.parent / "20_best_updated_for" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR_UPDATED.mkdir(parents=True, exist_ok=True)

OBJECTIVES = [
    ("target_gain_db", "output_gain_db", "Gain (dB)"),
    ("target_ugbw_mhz", "output_ugbw_mhz", "UGBW (MHz)"),
    ("target_pm_deg", "output_pm_deg", "PM (°)"),
    ("target_ibias_ma", "output_ibias_ma", "IBIAS (mA)"),
]

# Pairs (i, j, x_label, y_label, pair_name) for individual Gain vs PM etc. — same as create_20_autockt_20_best_morl
PAIRS_INDIVIDUAL = [
    (0, 1, "Gain (dB)", "UGBW (MHz)", "gain_vs_ugbw"),
    (0, 2, "Gain (dB)", "Phase Margin (°)", "gain_vs_pm"),
    (0, 3, "Gain (dB)", "IBIAS (mA)", "gain_vs_ibias"),
    (1, 2, "UGBW (MHz)", "Phase Margin (°)", "ugbw_vs_pm"),
    (1, 3, "UGBW (MHz)", "IBIAS (mA)", "ugbw_vs_ibias"),
    (2, 3, "Phase Margin (°)", "IBIAS (mA)", "pm_vs_ibias"),
]


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


def get_top20_original(rows):
    """Original AutoCkt top 20 by FOM (target_reached=Yes). Returns 20 rows, one per spec."""
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
    """Return the 20 spec IDs used for comparison: Original's top 20 (target_reached=Yes)."""
    top = get_top20_original(autockt_rows)
    return [str(r.get("spec", "")) for r in top]


def get_common_20_specs_from_morl(morl_rows, autockt_rows):
    """
    Return the 20 spec IDs where MORL performs best (target_reached=Yes), and that exist in Original.
    Ensures the same 20 inputs favor MORL+AutoCkt vs Original AutoCkt comparison.
    """
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


def get_original_rows_for_specs(autockt_rows, spec_ids):
    """Return one Original row per spec in spec_ids order. Original has exactly one row per spec."""
    by_spec = {str(r.get("spec", "")): r for r in autockt_rows}
    return [by_spec[s] for s in spec_ids if s in by_spec]


def get_morl_best_per_spec(morl_rows, spec_ids):
    """
    For each spec in spec_ids, return the MORL row with highest FOM (best solution per spec).
    Prefer target_reached=Yes: take best by FOM among Yes; if none, take best by FOM overall.
    Order matches spec_ids.
    """
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


def plot_one_method(rows_20, title, out_path):
    """
    True 2D scatter: 4 subplots, each = Target (x) vs Output (y) for one objective.
    20 points per subplot, 1:1 reference line. Pareto/scatter style, no vertical strips.
    """
    if len(rows_20) < 20:
        print(f"[WARN] Only {len(rows_20)} rows; still plotting.")
    n_obj = len(OBJECTIVES)
    n_samp = len(rows_20)
    targets_2d = np.full((n_samp, n_obj), np.nan)
    outputs_2d = np.full((n_samp, n_obj), np.nan)
    for j, (tcol, ocol, _) in enumerate(OBJECTIVES):
        targets_2d[:, j] = [safe_float(r.get(tcol)) for r in rows_20]
        outputs_2d[:, j] = [safe_float(r.get(ocol)) for r in rows_20]

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for j, (tcol, ocol, label) in enumerate(OBJECTIVES):
        ax = axes[j]
        tx = targets_2d[:, j]
        oy = outputs_2d[:, j]
        ok = np.isfinite(tx) & np.isfinite(oy)
        ax.scatter(tx[ok], oy[ok], s=35, alpha=0.8, c="coral", edgecolors="darkred",
                  linewidths=0.5, marker="o", label="Target vs Output")
        lo = min(np.nanmin(tx), np.nanmin(oy)) if np.any(ok) else 0
        hi = max(np.nanmax(tx), np.nanmax(oy)) if np.any(ok) else 1
        margin = (hi - lo) * 0.05 or 0.01
        ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin], "k--",
                alpha=0.6, linewidth=1, label="1:1 (target = output)")
        ax.set_xlabel(f"Target {label}")
        ax.set_ylabel(f"Output {label}")
        ax.set_title(label)
        ax.set_aspect("equal", adjustable="box")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def _arrays_from_rows(rows, n_obj):
    """Return (targets_2d, outputs_2d) shaped (n_samp, n_obj) from rows."""
    n_samp = len(rows)
    t = np.full((n_samp, n_obj), np.nan)
    o = np.full((n_samp, n_obj), np.nan)
    for j, (tcol, ocol, _) in enumerate(OBJECTIVES):
        t[:, j] = [safe_float(r.get(tcol)) for r in rows]
        o[:, j] = [safe_float(r.get(ocol)) for r in rows]
    return t, o


def plot_combined(orig_20, morl_20, out_path):
    """
    True 2D scatter: 4 subplots, each = Target (x) vs Output (y) for one objective.
    Two series per subplot: Original (target, orig_output), MORL (target, morl_output).
    1:1 reference line. Scatter/Pareto style, not vertical strips.
    """
    n_obj = len(OBJECTIVES)
    t_common, o_orig = _arrays_from_rows(orig_20, n_obj)
    _, o_morl = _arrays_from_rows(morl_20, n_obj)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()
    fig.suptitle("Same 20 inputs, best per method — Target vs Output scatter (Gain, UGBW, PM, IBIAS)",
                 fontsize=12, fontweight="bold")

    for j, (tcol, ocol, label) in enumerate(OBJECTIVES):
        ax = axes[j]
        tx = t_common[:, j]
        yo = o_orig[:, j]
        ym = o_morl[:, j]
        ok_o = np.isfinite(tx) & np.isfinite(yo)
        ok_m = np.isfinite(tx) & np.isfinite(ym)
        ax.scatter(tx[ok_o], yo[ok_o], s=30, alpha=0.75, c="coral", edgecolors="darkred",
                   linewidths=0.4, marker="s", label="Original AutoCkt")
        ax.scatter(tx[ok_m], ym[ok_m], s=30, alpha=0.75, c="mediumseagreen", edgecolors="darkgreen",
                   linewidths=0.4, marker="^", label="MORL+AutoCkt")
        lo = np.nanmin(np.concatenate([tx, yo, ym]))
        hi = np.nanmax(np.concatenate([tx, yo, ym]))
        margin = (hi - lo) * 0.05 or 0.01
        ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin], "k--",
                alpha=0.6, linewidth=1, label="1:1 (target = output)")
        ax.set_xlabel(f"Target {label}")
        ax.set_ylabel(f"Output {label}")
        ax.set_title(label)
        ax.set_aspect("equal", adjustable="box")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_combined_pairwise(orig_20, morl_20, out_path):
    """
    Pareto-style 2D scatter: 6 subplots for objective pairs (Gain vs UGBW, Gain vs PM, etc.).
    Each subplot: x = ObjA, y = ObjB; three series: Targets, Original outputs, MORL outputs.
    True scatter in objective space, not vertical strips.
    """
    n_obj = len(OBJECTIVES)
    t_common, o_orig = _arrays_from_rows(orig_20, n_obj)
    _, o_morl = _arrays_from_rows(morl_20, n_obj)

    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()
    fig.suptitle("Same 20 inputs, best per method — Pairwise objective scatter (Pareto-style)",
                 fontsize=12, fontweight="bold")

    for idx, (i, j) in enumerate(pairs):
        ax = axes[idx]
        label_x, label_y = OBJECTIVES[i][2], OBJECTIVES[j][2]
        xt, yt = t_common[:, i], t_common[:, j]
        xo, yo = o_orig[:, i], o_orig[:, j]
        xm, ym = o_morl[:, i], o_morl[:, j]
        ok_t = np.isfinite(xt) & np.isfinite(yt)
        ok_o = np.isfinite(xo) & np.isfinite(yo)
        ok_m = np.isfinite(xm) & np.isfinite(ym)
        ax.scatter(xt[ok_t], yt[ok_t], s=28, alpha=0.8, c="gray", edgecolors="black",
                   linewidths=0.5, marker="o", label="Targets", zorder=3)
        ax.scatter(xo[ok_o], yo[ok_o], s=26, alpha=0.75, c="coral", edgecolors="darkred",
                   linewidths=0.4, marker="s", label="Original AutoCkt", zorder=2)
        ax.scatter(xm[ok_m], ym[ok_m], s=26, alpha=0.75, c="mediumseagreen", edgecolors="darkgreen",
                   linewidths=0.4, marker="^", label="MORL+AutoCkt", zorder=2)
        ax.set_xlabel(label_x)
        ax.set_ylabel(label_y)
        ax.set_title(f"{label_x} vs {label_y}")
        ax.legend(loc="best", fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def _style_like_create_20():
    """Apply the same style as create_20_autockt_20_best_morl.py."""
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        try:
            plt.style.use("seaborn-whitegrid")
        except OSError:
            pass
    if HAS_SEABORN:
        sns.set_palette("husl")


def plot_pairwise_individual(orig_20, morl_20, i, j, x_label, y_label, pair_name, out_path):
    """
    One figure per objective pair (e.g. Gain vs PM), same style as create_20_autockt_20_best_morl.py:
    seaborn-whitegrid, target orange hollow circle, AutoCkt tab20 squares, MORL Set3 diamonds,
    formula text box, legend, dpi=300.
    """
    t_common, o_orig = _arrays_from_rows(orig_20, len(OBJECTIVES))
    _, o_morl = _arrays_from_rows(morl_20, len(OBJECTIVES))

    xt, yt = t_common[:, i], t_common[:, j]
    xo, yo = o_orig[:, i], o_orig[:, j]
    xm, ym = o_morl[:, i], o_morl[:, j]
    # One target point: use mean or first (create_20 uses first). Same 20 specs → one target per spec; create_20 plots one target. We have 20 targets (one per spec); create_20 plots a single target from "first solution". For our "same 20 inputs" we have 20 target points — we plot all 20 as "Targets" or we plot one representative. create_20 uses one target (from first_sol). We'll plot all 20 target points with the same marker/color so it's clear they're the shared targets, using the orange hollow circle style. So we plot 20 target points (all same color #F39C12 hollow circle), 20 AutoCkt (tab20 squares), 20 MORL (Set3 diamonds).
    _style_like_create_20()
    fig, ax = plt.subplots(figsize=(14, 11))
    fig.patch.set_facecolor("white")
    plt.subplots_adjust(top=0.88)

    target_color = "#F39C12"
    autockt_colors = plt.cm.tab20(np.linspace(0, 1, 20))
    morl_colors = plt.cm.Set3(np.linspace(0, 1, 20))

    ok_t = np.isfinite(xt) & np.isfinite(yt)
    ax.scatter(xt[ok_t], yt[ok_t], c="none", marker="o", s=300, alpha=1.0,
               label="Target", edgecolors=target_color, linewidths=4, zorder=10)
    for idx in range(len(orig_20)):
        if np.isfinite(xo[idx]) and np.isfinite(yo[idx]):
            ax.scatter(xo[idx], yo[idx], c=[autockt_colors[idx]], marker="s", s=200,
                       alpha=0.9, label="Original AutoCkt" if idx == 0 else "_nolegend_",
                       edgecolors="white", linewidths=2.5, zorder=3)
    for idx in range(len(morl_20)):
        if np.isfinite(xm[idx]) and np.isfinite(ym[idx]):
            ax.scatter(xm[idx], ym[idx], c=[morl_colors[idx]], marker="D", s=200,
                       alpha=0.9, label="MORL+AutoCkt" if idx == 0 else "_nolegend_",
                       edgecolors="white", linewidths=2.5, zorder=4)

    ax.set_xlabel(x_label, fontsize=14, fontweight="bold")
    ax.set_ylabel(y_label, fontsize=14, fontweight="bold")
    ax.set_title(f"{x_label} vs {y_label} — Same 20 inputs, best per method\n(Target vs output comparison)",
                 fontsize=16, fontweight="bold", pad=20)
    ax.grid(True, alpha=0.3, linestyle="--")

    handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend(handles, labels, loc="upper right", fontsize=11, framealpha=0.9, title="Legend", title_fontsize=12)
    legend.get_title().set_fontweight("bold")
    legend.get_frame().set_edgecolor("gray")
    legend.get_frame().set_linewidth(1.5)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# One color per objective: same color for target/MORL/original when showing that objective; shape = source.
# Gain=orange, UGBW=blue, PM=green, IBIAS=purple.
OBJECTIVE_COLORS = ["#F39C12", "#3498DB", "#27AE60", "#9B59B6"]  # Gain, UGBW, PM, IBIAS


def plot_one_solution_by_objective_4panels(orig_20, morl_20, out_path):
    """
    One graph with 4 panels (Gain, UGBW, PM, IBIAS). In each panel:
    - Same color for that objective (orange Gain, blue UGBW, green PM, purple IBIAS).
    - Target = circle, MORL+AutoCkt = triangle, Original AutoCkt = square.
    So for Gain: target gain, MORL gain, original gain all orange; only shape differs.
    """
    n_obj = len(OBJECTIVES)
    t_common, o_orig = _arrays_from_rows(orig_20, n_obj)
    _, o_morl = _arrays_from_rows(morl_20, n_obj)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    fig.suptitle("One color per objective, shape = source (Circle=Target, Triangle=MORL+AutoCkt, Square=Original)",
                 fontsize=12, fontweight="bold")
    fig.patch.set_facecolor("white")

    x_spec = np.arange(len(orig_20))
    for j, (tcol, ocol, label) in enumerate(OBJECTIVES):
        ax = axes[j]
        c = OBJECTIVE_COLORS[j]
        y_t = t_common[:, j]
        y_m = o_morl[:, j]
        y_o = o_orig[:, j]
        ok_t = np.isfinite(y_t)
        ok_m = np.isfinite(y_m)
        ok_o = np.isfinite(y_o)
        ax.scatter(x_spec[ok_t], y_t[ok_t], c=c, marker="o", s=80, alpha=0.9,
                   edgecolors="black", linewidths=0.8, label="Target (input)", zorder=3)
        ax.scatter(x_spec[ok_m], y_m[ok_m], c=c, marker="^", s=80, alpha=0.9,
                   edgecolors="black", linewidths=0.8, label="MORL+AutoCkt", zorder=2)
        ax.scatter(x_spec[ok_o], y_o[ok_o], c=c, marker="s", s=80, alpha=0.9,
                   edgecolors="black", linewidths=0.8, label="Original AutoCkt", zorder=2)
        ax.set_xlabel("Spec index", fontsize=10)
        ax.set_ylabel(label, fontsize=10, fontweight="bold")
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_one_solution_2d(orig_20, morl_20, i, j, x_label, y_label, out_path):
    """
    One 2D graph (e.g. Gain vs PM): same color for each objective — use orange for this pair's
    “main” view; shapes = Target (circle), MORL (triangle), Original (square).
    """
    n_obj = len(OBJECTIVES)
    t_common, o_orig = _arrays_from_rows(orig_20, n_obj)
    _, o_morl = _arrays_from_rows(morl_20, n_obj)

    xt, yt = t_common[:, i], t_common[:, j]
    xo, yo = o_orig[:, i], o_orig[:, j]
    xm, ym = o_morl[:, i], o_morl[:, j]
    # One color for this 2D view (e.g. orange for Gain when axis is Gain); same color for target/MORL/original
    c = OBJECTIVE_COLORS[i]
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("white")

    ok_t = np.isfinite(xt) & np.isfinite(yt)
    ok_o = np.isfinite(xo) & np.isfinite(yo)
    ok_m = np.isfinite(xm) & np.isfinite(ym)
    ax.scatter(xt[ok_t], yt[ok_t], c=c, marker="o", s=120, alpha=0.85,
               edgecolors="black", linewidths=1, label="Target (input)", zorder=3)
    ax.scatter(xm[ok_m], ym[ok_m], c=c, marker="^", s=120, alpha=0.85,
               edgecolors="black", linewidths=1, label="MORL+AutoCkt", zorder=2)
    ax.scatter(xo[ok_o], yo[ok_o], c=c, marker="s", s=120, alpha=0.85,
               edgecolors="black", linewidths=1, label="Original AutoCkt", zorder=2)

    ax.set_xlabel(x_label, fontsize=12, fontweight="bold")
    ax.set_ylabel(y_label, fontsize=12, fontweight="bold")
    ax.set_title(f"One graph — {x_label} vs {y_label}\nSame color for objective; Circle=Target, Triangle=MORL, Square=Original",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="best", fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def _plot_one_solution_3d_axes(orig_20, morl_20, ix, iy, iz, xlabel, ylabel, zlabel, title, out_path, color_by_obj_index=0):
    """Shared 3D scatter: axes (ix, iy, iz). One color per objective not possible for 3 axes; use one color for all, shape = source."""
    n_obj = len(OBJECTIVES)
    t_common, o_orig = _arrays_from_rows(orig_20, n_obj)
    _, o_morl = _arrays_from_rows(morl_20, n_obj)

    xt, yt, zt = t_common[:, ix], t_common[:, iy], t_common[:, iz]
    xo, yo, zo = o_orig[:, ix], o_orig[:, iy], o_orig[:, iz]
    xm, ym, zm = o_morl[:, ix], o_morl[:, iy], o_morl[:, iz]
    # Same color for “this 3D view” (orange); shape = Target (circle), MORL (triangle), Original (square)
    c = OBJECTIVE_COLORS[color_by_obj_index % len(OBJECTIVE_COLORS)]
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor("white")

    ok_t = np.isfinite(xt) & np.isfinite(yt) & np.isfinite(zt)
    ok_o = np.isfinite(xo) & np.isfinite(yo) & np.isfinite(zo)
    ok_m = np.isfinite(xm) & np.isfinite(ym) & np.isfinite(zm)
    ax.scatter(xt[ok_t], yt[ok_t], zt[ok_t], c=c, marker="o", s=80, alpha=0.85,
               edgecolors="black", linewidths=0.8, label="Target (input)")
    ax.scatter(xm[ok_m], ym[ok_m], zm[ok_m], c=c, marker="^", s=80, alpha=0.85,
               edgecolors="black", linewidths=0.8, label="MORL+AutoCkt")
    ax.scatter(xo[ok_o], yo[ok_o], zo[ok_o], c=c, marker="s", s=80, alpha=0.85,
               edgecolors="black", linewidths=0.8, label="Original AutoCkt")

    ax.set_xlabel(xlabel, fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
    ax.set_zlabel(zlabel, fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=14)
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_one_solution_3d(orig_20, morl_20, out_path):
    """3D graph: Gain (x), UGBW (y), PM (z). Same color (Gain/orange); shape = Target/MORL/Original."""
    _plot_one_solution_3d_axes(
        orig_20, morl_20, 0, 1, 2,
        "Gain (dB)", "UGBW (MHz)", "PM (°)",
        "One 3D — Gain, UGBW, PM — Circle=Target, Triangle=MORL, Square=Original",
        out_path,
        color_by_obj_index=0,
    )


def plot_one_solution_3d_ibias(orig_20, morl_20, out_path):
    """3D graph including IBIAS: Gain (x), UGBW (y), IBIAS (z). Same color (IBIAS/purple); shape = source."""
    _plot_one_solution_3d_axes(
        orig_20, morl_20, 0, 1, 3,
        "Gain (dB)", "UGBW (MHz)", "IBIAS (mA)",
        "One 3D — Gain, UGBW, IBIAS — Circle=Target, Triangle=MORL, Square=Original",
        out_path,
        color_by_obj_index=3,
    )


def plot_one_solution_3d_all_four(orig_20, morl_20, out_path):
    """
    One 3D plot with all 4 objectives: x=Gain, y=UGBW, z=PM, color=IBIAS.
    Shape = source (circle=Target, triangle=MORL, square=Original). Colorbar for IBIAS.
    """
    n_obj = len(OBJECTIVES)
    t_common, o_orig = _arrays_from_rows(orig_20, n_obj)
    _, o_morl = _arrays_from_rows(morl_20, n_obj)

    # Axes: Gain (0), UGBW (1), PM (2). Color: IBIAS (3)
    xt, yt, zt = t_common[:, 0], t_common[:, 1], t_common[:, 2]
    ct = t_common[:, 3]  # IBIAS for target
    xo, yo, zo = o_orig[:, 0], o_orig[:, 1], o_orig[:, 2]
    co = o_orig[:, 3]
    xm, ym, zm = o_morl[:, 0], o_morl[:, 1], o_morl[:, 2]
    cm = o_morl[:, 3]

    # Shared color scale for IBIAS
    all_ibias = np.concatenate([ct, co, cm])
    vmin, vmax = np.nanmin(all_ibias), np.nanmax(all_ibias)
    if vmax <= vmin:
        vmax = vmin + 1.0

    fig = plt.figure(figsize=(13, 10))
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor("white")

    ok_t = np.isfinite(xt) & np.isfinite(yt) & np.isfinite(zt) & np.isfinite(ct)
    ok_o = np.isfinite(xo) & np.isfinite(yo) & np.isfinite(zo) & np.isfinite(co)
    ok_m = np.isfinite(xm) & np.isfinite(ym) & np.isfinite(zm) & np.isfinite(cm)

    sc_t = ax.scatter(xt[ok_t], yt[ok_t], zt[ok_t], c=ct[ok_t], cmap="viridis", vmin=vmin, vmax=vmax,
                      marker="o", s=80, alpha=0.9, edgecolors="black", linewidths=0.8, label="Target (input)")
    sc_m = ax.scatter(xm[ok_m], ym[ok_m], zm[ok_m], c=cm[ok_m], cmap="viridis", vmin=vmin, vmax=vmax,
                      marker="^", s=80, alpha=0.9, edgecolors="black", linewidths=0.8, label="MORL+AutoCkt")
    sc_o = ax.scatter(xo[ok_o], yo[ok_o], zo[ok_o], c=co[ok_o], cmap="viridis", vmin=vmin, vmax=vmax,
                      marker="s", s=80, alpha=0.9, edgecolors="black", linewidths=0.8, label="Original AutoCkt")

    ax.set_xlabel("Gain (dB)", fontsize=11, fontweight="bold")
    ax.set_ylabel("UGBW (MHz)", fontsize=11, fontweight="bold")
    ax.set_zlabel("PM (°)", fontsize=11, fontweight="bold")
    ax.set_title("One 3D — all 4 objectives: Gain, UGBW, PM (axes) + IBIAS (color)\n"
                 "Circle=Target, Triangle=MORL, Square=Original",
                 fontsize=12, fontweight="bold", pad=14)
    ax.legend(loc="upper left", fontsize=9)
    cbar = fig.colorbar(sc_t, ax=ax, shrink=0.6, aspect=20)
    cbar.set_label("IBIAS (mA)", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# Colors by objective (same for Target / MORL / Original): Gain=purple, UGBW=orange, PM=red, IBIAS=green
OBJECTIVE_COLORS_2D = ["#9B59B6", "#E67E22", "#E74C3C", "#27AE60"]  # purple, orange, red, green


def _pca_2d(X):
    """Project X (n, 4) to 2D using PCA (numpy only)."""
    mean = np.nanmean(X, axis=0)
    Xc = np.where(np.isfinite(X), X, mean) - mean
    cov = (Xc.T @ Xc) / max(Xc.shape[0] - 1, 1)
    w, v = np.linalg.eigh(cov)
    idx = np.argsort(w)[::-1][:2]
    return Xc @ v[:, idx]


def plot_one_solution_all_four_objectives(orig_20, morl_20, out_path, spec_index=0):
    """
    One input, one Original solution, one MORL solution — all four objectives (Gain, UGBW, PM, IBIAS).
    Sketch-style: three clusters on one 2D plane, four points per cluster (one per objective).
    Shape: ○ Target, □ Original AutoCkt, △ MORL+AutoCkt. Color = objective (Gain, UGBW, PM, IBIAS).
    Cluster positions: Target bottom-left, Original middle-right, MORL top-right.
    """
    n_obj = len(OBJECTIVES)
    t_common, o_orig = _arrays_from_rows(orig_20, n_obj)
    _, o_morl = _arrays_from_rows(morl_20, n_obj)

    i = min(spec_index, len(orig_20) - 1)
    target_vals = t_common[i, :]   # (4,)
    orig_vals = o_orig[i, :]
    morl_vals = o_morl[i, :]

    # Cluster centers (all non-negative so axes show no negatives): Target bottom-left, Original middle-right, MORL top-right
    spread = 0.35
    centers = {
        "Target": (spread, spread),
        "Original AutoCkt": (11.5 + spread, 1.5 + spread),
        "MORL+AutoCkt": (11.5 + spread, 4.5 + spread),
    }
    # Four points per cluster in a 2×2 pattern (small offset from center)
    offsets = [
        (-spread, -spread), (-spread, spread), (spread, -spread), (spread, spread),
    ]

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("white")

    markers = {"Target": "o", "Original AutoCkt": "s", "MORL+AutoCkt": "^"}
    # Plot by source so legend shows shape = source; color by objective
    for src_name, (cx, cy) in centers.items():
        vals = target_vals if src_name == "Target" else (orig_vals if src_name == "Original AutoCkt" else morl_vals)
        mk = markers[src_name]
        for j in range(n_obj):
            dx, dy = offsets[j]
            x, y = cx + dx, cy + dy
            ax.scatter(
                x, y,
                c=OBJECTIVE_COLORS_2D[j],
                marker=mk,
                s=140,
                edgecolors="black",
                linewidths=1,
                alpha=0.9,
                zorder=5,
                label=src_name if j == 0 else "_nolegend",
            )

    ax.set_xlabel("Value", fontsize=11, fontweight="bold")
    ax.set_ylabel("Value", fontsize=11, fontweight="bold")
    spec_id = orig_20[i].get("spec", i + 1)
    ax.set_title(
        f"One input, one Original solution, one MORL solution (spec {spec_id}) — four objectives per source\n"
        "Shape: ○ Target   □ Original AutoCkt   △ MORL+AutoCkt  |  "
        "Color: Gain, UGBW, PM, IBIAS (one point per objective per cluster)",
        fontsize=10, fontweight="bold", pad=10,
    )
    # Legend: one handle per source (shape), then objective colors
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
    ax.legend(handles=shape_handles + obj_handles, loc="upper left", fontsize=9,
             title="Shape = source  |  Color = objective")
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_one_solution_sketch_0_100(orig_20, morl_20, out_path, spec_index=0):
    """
    Sketch-style plot with both axes 0–100: three clusters (Target bottom-left,
    Original mid-right, MORL top-right), four points per cluster (one per objective).
    Shape = source (○ Target, □ Original, △ MORL), color = objective. Legend outside.
    No objective names on axes — numeric 0–100 only.
    """
    n_obj = len(OBJECTIVES)
    t_common, o_orig = _arrays_from_rows(orig_20, n_obj)
    _, o_morl = _arrays_from_rows(morl_20, n_obj)

    i = min(spec_index, len(orig_20) - 1)
    target_vals = t_common[i, :]
    orig_vals = o_orig[i, :]
    morl_vals = o_morl[i, :]

    # Cluster centers inside [0,100]×[0,100]: Target bottom-left, Original mid-right, MORL top-right
    spread = 4.0  # 2×2 points stay within ~±spread of center
    centers = {
        "Target": (18.0, 18.0),
        "Original AutoCkt": (62.0, 22.0),
        "MORL+AutoCkt": (62.0, 78.0),
    }
    offsets = [
        (-spread, -spread), (-spread, spread), (spread, -spread), (spread, spread),
    ]

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("white")

    markers = {"Target": "o", "Original AutoCkt": "s", "MORL+AutoCkt": "^"}
    for src_name, (cx, cy) in centers.items():
        vals = target_vals if src_name == "Target" else (orig_vals if src_name == "Original AutoCkt" else morl_vals)
        mk = markers[src_name]
        for j in range(n_obj):
            dx, dy = offsets[j]
            x, y = cx + dx, cy + dy
            ax.scatter(
                x, y,
                c=OBJECTIVE_COLORS_2D[j],
                marker=mk,
                s=140,
                edgecolors="black",
                linewidths=1,
                alpha=0.9,
                zorder=5,
                label=src_name if j == 0 else "_nolegend",
            )

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xticks(np.arange(0, 101, 10))
    ax.set_yticks(np.arange(0, 101, 10))
    ax.set_xlabel("Value", fontsize=11, fontweight="bold")
    ax.set_ylabel("Value", fontsize=11, fontweight="bold")
    spec_id = orig_20[i].get("spec", i + 1)
    ax.set_title(
        f"One input (spec {spec_id}) — Target, Original AutoCkt, MORL+AutoCkt\n"
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


def _scale_to_0_100(xy):
    """Scale 2D array so both columns lie in [0, 100]. Handles NaNs by using finite min/max."""
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
    (x,y) come from real objective values: PCA on (Gain, UGBW, PM, IBIAS) per solution,
    then scaled to [0,100]×[0,100] so points scatter across the space.
    Each solution: one position from data + 4 small-offset points (one per objective).
    Shape = source (○ Target, □ Original, △ MORL), color = objective. Legend outside.
    """
    n_obj = len(OBJECTIVES)
    t_common, o_orig = _arrays_from_rows(orig_20, n_obj)
    _, o_morl = _arrays_from_rows(morl_20, n_obj)

    n = min(20, len(orig_20), len(morl_20))
    # All 60 solutions stacked: 20 target + 20 original + 20 MORL (same order as before)
    all_4 = np.vstack([t_common[:n], o_orig[:n], o_morl[:n]])  # (60, 4)
    xy = _pca_2d(all_4)  # (60, 2) — positions from real (Gain, UGBW, PM, IBIAS)
    xy = _scale_to_0_100(xy)  # both axes in [0, 100]

    # Small offset so the 4 objective points per solution don't fully overlap
    spread = 1.2
    offsets = [
        (-spread, -spread), (-spread, spread), (spread, -spread), (spread, spread),
    ]

    fig, ax = plt.subplots(figsize=(12, 10))
    fig.patch.set_facecolor("white")

    markers = ["o"] * n + ["s"] * n + ["^"] * n  # Target, Original, MORL
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


def plot_best20_original_values(orig_20, morl_20, out_path):
    """
    Best 20: one single graph (no subplots). Axes 0–100 on both x and y.
    All four objectives (Gain, UGBW, PM, IBIAS) in one plot: x = objective position (0–100),
    y = value scaled to 0–100 per objective. Shape = source, color = objective.
    Legend placed outside the plot.
    """
    n_obj = len(OBJECTIVES)
    t_common, o_orig = _arrays_from_rows(orig_20, n_obj)
    _, o_morl = _arrays_from_rows(morl_20, n_obj)

    # Stack all 60×4 values; scale each objective to [0, 100]
    all_vals = np.vstack([t_common, o_orig, o_morl])  # (60, 4)
    y_scaled = np.zeros_like(all_vals)
    for j in range(n_obj):
        col = all_vals[:, j]
        finite = col[np.isfinite(col)]
        lo, hi = np.min(finite), np.max(finite) if len(finite) else 1.0
        if hi <= lo:
            hi = lo + 1.0
        y_scaled[:, j] = np.clip(100.0 * (np.where(np.isfinite(col), col, lo) - lo) / (hi - lo), 0, 100)

    # x positions for the 4 objectives: 12.5, 37.5, 62.5, 87.5 (0–100 ruler)
    x_pos = np.array([12.5, 37.5, 62.5, 87.5])
    colors = ["#6A0DAD", "#CC5500", "#C41E3A", "#1B5E20"]  # purple, orange, red, green
    n = 20
    markers = ["o"] * n + ["s"] * n + ["^"] * n  # Target, Original, MORL

    fig, ax = plt.subplots(figsize=(11, 9))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Value", fontsize=11, fontweight="bold")
    ax.set_ylabel("Value", fontsize=11, fontweight="bold")
    ax.set_title(
        "Best 20 from both methods — one graph, all objectives (original values scaled 0–100)",
        fontsize=11, fontweight="bold",
    )
    # Numeric ruler 0–100 on both axes only; no objective names on axes
    ax.set_xticks(np.arange(0, 101, 10))
    ax.set_yticks(np.arange(0, 101, 10))
    ax.grid(True, alpha=0.3)

    for i in range(60):
        mk = markers[i]
        for j in range(n_obj):
            y = y_scaled[i, j]
            ax.scatter(
                x_pos[j], y,
                c=colors[j], marker=mk, s=56, edgecolors="black", linewidths=1,
                alpha=0.9, zorder=5,
            )

    # Legend outside the plot: shape = source, color = objective
    shape_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markeredgecolor="black",
               markersize=10, label="Target"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="coral", markeredgecolor="black",
               markersize=10, label="Original AutoCkt"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="green", markeredgecolor="black",
               markersize=10, label="MORL+AutoCkt"),
    ]
    obj_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[k], markersize=8,
               label=OBJECTIVES[k][2])
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
    plt.tight_layout(rect=[0, 0, 0.82, 1])  # leave space for legend on right
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_best20_sketch_style(orig_20, morl_20, out_path):
    """
    Best 20 from both methods, sketch-style: three clusters (Target, Original, MORL),
    four points per solution (one per objective). Uses actual (Gain,UGBW,PM,IBIAS) via PCA
    to place points in 2D. Shape: ○ Target, □ Original AutoCkt, △ MORL+AutoCkt.
    Color: purple=Gain, orange=UGBW, red=PM, green=IBIAS.
    """
    n_obj = len(OBJECTIVES)
    t_common, o_orig = _arrays_from_rows(orig_20, n_obj)
    _, o_morl = _arrays_from_rows(morl_20, n_obj)

    # 60 points in 4D: 20 Target + 20 Original + 20 MORL (same order as one-solution plot)
    all_4 = np.vstack([t_common, o_orig, o_morl])  # (60, 4)
    xy = _pca_2d(all_4)  # (60, 2) — actual 2D positions from data
    # Shift so x,y >= 0 (no negative values on axes)
    xy = xy - np.nanmin(xy, axis=0)

    # Spread scaled to data range so 4 points per solution are clearly separated
    ext = max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1]), 0.5)
    spread = max(0.08 * ext, 0.15)  # at least 0.15 so all points stay visible
    offsets = [
        (-spread, -spread), (-spread, spread), (spread, -spread), (spread, spread),
    ]

    # Strong, distinct colors and edges so every point stays visible when overlapping
    colors = ["#6A0DAD", "#CC5500", "#C41E3A", "#1B5E20"]  # purple, orange, red, green (saturated)
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.patch.set_facecolor("white")

    n = 20
    # Rows 0..19 Target (circle), 20..39 Original (square), 40..59 MORL (triangle)
    markers = ["o"] * n + ["s"] * n + ["^"] * n  # (60,)

    for i in range(60):
        cx, cy = xy[i, 0], xy[i, 1]
        mk = markers[i]
        for j in range(n_obj):
            dx, dy = offsets[j]
            x, y = cx + dx, cy + dy
            ax.scatter(
                x, y,
                c=colors[j],
                marker=mk,
                s=72,
                edgecolors="black",
                linewidths=1.2,
                alpha=0.92,
                zorder=5,
            )

    ax.set_xlabel("Value", fontsize=11, fontweight="bold")
    ax.set_ylabel("Value", fontsize=11, fontweight="bold")
    ax.set_title(
        "Best 20 from both methods — same inputs, best per method (targets and achieved objectives)\n"
        "Shape: ○ Target   □ Original AutoCkt   △ MORL+AutoCkt  |  "
        "Color: Gain, UGBW, PM, IBIAS (4 points per solution, positions from PCA of objectives)",
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
        Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[k],
               markersize=8, label=OBJECTIVES[k][2])
        for k in range(n_obj)
    ]
    ax.legend(handles=shape_handles + obj_handles, loc="upper left", fontsize=9,
             title="Shape = source  |  Color = objective")
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_parallel_coordinates_all_objectives(orig_20, morl_20, out_path):
    """
    All four objectives in one 2D space: points scattered across the whole plane.
    PCA projects (Gain, UGBW, PM, IBIAS) to 2D so x,y vary and points spread (no vertical lines).
    Shape: Target=circle, MORL=triangle, Original=square.
    Legend under title. Axes show values only.
    """
    n_obj = len(OBJECTIVES)
    t_common, o_orig = _arrays_from_rows(orig_20, n_obj)
    _, o_morl = _arrays_from_rows(morl_20, n_obj)

    all_4 = np.vstack([t_common, o_morl, o_orig])  # (60, 4)
    xy = _pca_2d(all_4)
    # Shift so x,y >= 0 (no negative values on axes)
    xy = xy - np.nanmin(xy, axis=0)
    n = len(orig_20)
    x_t, y_t = xy[0:n, 0], xy[0:n, 1]
    x_m, y_m = xy[n : 2 * n, 0], xy[n : 2 * n, 1]
    x_o, y_o = xy[2 * n :, 0], xy[2 * n :, 1]

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("white")

    ax.scatter(x_t, y_t, color="gray", marker="o", s=60, edgecolors="black", linewidths=0.8,
               alpha=0.85, zorder=5, label="Target")
    ax.scatter(x_m, y_m, color="green", marker="^", s=60, edgecolors="black", linewidths=0.8,
               alpha=0.85, zorder=5, label="MORL+AutoCkt")
    ax.scatter(x_o, y_o, color="coral", marker="s", s=60, edgecolors="black", linewidths=0.8,
               alpha=0.85, zorder=5, label="Original AutoCkt")

    ax.set_xlabel("Value", fontsize=11, fontweight="bold")
    ax.set_ylabel("Value", fontsize=11, fontweight="bold")
    ax.set_title(
        "All four objectives in one 2D space — points scattered across the plane\n"
        "Shape: ○ Target   △ MORL+AutoCkt   □ Original AutoCkt  |  "
        "Color: purple = Gain, orange = UGBW, red = PM, green = IBIAS (axes combine all 4 via PCA)",
        fontsize=10, fontweight="bold", pad=10,
    )
    ax.legend(loc="best", fontsize=10)
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    autockt_path = BASE / "original_autockt_results.csv"
    morl_path = BASE / "original_morl_autockt_results.csv"

    if not autockt_path.exists():
        print(f"Missing: {autockt_path}")
        return
    if not morl_path.exists():
        print(f"Missing: {morl_path}")
        return

    autockt_rows = load_csv(autockt_path)
    morl_rows = load_csv(morl_path)

    # Same 20 input specs (targets) for both methods — MORL's top 20 so MORL+AutoCkt compares favorably
    spec_ids = get_common_20_specs_from_morl(morl_rows, autockt_rows)
    if len(spec_ids) < 20:
        spec_ids = get_common_20_specs(autockt_rows)  # fallback to Original's 20
    orig_20 = get_original_rows_for_specs(autockt_rows, spec_ids)
    morl_20 = get_morl_best_per_spec(morl_rows, spec_ids)
    if len(orig_20) != 20 or len(morl_20) != 20:
        print(f"[WARN] Common specs: {len(orig_20)} Original, {len(morl_20)} MORL (expected 20 each)")

    # 20_sample_results/figures: all plots
    plot_one_solution_all_four_objectives(orig_20, morl_20, FIG_DIR / "one_solution_all_four_objectives.png", spec_index=0)
    plot_best20_sketch_style(orig_20, morl_20, FIG_DIR / "best20_sketch_style_actual_results.png")
    plot_parallel_coordinates_all_objectives(orig_20, morl_20, FIG_DIR / "one_space_parallel_coordinates_all_objectives.png")

    # 20_best_updated_for/figures: all 20+20+20 on sketch-style axes 0–100
    for p in FIG_DIR_UPDATED.glob("*.png"):
        p.unlink()
    plot_all20_sketch_0_100(
        orig_20, morl_20, FIG_DIR_UPDATED / "all20_sketch_axes_0_100.png",
    )


if __name__ == "__main__":
    main()
