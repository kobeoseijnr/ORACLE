"""
Compute Hypervolume and Sparsity: Original AutoCkt vs MORL Cosine.

Hypervolume (HV): Volume of objective space dominated by the Pareto front.
  - Higher HV = better coverage of trade-off space.
  - Single solution (Original AutoCkt) → HV = box volume from point to reference.
  - Multiple solutions (MORL) → HV captures quality AND diversity.

Sparsity: Measures spread of solutions along the Pareto front.
  - SP = (1/(|F|-1)) * sum_m ( sum_i (f_m^i - f_m^{i+1})^2 )
  - Lower = more uniform spread. 0 for single-point fronts.

Objectives (4D, converted to minimization):
  gain  (maximize) → negate
  ugbw  (maximize) → negate
  pm    (maximize) → negate
  ibias (minimize) → keep

Sources:
  1. Original AutoCkt — 1 solution per spec (single-objective RL)
  2. MORL Cosine DDQN — 10 solutions per spec (multi-objective RL)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import io

# ---------------------------------------------------------------------------
# Paths (Option A: match with_15%_final_New/morl_autockt/compute_hv_sparsity.py)
# ---------------------------------------------------------------------------
MORL_RESULTS = Path(r"C:\Users\kobeo\OneDrive\Desktop\trail\with_15%_final_New\with_15%\morl_autockt\results")
ORIG_RESULTS = Path(r"C:\Users\kobeo\OneDrive\Desktop\trail\with_15%_final_New\with_15%\original_autockt\results")

ORIG_JSON = ORIG_RESULTS / "original_autockt_results_15percent.json"
COS_JSON = MORL_RESULTS / "morl_raw_cosine_agent.json"
NW_JSON = MORL_RESULTS / "morl_raw_nw_agent.json"

# Included for reference-point consistency with historical summary
LLM_COS_JSON = MORL_RESULTS / "morl_raw_llm_cosine_agent.json"
LLM_NW_JSON = MORL_RESULTS / "morl_raw_llm_nw_agent.json"

OUTPUT_DIR = MORL_RESULTS


# ---------------------------------------------------------------------------
# Pareto front extraction (minimization)
# ---------------------------------------------------------------------------
def pareto_front(points):
    """Extract non-dominated points assuming minimization."""
    if len(points) == 0:
        return np.array([]).reshape(0, 4)
    if len(points) == 1:
        return points.copy()
    pts = np.array(points)
    n = len(pts)
    is_dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        if is_dominated[i]:
            continue
        for j in range(n):
            if i == j or is_dominated[j]:
                continue
            # j dominates i if j <= i in all dims AND j < i in at least one
            if np.all(pts[j] <= pts[i]) and np.any(pts[j] < pts[i]):
                is_dominated[i] = True
                break
    front = pts[~is_dominated]
    return front if len(front) > 0 else pts[:1]


# ---------------------------------------------------------------------------
# Hypervolume (Monte Carlo, 4D)
# ---------------------------------------------------------------------------
def hypervolume_mc(points, ref, n_samples=50000):
    """Monte Carlo hypervolume estimation."""
    if len(points) == 0:
        return 0.0
    pts = np.array(points)
    n, dim = pts.shape

    # Single point: exact box volume
    if n == 1:
        box = ref - pts[0]
        return float(np.prod(np.maximum(box, 0)))

    ideal = pts.min(axis=0)
    vol_box = np.prod(ref - ideal)
    if vol_box <= 0:
        return 0.0

    samples = np.random.uniform(ideal, ref, size=(n_samples, dim))

    # Check if any Pareto point dominates each sample
    dominated = np.zeros(n_samples, dtype=bool)
    for p in pts:
        dominated |= np.all(samples >= p, axis=1)

    return vol_box * dominated.sum() / n_samples


# ---------------------------------------------------------------------------
# Hypervolume (Option A: Monte Carlo with 10k samples, matches historical script)
# ---------------------------------------------------------------------------
def hypervolume_fast(points, ref, n_samples=10000):
    """Monte Carlo hypervolume with 10k samples (matches historical compute_hv_sparsity.py)."""
    if len(points) == 0:
        return 0.0
    pts = np.array(points)
    n, dim = pts.shape

    if n == 1:
        box = ref - pts[0]
        return float(np.prod(np.maximum(box, 0)))

    ideal = pts.min(axis=0)
    vol_box = np.prod(ref - ideal)
    if vol_box <= 0:
        return 0.0

    samples = np.random.uniform(ideal, ref, size=(n_samples, dim))
    dominated = np.zeros(n_samples, dtype=bool)
    for p in pts:
        dominated |= np.all(samples >= p, axis=1)
    return vol_box * dominated.sum() / n_samples


# ---------------------------------------------------------------------------
# Sparsity metric
# ---------------------------------------------------------------------------
def sparsity(front):
    """
    Sparsity of a Pareto front.
    SP = (1/(|F|-1)) * sum_m ( sum_i (f_m^i - f_m^{i+1})^2 )
    Lower = more uniform. 0 for single-point fronts.
    """
    n = len(front)
    if n <= 1:
        return 0.0
    dim = front.shape[1]
    sp = 0.0
    for m in range(dim):
        sorted_vals = np.sort(front[:, m])
        gaps = np.diff(sorted_vals)
        sp += np.sum(gaps ** 2)
    return sp / (n - 1)


# ---------------------------------------------------------------------------
# Convert to minimization objectives
# ---------------------------------------------------------------------------
def to_min_obj(row):
    """Convert row to minimization-space: [-gain, -ugbw, -pm, ibias]."""
    g = float(row['output_gain_linear'])
    u = float(row['output_ugbw_mhz'])
    p = float(row['output_pm_deg'])
    i = float(row['output_ibias_ma'])
    return np.array([-g, -u, -p, i])


def to_min_objectives_sol(sol):
    """Convert solution dict to minimization-space array: [-gain, -ugbw, -pm, ibias]."""
    g = float(sol['output_gain_linear'])
    u = float(sol['output_ugbw_mhz'])
    p = float(sol['output_pm_deg'])
    i = float(sol['output_ibias_ma'])
    return np.array([-g, -u, -p, i])


# ---------------------------------------------------------------------------
# Load CSVs
# ---------------------------------------------------------------------------
def load_csv_clean(path):
    """Load CSV, dropping summary/non-numeric rows."""
    with open(path) as f:
        lines = f.readlines()
    header = lines[0]
    data_lines = [l for l in lines[1:] if l.strip() and not l.startswith('summary') and not l.startswith('checksum')]
    df = pd.read_csv(io.StringIO(header + ''.join(data_lines)))
    return df


def group_by_spec_1indexed(df):
    """Group rows by spec, returning a dict keyed by 1-indexed spec IDs."""
    spec_raw = pd.to_numeric(df["spec"], errors="coerce")
    df = df.loc[spec_raw.notna()].copy()
    df["spec"] = spec_raw.loc[spec_raw.notna()].astype(int)

    by_spec = {}
    # Heuristic: if dataset is 0-indexed (min spec == 0), shift to 1-indexed.
    shift = 1 if df["spec"].min() == 0 else 0
    for _, row in df.iterrows():
        spec = int(row["spec"]) + shift
        by_spec.setdefault(spec, []).append(row)
    return by_spec


def load_morl_solutions(json_path):
    """Load and group MORL solutions by spec from a raw results JSON."""
    with open(json_path) as f:
        data = json.load(f)
    by_spec = {}
    for sol in data.get('all_solutions', []):
        spec = int(sol['spec'])
        by_spec.setdefault(spec, []).append(sol)
    return by_spec


def load_original_solutions(json_path):
    """Load original AutoCkt solutions (1 per spec) and convert spec to 1-indexed."""
    with open(json_path) as f:
        data = json.load(f)
    by_spec = {}
    for sol in data:
        spec = int(sol['spec']) + 1
        by_spec[spec] = [sol]
    return by_spec


def compute_metrics_per_spec(by_spec, ref_point):
    """Compute HV/sparsity/front size per spec; returns arrays."""
    hvs = []
    sps = []
    fronts = []
    for spec in sorted(by_spec.keys()):
        sols = by_spec[spec]
        points = []
        for sol in sols:
            try:
                pt = to_min_objectives_sol(sol)
            except (KeyError, TypeError, ValueError):
                continue
            if all(pt[d] < ref_point[d] for d in range(len(ref_point))):
                points.append(pt)

        if not points:
            hvs.append(0.0)
            sps.append(0.0)
            fronts.append(0)
            continue

        points = np.array(points)
        front = pareto_front(points)
        fronts.append(len(front))
        hvs.append(hypervolume_fast(front, ref_point))
        sps.append(sparsity(front))
    return np.array(hvs), np.array(sps), np.array(fronts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    np.random.seed(42)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  HYPERVOLUME & SPARSITY: AutoCkt vs MORL Cosine vs MORL NW")
    print("=" * 70)

    # Load data (Option A: JSON raw solutions)
    orig_by_spec = load_original_solutions(ORIG_JSON)
    cos_by_spec = load_morl_solutions(COS_JSON)
    nw_by_spec = load_morl_solutions(NW_JSON)
    llm_cos_by_spec = load_morl_solutions(LLM_COS_JSON) if LLM_COS_JSON.exists() else {}
    llm_nw_by_spec = load_morl_solutions(LLM_NW_JSON) if LLM_NW_JSON.exists() else {}

    print(f"\n  Original AutoCkt: {sum(len(v) for v in orig_by_spec.values())} rows (1 per spec)")
    print(f"  MORL Cosine:      {sum(len(v) for v in cos_by_spec.values())} rows")
    print(f"  MORL NW:          {sum(len(v) for v in nw_by_spec.values())} rows")

    # Common specs across the three requested methods
    common_specs = sorted(set(orig_by_spec.keys()) & set(cos_by_spec.keys()) & set(nw_by_spec.keys()))
    print(f"  Common specs: {len(common_specs)}")

    # --- Determine reference point (match historical compute_hv_sparsity.py) ---
    all_points = []
    for by_spec in [orig_by_spec, cos_by_spec, nw_by_spec, llm_cos_by_spec, llm_nw_by_spec]:
        for sols in by_spec.values():
            for sol in sols:
                try:
                    all_points.append(to_min_objectives_sol(sol))
                except (KeyError, TypeError, ValueError):
                    continue
    all_points = np.array(all_points)
    ref_point = all_points.max(axis=0) * 1.1
    for d in range(4):
        ref_point[d] = max(ref_point[d], all_points[:, d].max() + abs(all_points[:, d].max()) * 0.1)

    print(f"\n  Reference point (min-space): [{ref_point[0]:.4f}, {ref_point[1]:.4f}, {ref_point[2]:.4f}, {ref_point[3]:.4f}]")
    print(f"    = [gain≤{-ref_point[0]:.1f}, ugbw≤{-ref_point[1]:.2f}MHz, pm≤{-ref_point[2]:.1f}°, ibias≥{ref_point[3]:.3f}mA]")

    # --- Compute per-spec metrics ---
    print("\n  Computing HV & Sparsity per spec...\n")

    orig_common = {s: orig_by_spec[s] for s in common_specs}
    cos_common = {s: cos_by_spec[s] for s in common_specs}
    nw_common = {s: nw_by_spec[s] for s in common_specs}

    orig_hvs, orig_sps, orig_fronts = compute_metrics_per_spec(orig_common, ref_point)
    cos_hvs, cos_sps, cos_fronts = compute_metrics_per_spec(cos_common, ref_point)
    nw_hvs, nw_sps, nw_fronts = compute_metrics_per_spec(nw_common, ref_point)

    results = []
    for i, spec in enumerate(common_specs):
        hv_impr_cos = float(cos_hvs[i] - orig_hvs[i])
        hv_impr_nw = float(nw_hvs[i] - orig_hvs[i])
        hv_impr_nw_cos = float(nw_hvs[i] - cos_hvs[i])
        results.append({
            'spec': spec,
            # Legacy column names (used by generate_report.py / generate_conference_report.py)
            'hv_original': float(orig_hvs[i]),
            'hv_morl_cosine': float(cos_hvs[i]),
            'hv_improvement': hv_impr_cos,
            'sparsity_original': float(orig_sps[i]),
            'sparsity_morl_cosine': float(cos_sps[i]),
            'front_size_original': int(orig_fronts[i]),
            'front_size_morl_cosine': int(cos_fronts[i]),
            # New 3-way columns
            'hv_autockt': float(orig_hvs[i]),
            'hv_morl_nw': float(nw_hvs[i]),
            'hv_improvement_cosine_vs_autockt': hv_impr_cos,
            'hv_improvement_nw_vs_autockt': hv_impr_nw,
            'hv_improvement_nw_vs_cosine': hv_impr_nw_cos,
            'sparsity_autockt': float(orig_sps[i]),
            'sparsity_morl_nw': float(nw_sps[i]),
            'front_size_autockt': int(orig_fronts[i]),
            'front_size_morl_nw': int(nw_fronts[i]),
        })

    # --- Print results ---
    print("=" * 70)
    print("  RESULTS")
    print("=" * 70)

    header = f"{'Metric':<35} {'AutoCkt':>20} {'MORL Cosine':>20} {'MORL NW':>20}"
    print(header)
    print("-" * len(header))

    rows = [
        ("Mean Hypervolume",         f"{orig_hvs.mean():.6e}", f"{cos_hvs.mean():.6e}", f"{nw_hvs.mean():.6e}"),
        ("Std Hypervolume",          f"{orig_hvs.std():.6e}",  f"{cos_hvs.std():.6e}",  f"{nw_hvs.std():.6e}"),
        ("Median Hypervolume",       f"{np.median(orig_hvs):.6e}", f"{np.median(cos_hvs):.6e}", f"{np.median(nw_hvs):.6e}"),
        ("Mean Sparsity",            f"{orig_sps.mean():.6e}", f"{cos_sps.mean():.6e}", f"{nw_sps.mean():.6e}"),
        ("Std Sparsity",             f"{orig_sps.std():.6e}",  f"{cos_sps.std():.6e}",  f"{nw_sps.std():.6e}"),
        ("Median Sparsity",          f"{np.median(orig_sps):.6e}", f"{np.median(cos_sps):.6e}", f"{np.median(nw_sps):.6e}"),
        ("Mean Pareto Front Size",   f"{np.mean(orig_fronts):.2f}", f"{np.mean(cos_fronts):.2f}", f"{np.mean(nw_fronts):.2f}"),
        ("Specs where Cos HV > AutoCkt", "", f"{(cos_hvs > orig_hvs).sum()}/{len(common_specs)}", ""),
        ("Specs where NW HV > AutoCkt", "", "", f"{(nw_hvs > orig_hvs).sum()}/{len(common_specs)}"),
        ("Specs where NW HV > Cos", "", "", f"{(nw_hvs > cos_hvs).sum()}/{len(common_specs)}"),
    ]
    for name, v1, v2, v3 in rows:
        print(f"  {name:<33} {v1:>20} {v2:>20} {v3:>20}")

    # HV improvement
    if orig_hvs.mean() > 0:
        pct_cos = (cos_hvs.mean() - orig_hvs.mean()) / orig_hvs.mean() * 100
        pct_nw = (nw_hvs.mean() - orig_hvs.mean()) / orig_hvs.mean() * 100
        print(f"\n  MORL Cosine HV is {pct_cos:+.1f}% vs AutoCkt")
        print(f"  MORL NW HV is     {pct_nw:+.1f}% vs AutoCkt")

    # --- Statistical tests ---
    print(f"\n{'=' * 70}")
    print("  STATISTICAL TESTS")
    print(f"{'=' * 70}")
    try:
        from scipy import stats
        # Paired t-test on HV
        t_stat, t_pval = stats.ttest_rel(cos_hvs, orig_hvs)
        print(f"  Paired t-test (HV, Cos vs AutoCkt): t={t_stat:.4f}, p={t_pval:.2e}", end="")
        print("  **SIGNIFICANT**" if t_pval < 0.05 else "  not significant")

        t_stat_nw, t_pval_nw = stats.ttest_rel(nw_hvs, orig_hvs)
        print(f"  Paired t-test (HV, NW  vs AutoCkt): t={t_stat_nw:.4f}, p={t_pval_nw:.2e}", end="")
        print("  **SIGNIFICANT**" if t_pval_nw < 0.05 else "  not significant")

        # Wilcoxon signed-rank test on HV
        try:
            w_stat, w_pval = stats.wilcoxon(cos_hvs, orig_hvs)
            print(f"  Wilcoxon (HV, Cos vs AutoCkt): W={w_stat:.0f}, p={w_pval:.2e}", end="")
            print("  **SIGNIFICANT**" if w_pval < 0.05 else "  not significant")

            w_stat_nw, w_pval_nw = stats.wilcoxon(nw_hvs, orig_hvs)
            print(f"  Wilcoxon (HV, NW  vs AutoCkt): W={w_stat_nw:.0f}, p={w_pval_nw:.2e}", end="")
            print("  **SIGNIFICANT**" if w_pval_nw < 0.05 else "  not significant")
        except Exception as e:
            print(f"  Wilcoxon: {e}")

        # Paired t-test on Sparsity
        t_stat2, t_pval2 = stats.ttest_rel(cos_sps, orig_sps)
        print(f"\n  Paired t-test (Sparsity, Cos vs AutoCkt): t={t_stat2:.4f}, p={t_pval2:.2e}", end="")
        print("  **SIGNIFICANT**" if t_pval2 < 0.05 else "  not significant")

        t_stat2_nw, t_pval2_nw = stats.ttest_rel(nw_sps, orig_sps)
        print(f"  Paired t-test (Sparsity, NW  vs AutoCkt): t={t_stat2_nw:.4f}, p={t_pval2_nw:.2e}", end="")
        print("  **SIGNIFICANT**" if t_pval2_nw < 0.05 else "  not significant")

    except ImportError:
        print("  (scipy not available for statistical tests)")

    # --- Save CSV ---
    # Add summary rows
    results.append({
        'spec': 'MEAN',
        'hv_original': orig_hvs.mean(),
        'hv_morl_cosine': cos_hvs.mean(),
        'hv_improvement': (cos_hvs - orig_hvs).mean(),
        'sparsity_original': orig_sps.mean(),
        'sparsity_morl_cosine': cos_sps.mean(),
        'front_size_original': np.mean(orig_fronts),
        'front_size_morl_cosine': np.mean(cos_fronts),
        'hv_autockt': orig_hvs.mean(),
        'hv_morl_nw': nw_hvs.mean(),
        'hv_improvement_cosine_vs_autockt': (cos_hvs - orig_hvs).mean(),
        'hv_improvement_nw_vs_autockt': (nw_hvs - orig_hvs).mean(),
        'hv_improvement_nw_vs_cosine': (nw_hvs - cos_hvs).mean(),
        'sparsity_autockt': orig_sps.mean(),
        'sparsity_morl_nw': nw_sps.mean(),
        'front_size_autockt': np.mean(orig_fronts),
        'front_size_morl_nw': np.mean(nw_fronts),
    })
    results.append({
        'spec': 'MEDIAN',
        'hv_original': np.median(orig_hvs),
        'hv_morl_cosine': np.median(cos_hvs),
        'hv_improvement': np.median(cos_hvs - orig_hvs),
        'sparsity_original': np.median(orig_sps),
        'sparsity_morl_cosine': np.median(cos_sps),
        'front_size_original': np.median(orig_fronts),
        'front_size_morl_cosine': np.median(cos_fronts),
        'hv_autockt': np.median(orig_hvs),
        'hv_morl_nw': np.median(nw_hvs),
        'hv_improvement_cosine_vs_autockt': np.median(cos_hvs - orig_hvs),
        'hv_improvement_nw_vs_autockt': np.median(nw_hvs - orig_hvs),
        'hv_improvement_nw_vs_cosine': np.median(nw_hvs - cos_hvs),
        'sparsity_autockt': np.median(orig_sps),
        'sparsity_morl_nw': np.median(nw_sps),
        'front_size_autockt': np.median(orig_fronts),
        'front_size_morl_nw': np.median(nw_fronts),
    })
    results.append({
        'spec': 'STD',
        'hv_original': orig_hvs.std(),
        'hv_morl_cosine': cos_hvs.std(),
        'hv_improvement': (cos_hvs - orig_hvs).std(),
        'sparsity_original': orig_sps.std(),
        'sparsity_morl_cosine': cos_sps.std(),
        'hv_autockt': orig_hvs.std(),
        'hv_morl_nw': nw_hvs.std(),
        'hv_improvement_cosine_vs_autockt': (cos_hvs - orig_hvs).std(),
        'hv_improvement_nw_vs_autockt': (nw_hvs - orig_hvs).std(),
        'hv_improvement_nw_vs_cosine': (nw_hvs - cos_hvs).std(),
        'sparsity_autockt': orig_sps.std(),
        'sparsity_morl_nw': nw_sps.std(),
    })

    df_out = pd.DataFrame(results)
    csv_path = OUTPUT_DIR / "hypervolume_sparsity_comparison.csv"
    df_out.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")
    print("DONE")


if __name__ == "__main__":
    main()
