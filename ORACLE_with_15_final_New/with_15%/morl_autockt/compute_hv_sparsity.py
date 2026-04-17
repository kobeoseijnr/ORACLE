"""
Compute Hypervolume and Sparsity metrics for MORL vs Original AutoCkt.

Hypervolume (HV): Volume of objective space dominated by the Pareto front.
  - Higher HV = better coverage of the trade-off space.
  - For a single solution (AutoCkt), HV is just the box from that point to the reference.
  - For multiple solutions (MORL), HV captures both quality AND diversity.

Sparsity: Measures how spread out solutions are along the Pareto front.
  - Computed as: (1/(|F|-1)) * sum over objectives of sum of consecutive squared gaps.
  - Lower sparsity = more evenly distributed solutions.
  - Undefined (0) for single-point fronts.

Objectives (4D):
  gain  (maximize) → negate for minimization
  ugbw  (maximize) → negate for minimization
  pm    (maximize) → negate for minimization
  ibias (minimize) → keep as-is

Comparison:
  1. Original AutoCkt (single-objective RL) — 1 solution per spec
  2. MORL Standard DDQN (NW + Cosine combined) — up to 20 solutions per spec
  3. MORL LLM-Guided DDQN (NW + Cosine combined) — up to 20 solutions per spec
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
# (no extra imports needed)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MORL_RESULTS = BASE_DIR / "results"
ORIG_RESULTS = BASE_DIR.parent / "original_autockt" / "results"

# ---------------------------------------------------------------------------
# Pareto front extraction
# ---------------------------------------------------------------------------
def pareto_front(points):
    """Extract non-dominated points (minimization) using vectorized numpy."""
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
        # Check if any other point dominates i
        others = pts[~is_dominated]
        dom_mask = np.all(others <= pts[i], axis=1) & np.any(others < pts[i], axis=1)
        if dom_mask.any():
            is_dominated[i] = True
    front = pts[~is_dominated]
    return front if len(front) > 0 else pts[:1]


# ---------------------------------------------------------------------------
# Hypervolume (exact, 4D)
# ---------------------------------------------------------------------------
def hypervolume_fast(points, ref):
    """
    Fast hypervolume via Monte Carlo with vectorized dominance checking.
    Uses 10k samples with fully vectorized numpy — fast enough for 1000 specs.
    """
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

    n_samples = 10000
    samples = np.random.uniform(ideal, ref, size=(n_samples, dim))

    # Vectorized: for each sample, check if ANY point dominates it
    # pts[i] dominates sample[j] if pts[i] <= sample[j] for all dims
    # Shape: (n_samples, n_points, dim)
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
    where solutions are sorted per objective.
    Lower = more uniform spread. 0 for single-point fronts.
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
# Load solutions and normalize to minimization objectives
# ---------------------------------------------------------------------------
def to_min_objectives(sol):
    """Convert a solution dict to minimization-space array: [-gain, -ugbw, -pm, ibias]."""
    g = float(sol['output_gain_linear'])
    u = float(sol['output_ugbw_mhz'])
    p = float(sol['output_pm_deg'])
    i = float(sol['output_ibias_ma'])
    return np.array([-g, -u, -p, i])


def to_min_objectives_normalized(sol, target):
    """Normalize objectives by target before negation: ratio-based."""
    g = float(sol['output_gain_linear']) / max(float(target['gain']), 1e-9)
    u = float(sol['output_ugbw_mhz']) / max(float(target['ugbw']), 1e-9)
    p = float(sol['output_pm_deg']) / max(float(target['pm']), 1e-9)
    i = float(sol['output_ibias_ma']) / max(float(target['ibias']), 1e-9)
    return np.array([-g, -u, -p, i])  # minimize: want high g/u/p, low i


def load_morl_solutions(json_paths):
    """Load and group solutions by spec from one or more JSON files."""
    by_spec = {}
    for path in json_paths:
        with open(path) as f:
            data = json.load(f)
        for sol in data['all_solutions']:
            spec = sol['spec']
            if spec not in by_spec:
                by_spec[spec] = []
            by_spec[spec].append(sol)
    return by_spec


def load_original_solutions(json_path):
    """Load original AutoCkt solutions (1 per spec)."""
    with open(json_path) as f:
        data = json.load(f)
    by_spec = {}
    for sol in data:
        spec = sol['spec'] + 1  # original is 0-indexed, MORL is 1-indexed
        by_spec[spec] = [sol]
    return by_spec


# ---------------------------------------------------------------------------
# Compute metrics per spec and aggregate
# ---------------------------------------------------------------------------
def compute_metrics_per_spec(by_spec, ref_point, label, use_normalized=False):
    """Compute HV and sparsity per spec, return aggregate stats."""
    hvs = []
    sps = []
    front_sizes = []
    specs_computed = 0

    for spec in sorted(by_spec.keys()):
        sols = by_spec[spec]
        if not sols:
            continue

        # Convert to minimization-space arrays
        points = []
        for sol in sols:
            try:
                pt = to_min_objectives(sol)
                # Only include points that dominate the reference
                if all(pt[d] < ref_point[d] for d in range(len(ref_point))):
                    points.append(pt)
            except (KeyError, TypeError):
                continue

        if not points:
            hvs.append(0.0)
            sps.append(0.0)
            front_sizes.append(0)
            specs_computed += 1
            continue

        points = np.array(points)
        front = pareto_front(points)
        front_sizes.append(len(front))

        hv = hypervolume_fast(front, ref_point)
        hvs.append(hv)

        sp = sparsity(front)
        sps.append(sp)
        specs_computed += 1

    return {
        'label': label,
        'num_specs': specs_computed,
        'mean_hv': np.mean(hvs) if hvs else 0,
        'std_hv': np.std(hvs) if hvs else 0,
        'median_hv': np.median(hvs) if hvs else 0,
        'mean_sparsity': np.mean(sps) if sps else 0,
        'std_sparsity': np.std(sps) if sps else 0,
        'median_sparsity': np.median(sps) if sps else 0,
        'mean_front_size': np.mean(front_sizes) if front_sizes else 0,
        'hvs': hvs,
        'sps': sps,
        'front_sizes': front_sizes,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    np.random.seed(42)

    print("=" * 70)
    print("  HYPERVOLUME & SPARSITY: MORL vs Original AutoCkt")
    print("=" * 70)

    # ── Load all solution sets ──────────────────────────────────────────
    print("\nLoading solutions...")

    # Original AutoCkt (single-objective)
    orig_path = ORIG_RESULTS / "original_autockt_results_15percent.json"
    orig_by_spec = load_original_solutions(orig_path)
    print(f"  Original AutoCkt: {len(orig_by_spec)} specs, 1 solution each")

    # MORL Standard DDQN (NW + Cosine combined)
    std_nw_path = MORL_RESULTS / "morl_raw_nw_agent.json"
    std_cos_path = MORL_RESULTS / "morl_raw_cosine_agent.json"
    std_by_spec = load_morl_solutions([std_nw_path, std_cos_path])
    print(f"  MORL Standard DDQN: {len(std_by_spec)} specs, up to {max(len(v) for v in std_by_spec.values())} solutions each")

    cos_by_spec = load_morl_solutions([std_cos_path])
    print(f"  MORL Cosine:        {len(cos_by_spec)} specs, up to {max(len(v) for v in cos_by_spec.values())} solutions each")

    nw_by_spec = load_morl_solutions([std_nw_path])
    print(f"  MORL NW:            {len(nw_by_spec)} specs, up to {max(len(v) for v in nw_by_spec.values())} solutions each")

    # MORL LLM-Guided DDQN (NW + Cosine combined)
    llm_nw_path = MORL_RESULTS / "morl_raw_llm_nw_agent.json"
    llm_cos_path = MORL_RESULTS / "morl_raw_llm_cosine_agent.json"
    llm_by_spec = load_morl_solutions([llm_nw_path, llm_cos_path])
    print(f"  MORL LLM-Guided:   {len(llm_by_spec)} specs, up to {max(len(v) for v in llm_by_spec.values())} solutions each")

    # ── Determine reference point ──────────────────────────────────────
    # Reference point = worst value per objective across ALL solutions (minimization)
    # We use a conservative reference: slightly worse than worst observed
    all_points = []
    for by_spec in [orig_by_spec, std_by_spec, llm_by_spec]:
        for sols in by_spec.values():
            for sol in sols:
                try:
                    all_points.append(to_min_objectives(sol))
                except (KeyError, TypeError):
                    continue
    all_points = np.array(all_points)
    # Reference = max per dimension (worst in minimization) + 10% margin
    ref_point = all_points.max(axis=0) * 1.1
    # For negative objectives (gain, ugbw, pm are negated), max is closest to 0
    # Make sure ref is actually worse than all points
    for d in range(4):
        ref_point[d] = max(ref_point[d], all_points[:, d].max() + abs(all_points[:, d].max()) * 0.1)

    print(f"\n  Reference point (minimization): {ref_point}")
    print(f"    = [gain≤{-ref_point[0]:.1f}, ugbw≤{-ref_point[1]:.2f}MHz, pm≤{-ref_point[2]:.1f}°, ibias≥{ref_point[3]:.2f}mA]")

    # ── Compute metrics ────────────────────────────────────────────────
    # Use common specs (intersection)
    common_specs = set(orig_by_spec.keys()) & set(std_by_spec.keys()) & set(llm_by_spec.keys())
    print(f"\n  Common specs across all 3 approaches: {len(common_specs)}")

    # Filter to common specs
    orig_common = {s: orig_by_spec[s] for s in common_specs}
    std_common = {s: std_by_spec[s] for s in common_specs}
    llm_common = {s: llm_by_spec[s] for s in common_specs}
    cos_common = {s: cos_by_spec[s] for s in common_specs if s in cos_by_spec}
    nw_common = {s: nw_by_spec[s] for s in common_specs if s in nw_by_spec}

    print("\nComputing hypervolume and sparsity per spec...")
    print("  (This may take a minute for inclusion-exclusion on 1000 specs)\n")

    orig_metrics = compute_metrics_per_spec(orig_common, ref_point, "Original AutoCkt")
    std_metrics = compute_metrics_per_spec(std_common, ref_point, "MORL Standard DDQN")
    llm_metrics = compute_metrics_per_spec(llm_common, ref_point, "MORL LLM-Guided DDQN")
    cos_metrics = compute_metrics_per_spec(cos_common, ref_point, "MORL Cosine")
    nw_metrics = compute_metrics_per_spec(nw_common, ref_point, "MORL NW")

    # ── Print Results ──────────────────────────────────────────────────
    print("=" * 70)
    print("  RESULTS")
    print("=" * 70)

    header = f"{'Metric':<30} {'Original AutoCkt':>20} {'MORL Std DDQN':>20} {'MORL Cosine':>20} {'MORL LLM DDQN':>20} {'MORL NW':>20}"
    print(header)
    print("-" * len(header))

    rows = [
        ("Mean Hypervolume", f"{orig_metrics['mean_hv']:.4e}", f"{std_metrics['mean_hv']:.4e}", f"{cos_metrics['mean_hv']:.4e}", f"{llm_metrics['mean_hv']:.4e}", f"{nw_metrics['mean_hv']:.4e}"),
        ("Std Hypervolume", f"{orig_metrics['std_hv']:.4e}", f"{std_metrics['std_hv']:.4e}", f"{cos_metrics['std_hv']:.4e}", f"{llm_metrics['std_hv']:.4e}", f"{nw_metrics['std_hv']:.4e}"),
        ("Median Hypervolume", f"{orig_metrics['median_hv']:.4e}", f"{std_metrics['median_hv']:.4e}", f"{cos_metrics['median_hv']:.4e}", f"{llm_metrics['median_hv']:.4e}", f"{nw_metrics['median_hv']:.4e}"),
        ("Mean Sparsity", f"{orig_metrics['mean_sparsity']:.4e}", f"{std_metrics['mean_sparsity']:.4e}", f"{cos_metrics['mean_sparsity']:.4e}", f"{llm_metrics['mean_sparsity']:.4e}", f"{nw_metrics['mean_sparsity']:.4e}"),
        ("Std Sparsity", f"{orig_metrics['std_sparsity']:.4e}", f"{std_metrics['std_sparsity']:.4e}", f"{cos_metrics['std_sparsity']:.4e}", f"{llm_metrics['std_sparsity']:.4e}", f"{nw_metrics['std_sparsity']:.4e}"),
        ("Median Sparsity", f"{orig_metrics['median_sparsity']:.4e}", f"{std_metrics['median_sparsity']:.4e}", f"{cos_metrics['median_sparsity']:.4e}", f"{llm_metrics['median_sparsity']:.4e}", f"{nw_metrics['median_sparsity']:.4e}"),
        ("Mean Pareto Front Size", f"{orig_metrics['mean_front_size']:.2f}", f"{std_metrics['mean_front_size']:.2f}", f"{cos_metrics['mean_front_size']:.2f}", f"{llm_metrics['mean_front_size']:.2f}", f"{nw_metrics['mean_front_size']:.2f}"),
    ]
    for name, v1, v2, v3, v4, v5 in rows:
        print(f"  {name:<28} {v1:>20} {v2:>20} {v3:>20} {v4:>20} {v5:>20}")

    # ── Interpretation ─────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  INTERPRETATION")
    print(f"{'=' * 70}")
    print(f"  Hypervolume (higher = better): measures quality AND diversity of Pareto front")
    print(f"  Sparsity (lower = better): measures uniformity of solution spread")
    print(f"  Pareto Front Size: number of non-dominated solutions per spec")

    # HV comparison
    if std_metrics['mean_hv'] > orig_metrics['mean_hv']:
        pct = (std_metrics['mean_hv'] - orig_metrics['mean_hv']) / max(orig_metrics['mean_hv'], 1e-30) * 100
        print(f"\n  MORL Std DDQN HV is {pct:.1f}% higher than Original AutoCkt")
    if llm_metrics['mean_hv'] > std_metrics['mean_hv']:
        pct = (llm_metrics['mean_hv'] - std_metrics['mean_hv']) / max(std_metrics['mean_hv'], 1e-30) * 100
        print(f"  MORL LLM DDQN HV is {pct:.1f}% higher than Standard DDQN")

    # ── Statistical test on HV ─────────────────────────────────────────
    from scipy import stats
    print(f"\n{'=' * 70}")
    print("  STATISTICAL TESTS (Hypervolume)")
    print(f"{'=' * 70}")

    # MORL Std vs Original
    orig_hvs = np.array(orig_metrics['hvs'])
    std_hvs = np.array(std_metrics['hvs'])
    llm_hvs = np.array(llm_metrics['hvs'])

    # Use paired tests on common specs
    if len(orig_hvs) == len(std_hvs) and len(orig_hvs) > 0:
        t_stat, t_pval = stats.ttest_rel(std_hvs, orig_hvs)
        print(f"  MORL Std vs Original: t={t_stat:.4f}, p={t_pval:.6f}", end="")
        print("  **SIGNIFICANT**" if t_pval < 0.05 else "  not significant")

        try:
            w_stat, w_pval = stats.wilcoxon(std_hvs, orig_hvs)
            print(f"  Wilcoxon:             W={w_stat:.0f}, p={w_pval:.6f}", end="")
            print("  **SIGNIFICANT**" if w_pval < 0.05 else "  not significant")
        except:
            pass

    if len(std_hvs) == len(llm_hvs) and len(std_hvs) > 0:
        t_stat, t_pval = stats.ttest_rel(llm_hvs, std_hvs)
        print(f"\n  LLM DDQN vs Std DDQN: t={t_stat:.4f}, p={t_pval:.6f}", end="")
        print("  **SIGNIFICANT**" if t_pval < 0.05 else "  not significant")

        try:
            w_stat, w_pval = stats.wilcoxon(llm_hvs, std_hvs)
            print(f"  Wilcoxon:             W={w_stat:.0f}, p={w_pval:.6f}", end="")
            print("  **SIGNIFICANT**" if w_pval < 0.05 else "  not significant")
        except:
            pass

    if len(orig_hvs) == len(llm_hvs) and len(orig_hvs) > 0:
        t_stat, t_pval = stats.ttest_rel(llm_hvs, orig_hvs)
        print(f"\n  LLM DDQN vs Original: t={t_stat:.4f}, p={t_pval:.6f}", end="")
        print("  **SIGNIFICANT**" if t_pval < 0.05 else "  not significant")

    # ── Save CSV ───────────────────────────────────────────────────────
    csv_rows = []
    common_list = sorted(common_specs)
    for i, spec in enumerate(common_list):
        csv_rows.append({
            'spec': spec,
            'hv_original': orig_metrics['hvs'][i],
            'hv_morl_std': std_metrics['hvs'][i],
            'hv_morl_cosine': cos_metrics['hvs'][i],
            'hv_morl_llm': llm_metrics['hvs'][i],
            'hv_morl_nw': nw_metrics['hvs'][i],
            'sparsity_original': orig_metrics['sps'][i],
            'sparsity_morl_std': std_metrics['sps'][i],
            'sparsity_morl_cosine': cos_metrics['sps'][i],
            'sparsity_morl_llm': llm_metrics['sps'][i],
            'sparsity_morl_nw': nw_metrics['sps'][i],
            'front_size_original': orig_metrics['front_sizes'][i],
            'front_size_morl_std': std_metrics['front_sizes'][i],
            'front_size_morl_cosine': cos_metrics['front_sizes'][i],
            'front_size_morl_llm': llm_metrics['front_sizes'][i],
            'front_size_morl_nw': nw_metrics['front_sizes'][i],
        })

    # Summary row
    csv_rows.append({
        'spec': 'MEAN',
        'hv_original': orig_metrics['mean_hv'],
        'hv_morl_std': std_metrics['mean_hv'],
        'hv_morl_cosine': cos_metrics['mean_hv'],
        'hv_morl_llm': llm_metrics['mean_hv'],
        'hv_morl_nw': nw_metrics['mean_hv'],
        'sparsity_original': orig_metrics['mean_sparsity'],
        'sparsity_morl_std': std_metrics['mean_sparsity'],
        'sparsity_morl_cosine': cos_metrics['mean_sparsity'],
        'sparsity_morl_llm': llm_metrics['mean_sparsity'],
        'sparsity_morl_nw': nw_metrics['mean_sparsity'],
        'front_size_original': orig_metrics['mean_front_size'],
        'front_size_morl_std': std_metrics['mean_front_size'],
        'front_size_morl_cosine': cos_metrics['mean_front_size'],
        'front_size_morl_llm': llm_metrics['mean_front_size'],
        'front_size_morl_nw': nw_metrics['mean_front_size'],
    })
    csv_rows.append({
        'spec': 'MEDIAN',
        'hv_original': orig_metrics['median_hv'],
        'hv_morl_std': std_metrics['median_hv'],
        'hv_morl_cosine': cos_metrics['median_hv'],
        'hv_morl_llm': llm_metrics['median_hv'],
        'hv_morl_nw': nw_metrics['median_hv'],
        'sparsity_original': orig_metrics['median_sparsity'],
        'sparsity_morl_std': std_metrics['median_sparsity'],
        'sparsity_morl_cosine': cos_metrics['median_sparsity'],
        'sparsity_morl_llm': llm_metrics['median_sparsity'],
        'sparsity_morl_nw': nw_metrics['median_sparsity'],
    })

    df = pd.DataFrame(csv_rows)
    csv_path = MORL_RESULTS / "hypervolume_sparsity_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved per-spec CSV: {csv_path}")

    # ── Save summary text ──────────────────────────────────────────────
    txt_path = MORL_RESULTS / "hypervolume_sparsity_summary.txt"
    with open(txt_path, 'w') as f:
        f.write("Hypervolume & Sparsity: MORL vs Original AutoCkt\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Reference point (min-space): {ref_point.tolist()}\n\n")
        for m in [orig_metrics, std_metrics, cos_metrics, llm_metrics, nw_metrics]:
            f.write(f"{m['label']}:\n")
            f.write(f"  Mean HV:       {m['mean_hv']:.6e}\n")
            f.write(f"  Median HV:     {m['median_hv']:.6e}\n")
            f.write(f"  Std HV:        {m['std_hv']:.6e}\n")
            f.write(f"  Mean Sparsity: {m['mean_sparsity']:.6e}\n")
            f.write(f"  Mean Front:    {m['mean_front_size']:.2f} solutions\n\n")
    print(f"  Saved summary: {txt_path}")


if __name__ == "__main__":
    main()
