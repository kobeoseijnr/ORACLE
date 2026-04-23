"""Check if best-20 data has any negative values in objectives (Gain, UGBW, PM, IBIAS)."""
import csv
import numpy as np
from pathlib import Path

BASE = Path(__file__).parent
OBJECTIVES = [
    ("target_gain_db", "output_gain_db", "Gain (dB)"),
    ("target_ugbw_mhz", "output_ugbw_mhz", "UGBW (MHz)"),
    ("target_pm_deg", "output_pm_deg", "PM (°)"),
    ("target_ibias_ma", "output_ibias_ma", "IBIAS (mA)"),
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


def get_common_20_specs_from_morl(morl_rows, autockt_rows):
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
        if s not in autockt_specs or not r["_yes"]:
            continue
        f = r["_fom"] if r["_fom"] is not None else -np.inf
        if s not in by_spec_best or f > (by_spec_best[s]["_fom"] or -np.inf):
            by_spec_best[s] = r
    best_list = sorted(by_spec_best.values(), key=lambda r: r["_fom"] or -np.inf, reverse=True)
    return [str(r.get("spec", "")) for r in best_list[:20]]


def get_common_20_specs(autockt_rows):
    yes = [r for r in autockt_rows if (r.get("target_reached") or "").strip().lower() == "yes"]
    for r in yes:
        try:
            r["_fom"] = float((r.get("fom") or "").strip())
        except (ValueError, TypeError):
            r["_fom"] = None
    yes = [r for r in yes if r["_fom"] is not None]
    yes.sort(key=lambda r: r["_fom"], reverse=True)
    return [str(r.get("spec", "")) for r in yes[:20]]


def get_original_rows_for_specs(autockt_rows, spec_ids):
    by_spec = {str(r.get("spec", "")): r for r in autockt_rows}
    return [by_spec[s] for s in spec_ids if s in by_spec]


def get_morl_best_per_spec(morl_rows, spec_ids):
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


def main():
    autockt = load_csv(BASE / "original_autockt_results.csv")
    morl = load_csv(BASE / "original_morl_autockt_results.csv")
    spec_ids = get_common_20_specs_from_morl(morl, autockt)
    if len(spec_ids) < 20:
        spec_ids = get_common_20_specs(autockt)
    orig_20 = get_original_rows_for_specs(autockt, spec_ids)
    morl_20 = get_morl_best_per_spec(morl, spec_ids)

    n_obj = 4

    def to_array(rows, col_key):
        return np.array([[safe_float(r.get(OBJECTIVES[j][col_key])) for j in range(n_obj)] for r in rows])

    t_common = to_array(orig_20, 0)  # target columns
    o_orig = to_array(orig_20, 1)   # output columns from orig
    o_morl = to_array(morl_20, 1)   # output columns from morl

    obj_names = [OBJECTIVES[j][2] for j in range(n_obj)]
    print("Best 20 — min/max and negative count per objective (Target, Original output, MORL output):")
    print()
    any_neg = False
    for j in range(n_obj):
        for label, arr in [("Target", t_common), ("Original output", o_orig), ("MORL output", o_morl)]:
            v = arr[:, j]
            finite = v[np.isfinite(v)]
            mn = np.min(finite) if len(finite) else np.nan
            mx = np.max(finite) if len(finite) else np.nan
            n_neg = int(np.sum(finite < 0)) if len(finite) else 0
            if n_neg > 0:
                any_neg = True
            print(f"  {obj_names[j]} ({label}): min={mn:.6g}, max={mx:.6g}, negative count={n_neg}")
    print()
    if any_neg:
        print("Yes — there are negative values in the best-20 objectives.")
    else:
        print("No — there are no negative values in the best-20 objectives (all targets and outputs are >= 0).")


if __name__ == "__main__":
    main()
