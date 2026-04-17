"""
Export Original AutoCkt and MORL+AutoCkt JSON results to CSV with the same column structure.
Both CSVs share the same core columns for comparability; MORL CSV includes extra paper_* fields.
"""
import json
import csv
import math
from pathlib import Path


def gain_linear_to_db(linear):
    if linear is None or linear <= 0:
        return ""
    return 20 * math.log10(linear)

OUTPUT_DIR = Path(__file__).parent

# Both CSVs: 15 columns (spec through fom, then target_reached)
COLUMNS_15 = [
    "spec", "method", "solution",
    "target_gain_linear", "target_ugbw_mhz", "target_pm_deg", "target_ibias_ma", "target_gain_db",
    "output_gain_linear", "output_gain_db", "output_ugbw_mhz", "output_pm_deg", "output_ibias_ma",
    "fom", "target_reached",
]


def row_from_autockt(sol):
    """Build a CSV row from an Original AutoCkt solution (15 columns). Canonical: 962 have target_reached=Yes (same as FOM count)."""
    fom = sol.get("fom")
    has_fom = fom is not None and fom != ""
    # Original: 962 Yes = same 962 that have FOM
    target_reached = "Yes" if has_fom else ("No" if sol.get("target_gain_linear") is not None else "")
    return {
        "spec": sol.get("spec"),
        "method": "Original AutoCkt",
        "solution": 1,
        "target_gain_linear": sol.get("target_gain_linear"),
        "target_ugbw_mhz": sol.get("target_ugbw_mhz"),
        "target_pm_deg": sol.get("target_pm_deg"),
        "target_ibias_ma": sol.get("target_ibias_ma"),
        "target_gain_db": sol.get("target_gain_db"),
        "output_gain_linear": sol.get("output_gain_linear"),
        "output_gain_db": sol.get("output_gain_db"),
        "output_ugbw_mhz": sol.get("output_ugbw_mhz"),
        "output_pm_deg": sol.get("output_pm_deg"),
        "output_ibias_ma": sol.get("output_ibias_ma"),
        "fom": fom if has_fom else "",
        "target_reached": target_reached,
    }


def _load_morl_specs_reached():
    """Return set of spec IDs that have Has_Reached=True (982 specs). Used for MORL target_reached."""
    breakdown = OUTPUT_DIR.parent / "results" / "target_reached_breakdown.csv"
    if not breakdown.exists():
        return None
    reached = set()
    with open(breakdown, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (row.get("Has_Reached") or "").lower() == "true":
                try:
                    reached.add(int(row.get("Spec", 0)))
                except (TypeError, ValueError):
                    pass
    return reached if reached else None


def row_from_morl(sol, specs_reached=None):
    """Build a CSV row from a MORL+AutoCkt solution (15 columns). MORL: 982 specs have Yes (all 11 solutions per spec)."""
    spec = sol.get("spec")
    if specs_reached is not None and spec is not None:
        try:
            tr = "Yes" if int(spec) in specs_reached else "No"
        except (TypeError, ValueError):
            tr = sol.get("target_reached", "")
    else:
        tr = sol.get("target_reached", "")
    return {
        "spec": spec,
        "method": sol.get("method", "MORL+AutoCkt"),
        "solution": sol.get("solution"),
        "target_gain_linear": sol.get("target_gain_linear"),
        "target_ugbw_mhz": sol.get("target_ugbw_mhz"),
        "target_pm_deg": sol.get("target_pm_deg"),
        "target_ibias_ma": sol.get("target_ibias_ma"),
        "target_gain_db": sol.get("target_gain_db") if sol.get("target_gain_db") is not None else gain_linear_to_db(sol.get("target_gain_linear")),
        "output_gain_linear": sol.get("output_gain_linear"),
        "output_gain_db": sol.get("output_gain_db"),
        "output_ugbw_mhz": sol.get("output_ugbw_mhz"),
        "output_pm_deg": sol.get("output_pm_deg"),
        "output_ibias_ma": sol.get("output_ibias_ma"),
        "fom": sol.get("fom") if sol.get("fom") is not None else "",
        "target_reached": tr,
    }


def main():
    # 1) Original AutoCkt: JSON -> CSV (same results)
    autockt_json = OUTPUT_DIR / "original_autockt_results.json"
    autockt_csv = OUTPUT_DIR / "original_autockt_results.csv"
    if not autockt_json.exists():
        autockt_json = OUTPUT_DIR.parent / "results" / "autockt_results_1000.json"
        if not autockt_json.exists():
            print(f"[ERROR] AutoCkt JSON not found in {OUTPUT_DIR} or results/")
            return
    print(f"[1] Reading {autockt_json.name} ...")
    with open(autockt_json, "r", encoding="utf-8") as f:
        ad = json.load(f)
    rows_autockt = [row_from_autockt(s) for s in ad.get("solutions", [])]
    print(f"[1] Writing {autockt_csv.name} ({len(rows_autockt)} rows, 15 columns) ...")
    with open(autockt_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS_15, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows_autockt)
    print(f"     -> {autockt_csv}")

    # 2) MORL+AutoCkt: JSON -> CSV (same results, same columns)
    morl_json = OUTPUT_DIR / "original_morl_autockt_results.json"
    morl_csv = OUTPUT_DIR / "original_morl_autockt_results.csv"
    if not morl_json.exists():
        morl_json = OUTPUT_DIR.parent / "results" / "all_input_output_values_complete_with_paper_ranges.json"
        if not morl_json.exists():
            print(f"[ERROR] MORL JSON not found in {OUTPUT_DIR} or results/")
            return
    print(f"[2] Reading {morl_json.name} ...")
    with open(morl_json, "r", encoding="utf-8") as f:
        md = json.load(f)
    specs_reached = _load_morl_specs_reached()
    rows_morl = [row_from_morl(s, specs_reached) for s in md.get("solutions", [])]
    print(f"[2] Writing {morl_csv.name} ({len(rows_morl)} rows, 15 columns) ...")
    with open(morl_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS_15, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows_morl)
    print(f"     -> {morl_csv}")

    print("Done. Both CSVs have 15 columns (spec through fom, target_reached).")


if __name__ == "__main__":
    main()
