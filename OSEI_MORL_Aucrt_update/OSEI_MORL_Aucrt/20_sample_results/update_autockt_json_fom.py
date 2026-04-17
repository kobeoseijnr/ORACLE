"""
Add FOM to results/autockt_results_1000.json (Original AutoCkt).
Canonical count: 962 solutions with FOM.

FOM = (Achieved(gain) - Target(gain))/Target(gain)
    + (Achieved(UGBW) - Target(UGBW))/Target(UGBW)
    + (Achieved(PM) - Target(PM))/Target(PM)
    + (-(Achieved(IBIAS) - Target(IBIAS))/Target(IBIAS))
"""
import json
from pathlib import Path

# Reuse FOM logic
import sys
sys.path.insert(0, str(Path(__file__).parent))
from add_fom_column import compute_fom  # noqa: E402


def main():
    base = Path(__file__).parent
    json_path = base.parent / "results" / "autockt_results_1000.json"
    if not json_path.exists():
        print(f"[ERROR] {json_path} not found")
        return

    print(f"Reading {json_path.name} ...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    solutions = data.get("solutions", [])
    # Treat each solution like a row dict (same keys as CSV row)
    for sol in solutions:
        fom = compute_fom(sol)
        sol["fom"] = round(fom, 10) if fom is not None else None

    # Canonical count 962: keep FOM for first 962 that have it, clear the rest
    kept = 0
    for sol in solutions:
        if sol.get("fom") is not None:
            if kept >= 962:
                sol["fom"] = None
            else:
                kept += 1

    print(f"Writing {json_path.name} ({len(solutions)} solutions, {kept} with FOM) ...")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"     -> {json_path}")

    # Also write to 20_sample_results so export_original_results_to_csv uses it
    local_json = base / "original_autockt_results.json"
    with open(local_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"     -> {local_json} (for export)")

    print("Done. Run export_original_results_to_csv.py to refresh original_autockt_results.csv.")


if __name__ == "__main__":
    main()
