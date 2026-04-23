"""Build top-20 FOM tables in 20-Sample Analysis format for FOM_for_Yes_summary.md."""
import csv
from pathlib import Path

BASE = Path(__file__).parent

def fmt(x, decimals=2):
    if x is None or x == "":
        return ""
    try:
        v = float(x)
        return f"{v:.{decimals}f}" if decimals else str(v)
    except (ValueError, TypeError):
        return str(x) if x != "" else ""

def get_top20(csv_path):
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    yes = [r for r in rows if (r.get("target_reached") or "").strip().lower() == "yes"]
    for r in yes:
        try:
            r["_fom"] = float((r.get("fom") or "").strip())
        except (ValueError, TypeError):
            r["_fom"] = None
    yes = [r for r in yes if r["_fom"] is not None]
    yes.sort(key=lambda r: r["_fom"], reverse=True)
    return yes[:20]

HEADER = "| Sample ID | Target Gain (dB) | Output Gain (dB) | Target UGBW (MHz) | Output UGBW (MHz) | Target PM (°) | Output PM (°) | Target IBIAS (mA) | Output IBIAS (mA) | Avg FoM |"

def md_table(rows):
    lines = [HEADER, "|" + "|".join(["---"] * 10) + "|"]
    for r in rows:
        line = "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            r.get("spec", ""),
            fmt(r.get("target_gain_db")),
            fmt(r.get("output_gain_db")),
            fmt(r.get("target_ugbw_mhz")),
            fmt(r.get("output_ugbw_mhz")),
            fmt(r.get("target_pm_deg")),
            fmt(r.get("output_pm_deg")),
            fmt(r.get("target_ibias_ma")),
            fmt(r.get("output_ibias_ma")),
            fmt(r.get("fom"), 4),
        )
        lines.append(line)
    return "\n".join(lines)

def main():
    autockt = BASE / "original_autockt_results.csv"
    morl = BASE / "original_morl_autockt_results.csv"
    top_autockt = get_top20(autockt)
    top_morl = get_top20(morl)
    print("ORIGINAL_TABLE")
    print(md_table(top_autockt))
    print("MORL_TABLE")
    print(md_table(top_morl))

if __name__ == "__main__":
    main()
