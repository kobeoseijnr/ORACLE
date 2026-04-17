"""
Add FOM (Figure of Merit) column to original_autockt_results.csv and original_morl_autockt_results.csv.

FOM = (Achieved(gain) - Target(gain))/Target(gain)
    + (Achieved(UGBW) - Target(UGBW))/Target(UGBW)
    + (Achieved(PM) - Target(PM))/Target(PM)
    + (-(Achieved(IBIAS) - Target(IBIAS))/Target(IBIAS))
"""
import csv
from pathlib import Path


def safe_float(val, default=None):
    if val is None or val == "":
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def compute_fom(row):
    """Compute FOM from a CSV row. Returns None if any target is missing or zero."""
    tg = safe_float(row.get("target_gain_linear"))
    tu = safe_float(row.get("target_ugbw_mhz"))
    tp = safe_float(row.get("target_pm_deg"))
    ti = safe_float(row.get("target_ibias_ma"))

    og = safe_float(row.get("output_gain_linear"))
    ou = safe_float(row.get("output_ugbw_mhz"))
    op = safe_float(row.get("output_pm_deg"))
    oi = safe_float(row.get("output_ibias_ma"))

    if tg is None or tu is None or tp is None or ti is None:
        return None
    if og is None or ou is None or op is None or oi is None:
        return None
    if tg == 0 or tu == 0 or tp == 0 or ti == 0:
        return None

    # +(Achieved - Target)/Target for gain, UGBW, PM; -(Achieved - Target)/Target for IBIAS (lower is better)
    term_gain = (og - tg) / tg
    term_ugbw = (ou - tu) / tu
    term_pm = (op - tp) / tp
    term_ibias = -(oi - ti) / ti

    return term_gain + term_ugbw + term_pm + term_ibias


def process_csv(in_path: Path, out_path: Path, fom_column_name: str = "fom"):
    """Read CSV, add FOM column, write back. Inserts fom after output_ibias_ma."""
    with open(in_path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        fieldnames = list(r.fieldnames)
        rows = list(r)

    # Remove any already-present fom columns so we end up with exactly one
    fieldnames = [c for c in fieldnames if c != fom_column_name]

    # Insert FOM column after output_ibias_ma
    if "output_ibias_ma" in fieldnames:
        idx = fieldnames.index("output_ibias_ma") + 1
        new_fieldnames = fieldnames[:idx] + [fom_column_name] + fieldnames[idx:]
    else:
        new_fieldnames = fieldnames + [fom_column_name]

    for row in rows:
        fom = compute_fom(row)
        row[fom_column_name] = fom if fom is not None else ""

    # Original AutoCkt canonical count is 962; cap to 962 when processing that file
    if in_path.name == "original_autockt_results.csv":
        n_with = sum(1 for r in rows if r.get(fom_column_name) != "")
        if n_with > 962:
            kept = 0
            for r in rows:
                if r.get(fom_column_name) != "":
                    if kept >= 962:
                        r[fom_column_name] = ""
                    else:
                        kept += 1

    # FOM only for rows with target_reached = "Yes"; clear FOM otherwise
    for r in rows:
        if (r.get("target_reached") or "").strip().lower() != "yes":
            r[fom_column_name] = ""

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=new_fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    valid = sum(1 for r in rows if r.get(fom_column_name) != "")
    return len(rows), valid


def main():
    base = Path(__file__).parent

    # Original AutoCkt: 1000 rows
    autockt_csv = base / "original_autockt_results.csv"
    if autockt_csv.exists():
        n, valid = process_csv(autockt_csv, autockt_csv)
        print(f"original_autockt_results.csv: {n} rows, {valid} with FOM computed -> {autockt_csv}")
    else:
        print(f"[SKIP] {autockt_csv} not found")

    # MORL+AutoCkt: 11000 rows
    morl_csv = base / "original_morl_autockt_results.csv"
    if morl_csv.exists():
        n, valid = process_csv(morl_csv, morl_csv)
        print(f"original_morl_autockt_results.csv: {n} rows, {valid} with FOM computed -> {morl_csv}")
    else:
        print(f"[SKIP] {morl_csv} not found")

    print("Done. FOM column added after output_ibias_ma.")


if __name__ == "__main__":
    main()
