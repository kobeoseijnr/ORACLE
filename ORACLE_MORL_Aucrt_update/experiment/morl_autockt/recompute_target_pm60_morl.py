import csv
import json
import math
import shutil
from pathlib import Path


THRESHOLD = -0.456650


def _to_float(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def cosine_similarity_scalarization(reward_vector, preference):
    reward_norm = math.sqrt(sum(v * v for v in reward_vector))
    pref_norm = math.sqrt(sum(v * v for v in preference))

    if reward_norm == 0.0 or pref_norm == 0.0:
        return 0.0

    cosine_sim = sum(r * p for r, p in zip(reward_vector, preference)) / (reward_norm * pref_norm)
    return cosine_sim * reward_norm


def parse_preference(pref_str: str):
    if pref_str is None:
        return [0.25, 0.25, 0.25, 0.25]
    s = str(pref_str).strip()
    if not s:
        return [0.25, 0.25, 0.25, 0.25]

    # Typically stored like "[1.0, 0.0, 0.0, 0.0]" in the CSV
    try:
        pref = json.loads(s)
        if isinstance(pref, list) and len(pref) == 4:
            return [float(x) for x in pref]
    except Exception:
        pass

    return [0.25, 0.25, 0.25, 0.25]


def recompute(csv_path: Path, target_pm_deg: float = 60.0):
    bak_path = csv_path.with_suffix(csv_path.suffix + f".bak_pm{target_pm_deg:g}")
    shutil.copy2(csv_path, bak_path)

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    if not rows:
        raise RuntimeError(f"Empty CSV: {csv_path}")

    hdr = rows[0]
    data = rows[1:]

    i_spec = hdr.index("spec")
    had_total = bool(data) and str(data[-1][i_spec]).strip().upper() == "TOTAL"
    if had_total:
        data_core = data[:-1]
    else:
        data_core = data

    i_tgain = hdr.index("target_gain_linear")
    i_tugbw = hdr.index("target_ugbw_mhz")
    i_tpm = hdr.index("target_pm_deg")
    i_tib = hdr.index("target_ibias_ma")

    i_ogain = hdr.index("output_gain_linear")
    i_ougbw = hdr.index("output_ugbw_mhz")
    i_opm = hdr.index("output_pm_deg")
    i_oib = hdr.index("output_ibias_ma")

    i_scalar = hdr.index("scalarized_value")
    i_pref = hdr.index("preference")
    i_tr = hdr.index("target_reached")

    yes_count = 0
    new_rows = []

    for r in data_core:
        # Override PM target
        r[i_tpm] = str(float(target_pm_deg))

        gain_actual = _to_float(r[i_ogain])
        ugbw_actual = _to_float(r[i_ougbw])
        phm_actual = _to_float(r[i_opm])
        ibias_actual = _to_float(r[i_oib])

        gain_target = _to_float(r[i_tgain])
        ugbw_target = _to_float(r[i_tugbw])
        phm_target = float(target_pm_deg)
        ibias_target = _to_float(r[i_tib])

        gain_obj = (gain_actual - gain_target) / gain_target if gain_target > 0 else 0.0
        ugbw_obj = (ugbw_actual - ugbw_target) / ugbw_target if ugbw_target > 0 else 0.0
        phm_obj = (phm_actual - phm_target) / phm_target if phm_target > 0 else 0.0
        ibias_obj = -(ibias_actual - ibias_target) / ibias_target if ibias_target > 0 else 0.0

        reward_vector = [gain_obj, ugbw_obj, phm_obj, ibias_obj]

        preference = parse_preference(r[i_pref])
        pref_norm = math.sqrt(sum(p * p for p in preference))
        if pref_norm > 0.0:
            preference = [p / pref_norm for p in preference]

        scalarized_value = cosine_similarity_scalarization(reward_vector, preference)
        r[i_scalar] = str(float(scalarized_value))

        reached = "Yes" if scalarized_value > THRESHOLD else "No"
        r[i_tr] = reached
        if reached == "Yes":
            yes_count += 1

        new_rows.append(r)

    out = [hdr] + new_rows

    if had_total:
        total_row = [""] * len(hdr)
        total_row[i_spec] = "TOTAL"
        total_row[i_tr] = str(yes_count)
        out.append(total_row)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(out)

    return yes_count, len(new_rows), bak_path


if __name__ == "__main__":
    csv_path = Path(__file__).parent / "results" / "morl_autockt_results_original_morl.csv"
    yes, total, bak = recompute(csv_path, target_pm_deg=60.0)
    print(f"Backup: {bak}")
    print(f"New YES count: {yes} / {total}")
