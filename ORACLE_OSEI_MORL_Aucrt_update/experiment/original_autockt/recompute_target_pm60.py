import csv
import shutil
from pathlib import Path


def _to_float(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def lookup(spec, goal_spec):
    norm = []
    for s, g in zip(spec, goal_spec):
        denom = g + s
        if denom == 0:
            norm.append(0.0)
        else:
            norm.append((s - g) / denom)
    return norm


def reward(spec, goal_spec, specs_id):
    rel_specs = lookup(spec, goal_spec)
    reward_val = 0.0
    for i, rel_spec in enumerate(rel_specs):
        if specs_id[i] == "ibias_max":
            rel_spec = rel_spec * -1.0
        if rel_spec < 0:
            reward_val += rel_spec
    return reward_val if reward_val < -0.02 else 10


def recompute(csv_path: Path, target_pm_deg: float = 60.0) -> tuple[int, int, Path]:
    bak_path = csv_path.with_suffix(csv_path.suffix + f".bak_pm{target_pm_deg:g}")
    shutil.copy2(csv_path, bak_path)

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    if not rows:
        raise RuntimeError(f"Empty CSV: {csv_path}")

    hdr = rows[0]
    data = rows[1:]

    # If a TOTAL summary row exists, drop it before recomputing.
    i_spec = hdr.index("spec")
    had_total = bool(data) and str(data[-1][i_spec]).strip().upper() == "TOTAL"
    if had_total:
        data_core = data[:-1]
    else:
        data_core = data

    i_tr = hdr.index("target_reached")
    i_reward = hdr.index("reward")
    i_tgain = hdr.index("target_gain_linear")
    i_tugbw = hdr.index("target_ugbw_mhz")
    i_tpm = hdr.index("target_pm_deg")
    i_tib = hdr.index("target_ibias_ma")
    i_ogain = hdr.index("output_gain_linear")
    i_oib = hdr.index("output_ibias_ma")
    i_opm = hdr.index("output_pm_deg")
    i_ougbw = hdr.index("output_ugbw_mhz")

    specs_id = ["gain_min", "ibias_max", "phm_min", "ugbw_min"]

    yes_count = 0
    new_rows = []
    for r in data_core:
        # Override PM target
        r[i_tpm] = str(float(target_pm_deg))

        # Targets
        tgain = _to_float(r[i_tgain])
        tubgw_mhz = _to_float(r[i_tugbw])
        tib_ma = _to_float(r[i_tib])

        # Outputs
        ogain = _to_float(r[i_ogain])
        oib_ma = _to_float(r[i_oib])
        opm = _to_float(r[i_opm])
        ougbw_mhz = _to_float(r[i_ougbw])

        actual_specs = [
            ogain,
            oib_ma / 1000.0,
            opm,
            ougbw_mhz * 1e6,
        ]
        target_specs = [
            tgain,
            tib_ma / 1000.0,
            float(target_pm_deg),
            tubgw_mhz * 1e6,
        ]

        rew = reward(actual_specs, target_specs, specs_id)
        r[i_reward] = str(rew)
        r[i_tr] = "Yes" if rew == 10 else "No"
        if rew == 10:
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
    csv_path = Path(__file__).parent / "results" / "original_autockt_results_original_strict.csv"
    yes, total, bak = recompute(csv_path, target_pm_deg=60.0)
    print(f"Backup: {bak}")
    print(f"New YES count: {yes} / {total}")
