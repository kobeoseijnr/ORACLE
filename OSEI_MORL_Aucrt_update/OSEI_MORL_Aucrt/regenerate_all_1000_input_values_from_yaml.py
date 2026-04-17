import json
import random
from pathlib import Path

import re


def _as_float(x):
    try:
        return float(x)
    except Exception:
        return x


def _sample_range(lo, hi, n, *, integer=False):
    if integer:
        return [random.randint(int(lo), int(hi)) for _ in range(n)]
    return [random.uniform(float(lo), float(hi)) for _ in range(n)]


def _parse_target_specs_from_yaml_text(yaml_text):
    """Parse target_specs ranges from the repo's YAML without external deps.

    Expected YAML snippet:
      target_specs:
        gain_min: [200,400]
        ugbw_min: [1.0e6, 2.5e7]
        phm_min: [60,60.0000001]
        ibias_max: [0.0001, 0.01]

    Returns:
      dict[str, tuple[float,float]]
    """
    lines = yaml_text.splitlines()
    in_block = False
    indent = None
    out = {}

    for raw in lines:
        line = raw.rstrip("\n")
        if not in_block:
            if re.match(r"^\s*target_specs\s*:\s*$", line):
                in_block = True
                indent = None
            continue

        if not line.strip() or line.lstrip().startswith("#"):
            continue

        cur_indent = len(line) - len(line.lstrip(" "))
        if indent is None:
            indent = cur_indent
        if cur_indent < indent:
            break

        m = re.match(r"^\s*([A-Za-z0-9_]+)\s*:\s*(.*)$", line)
        if not m:
            continue

        key = m.group(1)
        rhs = m.group(2)

        # Strip tags like: !!python/tuple
        rhs = re.sub(r"!![^\s]+\s*", "", rhs).strip()

        # Support [a,b] or (a,b)
        m2 = re.search(r"[\[(]\s*([^,\]\)]+)\s*,\s*([^\]\)]+)\s*[\])]\s*$", rhs)
        if not m2:
            continue

        lo = float(m2.group(1))
        hi = float(m2.group(2))
        out[key] = (lo, hi)

    return out


def main():
    base_dir = Path(__file__).parent
    yaml_path = base_dir / "eval_engines" / "ngspice" / "ngspice_inputs" / "yaml_files" / "two_stage_opamp.yaml"
    out_path = base_dir / "data" / "all_1000_input_values.json"

    n = 1000

    with open(yaml_path, "r", encoding="utf-8") as f:
        yaml_text = f.read()

    target_specs = _parse_target_specs_from_yaml_text(yaml_text)

    # YAML units:
    # - gain_min: linear
    # - ugbw_min: Hz
    # - phm_min: degrees
    # - ibias_max: A
    gain_lo, gain_hi = map(_as_float, target_specs.get("gain_min", [200, 400]))
    ugbw_lo_hz, ugbw_hi_hz = map(_as_float, target_specs.get("ugbw_min", [1.0e6, 2.5e7]))
    phm_lo, phm_hi = map(_as_float, target_specs.get("phm_min", [60, 60.0000001]))
    ibias_lo_a, ibias_hi_a = map(_as_float, target_specs.get("ibias_max", [0.0001, 0.01]))

    # Output file uses the repo's existing naming convention and units:
    # - target_ugbw_mhz: MHz
    # - target_ibias_ma: mA
    gain_vals = _sample_range(gain_lo, gain_hi, n, integer=True)
    ugbw_vals_mhz = [v / 1e6 for v in _sample_range(ugbw_lo_hz, ugbw_hi_hz, n, integer=False)]

    # phm_min is essentially fixed ~60 in YAML; sample within its range (will be ~60)
    phm_vals = _sample_range(phm_lo, phm_hi, n, integer=False)

    ibias_vals_ma = [v * 1e3 for v in _sample_range(ibias_lo_a, ibias_hi_a, n, integer=False)]

    input_values = {}
    for i in range(1, n + 1):
        input_values[str(i)] = {
            "target_gain_linear": float(gain_vals[i - 1]),
            "target_ugbw_mhz": float(ugbw_vals_mhz[i - 1]),
            "target_pm_deg": float(phm_vals[i - 1]),
            "target_ibias_ma": float(ibias_vals_ma[i - 1]),
        }

    out = {
        "total_specifications": n,
        "input_values": input_values,
        "source": {
            "yaml": str(yaml_path.as_posix()),
            "units": {
                "target_gain_linear": "linear",
                "target_ugbw_mhz": "MHz (converted from YAML Hz)",
                "target_pm_deg": "deg",
                "target_ibias_ma": "mA (converted from YAML A)",
            },
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"Wrote: {out_path} ({n} specs)")
    print("Example spec #1:", out["input_values"]["1"])


if __name__ == "__main__":
    # Deterministic output
    random.seed(42)
    main()
