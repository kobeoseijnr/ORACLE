"""
Figure of Merit (FoM) calculation for the AutoCkt 20 + MORL+AutoCkt report.

Updated FoM formula (see with_updated_FoM.md §2):

  FoM = (G_out − G_tgt)/G_tgt + (U_out − U_tgt)/U_tgt + (P_out − P_tgt)/P_tgt − (I_out − I_tgt)/I_tgt

Where:
  - G = Gain (linear, e.g. 200–400 for 46–52 dB)
  - U = UGBW (MHz)
  - P = Phase Margin (°)
  - I = IBIAS (mA)

Interpretation:
  - Gain, UGBW, PM (maximize): term = (Achieved − Target) / Target
  - IBIAS (minimize):         term = −(Achieved − Target) / Target
  - Higher FoM = better.
"""


def _safe_float(val, default=None):
    if val is None or (isinstance(val, str) and val.strip() == ""):
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def compute_fom_values(
    gain_tgt,
    gain_out,
    ugbw_tgt,
    ugbw_out,
    pm_tgt,
    pm_out,
    ibias_tgt,
    ibias_out,
):
    """
    Compute FoM from raw numeric values.

    Args:
        gain_tgt, gain_out:  Target and achieved gain (linear, e.g. from 20*log10(A_v)).
        ugbw_tgt, ugbw_out:  Target and achieved UGBW (MHz).
        pm_tgt, pm_out:      Target and achieved phase margin (°).
        ibias_tgt, ibias_out: Target and achieved IBIAS (mA).

    Returns:
        FoM float, or None if any required value is missing or any target is zero.
    """
    if gain_tgt is None or ugbw_tgt is None or pm_tgt is None or ibias_tgt is None:
        return None
    if gain_out is None or ugbw_out is None or pm_out is None or ibias_out is None:
        return None
    if gain_tgt == 0 or ugbw_tgt == 0 or pm_tgt == 0 or ibias_tgt == 0:
        return None

    term_gain = (gain_out - gain_tgt) / gain_tgt
    term_ugbw = (ugbw_out - ugbw_tgt) / ugbw_tgt
    term_pm = (pm_out - pm_tgt) / pm_tgt
    term_ibias = -(ibias_out - ibias_tgt) / ibias_tgt

    return term_gain + term_ugbw + term_pm + term_ibias


# CSV column names used in original_autockt_results.csv / moral_autockt_results.csv
_COL_GAIN_TGT = "target_gain_linear"
_COL_GAIN_OUT = "output_gain_linear"
_COL_UGBW_TGT = "target_ugbw_mhz"
_COL_UGBW_OUT = "output_ugbw_mhz"
_COL_PM_TGT = "target_pm_deg"
_COL_PM_OUT = "output_pm_deg"
_COL_IBIAS_TGT = "target_ibias_ma"
_COL_IBIAS_OUT = "output_ibias_ma"


def compute_fom(row):
    """
    Compute FoM from a row-like dict (e.g. CSV row).

    Expects keys: target_gain_linear, output_gain_linear, target_ugbw_mhz,
    output_ugbw_mhz, target_pm_deg, output_pm_deg, target_ibias_ma, output_ibias_ma.

    Returns:
        FoM float, or None if any required value is missing or any target is zero.
    """
    gain_tgt = _safe_float(row.get(_COL_GAIN_TGT))
    gain_out = _safe_float(row.get(_COL_GAIN_OUT))
    ugbw_tgt = _safe_float(row.get(_COL_UGBW_TGT))
    ugbw_out = _safe_float(row.get(_COL_UGBW_OUT))
    pm_tgt = _safe_float(row.get(_COL_PM_TGT))
    pm_out = _safe_float(row.get(_COL_PM_OUT))
    ibias_tgt = _safe_float(row.get(_COL_IBIAS_TGT))
    ibias_out = _safe_float(row.get(_COL_IBIAS_OUT))

    return compute_fom_values(
        gain_tgt, gain_out,
        ugbw_tgt, ugbw_out,
        pm_tgt, pm_out,
        ibias_tgt, ibias_out,
    )


if __name__ == "__main__":
    # Quick sanity check: one term dominant
    # G_tgt=200, G_out=400 -> (400-200)/200 = 1.0; others 0 -> FoM = 1.0
    f = compute_fom_values(200, 400, 1.0, 1.0, 60, 60, 1.0, 1.0)
    assert abs(f - 1.0) < 1e-9, f"expected 1.0, got {f}"
    print("FoM formula check passed.")
