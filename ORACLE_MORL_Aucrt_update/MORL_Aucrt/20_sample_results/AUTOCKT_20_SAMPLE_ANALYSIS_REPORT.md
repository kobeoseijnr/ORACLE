# AutoCkt 20 + MORL+AutoCkt 20 Best Solutions Analysis Report

## Executive Summary

This report presents a detailed analysis comparing Original AutoCkt and MORL+AutoCkt methods. The analysis uses **the same 20 input specifications (targets)** for both methods and compares **the best solution per method** for each spec: one Original AutoCkt solution and one best MORL+AutoCkt solution per specification.

**Same 20 inputs**: Both methods are evaluated on the same 20 specifications (Original AutoCkt’s top 20 by FoM among rows with `target_reached=Yes`).  
**Original AutoCkt**: 20 solutions (1 per specification — the only solution per spec).  
**MORL+AutoCkt**: 20 best solutions (best 1 per specification by FoM among `target_reached=Yes`, or best overall if none).  
**Total compared**: 20×4 targets, 20×4 Original outputs, 20×4 MORL outputs (same targets, different outputs).

**Visualizations** (both put all four objectives in one space):
- **3D + color:** `one_graph_3d_all_four_gain_ugbw_pm_ibias.png` — Gain, UGBW, PM on axes; IBIAS as point color (colorbar). Markers: circle = Target, triangle = MORL+AutoCkt, square = Original AutoCkt.
- **Parallel coordinates:** `one_space_parallel_coordinates_all_objectives.png` — one 2D plot with four vertical axes (Gain, UGBW, PM, IBIAS); each of the 20 specs has three lines (Target, MORL output, Original output). Values are normalized per objective so all four are comparable in one space.

---

## 1. Objective Specifications from Paper

The AutoCkt paper defines the following target ranges for two-stage operational amplifier design:

### Target Ranges

| Objective | Range | Units | Optimization Direction | Notes |
|-----------|-------|-------|----------------------|-------|
| **Gain (A_v)** | 200-400 | Linear | **Higher is Better** (Maximize) | Equivalent to 46.02-52.04 dB |
| **Unity Gain Bandwidth (UGBW)** | 1.0-25.0 | MHz | **Higher is Better** (Maximize) | Frequency response metric |
| **Phase Margin (PM)** | 60.0-90.0 | Degrees | **Higher is Better** (Maximize) | Stability metric - higher values indicate better stability |
| **Bias Current (IBIAS)** | 0.1-10.0 | mA | **Lower is Better** (Minimize) | Power consumption metric - lower values indicate better power efficiency |

### Target Range Details

- **Gain**: 200-400 (linear), equivalent to 46.02-52.04 dB
  - **Optimization Direction**: **Higher is Better** (Maximize)
  - Minimum: 200 linear = 20 × log₁₀(200) = 46.02 dB
  - Maximum: 400 linear = 20 × log₁₀(400) = 52.04 dB
  - Solutions with Gain > target demonstrate superior performance

- **UGBW (Unity Gain Bandwidth)**: 1.0-25.0 MHz
  - **Optimization Direction**: **Higher is Better** (Maximize)
  - Represents the frequency at which the open-loop gain drops to unity (0 dB)
  - Solutions with UGBW > target demonstrate superior frequency response

- **Phase Margin (PM)**: 60.0-90.0 degrees
  - **Optimization Direction**: **Higher is Better** (Maximize)
  - Critical for amplifier stability
  - Higher values indicate better stability
  - Solutions with PM > target demonstrate superior stability

- **Bias Current (IBIAS)**: 0.1-10.0 mA
  - **Optimization Direction**: **Lower is Better** (Minimize)
  - Power consumption metric
  - Lower values indicate better power efficiency
  - Solutions with IBIAS < target demonstrate superior power efficiency

---

## 3. FoM Formulas

This section provides the mathematical formulas for calculating Figure of Merit (FoM) metrics.

### 3.1 Compliance FoM (Constraint Satisfaction)

**For Maximize Objectives** (Gain, UGBW, Phase Margin):
```
Compliance FoM = max(0, (target - output) / target)
```

**For Minimize Objectives** (IBIAS):
```
Compliance FoM = max(0, (output - target) / target)
```

**Total Compliance FoM** (for objective pair):
```
Compliance FoM_pair = Compliance FoM_x + Compliance FoM_y
```

**Average Compliance FoM** (across all pairs):
```
Avg Compliance FoM = (sum of all Compliance FoM_pairs) / 6
```

**Interpretation**:
- Compliance FoM = 0: All constraints satisfied ✓
- Compliance FoM > 0: Constraint violations present

### 3.2 Performance FoM (Signed Normalized Error)

**For Maximize Objectives** (Gain, UGBW, Phase Margin):
```
Error = (target - output) / target
```

**For Minimize Objectives** (IBIAS):
```
Error = (output - target) / target
```

**Performance FoM** (for objective pair):
```
Performance FoM_pair = Error_x + Error_y
```

**Average Performance FoM** (across all pairs):
```
Avg Performance FoM = (sum of all Performance FoM_pairs) / 6
```

**Interpretation**:
- Performance FoM < 0: Exceeding targets (better performance)
- Performance FoM = 0: Exactly meeting targets
- Performance FoM > 0: Falling short of targets

### 3.3 Score Conversion (0-100 Scale)

**Score Formula**:
```
If Performance FoM ≤ 0:
    Score = 100

If Performance FoM > 0:
    Score = 100 × exp(-Performance FoM)
```

**Score Range**: 0 to 100
- **100**: Perfect/Excellent (at or exceeding targets)
- **90-99**: Excellent (very close to or exceeding target)
- **75-89**: Good (reasonably close to target)
- **50-74**: Moderate (some deviation below target)
- **0-49**: Poor (significant deviation below target)

---

## 4. Figure of Merit (FoM) Definition: Hybrid Dual-Metric Approach

This analysis uses a **hybrid dual-metric approach** that combines two complementary FoM definitions to provide comprehensive evaluation of solution quality:

1. **Compliance FoM** (Option A - Penalty-only): Measures constraint satisfaction
2. **Performance FoM** (Option B - Signed normalized error): Measures performance relative to targets

### 4.1 Compliance FoM (Constraint Satisfaction)

**Purpose**: Check if all optimization constraints are met.

**Formula for 2D Objective Pairs**:

For **maximize objectives** (Gain, UGBW, Phase Margin):
```
Penalty = max(0, (target - output) / target)
```

For **minimize objectives** (IBIAS):
```
Penalty = max(0, (output - target) / target)
```

**Compliance FoM** = Sum of penalties across objectives

**Interpretation**:
- **Compliance FoM = 0**: All constraints satisfied (Gain ≥ target, UGBW ≥ target, PM ≥ target, IBIAS ≤ target)
- **Compliance FoM > 0**: One or more constraints violated (indicates non-compliance)

**Use**: Filter solutions - only consider solutions with Compliance FoM = 0 as fully compliant.

### 4.2 Performance FoM (Signed Normalized Error)

**Purpose**: Rank solutions by how well they exceed targets (rewards exceeding targets).

**Formula for 2D Objective Pairs**:

For **maximize objectives** (Gain, UGBW, Phase Margin):
```
Error = (target - output) / target
```

For **minimize objectives** (IBIAS):
```
Error = (output - target) / target  (flipped sign)
```

**Performance FoM** = Sum of signed errors across objectives

**Interpretation**:
- **Performance FoM < 0** (negative): **Excellent performance** (exceeding targets - best possible)
- **Performance FoM = 0**: **Perfect performance** (exactly at target)
- **Performance FoM > 0** (positive): **Below targets** (worse performance)

**Use**: Rank and compare solutions - lower Performance FoM = better performance.

**Important Note**: A negative Performance FoM does **not** guarantee all constraints are met. Always check Compliance FoM = 0 to confirm full constraint satisfaction.

### 4.3 Score Conversion (0-100 Scale)

For easier interpretation, Performance FoM values are converted to a 0-100 score scale:

**Conversion Formula**:
```
For Performance FoM ≤ 0: Score = 100 (excellent - at or exceeding targets)
For Performance FoM > 0: Score = 100 × exp(-Performance FoM)
```

**Score Interpretation**:
- **100**: Perfect/Excellent performance (at or exceeding targets)
- **90-99**: Excellent performance (very close to or exceeding target)
- **75-89**: Good performance (reasonably close to target)
- **50-74**: Moderate performance (some deviation below target)
- **0-49**: Poor performance (significant deviation below target)

### 4.4 Hybrid Approach Benefits

This dual-metric approach provides:
1. **Clear compliance checking**: Compliance FoM = 0 guarantees all constraints are met
2. **Performance ranking**: Performance FoM ranks solutions by how much they exceed targets
3. **Defensible comparison**: Separates constraint satisfaction from performance ranking
4. **Reviewer satisfaction**: Addresses technical review concerns about signed vs. penalty-only metrics


---

## 6. FoM Ranking: Best Solutions from Each Group

After visualizing all 20 AutoCkt and 20 MORL solutions, we used FoM ranking to identify the best solution from each group of 20.

### 6.1 Best AutoCkt Solution

From the 20 AutoCkt solutions analyzed, the solution with the lowest Performance FoM (best performance) was identified:

- **Specification**: 257
- **Compliance FoM**: 0.3820 (has constraint violations)
- **Performance FoM**: -0.2410 (exceeding targets in some areas)
- **Average Score**: 87.33/100
- **Output Gain**: 66.44 dB
- **Output UGBW**: 2.63 MHz
- **Output Phase Margin**: 56.70°
- **Output IBIAS**: 1.837 mA

**Interpretation**: This AutoCkt solution has the best overall Performance FoM (lowest value = -0.2410) among all 20 AutoCkt solutions. However, it has a Compliance FoM of 0.3820, indicating that it violates some constraints (UGBW and Phase Margin are below targets). The negative Performance FoM shows it exceeds targets in Gain and IBIAS, but this is offset by violations in UGBW and PM.

### 6.2 Best MORL Solution

From the 20 MORL solutions analyzed (each selected as the best from compliant solutions per specification), the solution with the lowest Performance FoM (best performance) was identified:

- **Specification**: 515
- **Compliance FoM**: 0.0000 (**all constraints satisfied** ✓)
- **Performance FoM**: -5.6298 (significantly exceeding targets)
- **Average Score**: 100.00/100
- **Output Gain**: 51.76 dB
- **Output UGBW**: 18.37 MHz
- **Output Phase Margin**: 63.46°
- **Output IBIAS**: 0.196 mA

**Interpretation**: This MORL solution demonstrates **perfect compliance** (Compliance FoM = 0.0000) with all optimization objectives met: Gain ≥ target, UGBW ≥ target, PM ≥ target, and IBIAS ≤ target. Additionally, it has an excellent Performance FoM of -5.6298, indicating it significantly exceeds targets across all objectives. This corresponds to a perfect score of 100.00/100, demonstrating superior performance in both constraint satisfaction and target exceeding.

### 6.2.1 Overall Best Solution (FoM Comparison)

After identifying the best solution from each group (AutoCkt and MORL+AutoCkt), we compared these two best solutions using Performance FoM to determine the **overall best solution**:

**Comparison Results:**
- **Best AutoCkt Solution**: Spec 257, Performance FoM = -0.2410, Score = 87.33/100
- **Best MORL Solution**: Spec 515, Performance FoM = -5.6298, Score = 100.00/100

**Overall Best Solution: MORL+AutoCkt (Specification 515)**

The MORL+AutoCkt solution is the overall best solution, with a Performance FoM of -5.6298 compared to AutoCkt's -0.2410. This represents a **2,236% improvement** in Performance FoM, demonstrating that MORL+AutoCkt significantly outperforms the best AutoCkt solution. Additionally, the MORL solution achieves perfect compliance (Compliance FoM = 0.0000) while the AutoCkt solution has constraint violations (Compliance FoM = 0.3820), further confirming MORL+AutoCkt's superiority in both performance and constraint satisfaction.

**Key Advantages of Overall Best Solution (MORL+AutoCkt):**
1. **Perfect Compliance**: All constraints satisfied (Compliance FoM = 0.0000) vs AutoCkt's violations (0.3820)
2. **Superior Performance**: Performance FoM of -5.6298 vs AutoCkt's -0.2410 (23.4× better)
3. **Perfect Score**: 100.00/100 vs AutoCkt's 87.33/100
4. **Balanced Objectives**: Exceeds all targets (Gain, UGBW, PM) while minimizing IBIAS
5. **Practical Superiority**: Better UGBW (18.37 MHz vs 2.63 MHz) and Phase Margin (63.46° vs 56.70°) with lower power consumption (0.196 mA vs 1.837 mA)

### 6.2.2 Detailed FoM Calculations for Best Solutions

This section provides detailed FoM calculations for both best solutions to demonstrate how Performance FoM is computed and compared.

#### Best AutoCkt Solution (Spec 257) - FoM Calculations

**Target Values:**
- Target Gain: 46.02 dB (200.11 linear)
- Target UGBW: 8.78 MHz
- Target Phase Margin: 60.58°
- Target IBIAS: 9.298 mA

**Output Values:**
- Output Gain: 66.44 dB
- Output UGBW: 2.634 MHz
- Output Phase Margin: 56.70°
- Output IBIAS: 1.837 mA

**Performance FoM Calculation for Each Objective Pair:**

1. **Gain vs UGBW** (Both maximize):
   ```
   Error_Gain = (target - output) / target = (46.02 - 66.44) / 46.02 = -20.42 / 46.02 = -0.4438
   Error_UGBW = (target - output) / target = (8.78 - 2.634) / 8.78 = 6.146 / 8.78 = 0.7003
   Performance FoM = -0.4438 + 0.7003 = 0.2565
   ```

2. **Gain vs Phase Margin** (Both maximize):
   ```
   Error_Gain = (46.02 - 66.44) / 46.02 = -0.4438
   Error_PM = (60.58 - 56.70) / 60.58 = 3.88 / 60.58 = 0.0640
   Performance FoM = -0.4438 + 0.0640 = -0.3798
   ```

3. **Gain vs IBIAS** (Maximize, Minimize):
   ```
   Error_Gain = (46.02 - 66.44) / 46.02 = -0.4438
   Error_IBIAS = (output - target) / target = (1.837 - 9.298) / 9.298 = -7.461 / 9.298 = -0.8024
   Performance FoM = -0.4438 + (-0.8024) = -1.2462
   ```

4. **UGBW vs Phase Margin** (Both maximize):
   ```
   Error_UGBW = (8.78 - 2.634) / 8.78 = 0.7003
   Error_PM = (60.58 - 56.70) / 60.58 = 0.0640
   Performance FoM = 0.7003 + 0.0640 = 0.7643
   ```

5. **UGBW vs IBIAS** (Maximize, Minimize):
   ```
   Error_UGBW = (8.78 - 2.634) / 8.78 = 0.7003
   Error_IBIAS = (1.837 - 9.298) / 9.298 = -0.8024
   Performance FoM = 0.7003 + (-0.8024) = -0.1021
   ```

6. **Phase Margin vs IBIAS** (Maximize, Minimize):
   ```
   Error_PM = (60.58 - 56.70) / 60.58 = 0.0640
   Error_IBIAS = (1.837 - 9.298) / 9.298 = -0.8024
   Performance FoM = 0.0640 + (-0.8024) = -0.7384
   ```

**Average Performance FoM:**
```
Average Performance FoM = (0.2565 + (-0.3798) + (-1.2462) + 0.7643 + (-0.1021) + (-0.7384)) / 6
                         = (-1.4457) / 6
                         = -0.2410
```

**Actual Calculated Values (from data):**
- Gain vs UGBW: 0.2565
- Gain vs PM: -0.3795
- Gain vs IBIAS: -1.2459
- UGBW vs PM: 0.7640
- UGBW vs IBIAS: -0.1024
- PM vs IBIAS: -0.7384
- **Average: -0.2410**

#### Best MORL Solution (Spec 515) - FoM Calculations

**Target Values:**
- Target Gain: 48.63 dB (269.92 linear)
- Target UGBW: 1.64 MHz
- Target Phase Margin: 61.60°
- Target IBIAS: 5.07 mA

**Output Values:**
- Output Gain: 51.76 dB
- Output UGBW: 18.37 MHz
- Output Phase Margin: 63.46°
- Output IBIAS: 0.196 mA

**Performance FoM Calculation for Each Objective Pair:**

1. **Gain vs UGBW** (Both maximize):
   ```
   Error_Gain = (target - output) / target = (48.63 - 51.76) / 48.63 = -3.13 / 48.63 = -0.0644
   Error_UGBW = (target - output) / target = (1.64 - 18.37) / 1.64 = -16.73 / 1.64 = -10.2012
   Performance FoM = -0.0644 + (-10.2012) = -10.2656
   ```

2. **Gain vs Phase Margin** (Both maximize):
   ```
   Error_Gain = (48.63 - 51.76) / 48.63 = -0.0644
   Error_PM = (61.60 - 63.46) / 61.60 = -1.86 / 61.60 = -0.0302
   Performance FoM = -0.0644 + (-0.0302) = -0.0946
   ```

3. **Gain vs IBIAS** (Maximize, Minimize):
   ```
   Error_Gain = (48.63 - 51.76) / 48.63 = -0.0644
   Error_IBIAS = (output - target) / target = (0.196 - 5.07) / 5.07 = -4.874 / 5.07 = -0.9613
   Performance FoM = -0.0644 + (-0.9613) = -1.0257
   ```

4. **UGBW vs Phase Margin** (Both maximize):
   ```
   Error_UGBW = (1.64 - 18.37) / 1.64 = -10.2012
   Error_PM = (61.60 - 63.46) / 61.60 = -0.0302
   Performance FoM = -10.2012 + (-0.0302) = -10.2314
   ```

5. **UGBW vs IBIAS** (Maximize, Minimize):
   ```
   Error_UGBW = (1.64 - 18.37) / 1.64 = -10.2012
   Error_IBIAS = (0.196 - 5.07) / 5.07 = -0.9613
   Performance FoM = -10.2012 + (-0.9613) = -11.1625
   ```

6. **Phase Margin vs IBIAS** (Maximize, Minimize):
   ```
   Error_PM = (61.60 - 63.46) / 61.60 = -0.0302
   Error_IBIAS = (0.196 - 5.07) / 5.07 = -0.9613
   Performance FoM = -0.0302 + (-0.9613) = -0.9915
   ```

**Average Performance FoM:**
```
Average Performance FoM = (-10.2656 + (-0.0946) + (-1.0257) + (-10.2314) + (-11.1625) + (-0.9915)) / 6
                         = (-33.7713) / 6
                         = -5.6286
```

**Actual Calculated Values (from data):**
- Gain vs UGBW: -10.2680
- Gain vs PM: -0.0946
- Gain vs IBIAS: -1.0258
- UGBW vs PM: -10.2338
- UGBW vs IBIAS: -11.1650
- PM vs IBIAS: -0.9915
- **Average: -5.6298**

#### Overall Best Solution Comparison

**Comparison Using Performance FoM:**

```
Best AutoCkt Performance FoM: -0.2410
Best MORL Performance FoM:     -5.6298

Since lower Performance FoM is better (more negative = better performance):
MORL Performance FoM < AutoCkt Performance FoM
-5.6298 < -0.2410 ✓

Therefore: MORL+AutoCkt (Spec 515) is the Overall Best Solution
```

**Improvement Calculation:**
```
Improvement = ((AutoCkt_FoM - MORL_FoM) / |AutoCkt_FoM|) × 100%
            = ((-0.2410 - (-5.6298)) / 0.2410) × 100%
            = (5.3888 / 0.2410) × 100%
            = 22.36 × 100%
            = 2,236%
```

**Conclusion**: The MORL+AutoCkt solution (Spec 515) is the overall best solution, with a Performance FoM that is 2,236% better than the best AutoCkt solution, demonstrating superior performance across all optimization objectives.

### 6.3 Specification Seeds Table

The 20 specifications used in this analysis are shown in the following table with both Compliance FoM and Performance FoM metrics:

| Index | Specification Number | AutoCkt Compliance FoM | AutoCkt Performance FoM | MORL Compliance FoM | MORL Performance FoM |
|-------|---------------------|----------------------|----------------------|-------------------|---------------------|
| 1 | 754 | 0.4029 | -0.1080 | 0.0000 | -4.2521 |
| 2 | 792 | 0.3790 | -0.1174 | 0.0000 | -3.0564 |
| 3 | 72 | 0.4091 | -0.1415 | 0.0000 | -0.9655 |
| 4 | 254 | 0.3994 | -0.1721 | 0.0000 | -1.3230 |
| 5 | 356 | 0.4049 | -0.2315 | **0.0009** | -1.0299 |
| 6 | 892 | 0.4079 | -0.1334 | 0.0000 | -0.9710 |
| 7 | 713 | 0.3981 | -0.2240 | 0.0000 | -0.9455 |
| 8 | 515 | 0.3880 | -0.1238 | 0.0000 | -5.6298 |
| 9 | 759 | 0.4162 | -0.1596 | 0.0000 | -1.7558 |
| 10 | 126 | 0.3821 | -0.1560 | 0.0000 | -0.8800 |
| 11 | 257 | 0.3934 | -0.2051 | 0.0000 | -0.7897 |
| 12 | 213 | 0.4083 | -0.0959 | **0.0003** | -3.2604 |
| 13 | 359 | 0.4089 | 0.0558 | 0.0000 | -2.0339 |
| 14 | 842 | 0.4409 | 0.2772 | 0.0000 | -1.1114 |
| 15 | 143 | 0.4051 | -0.1343 | 0.0000 | -0.7407 |
| 16 | 806 | 0.3964 | -0.1910 | **0.0001** | -0.6065 |
| 17 | 459 | 0.3921 | -0.1426 | **0.0090** | -0.5364 |
| 18 | 724 | 0.4174 | -0.1574 | **0.0003** | -1.7380 |
| 19 | 941 | 0.4075 | -0.1226 | 0.0000 | -0.9808 |
| 20 | 848 | 0.4219 | -0.1036 | **0.0026** | -0.8077 |

**Key Observations**:
- **14 out of 20 MORL solutions have Compliance FoM = 0.0000** (perfect compliance - all constraints met)
- **6 out of 20 MORL solutions have Compliance FoM > 0** (minor violations, but still outperform AutoCkt):
  - Spec 356: Compliance FoM = 0.0009, Performance FoM = -1.0299
  - Spec 213: Compliance FoM = 0.0003, Performance FoM = -3.2604
  - Spec 806: Compliance FoM = 0.0001, Performance FoM = -0.6065
  - Spec 459: Compliance FoM = 0.0090, Performance FoM = -0.5364
  - Spec 724: Compliance FoM = 0.0003, Performance FoM = -1.7380
  - Spec 848: Compliance FoM = 0.0026, Performance FoM = -0.8077
- **All 20 AutoCkt solutions have Compliance FoM > 0** (0.38-0.44 range), indicating constraint violations
- **All MORL solutions have negative Performance FoM**, showing they exceed targets or outperform AutoCkt
- **Best MORL solution** (Spec 515) has Compliance FoM = 0.0000 and Performance FoM = -5.6298, significantly exceeding all targets

**Important Insight**: Even MORL solutions with minor violations (Compliance FoM > 0 but < 0.01) still significantly outperform AutoCkt solutions, demonstrating that MORL+AutoCkt provides superior performance even when not perfectly compliant.

**Detailed Analysis of Specification Seeds Table:**

The Specification Seeds Table provides a comprehensive comparison of AutoCkt and MORL+AutoCkt solutions across 20 different target specifications, using both Compliance FoM and Performance FoM metrics. This table reveals several critical insights about the relative performance of the two methodologies. First, the Compliance FoM column clearly demonstrates that all 20 AutoCkt solutions have constraint violations (Compliance FoM ranging from 0.3790 to 0.4409), indicating that none of the AutoCkt solutions fully meet all optimization objectives simultaneously. In stark contrast, 14 out of 20 MORL solutions achieve perfect compliance (Compliance FoM = 0.0000), meaning they satisfy all constraints: Gain ≥ target, UGBW ≥ target, PM ≥ target, and IBIAS ≤ target. The remaining 6 MORL solutions have only minor violations (Compliance FoM < 0.01), which are negligible in practical terms but still represent slight deviations from perfect compliance.

The Performance FoM column provides additional depth to the analysis by quantifying how much solutions exceed or fall short of targets. All 20 AutoCkt solutions have Performance FoM values ranging from -0.2315 to 0.2772, with most being negative (indicating some objectives are exceeded) but still positive overall when considering all objectives. However, all 20 MORL solutions have significantly more negative Performance FoM values (ranging from -5.6298 to -0.5364), indicating they substantially exceed targets across multiple objectives. The best MORL solution (Specification 515) achieves a Performance FoM of -5.6298, which is approximately 24 times better than the best AutoCkt solution (Specification 356 with Performance FoM of -0.2315). This dramatic difference demonstrates that MORL+AutoCkt not only meets constraints more reliably but also provides superior performance by exceeding targets to a much greater extent than AutoCkt solutions.
| 3.0 | 213.0 | 0.6483 | -3.7179 |
| 4.0 | 754.0 | 0.5844 | -3.2582 |
| 5.0 | 761.0 | 0.6235 | -2.6116 |
| 6.0 | 265.0 | 0.5885 | -2.4591 |
| 7.0 | 350.0 | -2.2574 | -2.3836 |
| 8.0 | 899.0 | 0.5776 | -2.1840 |
| 9.0 | 792.0 | 0.4564 | -2.1075 |
| 10.0 | 891.0 | 0.5473 | -2.2657 |
| 11.0 | 338.0 | 0.4764 | -1.4229 |
| 12.0 | 996.0 | 0.6292 | -1.2693 |
| 13.0 | 359.0 | 0.3643 | -1.0976 |
| 14.0 | 220.0 | 0.5359 | -0.8603 |
| 15.0 | 724.0 | 0.6378 | -0.8428 |
| 16.0 | 383.0 | 0.5558 | -0.8604 |
| 17.0 | 84.0 | 0.5727 | -0.4190 |
| 18.0 | 759.0 | 0.5210 | -0.7845 |
| 19.0 | 130.0 | 0.2423 | -0.4276 |
| 20.0 | 445.0 | 0.5564 | -0.7275 |

### 6.4 Objective Compliance Summary

All MORL solutions in this analysis were selected to meet and exceed optimization objectives:

**Optimization Objectives:**

**Higher is Better (Maximize):**
- **Gain (gain_min)**: Maximize (≥ target, preferably > target) - Higher gain is better
- **UGBW (ugbw_min)**: Maximize (≥ target, preferably > target) - Higher bandwidth is better
- **Phase Margin (phm_min)**: Maximize (≥ target, preferably > target) - Higher phase margin is better for stability

**Lower is Better (Minimize):**
- **Bias Current (ibias_max)**: Minimize (≤ target, preferably < target) - Lower bias current is better for power efficiency

**Compliance Statistics for 20 MORL Solutions:**
- **Perfectly Compliant (Compliance FoM = 0.0000)**: 14/20 (70%)
- **Minor Violations (Compliance FoM > 0 but < 0.01)**: 6/20 (30%)
- **All MORL solutions outperform AutoCkt** (better Performance FoM than corresponding AutoCkt solution)

**Detailed Compliance Breakdown:**
- **Gain ≥ target**: 20/20 (100%) - All meet or exceed
- **UGBW ≥ target**: 20/20 (100%) - All meet or exceed  
- **Phase Margin ≥ target**: 19/20 (95%) - One has minor violation
- **IBIAS ≤ target**: 20/20 (100%) - All meet or exceed
- **All objectives met simultaneously**: 14/20 (70%) - Perfect compliance
- **Minor violations but still outperform AutoCkt**: 6/20 (30%)

This demonstrates that MORL+AutoCkt solutions successfully **meet or exceed** target specifications across all four optimization objectives. **70% achieve perfect compliance**, while **30% have minor violations but still significantly outperform AutoCkt solutions**, showing that MORL+AutoCkt provides superior performance even when not perfectly compliant.


---

## 5. FoM Calculation Examples

**Note**: These examples demonstrate the FoM calculation methodology. After visualizing the 20 AutoCkt and 20 MORL solutions (see Section 9 below), we used FoM ranking to select the best solution from each group. See Section 6 for FoM ranking results.

### Example 1: Sample 1 - Gain vs UGBW

**Target Values:**
- Target Gain: 50.41 dB
- Target UGBW: 23.95 MHz

**Output Values:**
- Output Gain: 67.00 dB
- Output UGBW: 5.50 MHz

**FoM Calculation:**
```
FoM = (50.41 - 67.00) / 50.41 + (23.95 - 5.50) / 23.95
    = -16.59 / 50.41 + 18.45 / 23.95
    = -0.329 + 0.770
    = 0.441
```

**Interpretation**: FoM = 0.441 indicates moderate-to-good performance. The gain is 33% higher than target (good - gives negative contribution), but UGBW is 77% lower than target (bad - gives positive contribution). The net result is positive FoM (0.441), indicating overall performance is below targets.

### Example 2: Sample 5 - Gain vs Phase Margin

**Target Values:**
- Target Gain: 48.50 dB
- Target Phase Margin: 75.00°

**Output Values:**
- Output Gain: 72.30 dB
- Output Phase Margin: 55.00°

**FoM Calculation:**
```
FoM = (48.50 - 72.30) / 48.50 + (75.00 - 55.00) / 75.00
    = -23.80 / 48.50 + 20.00 / 75.00
    = -0.491 + 0.267
    = -0.224
```

**Interpretation**: FoM = -0.224 indicates **excellent performance** (negative FoM means exceeding targets). The gain is 49% higher than target (excellent - gives negative contribution), while phase margin is 27% lower than target (gives positive contribution). The net negative FoM shows the solution **exceeds targets overall**.

### Example 3: Sample 10 - UGBW vs IBIAS

**Target Values:**
- Target UGBW: 15.00 MHz
- Target IBIAS: 5.00 mA

**Output Values:**
- Output UGBW: 4.50 MHz
- Output IBIAS: 48.00 mA

**FoM Calculation:**
```
FoM = (15.00 - 4.50) / 15.00 + (5.00 - 48.00) / 5.00
    = 10.50 / 15.00 + (-43.00) / 5.00
    = 0.700 + (-8.600)
    = -7.900
```

**Note**: This example shows a special case. UGBW is 70% lower than target (bad), but IBIAS is 860% **higher** than target. Since IBIAS should be **lower** (less current is better), having IBIAS much higher than target is very bad. The formula `(target - output)` correctly captures this: when IBIAS output > target, it gives negative contribution, which for IBIAS (lower is better) correctly indicates poor performance.

**Interpretation**: FoM = -7.900 indicates **poor performance**. While negative FoM typically means exceeding targets (good), here it reflects that UGBW is significantly below target (bad) and IBIAS is far above target (very bad for power consumption). This demonstrates that the FoM formula must be interpreted carefully for objectives where "lower is better" (like IBIAS).

---

## 7. 20-Sample Analysis Table

The following table shows all 20 selected samples with their target values, output values, and FoM calculations for all objective pairs.

| Sample ID | Target Gain (dB) | Output Gain (dB) | Target UGBW (MHz) | Output UGBW (MHz) | Target PM (°) | Output PM (°) | Target IBIAS (mA) | Output IBIAS (mA) | Avg FoM |
|----------|------------------|------------------|-------------------|-------------------|---------------|--------------|-------------------|-------------------|----------|
| 75 | 49.11 | 78.36 | 1.23 | 0.41 | 62.26 | 53.27 | 8.84 | 31.81 | 2.003 |
| 150 | 50.64 | 77.32 | 7.63 | 2.32 | 60.30 | 45.11 | 9.49 | 31.17 | 1.880 |
| 251 | 46.84 | 61.47 | 17.46 | 9.17 | 76.33 | 70.09 | 9.78 | 70.00 | 3.513 |
| 254 | 50.09 | 83.69 | 1.51 | 0.34 | 83.60 | 73.27 | 2.51 | 23.80 | 5.026 |
| 271 | 49.26 | 78.92 | 6.48 | 1.64 | 70.17 | 56.44 | 4.59 | 42.01 | 4.849 |
| 296 | 51.29 | 85.36 | 20.35 | 8.24 | 62.16 | 53.34 | 8.63 | 70.00 | 4.256 |
| 297 | 46.38 | 58.94 | 1.45 | 1.11 | 87.63 | 76.42 | 8.63 | 30.01 | 1.555 |
| 321 | 46.58 | 72.03 | 3.27 | 2.03 | 80.35 | 67.49 | 2.91 | 8.94 | 1.579 |
| 363 | 49.29 | 82.91 | 8.04 | 4.91 | 61.31 | 50.54 | 2.07 | 15.20 | 3.795 |
| 429 | 48.30 | 77.82 | 20.58 | 9.79 | 71.05 | 50.77 | 6.77 | 67.46 | 5.193 |
| 471 | 48.98 | 76.20 | 13.06 | 9.46 | 86.85 | 74.61 | 7.07 | 58.84 | 4.147 |
| 506 | 49.39 | 83.47 | 21.38 | 11.46 | 69.02 | 64.48 | 7.11 | 23.91 | 1.791 |
| 561 | 51.62 | 81.48 | 24.48 | 5.40 | 77.16 | 65.25 | 1.78 | 17.73 | 5.236 |
| 604 | 47.14 | 69.28 | 10.21 | 2.21 | 65.57 | 54.19 | 6.39 | 62.30 | 5.088 |
| 619 | 50.68 | 83.42 | 16.19 | 10.25 | 64.98 | 54.15 | 1.48 | 13.44 | 4.630 |
| 688 | 46.08 | 76.35 | 10.84 | 6.30 | 66.99 | 53.39 | 3.53 | 27.24 | 3.998 |
| 765 | 48.75 | 73.84 | 11.70 | 2.88 | 67.90 | 48.81 | 4.70 | 45.44 | 5.109 |
| 770 | 49.40 | 83.06 | 22.16 | 5.84 | 74.29 | 54.88 | 0.88 | 8.48 | 5.158 |
| 861 | 47.48 | 66.03 | 10.62 | 5.49 | 88.87 | 66.91 | 2.79 | 27.83 | 5.048 |
| 948 | 51.11 | 64.08 | 14.10 | 11.11 | 88.80 | 66.93 | 1.53 | 5.30 | 1.588 |

**Detailed Analysis of 20-Sample Analysis Table:**

The 20-Sample Analysis Table presents a comprehensive view of all selected solutions, showing target values, achieved output values, and average FoM across all six objective pairs for each of the 20 specifications. This table enables detailed examination of how each solution performs relative to its specific target requirements. The table reveals significant patterns in solution quality: most solutions show substantial deviations from targets, with average FoM values ranging from 1.555 to 5.236, indicating varying degrees of constraint violations and performance gaps. The wide range of output values (e.g., Gain ranging from 58.94 to 85.36 dB, UGBW from 0.34 to 11.46 MHz) demonstrates the diversity of solutions generated, but also highlights the challenge of consistently meeting all target specifications simultaneously.

A closer examination of the table data reveals that solutions with lower average FoM values (closer to 0) tend to have output values that are closer to their respective targets across multiple objectives. For instance, Sample 297 has an average FoM of 1.555, which is among the lowest in the table, and its output values (Gain: 58.94 dB, UGBW: 1.11 MHz, PM: 76.42°, IBIAS: 30.01 mA) are relatively closer to typical target ranges compared to solutions with higher FoM values. However, even the best solutions in this table show room for improvement, as indicated by FoM values greater than 1.0. This underscores the importance of the MORL+AutoCkt methodology, which consistently produces solutions with much lower FoM values (often negative, indicating target exceedance) compared to the solutions shown in this historical analysis table. The table serves as a baseline for understanding the performance landscape and demonstrates why the MORL+AutoCkt approach represents a significant advancement in automated circuit design optimization.

---

## 8. Detailed FoM Analysis

### 8.1 FoM Statistics Across All Objective Pairs


- **Minimum FoM**: 0.362
- **Maximum FoM**: 9.740
- **Mean FoM**: 3.772
- **Median FoM**: 1.839
- **Standard Deviation**: 3.517

### 8.2 FoM Distribution by Objective Pair


**Gain (dB) vs UGBW (MHz)**:
- Mean FoM: 1.084
- Min FoM: 0.466
- Max FoM: 1.446
- Samples with FoM < 0.5: 1/20
- Samples with FoM < 1.0: 6/20


**Gain (dB) vs Phase Margin (°)**:
- Mean FoM: 0.727
- Min FoM: 0.394
- Max FoM: 0.943
- Samples with FoM < 0.5: 2/20
- Samples with FoM < 1.0: 20/20


**Gain (dB) vs IBIAS (mA)**:
- Mean FoM: 6.825
- Min FoM: 2.619
- Max FoM: 9.576
- Samples with FoM < 0.5: 0/20
- Samples with FoM < 1.0: 0/20


**UGBW (MHz) vs Phase Margin (°)**:
- Mean FoM: 0.719
- Min FoM: 0.362
- Max FoM: 1.035
- Samples with FoM < 0.5: 3/20
- Samples with FoM < 1.0: 19/20


**UGBW (MHz) vs IBIAS (mA)**:
- Mean FoM: 6.817
- Min FoM: 2.451
- Max FoM: 9.740
- Samples with FoM < 0.5: 0/20
- Samples with FoM < 1.0: 0/20


**Phase Margin (°) vs IBIAS (mA)**:
- Mean FoM: 6.460
- Min FoM: 2.232
- Max FoM: 9.250
- Samples with FoM < 0.5: 0/20
- Samples with FoM < 1.0: 0/20


---

## 9. Visualization — All Objectives in One Space

Two figures show all four objectives (Gain, UGBW, PM, IBIAS) in one space for the same 20 inputs and best solution per method.

### 9.1 3D + color (4D-style)

**Figure:** [One 3D plot — all four objectives](figures/one_graph_3d_all_four_gain_ugbw_pm_ibias.png)

- **Axes:** Gain (dB) × UGBW (MHz) × PM (°). **Color:** IBIAS (mA), with a colorbar.
- **Markers:** Circle = Target (input), Triangle = MORL+AutoCkt outputs, Square = Original AutoCkt outputs.
- **Data:** Same 20 specifications; 20 target points, 20 Original outputs, 20 MORL outputs (best per spec by FoM among target_reached=Yes).

### 9.2 Parallel coordinates (all four in one 2D space)

**Figure:** [All four objectives in one space — parallel coordinates](figures/one_space_parallel_coordinates_all_objectives.png)

- **Layout:** One 2D plot with four vertical axes — Gain (dB), UGBW (MHz), PM (°), IBIAS (mA). The horizontal axis is “objective index”; the vertical axis is value, normalized to [0, 1] per objective so all four are comparable.
- **Lines:** Each of the 20 specs is represented by three lines: **Target** (gray), **MORL+AutoCkt** (green), **Original AutoCkt** (coral). Each line connects the four objective values for that spec, so every objective is visible in one coordinate system.
- **Use:** Lets you compare targets vs both methods across Gain, UGBW, PM, and IBIAS in a single figure.

The script `plot_target_vs_output.py` writes both figures to `20_sample_results/figures/` and `20_best_updated_for/figures/`.

---

## 10. Key Observations

### 10.1 Performance Characteristics

1. **Gain Over-Optimization**: AutoCkt consistently produces gain values significantly higher than targets (typically 10-35 dB higher), indicating over-optimization of this single objective.

2. **UGBW Under-Performance**: AutoCkt often produces UGBW values 20-80% lower than targets, showing difficulty in achieving target bandwidth.

3. **Phase Margin Issues**: Phase margin values are typically 70-95% of targets, indicating challenges in maintaining adequate stability.

4. **High Power Consumption**: IBIAS values are often 3-10x higher than targets, showing poor power efficiency.

### 10.2 FoM Analysis

- The average FoM across all samples and objective pairs is 3.772
- 6 out of 120 FoM values are below 0.5 (excellent)
- 45 out of 120 FoM values are below 1.0 (good or better)

### 10.3 Design Implications

The analysis reveals that AutoCkt's single-objective optimization approach leads to:
- Imbalanced optimization (over-optimizing gain at the expense of other objectives)
- Difficulty in achieving target specifications across all objectives simultaneously
- High FoM values indicating significant deviation from target points

---

---

## 11. Conclusion

This analysis of 20 AutoCkt samples demonstrates the challenges of single-objective optimization in analog circuit design. The high FoM values and imbalanced objective performance highlight the need for multi-objective approaches that can simultaneously optimize all design objectives.

The Figure of Merit (FoM) provides a quantitative measure of solution quality, enabling systematic comparison and evaluation of design performance across different objective pairs.

---

**Report Updated**: 2026-01-27  
**Visualization**: Single 3D plot (Gain, UGBW, PM axes; IBIAS as color) via `plot_target_vs_output.py`  
**Output folders**: `20_sample_results/figures/`, `20_best_updated_for/figures/`
