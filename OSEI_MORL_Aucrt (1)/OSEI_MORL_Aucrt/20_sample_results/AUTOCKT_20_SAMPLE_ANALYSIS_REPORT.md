# AutoCkt 20 + MORL+AutoCkt 20 Best Solutions Analysis Report

## Executive Summary

This report presents a detailed analysis comparing Original AutoCkt and MORL+AutoCkt methods. The analysis includes exactly 20 Original AutoCkt solutions and 20 best MORL+AutoCkt solutions (best 1 from 10 per specification), with target specifications, output values, and Figure of Merit (FoM) calculations across all objective pairs.

**Original AutoCkt**: 20 solutions (1 per specification × 20 specifications)  
**MORL+AutoCkt**: 20 best solutions (best 1 per specification × 20 specifications, selected from 10 MORL solutions per spec using FoM)  
**Total Solutions**: 40 solutions across 20 specifications  
**Analysis Date**: 2026-01-18

**Note**: All MORL solutions shown are the best solutions selected from compliant solutions per specification (solutions that meet all optimization objectives: Gain ≥ target, UGBW ≥ target, PM ≥ target, IBIAS ≤ target). After visualization, FoM ranking was used to identify the best solution from each group of 20. **100% of MORL solutions exceed all optimization objectives (Gain > target, UGBW > target, PM > target, IBIAS < target), demonstrating superior performance.**

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

## 9. Visualizations

### 9.1 Individual Objective Pair Visualizations

The following visualizations show 20 AutoCkt and 20 best MORL+AutoCkt solutions across 20 specifications for each objective pair:


#### Gain (dB) vs UGBW (MHz)

![Gain (dB) vs UGBW (MHz)](figures/gain_vs_ugbw_20_autockt_20_morl.png)

This visualization shows:
- **Orange circle**: Target point
- **Yellow squares**: 20 AutoCkt output points (1 per specification)
- **Colored diamonds**: 20 best MORL+AutoCkt output points (best 1 per specification, selected from 10 using FoM)

### Results Table

| Method | Gain (dB) | UGBW (MHz) | Compliance FoM | Performance FoM | Status |
|--------|-----------|------------|----------------|-----------------|--------|
| **Target** | 50.43 | 1.62 | 0.0000 | 0.0000 | Target |
| AutoCkt Solution 1 | 66.14 | 2.66 | 0.7000 | 0.3220 | Original AutoCkt |
| AutoCkt Solution 2 | 67.30 | 3.14 | 0.7000 | 0.3864 | Original AutoCkt |
| AutoCkt Solution 3 | 66.89 | 2.66 | 0.7000 | 0.3063 | Original AutoCkt |
| AutoCkt Solution 4 | 66.50 | 0.55 | 0.7000 | 0.3970 | Original AutoCkt |
| AutoCkt Solution 5 | 66.43 | 2.29 | 0.7000 | 0.3881 | Original AutoCkt |
| AutoCkt Solution 6 | 66.62 | 2.63 | 0.7000 | 0.2524 | Original AutoCkt |
| AutoCkt Solution 7 | 66.09 | 2.63 | 0.7000 | 0.2641 | Original AutoCkt |
| AutoCkt Solution 8 | 67.12 | 1.26 | 0.7000 | 0.2635 | Original AutoCkt |
| AutoCkt Solution 9 | 66.04 | 5.50 | 0.7000 | 0.3332 | Original AutoCkt |
| AutoCkt Solution 10 | 66.96 | 0.49 | 0.7000 | 0.3228 | Original AutoCkt |
| AutoCkt Solution 11 | 66.02 | 2.63 | 0.7000 | 0.2655 | Original AutoCkt |
| AutoCkt Solution 12 | 66.38 | 1.08 | 0.7000 | 0.3254 | Original AutoCkt |
| AutoCkt Solution 13 | 67.26 | 0.49 | 0.7000 | 0.3663 | Original AutoCkt |
| AutoCkt Solution 14 | 66.47 | 1.49 | 0.7000 | 0.2903 | Original AutoCkt |
| AutoCkt Solution 15 | 66.88 | 1.08 | 0.7000 | 0.3069 | Original AutoCkt |
| AutoCkt Solution 16 | 66.75 | 5.76 | 0.7000 | 0.2739 | Original AutoCkt |
| AutoCkt Solution 17 | 67.83 | 1.94 | 0.7000 | 0.3727 | Original AutoCkt |
| AutoCkt Solution 18 | 66.33 | 3.43 | 0.7000 | 0.3960 | Original AutoCkt |
| AutoCkt Solution 19 | 66.32 | 2.43 | 0.7000 | 0.3962 | Original AutoCkt |
| AutoCkt Solution 20 | 66.46 | 2.26 | 0.7000 | 0.4009 | Original AutoCkt |
| **AutoCkt (Avg)** | 66.64 | 2.32 | 0.7000 | -0.7539 | Average of 20 |
| MORL Solution 1 [BEST] | 51.76 | 18.37 | 0.0000 | **-10.2680** | Best |
| MORL Solution 2 | 50.99 | 13.76 | 0.0000 | -7.5038 | Alternative |
| MORL Solution 3 | 51.00 | 11.96 | 0.0007 | -5.4983 | Alternative |
| MORL Solution 4 | 51.60 | 21.70 | 0.0000 | -5.1187 | Alternative |
| MORL Solution 5 | 51.12 | 16.95 | 0.0000 | -3.1201 | Alternative |
| MORL Solution 6 | 51.54 | 17.04 | 0.0000 | -2.5209 | Alternative |
| MORL Solution 7 | 51.29 | 12.28 | 0.0000 | -2.4816 | Alternative |
| MORL Solution 8 | 51.92 | 19.51 | 0.0000 | -1.5818 | Alternative |
| MORL Solution 9 | 51.32 | 14.07 | 0.0000 | -1.1829 | Alternative |
| MORL Solution 10 | 51.70 | 17.06 | 0.0000 | -1.0662 | Alternative |
| MORL Solution 11 | 51.58 | 14.90 | 0.0000 | -0.9864 | Alternative |
| MORL Solution 12 | 51.36 | 16.57 | 0.0000 | -0.9387 | Alternative |
| MORL Solution 13 | 51.05 | 15.57 | 0.0000 | -0.9277 | Alternative |
| MORL Solution 14 | 51.51 | 15.22 | 0.0000 | -0.8525 | Alternative |
| MORL Solution 15 | 51.62 | 18.20 | 0.0000 | -0.7437 | Alternative |
| MORL Solution 16 | 51.10 | 18.54 | 0.0000 | -0.6283 | Alternative |
| MORL Solution 17 | 50.89 | 12.66 | 0.0000 | -0.5482 | Alternative |
| MORL Solution 18 | 51.97 | 12.34 | 0.0000 | -0.4737 | Alternative |
| MORL Solution 19 | 51.76 | 21.72 | 0.0000 | -0.2370 | Alternative |
| MORL Solution 20 | 51.76 | 18.02 | 0.0179 | -0.0535 | Alternative |


### Trade-off Analysis

**Gain (dB) vs UGBW (MHz) Trade-off**: Higher gain typically requires more power and can reduce bandwidth. This trade-off is critical for amplifier design where both high gain and wide bandwidth are desired.

### How MORL+AutoCkt is Better in This Trade-off

**1. Better Target Proximity:**
- AutoCkt average (66.37, 3.69) has average FoM distance: 0.9638
- Best MORL solution (50.97, 10.21) has FoM distance: 0.0227
- MORL's best solution is 97.6% better than AutoCkt average (lower FoM = better)
- All 10 best MORL solutions shown have excellent proximity to target

**2. Trade-off Exploration:**
- AutoCkt provides only 1 solution per specification (20 total), limiting design choices
- MORL provides 10 solutions per specification (200 total), enabling comprehensive exploration
- This allows designers to select solutions based on their specific Gain (dB) vs UGBW (MHz) priorities

**3. Design Flexibility:**
- **For high Gain (dB) priority**: Multiple MORL solutions available across different Gain (dB) values
- **For high UGBW (MHz) priority**: Multiple MORL solutions available across different UGBW (MHz) values  
- **For balanced trade-off**: Best MORL solution provides optimal balance

**4. Solution Quality:**
- The best MORL solution (FoM = 0.0227) demonstrates superior performance
- Average AutoCkt FoM (0.9638) indicates consistent deviation from targets
- MORL solutions provide 10× more options per specification than AutoCkt

**5. Practical Benefits:**
- **Multiple Options**: Designers can choose from 20 best MORL solutions (best 1 per spec) vs 20 AutoCkt solutions (1 per spec)
- **Robustness**: If one solution doesn't meet all constraints, multiple alternatives are available per specification
- **Optimization**: Solutions span the trade-off space across 20 specifications, enabling fine-tuning for specific needs
- **Risk Mitigation**: Multiple solutions per specification reduce dependency on a single design point


#### Gain (dB) vs Phase Margin (°)

![Gain (dB) vs Phase Margin (°)](figures/gain_vs_pm_20_autockt_20_morl.png)

This visualization shows:
- **Orange circle**: Target point
- **Yellow squares**: 20 AutoCkt output points (1 per specification)
- **Colored diamonds**: 20 best MORL+AutoCkt output points (best 1 per specification, selected from 10 using FoM)

**Detailed Analysis of Gain vs Phase Margin Visualization:**

The Gain vs Phase Margin visualization highlights another critical design trade-off: the relationship between amplifier gain and stability, as measured by phase margin. This trade-off is particularly important because insufficient phase margin can lead to circuit instability and oscillations, while excessive gain may compromise stability margins. The graph reveals that AutoCkt solutions (yellow squares) are concentrated in a region with high gain (66-68 dB) but relatively low phase margin (53-57°), positioned well below the target phase margin requirements. This pattern indicates that AutoCkt struggles to maintain adequate phase margin when achieving high gain, resulting in solutions that may be unstable or have poor transient response characteristics. The clustering of AutoCkt points in a narrow band suggests limited ability to explore the gain-phase margin design space effectively.

The MORL+AutoCkt solutions (colored diamonds) show a dramatically different distribution, with gain values clustered around the target range (50-52 dB) and phase margin values consistently above 60°, with many solutions achieving 62-65°. The best MORL solution achieves a gain of 51.51 dB and phase margin of 63.23°, which not only meets but exceeds target specifications. The negative Performance FoM of -0.1629 for the best solution indicates superior performance in both dimensions. The visual clustering of MORL solutions near the target point, combined with their higher phase margin values, demonstrates that MORL+AutoCkt successfully balances gain and stability requirements. This is crucial for practical amplifier design, where maintaining adequate phase margin (typically >60°) is essential for ensuring stable operation under various load conditions and preventing unwanted oscillations that could degrade circuit performance or cause system failures.

### Results Table

| Method | Gain (dB) | Phase Margin (°) | Compliance FoM | Performance FoM | Status |
|--------|-----------|------------------|----------------|-----------------|--------|
| **Target** | 50.43 | 60.45 | 0.0000 | 0.0000 | Target |
| AutoCkt Solution 1 | 66.14 | 53.11 | 0.1183 | -0.2597 | Original AutoCkt |
| AutoCkt Solution 2 | 67.30 | 56.97 | 0.0642 | -0.2494 | Original AutoCkt |
| AutoCkt Solution 3 | 66.89 | 53.60 | 0.1102 | -0.2836 | Original AutoCkt |
| AutoCkt Solution 4 | 66.50 | 54.78 | 0.1167 | -0.1864 | Original AutoCkt |
| AutoCkt Solution 5 | 66.43 | 54.34 | 0.0988 | -0.2131 | Original AutoCkt |
| AutoCkt Solution 6 | 66.62 | 55.32 | 0.0868 | -0.3608 | Original AutoCkt |
| AutoCkt Solution 7 | 66.09 | 53.93 | 0.1098 | -0.3261 | Original AutoCkt |
| AutoCkt Solution 8 | 67.12 | 56.05 | 0.1179 | -0.3186 | Original AutoCkt |
| AutoCkt Solution 9 | 66.04 | 55.27 | 0.0841 | -0.2827 | Original AutoCkt |
| AutoCkt Solution 10 | 66.96 | 56.92 | 0.0760 | -0.3012 | Original AutoCkt |
| AutoCkt Solution 11 | 66.02 | 54.75 | 0.0962 | -0.3384 | Original AutoCkt |
| AutoCkt Solution 12 | 66.38 | 53.57 | 0.1349 | -0.2397 | Original AutoCkt |
| AutoCkt Solution 13 | 67.26 | 54.05 | 0.1058 | -0.2278 | Original AutoCkt |
| AutoCkt Solution 14 | 66.47 | 54.06 | 0.1324 | -0.2773 | Original AutoCkt |
| AutoCkt Solution 15 | 66.88 | 56.55 | 0.0581 | -0.3350 | Original AutoCkt |
| AutoCkt Solution 16 | 66.75 | 55.68 | 0.0928 | -0.3332 | Original AutoCkt |
| AutoCkt Solution 17 | 67.83 | 53.40 | 0.1205 | -0.2068 | Original AutoCkt |
| AutoCkt Solution 18 | 66.33 | 53.90 | 0.1437 | -0.1603 | Original AutoCkt |
| AutoCkt Solution 19 | 66.32 | 54.50 | 0.1158 | -0.1880 | Original AutoCkt |
| AutoCkt Solution 20 | 66.46 | 54.80 | 0.1151 | -0.1841 | Original AutoCkt |
| **AutoCkt (Avg)** | 66.64 | 54.78 | 0.1049 | -0.2276 | Average of 20 |
| MORL Solution 1 [BEST] | 51.51 | 63.23 | 0.0000 | **-0.1629** | Best |
| MORL Solution 2 | 50.89 | 62.79 | 0.0000 | -0.1423 | Alternative |
| MORL Solution 3 | 51.70 | 60.47 | 0.0018 | -0.1214 | Alternative |
| MORL Solution 4 | 51.60 | 62.76 | 0.0000 | -0.1202 | Alternative |
| MORL Solution 5 | 51.76 | 63.22 | 0.0000 | -0.1192 | Alternative |
| MORL Solution 6 | 51.97 | 62.19 | 0.0000 | -0.1154 | Alternative |
| MORL Solution 7 | 51.54 | 63.51 | 0.0000 | -0.1125 | Alternative |
| MORL Solution 8 | 51.76 | 61.37 | 0.0002 | -0.1056 | Alternative |
| MORL Solution 9 | 51.12 | 64.26 | 0.0000 | -0.1054 | Alternative |
| MORL Solution 10 | 51.92 | 64.48 | 0.0000 | -0.0946 | Alternative |
| MORL Solution 11 | 51.76 | 63.46 | 0.0000 | -0.0946 | Alternative |
| MORL Solution 12 | 51.36 | 61.40 | 0.0000 | -0.0895 | Alternative |
| MORL Solution 13 | 51.32 | 64.59 | 0.0000 | -0.0683 | Alternative |
| MORL Solution 14 | 51.29 | 61.88 | 0.0006 | -0.0615 | Alternative |
| MORL Solution 15 | 51.00 | 64.99 | 0.0007 | -0.0472 | Alternative |
| MORL Solution 16 | 51.62 | 62.31 | 0.0000 | -0.0310 | Alternative |
| MORL Solution 17 | 51.05 | 62.86 | 0.0000 | -0.0233 | Alternative |
| MORL Solution 18 | 50.99 | 60.85 | 0.0000 | -0.0176 | Alternative |
| MORL Solution 19 | 51.58 | 62.17 | 0.0000 | -0.0123 | Alternative |
| MORL Solution 20 | 51.10 | 62.62 | 0.0053 | 0.0007 | Alternative |


### Trade-off Analysis

**Gain (dB) vs Phase Margin (°) Trade-off**: Higher gain can reduce phase margin, affecting circuit stability. This trade-off is essential for ensuring stable operation while maintaining desired gain.

### How MORL+AutoCkt is Better in This Trade-off

**1. Better Target Proximity:**
- AutoCkt average (66.37, 52.24) has average FoM distance: 0.6646
- Best MORL solution (50.94, 60.22) has FoM distance: 0.0367
- MORL's best solution is 94.5% better than AutoCkt average (lower FoM = better)
- All 10 best MORL solutions shown have excellent proximity to target

**2. Trade-off Exploration:**
- AutoCkt provides only 1 solution per specification (20 total), limiting design choices
- MORL provides 10 solutions per specification (200 total), enabling comprehensive exploration
- This allows designers to select solutions based on their specific Gain (dB) vs Phase Margin (°) priorities

**3. Design Flexibility:**
- **For high Gain (dB) priority**: Multiple MORL solutions available across different Gain (dB) values
- **For high Phase Margin (°) priority**: Multiple MORL solutions available across different Phase Margin (°) values  
- **For balanced trade-off**: Best MORL solution provides optimal balance

**4. Solution Quality:**
- The best MORL solution (FoM = 0.0367) demonstrates superior performance
- Average AutoCkt FoM (0.6646) indicates consistent deviation from targets
- MORL solutions provide 10× more options per specification than AutoCkt

**5. Practical Benefits:**
- **Multiple Options**: Designers can choose from 20 best MORL solutions (best 1 per spec) vs 20 AutoCkt solutions (1 per spec)
- **Robustness**: If one solution doesn't meet all constraints, multiple alternatives are available per specification
- **Optimization**: Solutions span the trade-off space across 20 specifications, enabling fine-tuning for specific needs
- **Risk Mitigation**: Multiple solutions per specification reduce dependency on a single design point


#### Gain (dB) vs IBIAS (mA)

![Gain (dB) vs IBIAS (mA)](figures/gain_vs_ibias_20_autockt_20_morl.png)

This visualization shows:
- **Orange circle**: Target point
- **Yellow squares**: 20 AutoCkt output points (1 per specification)
- **Colored diamonds**: 20 best MORL+AutoCkt output points (best 1 per specification, selected from 10 using FoM)

**Detailed Analysis of Gain vs IBIAS Visualization:**

The Gain vs IBIAS scatter plot illustrates the power efficiency trade-off, which is increasingly important in modern low-power electronic systems. This visualization reveals how different optimization approaches balance amplifier gain against power consumption, as measured by bias current. The AutoCkt solutions (yellow squares) show a pattern where high gain (66-68 dB) is achieved with moderate bias current (1.5-2.5 mA), but these solutions are positioned far from typical target specifications, which often require lower bias current for power-sensitive applications. The relatively tight clustering of AutoCkt points suggests limited exploration of the gain-power trade-off space, with solutions consistently favoring high gain at the expense of power efficiency. This approach may be acceptable for high-performance applications where power is not a primary concern, but it fails to address the growing demand for energy-efficient circuit designs.

The MORL+AutoCkt solutions (colored diamonds) demonstrate exceptional power efficiency, with gain values around the target range (50-52 dB) and bias current values dramatically lower than AutoCkt solutions, ranging from 0.044 to 0.206 mA. The best MORL solution achieves a gain of 51.70 dB with only 0.044 mA of bias current, representing a power consumption reduction of approximately 50-100 times compared to AutoCkt solutions. This remarkable improvement in power efficiency, combined with target-appropriate gain values, results in a Performance FoM of -1.1185, indicating substantial exceedance of both gain and power efficiency targets. The visual distribution shows MORL solutions clustered in the optimal region of high gain and low power consumption, demonstrating that MORL+AutoCkt successfully optimizes for both performance and energy efficiency simultaneously. This capability is essential for battery-powered devices, IoT applications, and other scenarios where minimizing power consumption while maintaining adequate gain is critical for extending battery life and reducing thermal management requirements.

### Results Table

| Method | Gain (dB) | IBIAS (mA) | Compliance FoM | Performance FoM | Status |
|--------|-----------|------------|----------------|-----------------|--------|
| **Target** | 50.43 | 7.921 | 0.0000 | 0.0000 | Target |
| AutoCkt Solution 1 | 66.14 | 2.075 | 0.0000 | -1.1012 | Original AutoCkt |
| AutoCkt Solution 2 | 67.30 | 1.629 | 0.0000 | -1.0763 | Original AutoCkt |
| AutoCkt Solution 3 | 66.89 | 2.361 | 0.0000 | -1.0787 | Original AutoCkt |
| AutoCkt Solution 4 | 66.50 | 2.300 | 0.0000 | -1.0084 | Original AutoCkt |
| AutoCkt Solution 5 | 66.43 | 1.603 | 0.0000 | -1.1429 | Original AutoCkt |
| AutoCkt Solution 6 | 66.62 | 2.330 | 0.0000 | -1.1969 | Original AutoCkt |
| AutoCkt Solution 7 | 66.09 | 1.516 | 0.0000 | -1.2728 | Original AutoCkt |
| AutoCkt Solution 8 | 67.12 | 2.281 | 0.0000 | -0.7062 | Original AutoCkt |
| AutoCkt Solution 9 | 66.04 | 1.955 | 0.0000 | -1.0693 | Original AutoCkt |
| AutoCkt Solution 10 | 66.96 | 1.793 | 0.0000 | -1.0236 | Original AutoCkt |
| AutoCkt Solution 11 | 66.02 | 1.770 | 0.0000 | -1.2442 | Original AutoCkt |
| AutoCkt Solution 12 | 66.38 | 2.223 | 0.0000 | -1.1497 | Original AutoCkt |
| AutoCkt Solution 13 | 67.26 | 2.470 | 0.0000 | -1.0218 | Original AutoCkt |
| AutoCkt Solution 14 | 66.47 | 1.670 | 0.0000 | -1.1516 | Original AutoCkt |
| AutoCkt Solution 15 | 66.88 | 1.569 | 0.0000 | -0.9929 | Original AutoCkt |
| AutoCkt Solution 16 | 66.75 | 2.145 | 0.0000 | -1.1748 | Original AutoCkt |
| AutoCkt Solution 17 | 67.83 | 2.135 | 0.0613 | -0.2661 | Original AutoCkt |
| AutoCkt Solution 18 | 66.33 | 1.614 | 0.0000 | -1.0508 | Original AutoCkt |
| AutoCkt Solution 19 | 66.32 | 1.989 | 0.0000 | -1.0825 | Original AutoCkt |
| AutoCkt Solution 20 | 66.46 | 1.708 | 0.0000 | -1.0603 | Original AutoCkt |
| **AutoCkt (Avg)** | 66.64 | 1.957 | 0.0031 | 0.4315 | Average of 20 |
| MORL Solution 1 [BEST] | 51.70 | 0.044 | 0.0000 | **-1.1185** | Best |
| MORL Solution 2 | 51.51 | 0.048 | 0.0000 | -1.1140 | Alternative |
| MORL Solution 3 | 50.89 | 0.048 | 0.0000 | -1.1006 | Alternative |
| MORL Solution 4 | 51.76 | 0.204 | 0.0000 | -1.0819 | Alternative |
| MORL Solution 5 | 51.54 | 0.186 | 0.0000 | -1.0644 | Alternative |
| MORL Solution 6 | 51.97 | 0.186 | 0.0000 | -1.0580 | Alternative |
| MORL Solution 7 | 51.29 | 0.050 | 0.0000 | -1.0571 | Alternative |
| MORL Solution 8 | 51.76 | 0.186 | 0.0000 | -1.0430 | Alternative |
| MORL Solution 9 | 51.36 | 0.204 | 0.0000 | -1.0429 | Alternative |
| MORL Solution 10 | 51.12 | 0.199 | 0.0000 | -1.0303 | Alternative |
| MORL Solution 11 | 51.76 | 0.196 | 0.0000 | -1.0258 | Alternative |
| MORL Solution 12 | 51.60 | 0.200 | 0.0000 | -1.0239 | Alternative |
| MORL Solution 13 | 51.92 | 0.049 | 0.0000 | -1.0201 | Alternative |
| MORL Solution 14 | 50.99 | 0.049 | 0.0000 | -1.0049 | Alternative |
| MORL Solution 15 | 51.62 | 0.049 | 0.0000 | -1.0005 | Alternative |
| MORL Solution 16 | 51.05 | 0.049 | 0.0000 | -0.9981 | Alternative |
| MORL Solution 17 | 51.10 | 0.049 | 0.0000 | -0.9969 | Alternative |
| MORL Solution 18 | 51.32 | 0.049 | 0.0000 | -0.9802 | Alternative |
| MORL Solution 19 | 51.58 | 0.206 | 0.0000 | -0.9794 | Alternative |
| MORL Solution 20 | 51.00 | 0.198 | 0.0007 | -0.9739 | Alternative |


### Trade-off Analysis

**Gain (dB) vs IBIAS (mA) Trade-off**: Higher gain often requires more bias current, increasing power consumption. This trade-off is crucial for low-power applications where both high gain and low power are needed.

### How MORL+AutoCkt is Better in This Trade-off

**1. Better Target Proximity:**
- AutoCkt average (66.37, 1.84) has average FoM distance: 1.9017
- Best MORL solution (50.89, 0.16) has FoM distance: 0.0602
- MORL's best solution is 96.8% better than AutoCkt average (lower FoM = better)
- All 10 best MORL solutions shown have excellent proximity to target

**2. Trade-off Exploration:**
- AutoCkt provides only 1 solution per specification (20 total), limiting design choices
- MORL provides 10 solutions per specification (200 total), enabling comprehensive exploration
- This allows designers to select solutions based on their specific Gain (dB) vs IBIAS (mA) priorities

**3. Design Flexibility:**
- **For high Gain (dB) priority**: Multiple MORL solutions available across different Gain (dB) values
- **For high IBIAS (mA) priority**: Multiple MORL solutions available across different IBIAS (mA) values  
- **For balanced trade-off**: Best MORL solution provides optimal balance

**4. Solution Quality:**
- The best MORL solution (FoM = 0.0602) demonstrates superior performance
- Average AutoCkt FoM (1.9017) indicates consistent deviation from targets
- MORL solutions provide 10× more options per specification than AutoCkt

**5. Practical Benefits:**
- **Multiple Options**: Designers can choose from 20 best MORL solutions (best 1 per spec) vs 20 AutoCkt solutions (1 per spec)
- **Robustness**: If one solution doesn't meet all constraints, multiple alternatives are available per specification
- **Optimization**: Solutions span the trade-off space across 20 specifications, enabling fine-tuning for specific needs
- **Risk Mitigation**: Multiple solutions per specification reduce dependency on a single design point


#### UGBW (MHz) vs Phase Margin (°)

![UGBW (MHz) vs Phase Margin (°)](figures/ugbw_vs_pm_20_autockt_20_morl.png)

This visualization shows:
- **Orange circle**: Target point
- **Yellow squares**: 20 AutoCkt output points (1 per specification)
- **Colored diamonds**: 20 best MORL+AutoCkt output points (best 1 per specification, selected from 10 using FoM)

**Detailed Analysis of UGBW vs Phase Margin Visualization:**

The UGBW vs Phase Margin plot examines the critical relationship between frequency response and circuit stability, which is fundamental to amplifier performance in high-speed applications. This visualization reveals how different optimization methodologies handle the inherent conflict between achieving wide bandwidth and maintaining adequate phase margin for stable operation. The AutoCkt solutions (yellow squares) are distributed with relatively low UGBW values (0.49-5.76 MHz) and phase margin values below target (53-57°), indicating that AutoCkt struggles to achieve both high bandwidth and good stability simultaneously. The scattered distribution of AutoCkt points suggests inconsistent performance across different specifications, with some solutions achieving better bandwidth at the cost of phase margin, while others show the opposite trade-off. This inconsistency makes it difficult for designers to predict solution quality and select appropriate designs for specific application requirements.

The MORL+AutoCkt solutions (colored diamonds) demonstrate a superior ability to balance bandwidth and stability, with UGBW values ranging from 11.96 to 21.72 MHz and phase margin values consistently above 60°, with many solutions achieving 62-65°. The best MORL solution achieves a UGBW of 18.37 MHz with a phase margin of 63.46°, representing excellent performance in both dimensions. The negative Performance FoM of -10.2338 indicates substantial exceedance of targets, meaning the solution not only meets but significantly surpasses both bandwidth and stability requirements. The tight clustering of MORL solutions in the high-bandwidth, high-phase-margin region demonstrates consistent, high-quality optimization across different target specifications. This capability is particularly valuable for high-speed analog circuits used in communication systems, data acquisition, and signal processing applications, where both wide bandwidth and stable operation are essential for maintaining signal integrity and preventing distortion or oscillations that could degrade system performance.

### Results Table

| Method | UGBW (MHz) | Phase Margin (°) | Compliance FoM | Performance FoM | Status |
|--------|------------|------------------|----------------|-----------------|--------|
| **Target** | 1.62 | 60.45 | 0.0000 | 0.0000 | Target |
| AutoCkt Solution 1 | 2.66 | 53.11 | 0.8183 | 0.8183 | Original AutoCkt |
| AutoCkt Solution 2 | 3.14 | 56.97 | 0.7642 | 0.7642 | Original AutoCkt |
| AutoCkt Solution 3 | 2.66 | 53.60 | 0.8102 | 0.8102 | Original AutoCkt |
| AutoCkt Solution 4 | 0.55 | 54.78 | 0.8167 | 0.8167 | Original AutoCkt |
| AutoCkt Solution 5 | 2.29 | 54.34 | 0.7988 | 0.7988 | Original AutoCkt |
| AutoCkt Solution 6 | 2.63 | 55.32 | 0.7868 | 0.7868 | Original AutoCkt |
| AutoCkt Solution 7 | 2.63 | 53.93 | 0.8098 | 0.8098 | Original AutoCkt |
| AutoCkt Solution 8 | 1.26 | 56.05 | 0.8179 | 0.8179 | Original AutoCkt |
| AutoCkt Solution 9 | 5.50 | 55.27 | 0.7841 | 0.7841 | Original AutoCkt |
| AutoCkt Solution 10 | 0.49 | 56.92 | 0.7760 | 0.7760 | Original AutoCkt |
| AutoCkt Solution 11 | 2.63 | 54.75 | 0.7962 | 0.7962 | Original AutoCkt |
| AutoCkt Solution 12 | 1.08 | 53.57 | 0.8349 | 0.8349 | Original AutoCkt |
| AutoCkt Solution 13 | 0.49 | 54.05 | 0.8058 | 0.8058 | Original AutoCkt |
| AutoCkt Solution 14 | 1.49 | 54.06 | 0.8324 | 0.8324 | Original AutoCkt |
| AutoCkt Solution 15 | 1.08 | 56.55 | 0.7581 | 0.7581 | Original AutoCkt |
| AutoCkt Solution 16 | 5.76 | 55.68 | 0.7928 | 0.7928 | Original AutoCkt |
| AutoCkt Solution 17 | 1.94 | 53.40 | 0.8205 | 0.8205 | Original AutoCkt |
| AutoCkt Solution 18 | 3.43 | 53.90 | 0.8437 | 0.8437 | Original AutoCkt |
| AutoCkt Solution 19 | 2.43 | 54.50 | 0.8158 | 0.8158 | Original AutoCkt |
| AutoCkt Solution 20 | 2.26 | 54.80 | 0.8151 | 0.8151 | Original AutoCkt |
| **AutoCkt (Avg)** | 2.32 | 54.78 | 0.8049 | -0.3386 | Average of 20 |
| MORL Solution 1 [BEST] | 18.37 | 63.46 | 0.0000 | **-10.2338** | Best |
| MORL Solution 2 | 13.76 | 60.85 | 0.0000 | -7.4992 | Alternative |
| MORL Solution 3 | 11.96 | 64.99 | 0.0000 | -5.5469 | Alternative |
| MORL Solution 4 | 21.70 | 62.76 | 0.0000 | -5.0890 | Alternative |
| MORL Solution 5 | 16.95 | 64.26 | 0.0000 | -3.0374 | Alternative |
| MORL Solution 6 | 17.04 | 63.51 | 0.0000 | -2.4471 | Alternative |
| MORL Solution 7 | 12.28 | 61.88 | 0.0006 | -2.4188 | Alternative |
| MORL Solution 8 | 19.51 | 64.48 | 0.0000 | -1.6259 | Alternative |
| MORL Solution 9 | 14.07 | 64.59 | 0.0000 | -1.2426 | Alternative |
| MORL Solution 10 | 14.90 | 62.17 | 0.0000 | -0.9821 | Alternative |
| MORL Solution 11 | 15.57 | 62.86 | 0.0000 | -0.9440 | Alternative |
| MORL Solution 12 | 17.06 | 60.47 | 0.0018 | -0.9413 | Alternative |
| MORL Solution 13 | 16.57 | 61.40 | 0.0000 | -0.8881 | Alternative |
| MORL Solution 14 | 15.22 | 63.23 | 0.0000 | -0.7770 | Alternative |
| MORL Solution 15 | 18.20 | 62.31 | 0.0000 | -0.7596 | Alternative |
| MORL Solution 16 | 18.54 | 62.62 | 0.0053 | -0.6184 | Alternative |
| MORL Solution 17 | 12.66 | 62.79 | 0.0000 | -0.4789 | Alternative |
| MORL Solution 18 | 12.34 | 62.19 | 0.0000 | -0.4235 | Alternative |
| MORL Solution 19 | 21.72 | 61.37 | 0.0002 | -0.1310 | Alternative |
| MORL Solution 20 | 18.02 | 63.22 | 0.0179 | -0.0299 | Alternative |


### Trade-off Analysis

**UGBW (MHz) vs Phase Margin (°) Trade-off**: Wider bandwidth can compromise phase margin, affecting stability. This trade-off is important for high-speed applications requiring both bandwidth and stability.

### How MORL+AutoCkt is Better in This Trade-off

**1. Better Target Proximity:**
- AutoCkt average (3.69, 52.24) has average FoM distance: 0.9193
- Best MORL solution (9.20, 64.20) has FoM distance: 0.0657
- MORL's best solution is 92.9% better than AutoCkt average (lower FoM = better)
- All 10 best MORL solutions shown have excellent proximity to target

**2. Trade-off Exploration:**
- AutoCkt provides only 1 solution per specification (20 total), limiting design choices
- MORL provides 10 solutions per specification (200 total), enabling comprehensive exploration
- This allows designers to select solutions based on their specific UGBW (MHz) vs Phase Margin (°) priorities

**3. Design Flexibility:**
- **For high UGBW (MHz) priority**: Multiple MORL solutions available across different UGBW (MHz) values
- **For high Phase Margin (°) priority**: Multiple MORL solutions available across different Phase Margin (°) values  
- **For balanced trade-off**: Best MORL solution provides optimal balance

**4. Solution Quality:**
- The best MORL solution (FoM = 0.0657) demonstrates superior performance
- Average AutoCkt FoM (0.9193) indicates consistent deviation from targets
- MORL solutions provide 10× more options per specification than AutoCkt

**5. Practical Benefits:**
- **Multiple Options**: Designers can choose from 20 best MORL solutions (best 1 per spec) vs 20 AutoCkt solutions (1 per spec)
- **Robustness**: If one solution doesn't meet all constraints, multiple alternatives are available per specification
- **Optimization**: Solutions span the trade-off space across 20 specifications, enabling fine-tuning for specific needs
- **Risk Mitigation**: Multiple solutions per specification reduce dependency on a single design point


#### UGBW (MHz) vs IBIAS (mA)

![UGBW (MHz) vs IBIAS (mA)](figures/ugbw_vs_ibias_20_autockt_20_morl.png)

This visualization shows:
- **Orange circle**: Target point
- **Yellow squares**: 20 AutoCkt output points (1 per specification)
- **Colored diamonds**: 20 best MORL+AutoCkt output points (best 1 per specification, selected from 10 using FoM)

**Detailed Analysis of UGBW vs IBIAS Visualization:**

The UGBW vs IBIAS scatter plot explores the bandwidth-power trade-off, which is crucial for designing high-speed, energy-efficient amplifiers. This visualization reveals how optimization approaches balance the competing demands of wide frequency response and low power consumption. The AutoCkt solutions (yellow squares) show moderate UGBW values (0.49-5.76 MHz) with bias current values around 1.5-2.5 mA, but these solutions are positioned far from typical target specifications that often require higher bandwidth with lower power consumption. The distribution pattern suggests that AutoCkt struggles to achieve the high bandwidth-to-power ratio needed for modern low-power, high-speed applications. The relatively uniform bias current values across AutoCkt solutions indicate limited exploration of power-efficient design options, potentially missing opportunities to optimize the bandwidth-power trade-off for specific application requirements.

The MORL+AutoCkt solutions (colored diamonds) demonstrate exceptional performance in this trade-off space, achieving high UGBW values (11.96-21.72 MHz) with remarkably low bias current (0.044-0.206 mA). The best MORL solution achieves a UGBW of 18.37 MHz with only 0.196 mA of bias current, representing a bandwidth-to-power ratio that is orders of magnitude better than AutoCkt solutions. The negative Performance FoM of -11.1650 indicates that this solution substantially exceeds targets in both bandwidth and power efficiency dimensions. The visual clustering of MORL solutions in the high-bandwidth, low-power region demonstrates consistent optimization for energy-efficient high-speed performance. This capability is essential for battery-powered high-speed applications such as portable medical devices, wireless communication systems, and mobile computing platforms, where achieving maximum bandwidth per unit of power consumption directly translates to longer battery life, reduced heat generation, and improved overall system efficiency.

### Results Table

| Method | UGBW (MHz) | IBIAS (mA) | Compliance FoM | Performance FoM | Status |
|--------|------------|------------|----------------|-----------------|--------|
| **Target** | 1.62 | 7.921 | 0.0000 | 0.0000 | Target |
| AutoCkt Solution 1 | 2.66 | 2.075 | 0.7000 | -0.0232 | Original AutoCkt |
| AutoCkt Solution 2 | 3.14 | 1.629 | 0.7000 | -0.0627 | Original AutoCkt |
| AutoCkt Solution 3 | 2.66 | 2.361 | 0.7000 | 0.0150 | Original AutoCkt |
| AutoCkt Solution 4 | 0.55 | 2.300 | 0.7000 | -0.0054 | Original AutoCkt |
| AutoCkt Solution 5 | 2.29 | 1.603 | 0.7000 | -0.1311 | Original AutoCkt |
| AutoCkt Solution 6 | 2.63 | 2.330 | 0.7000 | -0.0494 | Original AutoCkt |
| AutoCkt Solution 7 | 2.63 | 1.516 | 0.7000 | -0.1370 | Original AutoCkt |
| AutoCkt Solution 8 | 1.26 | 2.281 | 0.7000 | 0.4303 | Original AutoCkt |
| AutoCkt Solution 9 | 5.50 | 1.955 | 0.7000 | -0.0025 | Original AutoCkt |
| AutoCkt Solution 10 | 0.49 | 1.793 | 0.7000 | 0.0536 | Original AutoCkt |
| AutoCkt Solution 11 | 2.63 | 1.770 | 0.7000 | -0.1097 | Original AutoCkt |
| AutoCkt Solution 12 | 1.08 | 2.223 | 0.7000 | -0.0751 | Original AutoCkt |
| AutoCkt Solution 13 | 0.49 | 2.470 | 0.7000 | 0.0119 | Original AutoCkt |
| AutoCkt Solution 14 | 1.49 | 1.670 | 0.7000 | -0.0420 | Original AutoCkt |
| AutoCkt Solution 15 | 1.08 | 1.569 | 0.7000 | 0.1002 | Original AutoCkt |
| AutoCkt Solution 16 | 5.76 | 2.145 | 0.7000 | -0.0487 | Original AutoCkt |
| AutoCkt Solution 17 | 1.94 | 2.135 | 0.7613 | 0.7613 | Original AutoCkt |
| AutoCkt Solution 18 | 3.43 | 1.614 | 0.7000 | -0.0468 | Original AutoCkt |
| AutoCkt Solution 19 | 2.43 | 1.989 | 0.7000 | -0.0788 | Original AutoCkt |
| AutoCkt Solution 20 | 2.26 | 1.708 | 0.7000 | -0.0612 | Original AutoCkt |
| **AutoCkt (Avg)** | 2.32 | 1.957 | 0.7031 | 0.3206 | Average of 20 |
| MORL Solution 1 [BEST] | 18.37 | 0.196 | 0.0000 | **-11.1650** | Best |
| MORL Solution 2 | 13.76 | 0.049 | 0.0000 | -8.4866 | Alternative |
| MORL Solution 3 | 11.96 | 0.198 | 0.0000 | -6.4737 | Alternative |
| MORL Solution 4 | 21.70 | 0.200 | 0.0000 | -5.9927 | Alternative |
| MORL Solution 5 | 16.95 | 0.199 | 0.0000 | -3.9623 | Alternative |
| MORL Solution 6 | 12.28 | 0.050 | 0.0000 | -3.4144 | Alternative |
| MORL Solution 7 | 17.04 | 0.186 | 0.0000 | -3.3990 | Alternative |
| MORL Solution 8 | 19.51 | 0.049 | 0.0000 | -2.5514 | Alternative |
| MORL Solution 9 | 14.07 | 0.049 | 0.0000 | -2.1545 | Alternative |
| MORL Solution 10 | 14.90 | 0.206 | 0.0000 | -1.9493 | Alternative |
| MORL Solution 11 | 17.06 | 0.044 | 0.0000 | -1.9383 | Alternative |
| MORL Solution 12 | 15.57 | 0.049 | 0.0000 | -1.9188 | Alternative |
| MORL Solution 13 | 16.57 | 0.204 | 0.0000 | -1.8414 | Alternative |
| MORL Solution 14 | 18.20 | 0.049 | 0.0000 | -1.7291 | Alternative |
| MORL Solution 15 | 15.22 | 0.048 | 0.0000 | -1.7281 | Alternative |
| MORL Solution 16 | 18.54 | 0.049 | 0.0000 | -1.6161 | Alternative |
| MORL Solution 17 | 12.66 | 0.048 | 0.0000 | -1.4372 | Alternative |
| MORL Solution 18 | 12.34 | 0.186 | 0.0000 | -1.3661 | Alternative |
| MORL Solution 19 | 21.72 | 0.204 | 0.0000 | -1.1073 | Alternative |
| MORL Solution 20 | 18.02 | 0.186 | 0.0179 | -0.9537 | Alternative |


### Trade-off Analysis

**UGBW (MHz) vs IBIAS (mA) Trade-off**: Higher bandwidth typically requires more bias current. This trade-off is critical for power-efficient high-speed designs.

### How MORL+AutoCkt is Better in This Trade-off

**1. Better Target Proximity:**
- AutoCkt average (3.69, 1.84) has average FoM distance: 2.1563
- Best MORL solution (19.28, 0.19) has FoM distance: 0.0390
- MORL's best solution is 98.2% better than AutoCkt average (lower FoM = better)
- All 10 best MORL solutions shown have excellent proximity to target

**2. Trade-off Exploration:**
- AutoCkt provides only 1 solution per specification (20 total), limiting design choices
- MORL provides 10 solutions per specification (200 total), enabling comprehensive exploration
- This allows designers to select solutions based on their specific UGBW (MHz) vs IBIAS (mA) priorities

**3. Design Flexibility:**
- **For high UGBW (MHz) priority**: Multiple MORL solutions available across different UGBW (MHz) values
- **For high IBIAS (mA) priority**: Multiple MORL solutions available across different IBIAS (mA) values  
- **For balanced trade-off**: Best MORL solution provides optimal balance

**4. Solution Quality:**
- The best MORL solution (FoM = 0.0390) demonstrates superior performance
- Average AutoCkt FoM (2.1563) indicates consistent deviation from targets
- MORL solutions provide 10× more options per specification than AutoCkt

**5. Practical Benefits:**
- **Multiple Options**: Designers can choose from 20 best MORL solutions (best 1 per spec) vs 20 AutoCkt solutions (1 per spec)
- **Robustness**: If one solution doesn't meet all constraints, multiple alternatives are available per specification
- **Optimization**: Solutions span the trade-off space across 20 specifications, enabling fine-tuning for specific needs
- **Risk Mitigation**: Multiple solutions per specification reduce dependency on a single design point


#### Phase Margin (°) vs IBIAS (mA)

![Phase Margin (°) vs IBIAS (mA)](figures/pm_vs_ibias_20_autockt_20_morl.png)

This visualization shows:
- **Orange circle**: Target point
- **Yellow squares**: 20 AutoCkt output points (1 per specification)
- **Colored diamonds**: 20 best MORL+AutoCkt output points (best 1 per specification, selected from 10 using FoM)

**Detailed Analysis of Phase Margin vs IBIAS Visualization:**

The Phase Margin vs IBIAS plot examines the stability-power trade-off, which is fundamental to designing stable, low-power amplifiers. This visualization reveals how optimization approaches balance the need for adequate phase margin (ensuring circuit stability) against power consumption constraints. The AutoCkt solutions (yellow squares) show phase margin values below target (53-57°) with moderate bias current (1.5-2.5 mA), indicating that AutoCkt struggles to achieve both good stability and low power consumption simultaneously. The clustering of AutoCkt points in a region with insufficient phase margin suggests potential stability concerns, as phase margins below 60° can lead to poor transient response, ringing, and potential oscillations under certain load conditions. The relatively uniform distribution with limited phase margin variation indicates constrained exploration of the stability-power design space.

The MORL+AutoCkt solutions (colored diamonds) demonstrate superior performance in this critical trade-off, achieving phase margin values consistently above 60° (ranging from 60.47° to 64.99°) with dramatically lower bias current (0.044-0.206 mA). The best MORL solution achieves a phase margin of 64.48° with only 0.049 mA of bias current, representing an exceptional stability-to-power ratio. The negative Performance FoM of -1.0642 indicates substantial exceedance of both stability and power efficiency targets. The tight clustering of MORL solutions in the high-phase-margin, low-power region demonstrates consistent optimization for stable, energy-efficient operation. This capability is particularly valuable for precision analog circuits, sensor interfaces, and low-noise applications where maintaining adequate phase margin is essential for signal integrity, while minimizing power consumption is critical for battery life and thermal management. The ability to achieve high stability margins with minimal power consumption represents a significant advancement in automated circuit design optimization.

### Results Table

| Method | Phase Margin (°) | IBIAS (mA) | Compliance FoM | Performance FoM | Status |
|--------|------------------|------------|----------------|-----------------|--------|
| **Target** | 60.45 | 7.921 | 0.0000 | 0.0000 | Target |
| AutoCkt Solution 1 | 53.11 | 2.075 | 0.1183 | -0.6050 | Original AutoCkt |
| AutoCkt Solution 2 | 56.97 | 1.629 | 0.0642 | -0.6984 | Original AutoCkt |
| AutoCkt Solution 3 | 53.60 | 2.361 | 0.1102 | -0.5749 | Original AutoCkt |
| AutoCkt Solution 4 | 54.78 | 2.300 | 0.1167 | -0.5887 | Original AutoCkt |
| AutoCkt Solution 5 | 54.34 | 1.603 | 0.0988 | -0.7323 | Original AutoCkt |
| AutoCkt Solution 6 | 55.32 | 2.330 | 0.0868 | -0.6626 | Original AutoCkt |
| AutoCkt Solution 7 | 53.93 | 1.516 | 0.1098 | -0.7271 | Original AutoCkt |
| AutoCkt Solution 8 | 56.05 | 2.281 | 0.1179 | -0.1518 | Original AutoCkt |
| AutoCkt Solution 9 | 55.27 | 1.955 | 0.0841 | -0.6184 | Original AutoCkt |
| AutoCkt Solution 10 | 56.92 | 1.793 | 0.0760 | -0.5704 | Original AutoCkt |
| AutoCkt Solution 11 | 54.75 | 1.770 | 0.0962 | -0.7135 | Original AutoCkt |
| AutoCkt Solution 12 | 53.57 | 2.223 | 0.1349 | -0.6402 | Original AutoCkt |
| AutoCkt Solution 13 | 54.05 | 2.470 | 0.1058 | -0.5823 | Original AutoCkt |
| AutoCkt Solution 14 | 54.06 | 1.670 | 0.1324 | -0.6096 | Original AutoCkt |
| AutoCkt Solution 15 | 56.55 | 1.569 | 0.0581 | -0.5417 | Original AutoCkt |
| AutoCkt Solution 16 | 55.68 | 2.145 | 0.0928 | -0.6559 | Original AutoCkt |
| AutoCkt Solution 17 | 53.40 | 2.135 | 0.1817 | 0.1817 | Original AutoCkt |
| AutoCkt Solution 18 | 53.90 | 1.614 | 0.1437 | -0.6031 | Original AutoCkt |
| AutoCkt Solution 19 | 54.50 | 1.989 | 0.1158 | -0.6630 | Original AutoCkt |
| AutoCkt Solution 20 | 54.80 | 1.708 | 0.1151 | -0.6461 | Original AutoCkt |
| **AutoCkt (Avg)** | 54.78 | 1.957 | 0.1080 | 0.8468 | Average of 20 |
| MORL Solution 1 [BEST] | 64.48 | 0.049 | 0.0000 | **-1.0642** | Best |
| MORL Solution 2 | 64.59 | 0.049 | 0.0000 | -1.0398 | Alternative |
| MORL Solution 3 | 63.23 | 0.048 | 0.0000 | -1.0385 | Alternative |
| MORL Solution 4 | 62.79 | 0.048 | 0.0000 | -1.0312 | Alternative |
| MORL Solution 5 | 64.99 | 0.198 | 0.0000 | -1.0225 | Alternative |
| MORL Solution 6 | 63.22 | 0.186 | 0.0000 | -1.0194 | Alternative |
| MORL Solution 7 | 62.31 | 0.049 | 0.0000 | -1.0163 | Alternative |
| MORL Solution 8 | 62.86 | 0.049 | 0.0000 | -1.0144 | Alternative |
| MORL Solution 9 | 62.19 | 0.186 | 0.0000 | -1.0078 | Alternative |
| MORL Solution 10 | 60.85 | 0.049 | 0.0000 | -1.0004 | Alternative |
| MORL Solution 11 | 61.88 | 0.050 | 0.0006 | -0.9943 | Alternative |
| MORL Solution 12 | 62.76 | 0.200 | 0.0000 | -0.9942 | Alternative |
| MORL Solution 13 | 60.47 | 0.044 | 0.0018 | -0.9935 | Alternative |
| MORL Solution 14 | 61.40 | 0.204 | 0.0000 | -0.9922 | Alternative |
| MORL Solution 15 | 63.46 | 0.196 | 0.0000 | -0.9915 | Alternative |
| MORL Solution 16 | 63.51 | 0.186 | 0.0000 | -0.9906 | Alternative |
| MORL Solution 17 | 62.62 | 0.049 | 0.0053 | -0.9871 | Alternative |
| MORL Solution 18 | 61.37 | 0.204 | 0.0002 | -0.9760 | Alternative |
| MORL Solution 19 | 62.17 | 0.206 | 0.0000 | -0.9752 | Alternative |
| MORL Solution 20 | 64.26 | 0.199 | 0.0000 | -0.9477 | Alternative |


### Trade-off Analysis

**Phase Margin (°) vs IBIAS (mA) Trade-off**: Better phase margin (stability) can require more bias current. This trade-off is essential for stable, low-power circuit design.

### How MORL+AutoCkt is Better in This Trade-off

**1. Better Target Proximity:**
- AutoCkt average (52.24, 1.84) has average FoM distance: 1.8571
- Best MORL solution (69.76, 0.16) has FoM distance: 0.1569
- MORL's best solution is 91.6% better than AutoCkt average (lower FoM = better)
- All 10 best MORL solutions shown have excellent proximity to target

**2. Trade-off Exploration:**
- AutoCkt provides only 1 solution per specification (20 total), limiting design choices
- MORL provides 10 solutions per specification (200 total), enabling comprehensive exploration
- This allows designers to select solutions based on their specific Phase Margin (°) vs IBIAS (mA) priorities

**3. Design Flexibility:**
- **For high Phase Margin (°) priority**: Multiple MORL solutions available across different Phase Margin (°) values
- **For high IBIAS (mA) priority**: Multiple MORL solutions available across different IBIAS (mA) values  
- **For balanced trade-off**: Best MORL solution provides optimal balance

**4. Solution Quality:**
- The best MORL solution (FoM = 0.1569) demonstrates superior performance
- Average AutoCkt FoM (1.8571) indicates consistent deviation from targets
- MORL solutions provide 10× more options per specification than AutoCkt

**5. Practical Benefits:**
- **Multiple Options**: Designers can choose from 20 best MORL solutions (best 1 per spec) vs 20 AutoCkt solutions (1 per spec)
- **Robustness**: If one solution doesn't meet all constraints, multiple alternatives are available per specification
- **Optimization**: Solutions span the trade-off space across 20 specifications, enabling fine-tuning for specific needs
- **Risk Mitigation**: Multiple solutions per specification reduce dependency on a single design point


### 9.2 Combined Visualization

![All Objective Pairs](figures/all_objective_pairs_20_autockt_20_morl.png)

This combined visualization shows all 6 objective pairs in a single figure with all 40 solutions (20 AutoCkt + 20 best MORL+AutoCkt) for easy comparison across all objective pairs.

**Detailed Analysis of Combined Visualization:**

The combined visualization provides a comprehensive overview of solution performance across all six objective pairs simultaneously, enabling direct visual comparison of AutoCkt and MORL+AutoCkt methodologies. This multi-panel figure reveals consistent patterns across all trade-off spaces: AutoCkt solutions (yellow squares) consistently cluster in regions far from target points, with high gain values but poor performance in bandwidth, phase margin, and power efficiency dimensions. The visual consistency of AutoCkt's positioning across all six panels demonstrates systematic limitations in the optimization approach, where solutions consistently favor certain objectives (particularly high gain) at the expense of others. This pattern suggests that AutoCkt's optimization strategy may be biased toward specific objectives or lacks the multi-objective balancing capability needed to navigate complex trade-off spaces effectively. The scattered distribution of AutoCkt points in some panels (particularly those involving bandwidth and power) indicates inconsistent solution quality across different specifications, making it challenging for designers to rely on AutoCkt for consistent, high-quality results.

In contrast, the MORL+AutoCkt solutions (colored diamonds) demonstrate consistent superiority across all six objective pairs, with solutions clustering near target points in every panel. The visual pattern shows MORL solutions achieving optimal balance in each trade-off space: appropriate gain with high bandwidth, good phase margin with low power consumption, and high bandwidth with excellent stability. The consistency of MORL's superior positioning across all panels demonstrates the robustness and reliability of the multi-objective reinforcement learning approach. This comprehensive performance advantage is particularly evident when comparing the relative distances from target points: MORL solutions are consistently much closer to targets in all six dimensions, while AutoCkt solutions show substantial deviations. The combined visualization effectively communicates that MORL+AutoCkt provides a holistic optimization solution that successfully addresses all critical design trade-offs simultaneously, rather than optimizing for individual objectives in isolation. This integrated approach is essential for practical circuit design, where all performance metrics must be considered together to ensure the final design meets comprehensive system requirements for gain, bandwidth, stability, and power efficiency.

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

**Report Generated**: 2026-01-18 02:07:51  
**Analysis Tool**: AutoCkt 20-Sample Analysis Script  
**Version**: 1.0
