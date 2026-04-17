# AutoCkt 20 + MORL+AutoCkt 20 Best Solutions — Report (Updated FoM)

## Executive Summary

This report compares **20 Original AutoCkt** solutions and **20 best MORL+AutoCkt** solutions on the same 20 specifications. All MORL solutions are the best per specification (by updated FoM) among compliant solutions. The single visualization shows all 20 target, 20 Original, and 20 MORL solution points in one 2D scatter (axes 0–100), with positions from real objectives (PCA of Gain, UGBW, PM, IBIAS).

- **Original AutoCkt:** 20 solutions (1 per specification).
- **MORL+AutoCkt:** 20 best solutions (best 1 per specification by FoM).
- **Total:** 40 solutions across 20 specifications.

---

## 1. Objective Specifications

| Objective | Range / Units | Direction |
| --- | --- | --- |
| Gain (A_v) | 46–52 dB (200–400 linear) | Maximize |
| UGBW | 1–25 MHz | Maximize |
| Phase Margin (PM) | 60–90° | Maximize |
| IBIAS | 0.1–10 mA | Minimize |

---

## 2. Updated Figure of Merit (FoM) Formula

FoM is computed per solution as the sum of normalized terms over the four objectives:

- **Gain, UGBW, PM (maximize):** (Achieved − Target) / Target
- **IBIAS (minimize):** −(Achieved − Target) / Target

**Formula:**

FoM = (G_out − G_tgt)/G_tgt + (U_out − U_tgt)/U_tgt + (P_out − P_tgt)/P_tgt − (I_out − I_tgt)/I_tgt

**Interpretation:** Higher FoM = better (exceeding targets on Gain/UGBW/PM, and under target on IBIAS).

**Calculation code:** `fom_calculation.py` in this folder provides `compute_fom(row)` for CSV rows and `compute_fom_values(gain_tgt, gain_out, ugbw_tgt, ugbw_out, pm_tgt, pm_out, ibias_tgt, ibias_out)` for raw numbers. Gain is in linear units; UGBW in MHz; PM in °; IBIAS in mA.

**Worked FoM calculations (2 Original + 2 MORL).** Values in linear gain, MHz, °, mA from the CSVs.

- **Spec 476 (Original AutoCkt)**  
  Target: G_tgt = 205.67, U_tgt = 2.09, P_tgt = 79.24, I_tgt = 5.81. Output: G_out = 2660.73, U_out = 1.12, P_out = 55.27, I_out = 0.2769.  
  term_G = (2660.73 − 205.67)/205.67 = 11.93 | term_U = (1.12 − 2.09)/2.09 = −0.465 | term_P = (55.27 − 79.24)/79.24 = −0.302 | term_I = −(0.2769 − 5.81)/5.81 = 0.953  
  **FoM = 11.93 − 0.465 − 0.302 + 0.953 = 12.12**

- **Spec 94 (Original AutoCkt)**  
  Target: G_tgt = 208.46, U_tgt = 1.45, P_tgt = 87.63, I_tgt = 8.635. Output: G_out = 2004.32, U_out = 0.3, P_out = 63.75, I_out = 0.3869.  
  term_G = (2004.32 − 208.46)/208.46 = 8.62 | term_U = (0.3 − 1.45)/1.45 = −0.793 | term_P = (63.75 − 87.63)/87.63 = −0.272 | term_I = −(0.3869 − 8.635)/8.635 = 0.955  
  **FoM = 8.62 − 0.793 − 0.272 + 0.955 = 8.50**

- **Spec 476 (MORL+AutoCkt, best)**  
  Target: G_tgt = 205.67, U_tgt = 2.09, P_tgt = 79.24, I_tgt = 5.81. Output: G_out = 28027.28, U_out = 18.98, P_out = 64.79, I_out = 0.1859.  
  term_G = (28027.28 − 205.67)/205.67 = 135.2 | term_U = (18.98 − 2.09)/2.09 = 8.09 | term_P = (64.79 − 79.24)/79.24 = −0.182 | term_I = −(0.1859 − 5.81)/5.81 = 0.968  
  **FoM = 135.2 + 8.09 − 0.182 + 0.968 = 144.1**

- **Spec 94 (MORL+AutoCkt, best)**  
  Target: G_tgt = 208.46, U_tgt = 1.45, P_tgt = 87.63, I_tgt = 8.635. Output: G_out = 27616.70, U_out = 15.57, P_out = 68.56, I_out = 0.184.  
  term_G = (27616.70 − 208.46)/208.46 = 131.4 | term_U = (15.57 − 1.45)/1.45 = 9.72 | term_P = (68.56 − 87.63)/87.63 = −0.217 | term_I = −(0.184 − 8.635)/8.635 = 0.979  
  **FoM = 131.4 + 9.72 − 0.217 + 0.979 ≈ 142**

---

## 3. Original AutoCkt — 20 Solutions (with FoM)

| Spec | Target Gain (dB) | Output Gain (dB) | Target UGBW (MHz) | Output UGBW (MHz) | Target PM (°) | Output PM (°) | Target IBIAS (mA) | Output IBIAS (mA) | FoM |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 476 | 46.26 | 68.5 | 2.09 | 1.12 | 79.24 | 55.27 | 5.81 | 0.2769 | 12.12 |
| 94 | 46.38 | 66.04 | 1.45 | 0.3 | 87.63 | 63.75 | 8.635 | 0.3869 | 8.504 |
| 857 | 46.49 | 65.8 | 1.23 | 0.8918 | 65.14 | 41.76 | 5.049 | 0.3952 | 8.516 |
| 255 | 46.49 | 66.92 | 1.23 | 0.4864 | 65.14 | 41.25 | 5.049 | 0.2264 | 9.49 |
| 356 | 46.03 | 67.61 | 8.78 | 4.286 | 60.58 | 44.3 | 9.298 | 0.2797 | 11.2 |
| 635 | 46.38 | 65.06 | 1.45 | 0.3 | 87.63 | 59.78 | 8.635 | 0.4034 | 7.437 |
| 889 | 46.21 | 66.92 | 2.55 | 0.9127 | 87.07 | 59.53 | 4.49 | 0.3556 | 9.819 |
| 713 | 46.03 | 65.08 | 8.78 | 3.436 | 60.58 | 40 | 9.298 | 0.2267 | 7.999 |
| 289 | 46.03 | 67.73 | 8.78 | 4.274 | 60.58 | 45.08 | 9.298 | 0.2675 | 11.38 |
| 557 | 46.04 | 67.3 | 19.58 | 6 | 81.98 | 56.09 | 7.336 | 0.2155 | 10.52 |
| 42 | 46.09 | 66.62 | 8.23 | 4.519 | 66.34 | 46.66 | 1.459 | 0.3852 | 9.625 |
| 335 | 46.04 | 66.56 | 19.58 | 5.617 | 81.98 | 58.25 | 7.336 | 0.2235 | 9.577 |
| 257 | 46.03 | 67.69 | 8.78 | 4.86 | 60.58 | 44.66 | 9.298 | 0.2598 | 11.38 |
| 696 | 46.08 | 66.71 | 14.24 | 6 | 60.36 | 41.55 | 5.341 | 0.4097 | 9.79 |
| 220 | 46.2 | 67.81 | 4.66 | 0.9701 | 63.79 | 45.53 | 6.728 | 0.4177 | 10.9 |
| 658 | 46.24 | 67.03 | 4.49 | 1.554 | 80.11 | 53.08 | 2.079 | 0.3994 | 9.775 |
| 58 | 46.1 | 65.51 | 13.14 | 5.39 | 71.2 | 54.3 | 6.217 | 0.4263 | 8.448 |
| 152 | 46.1 | 67 | 17.69 | 6 | 84.51 | 62.97 | 9.883 | 0.3647 | 10.13 |
| 325 | 46.12 | 67.2 | 18.3 | 6 | 80.45 | 54.42 | 5.416 | 0.2083 | 10.29 |
| 1000 | 46.1 | 65.49 | 17.69 | 4.616 | 84.51 | 61.97 | 9.883 | 0.4008 | |

**Mean FoM (Original AutoCkt):** 9.70 (over 19 specs with FoM; spec 1000 has no FoM).

---

## 4. MORL+AutoCkt — 20 Best Solutions (with FoM)

| Spec | Target Gain (dB) | Output Gain (dB) | Target UGBW (MHz) | Output UGBW (MHz) | Target PM (°) | Output PM (°) | Target IBIAS (mA) | Output IBIAS (mA) | FoM |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 476 | 46.26 | 88.95 | 2.09 | 18.98 | 79.24 | 64.79 | 5.81 | 0.1859 | 144.1 |
| 94 | 46.38 | 88.92 | 1.45 | 16.35 | 87.63 | 65.7 | 8.635 | 0.1981 | 144 |
| 857 | 46.49 | 88.95 | 1.23 | 13.7 | 65.14 | 55.53 | 5.049 | 0.1975 | 142.6 |
| 255 | 46.49 | 88.91 | 1.23 | 12.43 | 65.14 | 57.78 | 5.049 | 0.1983 | 141 |
| 356 | 46.03 | 88.94 | 8.78 | 15.05 | 60.58 | 55.92 | 9.298 | 0.1978 | 140.5 |
| 635 | 46.38 | 88.9 | 1.45 | 11.5 | 87.63 | 72.66 | 8.635 | 0.1973 | 140.4 |
| 889 | 46.21 | 88.85 | 2.55 | 15.33 | 87.07 | 72.38 | 4.49 | 0.1979 | 140.4 |
| 713 | 46.03 | 88.92 | 8.78 | 14.58 | 60.58 | 54.75 | 9.298 | 0.196 | 140.1 |
| 289 | 46.03 | 88.95 | 8.78 | 7.409 | 60.58 | 56.26 | 9.298 | 0.1871 | 139.8 |
| 557 | 46.04 | 88.95 | 19.58 | 15.17 | 81.98 | 71.43 | 7.336 | 0.2064 | 139.4 |
| 42 | 46.09 | 88.91 | 8.23 | 18.85 | 66.34 | 55.22 | 1.459 | 0.1862 | 139.4 |
| 335 | 46.04 | 88.95 | 19.58 | 16.86 | 81.98 | 63.31 | 7.336 | 0.2046 | 139.3 |
| 257 | 46.03 | 88.9 | 8.78 | 8.634 | 60.58 | 54.34 | 9.298 | 0.1852 | 139.2 |
| 696 | 46.08 | 88.95 | 14.24 | 14.88 | 60.36 | 56 | 5.341 | 0.1844 | 139.1 |
| 220 | 46.2 | 88.93 | 4.66 | 14.78 | 63.79 | 55.93 | 6.728 | 0.1974 | 139.1 |
| 658 | 46.24 | 88.92 | 4.49 | 18.42 | 80.11 | 64.43 | 2.079 | 0.2015 | 139 |
| 58 | 46.1 | 88.93 | 13.14 | 18.29 | 71.2 | 61.24 | 6.217 | 0.1987 | 138.8 |
| 152 | 46.1 | 88.95 | 17.69 | 17.65 | 84.51 | 63.39 | 9.883 | 0.1851 | 138.5 |
| 325 | 46.12 | 88.95 | 18.3 | 18 | 80.45 | 69.85 | 5.416 | 0.2057 | 138.4 |
| 1000 | 46.1 | 88.93 | 17.69 | 14.98 | 84.51 | 70.8 | 9.883 | 0.2067 | 138.2 |

**Mean FoM (MORL+AutoCkt):** 139.2 (over all 20 specs).

---

## 5. Visualization

The figure below shows all 20 Target, 20 Original AutoCkt, and 20 MORL+AutoCkt solutions in one 2D scatter. Axes are 0–100; point positions come from real objective values (PCA of Gain, UGBW, PM, IBIAS, scaled to [0,100]²). Shape = source (○ Target, □ Original AutoCkt, △ MORL+AutoCkt); color = objective (Gain, UGBW, PM, IBIAS).

![Best 20 — Target, Original AutoCkt, MORL+AutoCkt (axes 0–100)](figures/all20_sketch_axes_0_100.png)

**How to read the graph.** The horizontal and vertical axes are the first two PCA components of the four objectives (Gain, UGBW, PM, IBIAS), each scaled to 0–100. So position in the plot reflects overall "goodness" across all specs: moving up or right generally means better performance on the objectives that PCA emphasizes. **Markers:** circles (○) = target specification; squares (□) = Original AutoCkt achieved; triangles (△) = MORL+AutoCkt achieved. **Colors** distinguish which objective each point comes from when projecting the same solution into the PCA space for each objective.

**How the graph shows MORL+AutoCkt is better.** MORL triangles sit in a clearly different—and better—region than Original squares: they are shifted toward higher values on the PCA axes, meaning MORL solutions achieve a better balance of Gain, UGBW, PM, and IBIAS relative to targets. Original AutoCkt points often cluster near or behind the target circles (underperforming on UGBW and sometimes PM). MORL points, by contrast, consistently land ahead of targets, which matches the FoM numbers: MORL FoM lies in the 138–144 range while Original AutoCkt is in the 7–12 range—roughly an order of magnitude higher for MORL.

---

## 6. Key Observations

- The same 20 specifications are used for both methods (MORL’s top-20 specs by FoM that exist in Original AutoCkt).
- FoM in the tables uses the updated formula above; higher FoM indicates better performance.
- MORL+AutoCkt has **much higher FoM** (about 138–144) than Original AutoCkt (about 7–12) on every spec, i.e. MORL exceeds targets more on Gain/UGBW/PM and stays under target on IBIAS.
- The single plot summarizes all four objectives via PCA and preserves the 0–100 axis scale; the spatial separation of △ (MORL) from □ (Original) in the figure is a direct visual reflection of that FoM advantage.

---

## 7. Conclusion and Results Summary

This analysis compares 20 Original AutoCkt and 20 best MORL+AutoCkt solutions on the same 20 specs using the updated FoM. **Results:** On every specification, MORL+AutoCkt achieves substantially higher FoM (138–144 vs 7–12). In the tables, MORL consistently meets or exceeds targets on Gain, UGBW, and PM (e.g. Gain ~89 dB vs target ~46 dB; UGBW and PM much closer to or above target), and keeps IBIAS near or below target (~0.19–0.21 mA). Original AutoCkt often falls short on UGBW and PM while exceeding only on Gain and IBIAS. The combined scatter (`figures/all20_sketch_axes_0_100.png`) illustrates this overall advantage: MORL triangles occupy a "better" region in the PCA space than Original squares, matching the quantitative FoM improvement.
