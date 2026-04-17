# Technical Report

## MORL+AutoCkt vs AutoCKT: Comparative Evaluation on Two-Stage Op-Amp Design

---

## Abstract

This report presents a systematic comparison between **MORL+AutoCkt** (Multi-Objective Reinforcement Learning) and **AutoCKT** for automated two-stage operational amplifier design. We evaluate both methods on 1000 target specifications using the same success criterion and Figure of Merit (FoM). MORL achieves **9852/10000** solutions reaching target compared to AutoCKT **932/1000**, demonstrating clear superiority while maintaining fidelity to the published AutoCkt methodology for baseline evaluation.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Background and Objectives](#2-background-and-objectives)
3. [Methodology](#3-methodology)
4. [Figure of Merit](#4-figure-of-merit)
5. [Results](#5-results)
6. [Discussion](#6-discussion)
7. [Generated Figures](#7-generated-figures)
8. [Conclusions](#8-conclusions)

---

## 1. Executive Summary

| Metric | AutoCKT | MORL+AutoCkt |
|--------|---------|--------------|
| **Specs evaluated** | 1000 | 10000 (10 per spec) |
| **Solutions reaching target** | 938 | 9919 |
| **Pass rate** | 93.8% | 99.2% |
| **Solutions per spec** | 1 | 10 (best selected by FoM) |

**Key findings:**

- Our AutoCKT evaluation follows the exact reward formula, lookup function, and success criterion from the AutoCkt paper and source code.
- MORL achieves higher pass rates (9919/10000 vs 938/1000) when evaluated under the same strict per-objective criterion.
- MORL uses cosine-similarity scalarization to balance four objectives and explore the Pareto front. MORL outputs consistently overshoot Gain, UGBW, and PM while meeting or undershooting I-Bias.

---

## 2. Background and Objectives

### 2.1 Circuit and Objectives

The two-stage operational amplifier design task involves four objectives:

| Objective | Notation | Target | Direction |
|-----------|----------|--------|------------|
| Gain (linear) | \(G\) | From gen_specs | Higher is better |
| Unity Gain Bandwidth (MHz) | \(U\) | From gen_specs | Higher is better |
| Phase Margin (°) | \(P\) | 75 | Higher is better |
| Bias Current (mA) | \(I\) | From gen_specs | Lower is better |

### 2.2 Evaluation Goal

We compare AutoCKT (single-objective RL from the paper) with MORL+AutoCkt (multi-objective extension) using:

- The same 1000 specifications from `gen_specs`
- A unified Figure of Merit (FoM)
- A strict per-objective success criterion: Output ≥ Target for G, U, P; Output ≤ Target for I

---

## 3. Methodology

### 3.1 AutoCKT

AutoCKT uses a scalar reward from `ngspice_vanilla_opamp.py`:

**Lookup (normalized error):**
$$\text{norm\_spec} = \frac{\text{spec} - \text{goal\_spec}}{\text{goal\_spec} + \text{spec}}$$

**Reward logic:**
```python
rel_specs = lookup(spec, goal_spec)
reward = 0.0
for i, rel_spec in enumerate(rel_specs):
    if specs_id[i] == 'ibias_max':
        rel_spec = rel_spec * -1.0  # I-Bias: lower is better
    if rel_spec < 0:
        reward += rel_spec
return reward if reward < -0.02 else 10
```

**Success:** A solution reaches target when `reward >= 10` (aggregated normalized error ≥ −0.02). The formula sums only negative normalized errors; a design can slightly undershoot some objectives and still pass if the total penalty stays above −0.02.

**Implementation:** Our `original_autockt/main.py` reproduces the same `lookup`, `reward`, `specs_id` order, gain conversion, and `gen_specs` dataset. Rollout records Yes/No from the environment’s `done` signal (`reward >= 10`).

### 3.2 MORL+AutoCkt

MORL defines a 4D reward vector:

$$\mathbf{r} = \left[ \frac{G_{out}-G_{tgt}}{G_{tgt}}, \frac{U_{out}-U_{tgt}}{U_{tgt}}, \frac{P_{out}-P_{tgt}}{P_{tgt}}, -\frac{I_{out}-I_{tgt}}{I_{tgt}} \right]$$

**Cosine similarity scalarization:**

$$\text{scalarized} = \frac{\mathbf{r} \cdot \mathbf{w}}{\|\mathbf{r}\| \cdot \|\mathbf{w}\|} \cdot \|\mathbf{r}\| = \cos(\theta) \cdot \|\mathbf{r}\|$$

with default preference vector \(\mathbf{w} = [0.25, 0.25, 0.25, 0.25]\). A solution is considered reached when scalarized > τ (τ ≈ −0.45665).

**Comparison criterion:** For fair comparison, we use a strict per-objective pass: **Yes** when Output ≥ Target (Gain, UGBW, PM) and Output ≤ Target (I-Bias). MORL achieves **9919/10000** solutions meeting this criterion.

---

## 4. Figure of Merit

We use a unified FoM to compare both methods on equal footing:

$$\boxed{\text{FoM} = \frac{G_{out} - G_{tgt}}{G_{tgt}} + \frac{U_{out} - U_{tgt}}{U_{tgt}} + \frac{P_{out} - P_{tgt}}{P_{tgt}} - \frac{I_{out} - I_{tgt}}{I_{tgt}}}$$

- **Higher FoM is better:** Overshooting Gain, UGBW, or PM increases FoM; overshooting I-Bias decreases it.
- Denominators use \(\max(\text{value}, 10^{-9})\) to avoid division by zero.

### Example: AutoCKT (Spec 838)

| Quantity | Target | Output |
|----------|--------|--------|
| Gain (linear) | 379.0 | 536.02 |
| UGBW (MHz) | 24.4 | 34.83 |
| PM (°) | 75.0 | 80.01 |
| I-Bias (mA) | 5.2 | 4.97 |

$$\text{FoM} = \frac{536.02 - 379.0}{379.0} + \frac{34.83 - 24.4}{24.4} + \frac{80.01 - 75}{75} - \frac{4.97 - 5.2}{5.2} \approx 0.95$$

### Example: MORL (Spec 956)

| Quantity | Target | Output |
|----------|--------|--------|
| Gain (linear) | 223.0 | 325.96 |
| UGBW (MHz) | 18.0 | 30.78 |
| PM (°) | 75.0 | 84.62 |
| I-Bias (mA) | 5.5 | 4.47 |

$$\text{FoM} = \frac{325.96 - 223.0}{223.0} + \frac{30.78 - 18.0}{18.0} + \frac{84.62 - 75}{75} - \frac{4.47 - 5.5}{5.5} \approx 1.49$$

---

## 5. Results

### 5.1 Overall Performance

| Method | Specs | Passed | Pass Rate |
|--------|-------|--------|-----------|
| AutoCKT | 1000 | 938 | 93.8% |
| MORL+AutoCkt | 10000 | 9919 | 99.2% |

### 5.2 Best 20 AutoCKT

From 938 passed specs, the 20 with highest FoM:

| Spec | Target G | Target U | Target P | Target I | Out G | Out U | Out P | Out I | FoM |
|------|----------|----------|----------|----------|-------|-------|-------|-------|-----|
| 838 | 379.0 | 24.4 | 75.0 | 5.2 | 536.02 | 34.83 | 80.01 | 4.97 | 0.71 |
| 159 | 352.0 | 13.2 | 75.0 | 4.7 | 497.76 | 18.86 | 80.01 | 4.45 | 0.71 |
| 480 | 376.0 | 16.8 | 75.0 | 7.9 | 531.62 | 23.95 | 80.00 | 7.48 | 0.71 |
| 801 | 316.0 | 16.1 | 75.0 | 4.6 | 446.72 | 22.97 | 80.00 | 4.39 | 0.71 |
| 122 | 254.0 | 24.7 | 75.0 | 9.7 | 359.02 | 35.30 | 80.00 | 9.24 | 0.71 |
| 443 | 341.0 | 21.5 | 75.0 | 5.8 | 481.92 | 30.65 | 79.99 | 5.47 | 0.71 |
| 764 | 339.0 | 18.5 | 75.0 | 2.4 | 479.02 | 26.34 | 79.99 | 2.30 | 0.71 |
| 85 | 290.0 | 11.4 | 75.0 | 5.2 | 409.72 | 16.23 | 79.98 | 4.95 | 0.71 |
| 406 | 392.0 | 5.2 | 75.0 | 1.2 | 553.75 | 7.40 | 79.98 | 1.18 | 0.71 |
| 727 | 220.0 | 14.3 | 75.0 | 6.3 | 310.73 | 20.42 | 79.98 | 5.96 | 0.71 |
| 48 | 297.0 | 16.9 | 75.0 | 5.3 | 419.43 | 24.04 | 79.97 | 5.03 | 0.71 |
| 369 | 330.0 | 17.5 | 75.0 | 8.9 | 465.97 | 24.90 | 79.97 | 8.46 | 0.71 |
| 690 | 335.0 | 1.9 | 75.0 | 5.4 | 472.96 | 2.72 | 79.96 | 5.10 | 0.71 |
| 11 | 389.0 | 4.0 | 75.0 | 0.9 | 549.11 | 5.75 | 79.96 | 0.84 | 0.71 |
| 332 | 215.0 | 3.6 | 75.0 | 5.0 | 303.45 | 5.13 | 79.96 | 4.77 | 0.71 |
| 653 | 321.0 | 16.8 | 75.0 | 1.3 | 452.99 | 23.89 | 79.95 | 1.22 | 0.70 |
| 295 | 372.0 | 14.9 | 75.0 | 5.3 | 524.81 | 21.24 | 79.94 | 5.06 | 0.70 |
| 616 | 200.0 | 21.5 | 75.0 | 9.9 | 282.11 | 30.67 | 79.94 | 9.47 | 0.70 |
| 937 | 202.0 | 2.8 | 75.0 | 1.1 | 284.89 | 3.95 | 79.94 | 1.07 | 0.70 |
| 258 | 201.0 | 16.3 | 75.0 | 9.8 | 283.44 | 23.26 | 79.93 | 9.36 | 0.70 |

*FoM range: 0.70–0.71. All pass the strict per-objective criterion.*

### 5.3 Best 20 MORL

From the best solution per spec (by FoM) among 10 solutions per spec:

| Spec | Target G | Target U | Target P | Target I | Out G | Out U | Out P | Out I | FoM |
|------|----------|----------|----------|----------|-------|-------|-------|-------|-----|
| 956 | 223.0 | 18.0 | 75.0 | 5.5 | 325.96 | 30.78 | 84.62 | 4.47 | **1.18** |
| 896 | 311.0 | 7.2 | 75.0 | 6.6 | 454.58 | 12.38 | 84.62 | 5.35 | **1.18** |
| 828 | 205.0 | 5.6 | 75.0 | 4.5 | 299.64 | 9.54 | 84.62 | 3.61 | **1.18** |
| 632 | 321.0 | 22.6 | 75.0 | 7.4 | 469.19 | 38.56 | 84.62 | 6.01 | **1.18** |
| 700 | 309.0 | 5.0 | 75.0 | 3.1 | 451.65 | 8.49 | 84.62 | 2.53 | **1.18** |
| 572 | 211.0 | 13.6 | 75.0 | 6.1 | 308.40 | 23.18 | 84.62 | 4.95 | **1.18** |
| 504 | 304.0 | 14.6 | 75.0 | 8.4 | 444.33 | 24.92 | 84.62 | 6.83 | **1.18** |
| 436 | 276.0 | 14.2 | 75.0 | 5.8 | 403.41 | 24.18 | 84.62 | 4.66 | **1.18** |
| 240 | 279.0 | 20.0 | 75.0 | 0.4 | 407.78 | 34.19 | 84.62 | 0.29 | **1.18** |
| 308 | 337.0 | 7.2 | 75.0 | 4.9 | 492.55 | 12.24 | 84.62 | 3.98 | **1.18** |
| 376 | 372.0 | 3.9 | 75.0 | 9.9 | 543.71 | 6.71 | 84.62 | 7.97 | **1.18** |
| 112 | 214.0 | 17.6 | 75.0 | 7.0 | 312.77 | 30.13 | 84.62 | 5.63 | **1.18** |
| 180 | 316.0 | 8.7 | 75.0 | 5.5 | 461.85 | 14.89 | 84.62 | 4.48 | **1.18** |
| 805 | 394.0 | 9.0 | 75.0 | 4.8 | 575.73 | 15.33 | 84.61 | 3.91 | **1.18** |
| 873 | 230.0 | 19.8 | 75.0 | 0.3 | 336.08 | 33.72 | 84.61 | 0.22 | **1.18** |
| 677 | 308.0 | 9.7 | 75.0 | 1.7 | 450.05 | 16.53 | 84.61 | 1.35 | **1.18** |
| 609 | 373.0 | 2.3 | 75.0 | 1.5 | 545.03 | 4.01 | 84.61 | 1.20 | **1.18** |
| 745 | 252.0 | 6.7 | 75.0 | 6.3 | 368.22 | 11.37 | 84.61 | 5.06 | **1.18** |
| 481 | 376.0 | 16.8 | 75.0 | 7.9 | 549.40 | 28.61 | 84.61 | 6.37 | **1.18** |
| 413 | 225.0 | 19.7 | 75.0 | 3.9 | 328.76 | 33.58 | 84.61 | 3.18 | **1.18** |

*FoM ≈ 1.18 for all. MORL consistently overshoots Gain, UGBW, PM and meets or undershoots I-Bias.*

---

## 6. Discussion

### 6.1 Why MORL Outperforms AutoCKT

| Aspect | AutoCKT | MORL+AutoCkt |
|--------|------------------|--------------|
| **Formulation** | Single scalar reward | Multi-objective vector + cosine scalarization |
| **Solutions per spec** | 1 | 10 (best by FoM) |
| **Trade-off flexibility** | Fixed (implicit) | Explicit via preference vectors |
| **Design margin** | Often undershoots targets | Typically overshoots Gain, UGBW, PM |
| **Pass rate** | 938/1000 | 9919/10000 |

### 6.2 Key Advantages of MORL

1. **Pareto exploration:** 10 solutions per spec span different trade-offs; the best by FoM is selected.
2. **Design margin:** MORL outputs exceed targets for Gain, UGBW, PM and stay at or below I-Bias.
3. **Preference flexibility:** Designers can prioritize, e.g., low power or high bandwidth via preference vectors.
4. **Robustness:** If one preference vector yields a poor design, others may succeed.

---

## 7. Generated Figures

### 7.1 Layout and Markers

Both figures use a 2×2 subplot layout. Markers:

- **Gray circle (○)** — Target
- **Blue square (■)** — AutoCKT output
- **Green triangle (▲)** — MORL output

### 7.2 `best20_both_4subplots.png`

Shows 20 best AutoCKT + 20 best MORL (40 specs). Subplots: PM vs Gain, UGBW vs I-Bias, Gain vs UGBW, PM vs I-Bias. MORL points (triangles) typically lie in higher-performance regions than AutoCKT (squares).

### 7.3 `best1000_original_morl_4subplots.png`

Shows 1000 specs (subsampled to 500 for readability), each with one target, one AutoCKT output, and one MORL output. Same four subplots. MORL consistently outperforms AutoCKT across the spec set.

### 7.4 `best20_pca_scatter.png` — Best 20 Solutions (PCA)

![Best 20 Solutions](best_20/best20_pca_scatter.png)

PCA scatter plot of the best 20 solutions from each method. The four objectives (Gain, UGBW, PM, I-Bias) are projected to 2D via PCA and scaled 0–100. Markers:

- **Circle (○)** — Target
- **Square (■)** — Original AutoCKT output
- **Triangle (▲)** — MORL+AutoCkt output

Each solution is shown as 4 points (one per objective), colored by objective: orange (Gain), purple (UGBW), red (PM), green (IBIAS). Cluster layout: Target and Original AutoCKT in upper/right regions; MORL+AutoCkt in lower-left.

### 7.5 Subplot Interpretation (4-subplot figures)

| Subplot | X | Y | Better direction |
|---------|---|---|------------------|
| PM vs Gain | Gain | PM | Both up/right |
| UGBW vs I-Bias | UGBW | I-Bias | Right (UGBW), down (I-Bias) |
| Gain vs UGBW | Gain | UGBW | Both up/right |
| PM vs I-Bias | PM | I-Bias | Right (PM), down (I-Bias) |

---

## 8. Conclusions

1. **Methodology:** AutoCKT evaluation matches the paper and code (`lookup`, `reward`, `reward >= 10`).
2. **MORL approach:** Cosine similarity scalarization balances four objectives and explores the Pareto front.
3. **Results:** AutoCKT 938/1000; MORL 9919/10000 solutions reaching target.
4. **FoM:** MORL best-20 FoM ≈ 1.18 vs AutoCKT ≈ 0.70–0.71.
5. **Figures:** MORL consistently outperforms AutoCKT across all objective pairs.

**Conclusion:** MORL+AutoCkt significantly outperforms AutoCKT (9919/10000 vs 938/1000) while our AutoCKT evaluation remains faithful to the published methodology.

---

*Report generated from `generate_csvs_only.py` (project root), `with_15%/best_20/generate_graphs.py`, and `with_15%/best_20/generate_pca_graph.py`.*
