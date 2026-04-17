# Milestone 2: 2D Objective Scatter Plots Comparison

## Overview

This milestone implements metric-based comparison visualizations using saved input/output data from 1,000 target specifications, focusing on the four optimization objectives (Gain, UGBW, Phase Margin, IBIAS).

## Generated Files

### Scripts
- `create_2d_objective_comparison.py` - Main script that generates all comparison plots

### Output Figures
- `figures/AutoCkt_2D_Objective_Scatter_Plots.png` - AutoCkt 2D scatter plots (6 objective pairs)
- `figures/MORL_2D_Objective_Scatter_Plots.png` - MORL 2D scatter plots (6 objective pairs)
- `figures/AutoCkt_vs_MORL_2D_Comparison.png` - Side-by-side comparison plots

### Data Files
- `data/selected_targets_info.json` - Information about the 10 randomly selected target specifications

## Selected Target Specifications

The following 10 target specifications were randomly selected from the 1,000 available:
- 655, 115, 26, 760, 282, 251, 229, 143, 755, 105

## Plot Details

### Marker Scheme
- **Triangle (^)**: Input/Target values (the randomly sampled target specification)
- **Square (s)**: Output values (what the method produced)
- **Circle (o)**: Target values (the exact target specification for reference)

### Color Scheme
- **AutoCkt**: Red (#e74c3c)
- **MORL**: Blue (#3498db)
- **Target**: Green (#2ecc71)

### Objective Pairs Visualized
1. Gain vs UGBW
2. Gain vs Phase Margin
3. Gain vs IBIAS
4. UGBW vs Phase Margin
5. UGBW vs IBIAS
6. Phase Margin vs IBIAS

## How to Run

```bash
cd experiment_5/milestone_2
python create_2d_objective_comparison.py
```

## Data Sources

The script reads from:
- `../results/all_input_output_values_complete.json` - Contains all 11,000 solutions (11 per spec × 1000 specs)

## Best MORL Solution Selection

For each target, the best MORL solution is selected from 11 candidates using:
- Normalized scores for each objective (gain, ugbw, phm, ibias)
- Overall score = average of all four normalized scores
- Solution with highest overall score is selected as the "best" solution

## AutoCkt Data

Since AutoCkt data is embedded in MORL results, the first MORL solution for each target is used as a proxy for AutoCkt results.

