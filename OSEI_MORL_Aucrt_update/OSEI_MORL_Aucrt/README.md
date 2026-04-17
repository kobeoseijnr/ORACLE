# OSEI MORL/AutoCkt

## Overview

This project implements a comparison framework for Multi-Objective Reinforcement Learning (MORL) and AutoCkt methods for circuit optimization. The framework evaluates and compares performance across multiple optimization objectives including Gain, UGBW (Unity Gain Bandwidth), Phase Margin, and IBIAS (Bias Current).

## Project Structure

### Scripts
- `create_2d_objective_comparison.py` - Generates objective scatter plot comparisons
- `train_and_save_model.py` - Training script for MORL models
- `evaluate_with_saved_model.py` - Evaluation script using saved models
- `verify_best_solution.py` - Verification script for best solutions
- `verify_reproducibility.py` - Reproducibility verification script
- `visualize_results.py` - Visualization script for results
- `target_count.py` - Target counting and analysis script

### Directories
- `data/` - Input data files and analysis results
- `results/` - Evaluation results, models, and generated figures
- `figures/` - Visualization outputs
- `eval_engines/` - Evaluation engine implementations (BAG, ngspice)
- `methodology/` - MORL methodology implementations
- `dataset/` - Dataset files and configurations

## Key Features

### Optimization Objectives
1. **Gain** - Amplifier gain
2. **UGBW** - Unity Gain Bandwidth
3. **Phase Margin** - Stability metric
4. **IBIAS** - Bias current

### Evaluation Metrics
- Target compliance analysis
- Pareto front comparisons
- Objective trade-off visualizations
- Time efficiency analysis

## Data Sources

The project uses:
- `results/all_input_output_values_complete.json` - Complete input/output evaluation data
- `data/all_1000_input_values.json` - Input values for 1,000 target specifications
- `data/selected_targets_info.json` - Selected target specification information

## Best Solution Selection

For each target, the best MORL solution is selected from multiple candidates using:
- Normalized scores for each objective (gain, ugbw, phm, ibias)
- Overall score = average of all four normalized scores
- Solution with highest overall score is selected as the "best" solution

## Usage

### Training
```bash
python train_and_save_model.py
```

### Evaluation
```bash
python evaluate_with_saved_model.py
```

### Visualization
```bash
python visualize_results.py
```

## Requirements

See `requirements.txt` for required Python packages.

## Results

Generated figures and analysis results are stored in:
- `results/figures/` - Paper-format figures and visualizations
- `results/` - CSV and JSON files with evaluation data

