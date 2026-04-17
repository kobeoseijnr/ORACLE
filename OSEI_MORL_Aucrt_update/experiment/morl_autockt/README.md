# MORL+AutoCkt Evaluation

## Overview

This folder contains scripts to evaluate MORL+AutoCkt on the Original AutoCkt dataset using MORL methodology.

## Files

- **`evaluate.py`**: Run MORL evaluation on the dataset
- **`main.py`**: Process results, generate graphs, and compare with Original AutoCkt

## Usage

### Step 1: Run Evaluation

```bash
python evaluate.py <model_checkpoint_path> [options]
```

Options:
- `--num_targets`: Number of targets to evaluate (default: 1000)
- `--num_preferences`: Number of preference vectors per target (default: 10)
- `--max_steps`: Maximum steps per preference (default: 120)
- `--dataset`: Path to dataset (default: auto-detect)
- `--target_specs`: Path to target specs JSON (default: auto-detect)

This generates: `results/morl_autockt_results_raw.json`

### Step 2: Process Results and Generate Graphs

```bash
python main.py
```

This will:
1. Process raw results for original targets
2. Process raw results for 15% increased targets
3. Generate graphs for both scenarios
4. Compare with Original AutoCkt results (if available)

Outputs:
- `results/morl_autockt_results_original.csv/json`
- `results/morl_autockt_results_15percent.csv/json`
- `graphs/morl_autockt_4_subplots_original.png`
- `graphs/morl_autockt_4_subplots_15percent.png`

## Requirements

- Python 3.7+
- numpy, pandas, matplotlib, torch
- MORL methodology code in `methodology/` folder
- Dataset: `dataset/ngspice_specs_gen_two_stage_opamp`
- Target specs: `data/target_specs_original.json`

## Methodology

MORL uses multi-objective reinforcement learning:
- Explores Pareto frontier (multiple solutions per target)
- Uses preference vectors to guide optimization
- Evaluates using MORL's native multi-objective evaluation method
