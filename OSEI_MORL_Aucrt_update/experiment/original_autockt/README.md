# Original AutoCkt Evaluation

## Overview

This folder contains scripts to evaluate Original AutoCkt on the dataset.

## Files

- **`evaluate.py`**: Run Original AutoCkt evaluation
- **`main.py`**: Process results, generate graphs

## Usage

### Step 1: Run Evaluation

```bash
python evaluate.py <checkpoint_path> [options]
```

Options:
- `--num_val_specs`: Number of validation specs (default: 1000)
- `--traj_len`: Trajectory length (default: 60)

This generates:
- `opamp_obs_reached_test` - Solutions that passed strict evaluation
- `opamp_obs_nreached_test` - Solutions that failed strict evaluation

### Step 2: Process Results and Generate Graphs

```bash
python main.py
```

This will:
1. Process results for 15% increased targets (strict and tolerance)
2. Process results for original targets (strict and tolerance)
3. Generate graphs for all scenarios

Outputs:
- `results/original_autockt_results_15percent_strict.csv/json`
- `results/original_autockt_results_15percent_tolerance.csv/json`
- `results/original_autockt_results_original_strict.csv/json`
- `results/original_autockt_results_original_tolerance.csv/json`
- `graphs/original_autockt_4_subplots_15percent_strict.png`
- `graphs/original_autockt_4_subplots_15percent_tolerance.png`
- `graphs/original_autockt_4_subplots_original_strict.png`
- `graphs/original_autockt_4_subplots_original_tolerance.png`

## Evaluation Methods

1. **Strict Evaluation**: Uses `reward >= 10` (from original code)
2. **Tolerance Evaluation**: Uses 15% tolerance (85%/80% rules)

## Requirements

- Python 3.7+
- numpy, pandas, matplotlib
- AutoCkt environment setup
- NGSpice installed and configured
- Trained model checkpoint
