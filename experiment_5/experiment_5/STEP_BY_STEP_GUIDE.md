# Complete Step-by-Step Guide (Windows)

## For First-Time Setup on Any System

This guide assumes you're starting from scratch on a new Windows system.

---

## Step 1: Install Python

1. Download Python 3.8 or higher from [python.org](https://www.python.org/downloads/)
2. During installation, check "Add Python to PATH"
3. Verify installation:
   ```powershell
   python --version
   ```
   Should show: `Python 3.x.x`

---

## Step 2: Navigate to Project Directory

```powershell
cd D:\comparison\experiment_5
```
(Adjust path to your actual project location)

---

## Step 3: Install Required Dependencies

```powershell
pip install numpy pandas matplotlib seaborn scipy torch gym pyyaml
```

**Verify installation:**
```powershell
python -c "import numpy, pandas, matplotlib, seaborn, scipy, torch, gym, yaml; print('All packages installed')"
```

---

## Step 4: Verify Dataset Files Exist

Check these files exist:
```powershell
# Check dataset file
Test-Path dataset\ngspice_specs_gen_two_stage_opamp

# Check YAML config
Test-Path dataset\yaml_config\two_stage_opamp.yaml
```

If files are missing, you need to generate them first or obtain them from the source.

---

## Step 5: Set Environment Variable (Optional but Recommended)

```powershell
$env:AUTOCKT_USE_SURROGATE = "true"
```

This uses a surrogate model instead of NGSpice (faster for testing). To make it permanent:
```powershell
[System.Environment]::SetEnvironmentVariable('AUTOCKT_USE_SURROGATE', 'true', 'User')
```

---

## Step 6: Train the Model

**Run this file:**
```powershell
python train_and_save_model.py
```

**What happens:**
1. Loads 1000 target specifications from `dataset/ngspice_specs_gen_two_stage_opamp`
2. Preprocesses data for training
3. Trains MORL agent on 50 randomly selected specifications
4. Saves trained model to `results/models/trained_morl_model_final.pth`

**Expected time:** 1-2 hours (depending on hardware)

**Output file:**
- `results/models/trained_morl_model_final.pth`

**Wait for:** Message saying "Training complete" or "Model saved"

---

## Step 7: Evaluate on 1000 Targets

**Run this file:**
```powershell
python evaluate_with_saved_model.py
```

**What happens:**
1. Loads trained model from `results/models/trained_morl_model_final.pth`
2. Evaluates on all 1000 target specifications
3. Generates 10 solutions per specification (using 10 preference vectors)
4. Saves evaluation results

**Expected time:** 20-30 minutes

**Output files:**
- `results/evaluation_on_1000.json` (detailed JSON results)
- `results/evaluation_on_1000.csv` (tabular CSV format)
- `results/all_input_output_values_complete.json` (complete input/output data)

**Wait for:** Message saying "Evaluation complete" or "Results saved"

---

## Step 8: Generate Main Visualizations

**Run this file:**
```powershell
python visualize_results.py
```

**What happens:**
1. Reads results from `results/evaluation_on_1000.csv`
2. Generates comprehensive visualizations
3. Creates scatter plots, line graphs, Pareto fronts, etc.

**Output files (in `results/figures/`):**
- Scatter plots for all objective pairs
- Line graphs showing distributions
- 3D Pareto front visualization
- Few-shot fine-tuning comparison
- RL convergence curves
- Pareto front comparison
- Objectives comparison over specifications

**Wait for:** Message saying "All visualizations generated successfully"

---

## Step 9: Generate 2D Comparison Plots

**Run this file:**
```powershell
python create_2d_objective_comparison.py
```

**What happens:**
1. Selects 10 random target specifications (seed=42 for reproducibility)
2. Generates 2D scatter plots comparing AutoCkt vs MORL
3. Counts total targets reached
4. Saves all 1000 input values

**Output files:**
- `figures/AutoCkt_2D_Objective_Scatter_Plots.png`
- `figures/MORL_2D_Objective_Scatter_Plots.png`
- `figures/AutoCkt_vs_MORL_2D_Comparison.png`
- `data/all_1000_input_values.json`
- `data/MORL_Time_Efficiency_Analysis.md`
- `data/selected_targets_info.json`

**Wait for:** Message saying "ALL PLOTS GENERATED SUCCESSFULLY"

---

## Step 10: Count Total Targets Reached

**Run this file:**
```powershell
python target_count.py
```

**What happens:**
1. Counts total solutions that reached target
2. Counts total specifications that reached target
3. Lists unreached specifications

**Output files:**
- `data/target_count_results.json`
- `data/target_count_summary.txt`

**Expected output displayed:**
```
Solutions reached: 9,919/10,000 (99.2%)
Specifications reached: 982/1000 (98.2%)
```

**Wait for:** Message saying "COUNTING COMPLETE"

---

## Step 11: Verify Reproducibility (Optional)

**Run this file:**
```powershell
python verify_reproducibility.py
```

**What happens:**
1. Verifies results are reproducible
2. Checks seed settings
3. Validates data consistency

**Wait for:** Message saying "All reproducibility checks passed"

---

## Step 12: View Results

### Main Report
Open: `results/TECHNICAL_REPORT_UPDATED.md`
- Complete analysis and comparison
- All statistics and findings

### Results Data
- `results/evaluation_on_1000.csv` - Complete results table (10,000 rows)
- `results/evaluation_on_1000.json` - Detailed JSON results
- `data/all_1000_input_values.json` - All input target values

### Visualizations
- `results/figures/` - Main visualizations (all PNG files)
- `figures/` - 2D comparison plots

### Summary Files
- `data/target_count_summary.txt` - Target count summary
- `data/MORL_Time_Efficiency_Analysis.md` - Time efficiency analysis

---

## Troubleshooting

### Issue: "Module not found"
**Solution:** Run Step 3 again to install dependencies

### Issue: "Dataset not found"
**Solution:** Verify Step 4 - ensure dataset files exist

### Issue: "Model file not found"
**Solution:** Run Step 6 first to train the model

### Issue: "Results file not found"
**Solution:** Run Step 7 first to generate evaluation results

---

## Quick Command Reference

```powershell
# Install dependencies
pip install numpy pandas matplotlib seaborn scipy torch gym pyyaml

# Full workflow (run in order)
python train_and_save_model.py
python evaluate_with_saved_model.py
python visualize_results.py
python create_2d_objective_comparison.py
python target_count.py
python verify_reproducibility.py
```

---

## Expected Final Results

- **Generalization**: 982/1000 specifications (98.2%)
- **Sample Efficiency**: 12 steps/solution (55.6% improvement)
- **Total Solutions**: 10,000 solutions across 1,000 specifications
- **Solutions per Spec**: 10 solutions per specification
- **Selected targets for plots**: [655, 115, 26, 760, 282, 251, 229, 143, 755, 105]

---

## File Structure After Running All Steps

```
experiment_5/
├── results/
│   ├── models/
│   │   └── trained_morl_model_final.pth
│   ├── figures/
│   │   └── [All visualization PNG files]
│   ├── evaluation_on_1000.json
│   ├── evaluation_on_1000.csv
│   └── all_input_output_values_complete.json
├── figures/
│   └── [2D comparison plots]
├── data/
│   ├── all_1000_input_values.json
│   ├── target_count_results.json
│   └── [Other data files]
└── [Python scripts]
```
