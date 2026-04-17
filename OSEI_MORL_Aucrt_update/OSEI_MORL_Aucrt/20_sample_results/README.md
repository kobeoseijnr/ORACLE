# AutoCkt 20-Sample Analysis

This folder contains a comprehensive analysis of 20 AutoCkt samples selected from 1000 generated samples.

## Contents

### 📊 Report
- **`AUTOCKT_20_SAMPLE_ANALYSIS_REPORT.md`** - Markdown version of the complete analysis report
- **`AUTOCKT_20_SAMPLE_ANALYSIS_REPORT.html`** - HTML version with embedded styling and images (recommended for viewing)

Both reports include:
  - Objective specifications from the paper
  - Figure of Merit (FoM) definition and formula
  - Detailed FoM calculation examples
  - 20-sample analysis table
  - Visualizations and key observations

### 📈 Visualizations (`figures/`)
- **`all_objective_pairs_20_samples.png`** - Combined visualization of all 6 objective pairs
- **`gain_vs_ugbw_20_samples.png`** - Gain vs Unity Gain Bandwidth
- **`gain_vs_pm_20_samples.png`** - Gain vs Phase Margin
- **`gain_vs_ibias_20_samples.png`** - Gain vs Bias Current
- **`ugbw_vs_pm_20_samples.png`** - UGBW vs Phase Margin
- **`ugbw_vs_ibias_20_samples.png`** - UGBW vs Bias Current
- **`pm_vs_ibias_20_samples.png`** - Phase Margin vs Bias Current

### 📁 Data Files (`data/`)
- **`autockt_1000_samples.json`** - Complete dataset of 1000 generated AutoCkt samples
- **`autockt_20_selected_samples.json`** - Selected 20 samples with FoM calculations
- **`autockt_20_samples_table.csv`** - Tabular format of 20 samples (easy to view in Excel)

### 🔧 Script
- **`generate_autockt_20_samples.py`** - Python script that generates all outputs

## Quick Start

1. **View the Report**: Open `AUTOCKT_20_SAMPLE_ANALYSIS_REPORT.md` in any markdown viewer
2. **View Visualizations**: Check the `figures/` folder for all graphs
3. **View Data**: Open `data/autockt_20_samples_table.csv` in Excel or any spreadsheet viewer

## Key Features

✅ **1000 AutoCkt Samples Generated** - Complete dataset following paper specifications  
✅ **20 Representative Samples Selected** - Diverse selection across different FoM ranges  
✅ **6 Objective Pair Visualizations** - All combinations of design objectives  
✅ **FoM Calculations** - Figure of Merit for all objective pairs  
✅ **Comprehensive Report** - Includes objectives specification, FoM formula, and examples  

## Objective Specifications (from Paper)

| Objective | Range | Units |
|-----------|-------|-------|
| Gain (A_v) | 200-400 | Linear (46.02-52.04 dB) |
| UGBW | 1.0-25.0 | MHz |
| Phase Margin | 60.0-90.0 | Degrees |
| IBIAS | 0.1-10.0 | mA |

## Figure of Merit (FoM) Formula

For a 2D objective pair:
```
FoM = |output_x - target_x| / target_x + |output_y - target_y| / target_y
```

Lower FoM values indicate better performance (closer to target).

## Regenerating Results

To regenerate all outputs, run:
```bash
python generate_autockt_20_samples.py
```

This will:
1. Generate 1000 AutoCkt samples
2. Select 20 representative samples
3. Create all visualizations
4. Calculate FoM for all samples
5. Generate the comprehensive report

---

**Generated**: 2025-01-17  
**Analysis Tool**: AutoCkt 20-Sample Analysis Script v1.0
