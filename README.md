# ORACLE: Multi-Objective RL for Analog Circuit Optimization

This repository contains the code used to train and evaluate **ORACLE**, a preference-conditioned multi-objective reinforcement learning (MORL) approach for discrete analog circuit design (two-stage operational amplifier). The workflow includes:

- Training MORL agents with different scalarizations (e.g., cosine, NW)
- Optional **LLM-guided action masking** (served locally via **Ollama**) to improve exploration efficiency
- Evaluation on a large benchmark of target specifications
- Post-processing utilities for Pareto metrics (hypervolume, sparsity) and report/figure generation

## Repository Structure (high-level)

- `with_15%_with_20/`, `with_15%_final_New/`: experiment folders (training/evaluation scripts, configs, and analysis utilities)
- `AutoCkt/`: baseline/original AutoCkt components and scripts
- `OSEI_MORL_Aucrt (1)/`, `OSEI_MORL_Aucrt_update/`: MORL + LLM masking training/evaluation code
- `compute_hv_sparsity_comparison.py`, `plot_hv_sparsity_figures.py`: Pareto metric computation and plotting 

## Setup

1. Create and activate a Python environment.
2. Install dependencies required by the scripts you plan to run (PyTorch, Gym, NumPy, etc.).
3. For LLM-guided action masking, install and run **Ollama** locally and pull a supported model (e.g., Llama 3.2).

## Running Training and Evaluation (examples)

- Train/evaluate MORL scalarizations:
  - See `with_15%_with_20/morl_experiments/morl_autockt/train_nw_vs_cosine.py`

- Train/evaluate MORL with LLM-guided action masking:
  - See `OSEI_MORL_Aucrt (1)/OSEI_MORL_Aucrt/train_llm_masked.py`


## Published Table Provenance (CSV sources)

This codebase contains multiple experiment runs stored under different folders. 
- **Hypervolume / Sparsity / PF Size table**
  - `with_15%_final_New/with_15%/morl_autockt/results/hypervolume_sparsity_comparison.csv` (use the `MEAN` row)

- **FoM table **
  - **Cosine **
    - `with_15%_with_20/morl_experiments/morl_autockt/results/morl_autockt_results_trained_cosine_with_llm.csv`
  - **NW **
    - `with_15%_final_New/with_15%/morl_autockt/results/morl_original_standard_cosine.csv`
  - **LLM **
    - `with_15%_with_20/morl_experiments/morl_autockt/results/morl_best_per_spec_nw.csv`


## Hardware

Experiments were executed on a machine equipped with an **NVIDIA GA102 GPU**.

