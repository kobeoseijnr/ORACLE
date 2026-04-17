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

## Project Structure (detailed)

The repository contains multiple experiment snapshots. The most important entry points for reproducing MORL runs and generating result artifacts are:

- `with_15%_with_20/morl_experiments/morl_autockt/`
  - `train_nw_vs_cosine.py`: train + evaluate MORL agents with NW vs cosine scalarization.
  - `evaluate.py`: evaluation pipeline (generates per-solution rollouts / raw results).
  - `results/`: output folder used by scripts in this directory.

- `with_15%_final_New/with_15%/morl_autockt/`
  - `train_nw_vs_cosine.py`: train + evaluate MORL NW vs cosine (writes models + raw results + comparison CSV).
  - `train_llm_cosine_original.py`: evaluate baseline cosine vs LLM-guided cosine and generate CSVs.
  - `train_llm_ddqn.py`: train/evaluate LLM-guided DDQN variants (cosine + NW) and generate comparisons.
  - `evaluate.py`: evaluation utility used by the above scripts.
  - `results/`: output folder used by scripts in this directory.
  - `methodology/`: MORL/AutoCkt environment + agent implementation used by the scripts above.

In general, training scripts write:

- Model checkpoints (`.pth`) under a `results/models_*` directory.
- Raw rollouts (`morl_raw_*.json`) under the corresponding `results/` folder.
- Aggregated results CSVs under the corresponding `results/` folder.

## Setup

1. Create and activate a Python environment.
2. Install dependencies required by the scripts you plan to run (PyTorch, Gym, NumPy, etc.).
3. For LLM-guided action masking, install and run **Ollama** locally and pull a supported model (e.g., Llama 3.2).

## Running Training and Evaluation (examples)

- Train/evaluate MORL scalarizations:
  - See `with_15%_with_20/morl_experiments/morl_autockt/train_nw_vs_cosine.py`

- Train/evaluate MORL with LLM-guided action masking:
  - See `OSEI_MORL_Aucrt (1)/OSEI_MORL_Aucrt/train_llm_masked.py`

## How to Run Experiments

The codebase contains multiple experiment folders; the commands below reference the most commonly used entrypoints.

MORL NW vs cosine (train + evaluate):

- `with_15%_final_New/with_15%/morl_autockt/train_nw_vs_cosine.py`
  - Train + evaluate:
    - `python with_15%_final_New/with_15%/morl_autockt/train_nw_vs_cosine.py`
  - Evaluate only (use existing trained models):
    - `python with_15%_final_New/with_15%/morl_autockt/train_nw_vs_cosine.py --evaluate-only`
  - Train only:
    - `python with_15%_final_New/with_15%/morl_autockt/train_nw_vs_cosine.py --train-only`
  - Outputs:
    - Models: `with_15%_final_New/with_15%/morl_autockt/results/models_nw_vs_cosine/`
    - Raw JSON: `with_15%_final_New/with_15%/morl_autockt/results/morl_raw_{nw|cosine}_agent.json`
    - Comparison CSV: `with_15%_final_New/with_15%/morl_autockt/results/morl_compare_nw_agent_vs_cosine_agent.csv`

Baseline cosine vs LLM-guided cosine (evaluate + export CSV):

- `with_15%_final_New/with_15%/morl_autockt/train_llm_cosine_original.py`
  - Train + evaluate:
    - `python with_15%_final_New/with_15%/morl_autockt/train_llm_cosine_original.py`
  - Evaluate only:
    - `python with_15%_final_New/with_15%/morl_autockt/train_llm_cosine_original.py --evaluate-only`
  - Outputs:
    - `with_15%_final_New/with_15%/morl_autockt/results/morl_original_standard_cosine.csv`
    - `with_15%_final_New/with_15%/morl_autockt/results/morl_original_llm_cosine.csv`

LLM-guided DDQN (cosine + NW):

- `with_15%_final_New/with_15%/morl_autockt/train_llm_ddqn.py`
  - Train + evaluate:
    - `python with_15%_final_New/with_15%/morl_autockt/train_llm_ddqn.py`
  - Evaluate only:
    - `python with_15%_final_New/with_15%/morl_autockt/train_llm_ddqn.py --evaluate-only`
  - Outputs (under the same `results/` directory):
    - Models: `results/models_llm_ddqn/`
    - Raw JSON: `results/morl_raw_llm_{cosine|nw}_agent.json`


The following small CSV artifacts are tracked specifically to reproduce the paper's **solution-level comparison table** (1,000 target specifications / MO benchmark). These correspond to selecting the **best FoM per spec** (1 row per `spec`) and then reporting:

- Pass-rate: fraction of best-per-spec rows with `complete_pass == Yes`
- Average FoM: mean of FoM over the 1,000 best-per-spec rows
- Top-20 FoM: mean of the top-20 FoM values among the 1,000 best-per-spec rows

Tracked CSVs:

- ORACLE (Cosine): `with_15%_with_20/morl_experiments/morl_autockt/results/morl_autockt_results_original_cosine_with_llm.csv` (use column `fom`)
- ORACLE (Cosine + LLM): `with_15%_with_20/morl_experiments/morl_autockt/results/morl_best_per_spec_llm_cosine.csv`
- ORACLE (NW): `with_15%_with_20/morl_experiments/morl_autockt/results/morl_best_per_spec_nw.csv`

## Table I (solution-level) reproducibility

Table I in the paper is a **solution-level comparison** on the 1,000-problem (1,000 target specifications) benchmark. Each method generates 10 candidate solutions per spec (10,000 total), but Table I reports **one solution per spec** by selecting the **best FoM per spec**.

Computation (per method):

- Select the best solution per spec: `best_fom(spec) = max FoM over the 10 solutions for that spec`.
- Pass-rate: fraction of these 1,000 best-per-spec solutions with `complete_pass == Yes`.
- Average FoM: mean FoM over the 1,000 best-per-spec solutions.
- Top-20 FoM: mean of the top-20 FoM values among the 1,000 best-per-spec solutions.

Paper-reported Table I values:

- AutoCKT [12]: Pass-rate 93.8%, Avg FoM 0.434, Top-20 FoM 0.708
- ORACLE (Cosine): Pass-rate 100.0%, Avg FoM 1.453, Top-20 FoM 1.489
- ORACLE (Cosine + LLM): Pass-rate 100.0%, Avg FoM 132.3, Top-20 FoM 384.3
- ORACLE (NW): Pass-rate 100.0%, Avg FoM 138.3, Top-20 FoM 450.1

Reproducible CSV artifacts in this repository:

- ORACLE (Cosine): compute best-per-spec from `with_15%_with_20/morl_experiments/morl_autockt/results/morl_autockt_results_original_cosine_with_llm.csv` using `fom`.
- ORACLE (Cosine + LLM): `with_15%_with_20/morl_experiments/morl_autockt/results/morl_best_per_spec_llm_cosine.csv` (already best-per-spec).
- ORACLE (NW): `with_15%_with_20/morl_experiments/morl_autockt/results/morl_best_per_spec_nw.csv` (already best-per-spec).

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


## Runtime (paper-reported)

The following wall-clock runtimes (minutes) are reported in the paper's runtime comparison table:

- AutoCKT [12]: 85
- ORACLE (Cosine): 6.5
- ORACLE (Cos + LLM): 3.2
- ORACLE (NW): 2.4

These runtimes are not derived from the tracked CSV artifacts above and may depend on hardware/software configuration.


## Hardware

Experiments were executed on a machine equipped with an **NVIDIA GA102 GPU**.

