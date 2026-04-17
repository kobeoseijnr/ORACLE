# ORACLE: A Multi-Objective Reinforcement Learning-Based Analog Circuit Design Optimizer with LLM-Guided Exploration

ORACLE is an open-source framework for **multi-objective analog circuit design optimization** using **preference-conditioned reinforcement learning** and **LLM-guided exploration**.

Unlike conventional RL-based analog sizing methods that optimize a single scalar reward, ORACLE keeps objectives separate during learning through **vector-valued rewards**. This allows a single trained model to generate multiple trade-off solutions for the same target specification without retraining.

## Highlights

- **True multi-objective optimization** for analog circuit design
- **Preference-conditioned MO-DDQN** for controllable trade-off generation
- **Vector-valued reward learning** instead of scalar reward compression
- **Cosine-aligned** and **normalized-weight (NW)** preference guidance
- **LLM-guided action masking** to reduce unproductive exploration
- **Multiple solutions per target specification** from one trained model
- Evaluated on a **two-stage op-amp benchmark** with **1,000 target specifications**

## Motivation

Analog circuit design is naturally multi-objective. Designers often need to satisfy and trade off objectives such as:

- Gain
- Unity-gain bandwidth (UGBW)
- Phase margin (PM)
- Bias current (Ibias)

Many prior RL-based approaches reduce these objectives to a single scalar reward. That simplification can hide Pareto trade-offs, bias optimization toward certain objectives, and force retraining when preferences change.

ORACLE addresses this by learning with **multi-objective reward vectors** and conditioning the agent on a **preference vector**, enabling flexible trade-off control at inference time.

## Method Overview

ORACLE consists of three main components:

### 1. Multi-Objective Circuit Environment

The environment returns a **normalized reward vector**, where each element measures progress toward one target specification.

- Maximization objectives such as gain, UGBW, and PM are rewarded positively when they improve toward or beyond target
- Minimization objectives such as Ibias are transformed so that lower values correspond to higher rewards
- This keeps each objective explicit during training and reduces domination from scale differences

### 2. Preference-Conditioned MO-DDQN

ORACLE uses a **multi-objective Double Deep Q-Network (MO-DDQN)** that takes:

- the current circuit state, and
- a user-defined **preference vector**

and predicts vector-valued Q-values for all candidate actions.

Two preference-guidance strategies are supported:

- **Cosine-aligned guidance**: selects actions whose predicted value vectors align with the desired trade-off direction
- **Normalized-weight (NW) guidance**: uses direct weighted scoring for action selection

This design allows one trained model to produce solutions for multiple trade-off settings by changing only the preference vector.

### 3. LLM-Guided Action Masking

Circuit simulation is expensive. ORACLE improves efficiency by using an LLM to filter actions that are unlikely to help in the current state.

Examples:

- block upsizing when **Ibias** is already too high
- block downsizing when **gain**, **UGBW**, or **PM** are still below target

This reduces wasted simulation calls and improves search efficiency.

## Framework

At a high level, ORACLE operates as follows:

1. Receive target specifications for the analog circuit
2. Build the current circuit state from observed specs and design parameters
3. Compute a normalized multi-objective reward vector
4. Use a preference-conditioned MO-DDQN to score actions
5. Optionally filter harmful or low-value actions using LLM-guided masking
6. Simulate the updated circuit
7. Repeat until the design satisfies the target or the episode ends

## Benchmark

ORACLE is evaluated on a **two-stage operational amplifier** benchmark in **45nm BSIM technology**.

### Target objectives

- **Maximize**: Gain, UGBW, PM
- **Minimize**: Ibias

### Success criterion

A solution is successful only if all four constraints are satisfied simultaneously:

- `G >= G*` 
- `UGBW >= UGBW*` 
- `PM >= PM*` 
- `Ibias <= Ibias*` 

### Evaluation setup

- **1,000 multi-objective target specifications**
- ORACLE generates **10 solutions per target specification**
- AutoCKT baseline generates **1 solution per target**

## Main Results

### Solution-level comparison

| Method | Pass-rate | Average FoM | Top-20 FoM |
|---|---:|---:|---:|
| AutoCKT | 93.8% | 0.434 | 0.708 |
| ORACLE (Cosine) | 100.0% | 1.453 | 1.489 |
| ORACLE (Cosine + LLM) | 100.0% | 132.3 | 384.3 |
| ORACLE (NW) | 100.0% | 138.3 | 450.1 |

### Pareto trade-off quality

| Method | Mean Hypervolume | Mean Sparsity | Mean PF Size |
|---|---:|---:|---:|
| AutoCKT | 1.40 × 10^5 | 0.00 | 1.00 |
| ORACLE (Cos) | 3.55 × 10^9 | 8.81 × 10^5 | 10.00 |
| ORACLE (NW) | 3.57 × 10^9 | 9.04 × 10^5 | 10.00 |

### Runtime comparison

| Method | Runtime (Minutes) |
|---|---:|
| AutoCKT | 85.0 |
| ORACLE (Cosine) | 6.5 |
| ORACLE (Cos + LLM) | 3.2 |
| ORACLE (NW) | 2.4 |

## Key Contributions

- Reformulates analog circuit optimization from **single scalar reward learning** to **multi-objective vector-valued learning**
- Enables **preference controllability** using a single trained policy
- Produces **multiple trade-off solutions** per target specification
- Integrates **LLM-guided exploration** for more efficient action selection
- Demonstrates strong improvements in **pass-rate**, **FoM**, **hypervolume**, and **runtime**

## Repository Structure

A typical project structure may look like this:

```text
ORACLE/
├── env/                # multi-objective circuit environment
├── agents/             # preference-conditioned MO-DDQN agents
├── models/             # neural network definitions
├── llm/                # LLM-guided action masking utilities
├── configs/            # experiment and benchmark configs
├── scripts/            # training and evaluation scripts
├── results/            # logs, checkpoints, plots, tables
├── notebooks/          # analysis notebooks
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/medal-ece/ORACLE.git
cd ORACLE
pip install numpy pandas matplotlib torch
```

## Train

```bash
cd ORACLE_with_15_with_20/morl_experiments/morl_autockt
python main.py
```

## Evaluate

```bash
cd ORACLE_with_15_with_20/morl_experiments/morl_autockt
python evaluate.py
```

## Generate Graphs

```bash
cd ORACLE_with_15_with_20/morl_experiments/best_20
python generate_std_vs_llm_graphs.py
python generate_all_report_graphs.py
```

## Project Structure

```text
ORACLE_with_15_with_20/
└── morl_experiments/
    ├── morl_autockt/                          # MORL code and results
    │   ├── autockt/                           # OpenAI Gym environment for the op-amp
    │   │   ├── envs/                          # Environment definitions
    │   │   └── gen_specs/                     # Target spec generator
    │   ├── methodology/                       # Agent implementations
    │   │   ├── autockt/
    │   │   │   ├── models/                    # DDQN architectures
    │   │   │   ├── evaluation/                # Hypervolume, sparsity evaluators
    │   │   │   └── utils/                     # Utility functions
    │   │   └── eval_engines/                  # NGSpice and surrogate wrapper
    │   │       └── ngspice/
    │   │           ├── ngspice_inputs/        # Netlists, SPICE models, configs
    │   │           ├── ngspice_wrapper.py     # Direct NGSpice interface
    │   │           └── surrogate_wrapper.py   # Fast surrogate evaluator
    │   ├── data/                              # Target spec files (JSON)
    │   ├── results/                           # All outputs (CSV, JSON, models)
    │   ├── main.py                            # Training entry point
    │   ├── evaluate.py                        # Evaluation script
    │   ├── train_nw_vs_cosine.py              # NW vs Cosine agent comparison
    │   ├── gen_nw_original.py                 # NW results on original specs
    │   └── merge_llm_to_cosine.py             # Merge LLM results into cosine CSV
    ├── original_autockt/                      # AutoCkt baseline
    │   ├── autockt/                           # Original environment
    │   ├── eval_engines/                      # Original NGSpice engine
    │   ├── results/                           # Baseline results
    │   ├── graphs/                            # Baseline plots
    │   ├── main.py                            # Training script
    │   └── evaluate.py                        # Evaluation script
    ├── best_20/                               # Analysis and visualization
    │   ├── generate_std_vs_llm_graphs.py      # 4-group comparison graphs
    │   ├── generate_all_report_graphs.py      # Full report figures
    │   ├── Comprehensive_Comparison_Report.md # Written comparison
    │   └── std_vs_llm_figures/                # Output figures
    └── create_best_comparison.py              # Best-of-1000 comparison
```
