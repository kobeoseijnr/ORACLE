"""
Load original ep5000 checkpoint, evaluate with CORRECT epsilon schedule
(same as evaluate_with_saved_model.py: max(0.05, 0.15*(1-step/max_steps))).
Evaluate with both Cosine and NW scalarization.
Cosine should reproduce ~1.488 FOM. Then compare NW against it.
"""
import os
import sys
import pickle
import numpy as np
import json
import torch
import gym
import pandas as pd
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).parent
METHODOLOGY_DIR = BASE_DIR / "methodology"
sys.path.insert(0, str(METHODOLOGY_DIR))

os.environ['AUTOCKT_USE_SURROGATE'] = 'true'

from autockt.envs.autockt_mo_env import AutoCktMOEnv
from autockt.agents.mo_agent import MO_DQN_Agent
from autockt.utils.mo_utils import generate_preference_vectors

DATASET_PATH = BASE_DIR / "dataset" / "ngspice_specs_gen_two_stage_opamp"
RESULTS_DIR = BASE_DIR / "results"
CHECKPOINT = BASE_DIR / "results" / "models" / "model_checkpoint_ep5000.pth"

EVAL_CONFIG = {
    'num_targets': 1000,
    'num_preferences': 10,
    'max_steps': 120,
}


def _fom(g, u, p, i, gt, ut, pt, it):
    def s(x): return max(float(x), 1e-9)
    gt, ut, pt, it = s(gt), s(ut), s(pt), s(it)
    return (float(g)-gt)/gt + (float(u)-ut)/ut + (float(p)-pt)/pt - (float(i)-it)/it


def load_checkpoint(scalarization):
    """Load ep5000 checkpoint with given scalarization."""
    cp = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    state_dim = cp.get('state_dim', 15)
    action_dim = cp.get('action_dim', 3)
    reward_dim = cp.get('reward_dim', 4)

    agent = MO_DQN_Agent(state_dim, action_dim, reward_dim, scalarization=scalarization)
    if 'agent_state_dict' in cp:
        agent.q_network.load_state_dict(cp['agent_state_dict'])
    if 'preference_net_state_dict' in cp:
        agent.preference_net.load_state_dict(cp['preference_net_state_dict'])
    agent.q_network.eval()
    # Set epsilon to match original evaluation behavior
    agent.epsilon = 0.05  # Will be overridden per-step anyway
    return agent


def evaluate_agent(agent, scalarization, env, specs, preferences):
    """
    Evaluate using the EXACT same epsilon schedule as evaluate_with_saved_model.py:
        epsilon = max(0.05, 0.15 * (1 - step / max_steps))
    10 preferences per spec, NO early stop = 10000 solutions.
    """
    tag = scalarization.upper()
    max_steps = EVAL_CONFIG['max_steps']
    num_specs_total = len(list(specs.values())[0])
    num_targets = min(EVAL_CONFIG['num_targets'], num_specs_total)

    print(f"\n{'='*70}")
    print(f"  EVALUATING — {tag} (correct epsilon schedule)")
    print(f"  epsilon = max(0.05, 0.15*(1-step/{max_steps}))")
    print(f"{'='*70}\n")

    target_specs_json = {}
    for i in range(num_specs_total):
        target_specs_json[str(i)] = {
            'target_gain_linear': float(specs['gain_min'][i]),
            'target_ugbw_mhz': float(specs['ugbw_min'][i]) / 1e6,
            'target_pm_deg': float(specs['phm_min'][i]),
            'target_ibias_ma': float(specs['ibias_max'][i]) * 1000.0,
        }

    all_solutions = []

    for target_idx in range(num_targets):
        if target_idx % 100 == 0 and target_idx > 0:
            print(f"  [{tag}] Progress: {target_idx}/{num_targets}")

        try:
            env.base_env.obj_idx = target_idx
            env.reset()
            env_specs_id = env.base_env.specs_id
        except Exception:
            continue

        num_params = len(env.action_space.spaces) if isinstance(env.action_space, gym.spaces.Tuple) else 1

        for pref_idx, preference in enumerate(preferences):
            try:
                agent.set_preference(preference)
                state = env.reset()
                done = False

                for step in range(max_steps):
                    # EXACT same epsilon schedule as original evaluate_with_saved_model.py
                    epsilon = max(0.05, 0.15 * (1 - step / max_steps))
                    action = agent.select_action(state, preference, epsilon=epsilon)

                    # Convert action for Tuple space
                    if isinstance(env.action_space, gym.spaces.Tuple):
                        if isinstance(action, (int, np.integer)):
                            action = tuple([int(np.clip(action, 0, 2))] * num_params)
                        elif isinstance(action, (list, np.ndarray)):
                            action_list = [int(np.clip(a, 0, 2)) for a in action[:num_params]]
                            while len(action_list) < num_params:
                                action_list.append(0)
                            action = tuple(action_list)
                        elif not isinstance(action, tuple):
                            action = tuple([int(np.clip(action, 0, 2))] * num_params)

                    step_result = env.step(action)
                    if len(step_result) == 5:
                        next_state, rw, done, truncated, info = step_result
                        done = done or truncated
                    elif len(step_result) == 4:
                        next_state, rw, done, info = step_result
                    else:
                        break
                    state = next_state
                    if done:
                        break

                if hasattr(env.base_env, 'cur_specs'):
                    actual = env.base_env.cur_specs.copy()
                    sd = dict(zip(env_specs_id, actual))
                    gv = sd.get('gain', sd.get('gain_min', actual[0]))
                    uv = sd.get('ugbw', sd.get('ugbw_min', actual[1] if len(actual) > 1 else 0))
                    pv = sd.get('phm', sd.get('phm_min', actual[2] if len(actual) > 2 else 0))
                    iv = sd.get('ibias_max', sd.get('ibias', actual[3] if len(actual) > 3 else 0))

                    if gv < 100:
                        gl = 10 ** (gv / 20); gd = gv
                    else:
                        gl = gv; gd = 20 * np.log10(gv) if gv > 0 else 0

                    td = target_specs_json.get(str(target_idx), {})
                    gt = td.get('target_gain_linear', 1e-9)
                    ut = td.get('target_ugbw_mhz', 1e-9)
                    pt = td.get('target_pm_deg', 1e-9)
                    it = td.get('target_ibias_ma', 1e-9)

                    go = float(gl); uo = float(uv / 1e6); po = float(pv); io = float(iv * 1000)
                    f = _fom(go, uo, po, io, gt, ut, pt, it)

                    all_solutions.append({
                        'spec': target_idx + 1,
                        'solution': pref_idx + 1,
                        'target_gain_linear': gt, 'target_ugbw_mhz': ut,
                        'target_pm_deg': pt, 'target_ibias_ma': it,
                        'output_gain_linear': round(go, 6), 'output_gain_db': round(float(gd), 6),
                        'output_ugbw_mhz': round(uo, 6), 'output_pm_deg': round(po, 4),
                        'output_ibias_ma': round(io, 6),
                        'fom': round(f, 6),
                        'scalarization': scalarization,
                    })
            except Exception:
                continue

    print(f"  [{tag}] Done: {len(all_solutions)} solutions")
    return all_solutions


def save_and_summarize(results, label, filename):
    df = pd.DataFrame(results)
    n = len(df); ns = df['spec'].nunique()
    avg_fom = df['fom'].astype(float).mean()
    best = df.groupby('spec')['fom'].max()
    avg_best = best.mean()

    sums = pd.DataFrame([
        {'spec': f'summary_avg_fom_{n}_solutions', 'fom': round(avg_fom, 6)},
        {'spec': f'summary_avg_best_fom_{ns}_specs', 'fom': round(avg_best, 6)},
    ])
    out = pd.concat([df, sums], ignore_index=True)
    path = RESULTS_DIR / filename
    out.to_csv(path, index=False)
    print(f"  {label}: {n} solutions, avg FOM={avg_fom:.6f}, avg best FOM={avg_best:.6f}")
    print(f"  Gain range: {df['output_gain_linear'].min():.1f} - {df['output_gain_linear'].max():.1f}")
    print(f"  Gain mean:  {df['output_gain_linear'].mean():.1f}")
    return avg_best, path


def compare(cos_results, nw_results):
    def best_per_spec(results):
        by_spec = {}
        for sol in results:
            s = sol['spec']; f = sol['fom']
            if s not in by_spec or f > by_spec[s]:
                by_spec[s] = f
        return by_spec

    cos_best = best_per_spec(cos_results)
    nw_best = best_per_spec(nw_results)
    common = sorted(set(cos_best.keys()) & set(nw_best.keys()))

    rows = []
    wins_cos = wins_nw = ties = 0
    for sp in common:
        cf = cos_best[sp]; nf = nw_best[sp]
        if abs(cf - nf) < 1e-9: w = 'tie'; ties += 1
        elif cf > nf: w = 'cosine'; wins_cos += 1
        else: w = 'nw'; wins_nw += 1
        rows.append({'spec': sp, 'fom_cosine': round(cf, 6), 'fom_nw': round(nf, 6), 'winner': w})

    cos_avg = np.mean([cos_best[s] for s in common])
    nw_avg = np.mean([nw_best[s] for s in common])

    cmp = pd.DataFrame(rows)
    sums = pd.DataFrame([
        {'spec': 'summary_specs', 'fom_cosine': len(common)},
        {'spec': 'summary_wins_cosine', 'fom_cosine': wins_cos},
        {'spec': 'summary_wins_nw', 'fom_nw': wins_nw},
        {'spec': 'summary_ties', 'fom_cosine': ties},
        {'spec': 'summary_avg_best_fom_cosine', 'fom_cosine': round(cos_avg, 6)},
        {'spec': 'summary_avg_best_fom_nw', 'fom_nw': round(nw_avg, 6)},
    ])
    out = pd.concat([cmp, sums], ignore_index=True)
    path = RESULTS_DIR / "morl_compare_cosine_vs_nw_correct_epsilon.csv"
    out.to_csv(path, index=False)

    print(f"\n{'='*70}")
    print(f"  FINAL: COSINE vs NW (same checkpoint, correct epsilon)")
    print(f"{'='*70}")
    print(f"  Cosine avg best FOM: {cos_avg:.6f}")
    print(f"  NW avg best FOM:     {nw_avg:.6f}")
    print(f"  Cosine wins: {wins_cos}")
    print(f"  NW wins:     {wins_nw}")
    print(f"  Ties:        {ties}")
    winner = 'NW' if nw_avg > cos_avg else 'COSINE'
    print(f"  WINNER: {winner} (by {abs(nw_avg - cos_avg):.6f})")
    print(f"  Saved: {path}")


if __name__ == "__main__":
    start = datetime.now()
    print(f"\n{'#'*70}")
    print(f"  EVALUATE ep5000 — CORRECT EPSILON SCHEDULE")
    print(f"  Checkpoint: {CHECKPOINT}")
    print(f"  epsilon = max(0.05, 0.15*(1-step/max_steps))")
    print(f"  Start: {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*70}")

    with open(DATASET_PATH, 'rb') as f:
        specs = pickle.load(f)
    num_specs = len(list(specs.values())[0])
    print(f"\nDataset: {num_specs} specs")

    env = AutoCktMOEnv(generalize=True, num_valid=num_specs, run_valid=True)

    try:
        preferences = generate_preference_vectors(4, method='focused', num_vectors=EVAL_CONFIG['num_preferences'])
        if len(preferences) > EVAL_CONFIG['num_preferences']:
            preferences = preferences[:EVAL_CONFIG['num_preferences']]
    except Exception:
        preferences = generate_preference_vectors(4, method='random', num_vectors=EVAL_CONFIG['num_preferences'])
    print(f"Preferences: {len(preferences)} vectors")

    # Evaluate COSINE (should reproduce ~1.488)
    cos_agent = load_checkpoint('cosine')
    cos_results = evaluate_agent(cos_agent, 'cosine', env, specs, preferences)

    # Evaluate NW (same checkpoint, different scalarization)
    nw_agent = load_checkpoint('nw')
    nw_results = evaluate_agent(nw_agent, 'nw', env, specs, preferences)

    # Save
    print(f"\n{'#'*70}")
    print("  RESULTS")
    print(f"{'#'*70}")
    save_and_summarize(cos_results, "Cosine", "morl_cosine_correct_epsilon.csv")
    save_and_summarize(nw_results, "NW", "morl_nw_correct_epsilon.csv")

    # Compare
    compare(cos_results, nw_results)

    elapsed = datetime.now() - start
    print(f"\nTotal time: {elapsed}")
    print("DONE")
