"""
Train two separate MORL agents: one using NW scalarization, one using cosine.
Then evaluate both on the full 1000-spec dataset and generate comparison results.

Usage:
    python train_nw_vs_cosine.py                  # Train + evaluate both
    python train_nw_vs_cosine.py --evaluate-only   # Skip training, evaluate existing models
    python train_nw_vs_cosine.py --train-only      # Train only, no evaluation
"""

import os
import sys
import pickle
import numpy as np
import json
import torch
import gym
from pathlib import Path
from datetime import datetime

# Add methodology to path
BASE_DIR = Path(__file__).parent
METHODOLOGY_DIR = BASE_DIR / "methodology"
sys.path.insert(0, str(METHODOLOGY_DIR))

# Enable surrogate simulation
os.environ['AUTOCKT_USE_SURROGATE'] = 'true'

from autockt.envs.autockt_mo_env import AutoCktMOEnv
from autockt.agents.mo_agent import MO_DQN_Agent
from autockt.utils.mo_utils import generate_preference_vectors


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR  = RESULTS_DIR / "models_nw_vs_cosine"
DATASET_PATH = BASE_DIR / "autockt" / "gen_specs" / "ngspice_specs_gen_two_stage_opamp"

TRAIN_CONFIG = {
    'num_training_specs': 50,
    'num_preferences': 10,
    'max_steps': 30,
    'training_episodes': 100,
    'save_frequency': 50,
}

EVAL_CONFIG = {
    'num_targets': 1000,
    'num_preferences': 10,
    'max_steps': 120,
}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_agent(scalarization: str, config: dict, env, specs, preferences):
    """Train a single MO_DQN_Agent with the given scalarization method."""
    tag = scalarization.upper()
    print(f"\n{'='*80}")
    print(f"  TRAINING AGENT — {tag} SCALARIZATION")
    print(f"{'='*80}\n")

    state_dim = env.observation_space.shape[0] if hasattr(env.observation_space, 'shape') else 20
    if isinstance(env.action_space, gym.spaces.Tuple):
        action_dim = env.action_space.spaces[0].n
        num_params = len(env.action_space.spaces)
    else:
        action_dim = env.action_space.n if hasattr(env.action_space, 'n') else 7
        num_params = 1
    reward_dim = 4

    agent = MO_DQN_Agent(state_dim, action_dim, reward_dim, scalarization=scalarization)
    print(f"Agent created: state_dim={state_dim}, action_dim={action_dim}, reward_dim={reward_dim}, scalarization={scalarization}")

    training_specs = list(range(min(config['num_training_specs'], len(list(specs.values())[0]))))
    episode_count = 0
    history = {'episodes': [], 'rewards': [], 'scalarization': scalarization}

    for spec_idx, spec_num in enumerate(training_specs):
        if spec_idx % 10 == 0:
            print(f"  [{tag}] Training spec {spec_idx+1}/{len(training_specs)} ...")

        try:
            env.base_env.obj_idx = spec_num
            env.reset()
        except Exception as e:
            print(f"  [{tag}] Skip spec {spec_num}: {e}")
            continue

        for pref_idx, preference in enumerate(preferences):
            agent.set_preference(preference)
            episodes_per_pref = max(1, config['training_episodes'] // len(preferences))

            for episode in range(episodes_per_pref):
                state = env.reset()
                episode_reward = 0.0

                for step in range(config['max_steps']):
                    epsilon = max(0.1, 1.0 - episode_count / 1000)
                    try:
                        action = agent.select_action(state, preference, epsilon=epsilon)
                        if isinstance(env.action_space, gym.spaces.Tuple):
                            if isinstance(action, (int, np.integer)):
                                action = tuple([int(np.clip(action, 0, 2))] * num_params)
                            elif isinstance(action, (list, np.ndarray)):
                                action = tuple([int(np.clip(a, 0, 2)) for a in action[:num_params]])
                                while len(action) < num_params:
                                    action = action + (0,)
                    except Exception:
                        if isinstance(env.action_space, gym.spaces.Tuple):
                            action = tuple([np.random.randint(0, 3) for _ in range(num_params)])
                        else:
                            action = env.action_space.sample()

                    step_result = env.step(action)
                    if len(step_result) == 5:
                        next_state, reward_vec, done, truncated, info = step_result
                        done = done or truncated
                    elif len(step_result) == 4:
                        next_state, reward_vec, done, info = step_result
                    else:
                        break

                    if hasattr(agent, 'memory') and hasattr(agent.memory, 'push'):
                        agent.memory.push(state, action, reward_vec, next_state, done, preference)

                    if hasattr(agent, 'memory') and len(agent.memory) > 32 and episode_count % 4 == 0:
                        try:
                            agent.update(batch_size=32)
                        except Exception:
                            pass

                    state = next_state
                    episode_reward += float(np.sum(reward_vec)) if isinstance(reward_vec, (list, np.ndarray)) else float(reward_vec)
                    if done:
                        break

                episode_count += 1
                history['episodes'].append(episode_count)
                history['rewards'].append(episode_reward)

                if episode_count % config['save_frequency'] == 0:
                    _save_model(agent, MODELS_DIR / f"checkpoint_{scalarization}_ep{episode_count}.pth", episode_count)

    # Save final model
    final_path = MODELS_DIR / f"trained_morl_{scalarization}_final.pth"
    _save_model(agent, final_path, episode_count)
    print(f"  [{tag}] Training complete — {episode_count} episodes. Model: {final_path}")

    # Save history
    hist_path = RESULTS_DIR / f"training_history_{scalarization}.json"
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)

    return final_path, agent


def _save_model(agent, path, episode_count):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'agent_state_dict': agent.q_network.state_dict(),
        'preference_net_state_dict': agent.preference_net.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'state_dim': agent.state_dim,
        'action_dim': agent.action_dim,
        'reward_dim': agent.reward_dim,
        'scalarization': agent.scalarization,
        'episode_count': episode_count,
    }, path)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def check_target_reached(actual_specs, target_specs, specs_id):
    """Tolerance-based target check (same as evaluate.py)."""
    actual_dict = dict(zip(specs_id, actual_specs))
    target_dict = dict(zip(specs_id, target_specs))
    gain_a = actual_dict.get('gain_min', actual_specs[0] if len(actual_specs) > 0 else 0)
    ugbw_a = actual_dict.get('ugbw_min', actual_specs[3] if len(actual_specs) > 3 else 0)
    phm_a  = actual_dict.get('phm_min',  actual_specs[2] if len(actual_specs) > 2 else 0)
    ibias_a = actual_dict.get('ibias_max', actual_specs[1] if len(actual_specs) > 1 else 0)
    gain_t = target_dict.get('gain_min', target_specs[0] if len(target_specs) > 0 else 0)
    ugbw_t = target_dict.get('ugbw_min', target_specs[3] if len(target_specs) > 3 else 0)
    phm_t  = target_dict.get('phm_min',  target_specs[2] if len(target_specs) > 2 else 0)
    ibias_t = target_dict.get('ibias_max', target_specs[1] if len(target_specs) > 1 else 0)
    if gain_a < 100:
        gain_a = 10 ** (gain_a / 20)
    if gain_t < 100:
        gain_t = 10 ** (gain_t / 20)
    tol = 0.15
    return (gain_a >= gain_t * (1 - tol) and ugbw_a >= ugbw_t * (1 - tol)
            and phm_a >= phm_t * (1 - tol) and ibias_a <= ibias_t * (1 + tol))


def evaluate_agent(model_path, scalarization, env, specs, config):
    """Evaluate a trained agent using the given scalarization during rollout."""
    tag = scalarization.upper()
    print(f"\n{'='*80}")
    print(f"  EVALUATING AGENT — {tag} SCALARIZATION")
    print(f"{'='*80}\n")

    # Load model
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    state_dim = checkpoint.get('state_dim', 64)
    action_dim = checkpoint.get('action_dim', 64)
    reward_dim = checkpoint.get('reward_dim', 4)

    agent = MO_DQN_Agent(state_dim, action_dim, reward_dim, scalarization=scalarization)
    if 'agent_state_dict' in checkpoint and checkpoint['agent_state_dict'] is not None:
        agent.q_network.load_state_dict(checkpoint['agent_state_dict'])
    if 'preference_net_state_dict' in checkpoint and checkpoint['preference_net_state_dict'] is not None:
        agent.preference_net.load_state_dict(checkpoint['preference_net_state_dict'])
    agent.q_network.eval()
    print(f"  [{tag}] Model loaded from {model_path}")

    num_specs_total = len(list(specs.values())[0])
    num_targets = min(config['num_targets'], num_specs_total)

    # Build target specs dict (same as main.py)
    target_specs_json = {}
    for i in range(num_specs_total):
        target_specs_json[str(i)] = {
            'target_gain_linear': float(specs['gain_min'][i]),
            'target_ugbw_mhz': float(specs['ugbw_min'][i]) / 1e6,
            'target_pm_deg': float(specs['phm_min'][i]),
            'target_ibias_ma': float(specs['ibias_max'][i]) * 1000.0,
        }

    try:
        preferences = generate_preference_vectors(4, method='focused', num_vectors=config['num_preferences'])
        if len(preferences) > config['num_preferences']:
            preferences = preferences[:config['num_preferences']]
    except Exception:
        preferences = generate_preference_vectors(4, method='random', num_vectors=config['num_preferences'])

    results = {'all_solutions': [], 'reached_count': 0, 'total_evaluated': 0}

    for target_idx in range(num_targets):
        if target_idx % 100 == 0 and target_idx > 0:
            print(f"  [{tag}] Progress: {target_idx}/{num_targets}")

        try:
            env.base_env.obj_idx = target_idx
            env.reset()
            target_spec = env.base_env.specs_ideal.copy()
            specs_id = env.base_env.specs_id
        except Exception as e:
            print(f"  [{tag}] Skip spec {target_idx}: {e}")
            continue

        reached = False
        for pref_idx, preference in enumerate(preferences):
            try:
                agent.set_preference(preference)
                state = env.reset()
                done = False
                steps = 0

                while not done and steps < config['max_steps']:
                    action = agent.select_action(state, preference)
                    step_result = env.step(action)
                    if len(step_result) == 5:
                        next_state, reward, done, truncated, info = step_result
                        done = done or truncated
                    elif len(step_result) == 4:
                        next_state, reward, done, info = step_result
                    else:
                        break
                    state = next_state
                    steps += 1

                if hasattr(env.base_env, 'cur_specs'):
                    actual_specs = env.base_env.cur_specs.copy()
                    sol_reached = check_target_reached(actual_specs, target_spec, specs_id)
                    if sol_reached:
                        reached = True

                    spec_dict = dict(zip(specs_id, actual_specs))
                    gain_val = spec_dict.get('gain', spec_dict.get('gain_min', actual_specs[0] if len(actual_specs) > 0 else 0))
                    ugbw_val = spec_dict.get('ugbw', spec_dict.get('ugbw_min', actual_specs[1] if len(actual_specs) > 1 else 0))
                    phm_val  = spec_dict.get('phm',  spec_dict.get('phm_min',  actual_specs[2] if len(actual_specs) > 2 else 0))
                    ibias_val = spec_dict.get('ibias_max', actual_specs[3] if len(actual_specs) > 3 else 0)

                    if gain_val < 100:
                        gain_linear = 10 ** (gain_val / 20)
                        gain_db = gain_val
                    else:
                        gain_linear = gain_val
                        gain_db = 20 * np.log10(gain_val) if gain_val > 0 else 0

                    td = target_specs_json.get(str(target_idx), {})
                    results['all_solutions'].append({
                        'spec': target_idx + 1,
                        'solution': pref_idx + 1,
                        'target_gain_linear': td.get('target_gain_linear'),
                        'target_ugbw_mhz': td.get('target_ugbw_mhz'),
                        'target_pm_deg': td.get('target_pm_deg'),
                        'target_ibias_ma': td.get('target_ibias_ma'),
                        'output_gain_linear': float(gain_linear),
                        'output_gain_db': float(gain_db),
                        'output_ugbw_mhz': float(ugbw_val / 1e6),
                        'output_pm_deg': float(phm_val),
                        'output_ibias_ma': float(ibias_val * 1000),
                        'target_reached': bool(sol_reached),
                        'preference': preference.tolist() if hasattr(preference, 'tolist') else list(preference),
                    })
            except Exception as e:
                print(f"  [{tag}] Error spec {target_idx} pref {pref_idx}: {e}")
                continue

        if reached:
            results['reached_count'] += 1
        results['total_evaluated'] += 1

    # Save raw results
    raw_path = RESULTS_DIR / f"morl_raw_{scalarization}_agent.json"
    with open(raw_path, 'w') as f:
        json.dump(results, f, indent=2, default=_convert)
    print(f"  [{tag}] Saved {len(results['all_solutions'])} solutions to {raw_path}")
    print(f"  [{tag}] Reached: {results['reached_count']}/{results['total_evaluated']}")

    return raw_path


def _convert(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------
def generate_comparison(nw_raw_path, cosine_raw_path):
    """Generate comparison CSV between NW-agent and cosine-agent results."""
    import pandas as pd

    print(f"\n{'='*80}")
    print("  GENERATING COMPARISON: NW-AGENT vs COSINE-AGENT")
    print(f"{'='*80}\n")

    with open(nw_raw_path) as f:
        nw_data = json.load(f)
    with open(cosine_raw_path) as f:
        cos_data = json.load(f)

    def _fom(g, u, p, i, gt, ut, pt, it):
        def s(x): return max(float(x), 1e-9) if x is not None else 1e-9
        gt, ut, pt, it = s(gt), s(ut), s(pt), s(it)
        g, u, p, i = float(g), float(u), float(p), float(i)
        return (g - gt)/gt + (u - ut)/ut + (p - pt)/pt - (i - it)/it

    def best_per_spec(solutions):
        """Select best solution per spec by highest FOM."""
        by_spec = {}
        for sol in solutions:
            spec = sol['spec']
            fom = _fom(sol['output_gain_linear'], sol['output_ugbw_mhz'],
                       sol['output_pm_deg'], sol['output_ibias_ma'],
                       sol['target_gain_linear'], sol['target_ugbw_mhz'],
                       sol['target_pm_deg'], sol['target_ibias_ma'])
            sol['fom'] = fom

            # Per-objective pass
            g_out, g_tgt = sol['output_gain_linear'], sol['target_gain_linear']
            u_out, u_tgt = sol['output_ugbw_mhz'], sol['target_ugbw_mhz']
            p_out, p_tgt = sol['output_pm_deg'], sol['target_pm_deg']
            i_out, i_tgt = sol['output_ibias_ma'], sol['target_ibias_ma']
            g_ok = g_out >= g_tgt
            u_ok = u_out >= u_tgt
            p_ok = p_out >= p_tgt
            i_ok = i_out <= i_tgt
            sol['complete_pass'] = 'Yes' if (g_ok and u_ok and p_ok and i_ok) else 'No'

            if spec not in by_spec or fom > by_spec[spec]['fom']:
                by_spec[spec] = sol
        return by_spec

    nw_best = best_per_spec(nw_data['all_solutions'])
    cos_best = best_per_spec(cos_data['all_solutions'])

    all_specs = sorted(set(nw_best.keys()) | set(cos_best.keys()))
    rows = []
    wins_nw, wins_cos, ties = 0, 0, 0

    for spec in all_specs:
        nw_sol = nw_best.get(spec)
        cos_sol = cos_best.get(spec)
        if nw_sol is None or cos_sol is None:
            continue

        nw_fom = nw_sol['fom']
        cos_fom = cos_sol['fom']

        if abs(nw_fom - cos_fom) < 1e-9:
            winner = 'tie'
            ties += 1
        elif nw_fom > cos_fom:
            winner = 'nw_agent'
            wins_nw += 1
        else:
            winner = 'cosine_agent'
            wins_cos += 1

        rows.append({
            'spec': spec,
            'solution_nw_agent': nw_sol['solution'],
            'fom_nw_agent': nw_fom,
            'complete_pass_nw_agent': nw_sol['complete_pass'],
            'preference_nw_agent': str(nw_sol['preference']),
            'solution_cosine_agent': cos_sol['solution'],
            'fom_cosine_agent': cos_fom,
            'complete_pass_cosine_agent': cos_sol['complete_pass'],
            'preference_cosine_agent': str(cos_sol['preference']),
            'fom_winner': winner,
        })

    df = pd.DataFrame(rows)

    # Add summary rows
    summary_rows = [
        {'spec': 'summary_specs', 'solution_nw_agent': len(rows)},
        {'spec': 'summary_wins_nw_agent', 'solution_nw_agent': wins_nw},
        {'spec': 'summary_wins_cosine_agent', 'solution_nw_agent': wins_cos},
        {'spec': 'summary_ties', 'solution_nw_agent': ties},
    ]
    df = pd.concat([df, pd.DataFrame(summary_rows)], ignore_index=True)

    out_path = RESULTS_DIR / "morl_compare_nw_agent_vs_cosine_agent.csv"
    df.to_csv(out_path, index=False)
    print(f"Comparison saved: {out_path}")
    print(f"  NW-agent wins:     {wins_nw}")
    print(f"  Cosine-agent wins: {wins_cos}")
    print(f"  Ties:              {ties}")

    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train & evaluate NW vs Cosine MORL agents')
    parser.add_argument('--evaluate-only', action='store_true', help='Skip training, evaluate existing models')
    parser.add_argument('--train-only', action='store_true', help='Train only, skip evaluation')
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if not DATASET_PATH.exists():
        print(f"ERROR: Dataset not found: {DATASET_PATH}")
        sys.exit(1)

    with open(DATASET_PATH, 'rb') as f:
        specs = pickle.load(f)
    print(f"Loaded {len(list(specs.values())[0])} specs from {DATASET_PATH}")

    # Initialize environment
    env = AutoCktMOEnv(generalize=True, num_valid=len(list(specs.values())[0]), run_valid=True)
    print("Environment initialized")

    # Generate preferences
    try:
        preferences = generate_preference_vectors(4, method='focused', num_vectors=TRAIN_CONFIG['num_preferences'])
        if len(preferences) > TRAIN_CONFIG['num_preferences']:
            preferences = preferences[:TRAIN_CONFIG['num_preferences']]
    except Exception:
        preferences = generate_preference_vectors(4, method='random', num_vectors=TRAIN_CONFIG['num_preferences'])
    print(f"Using {len(preferences)} preference vectors")

    nw_model_path = MODELS_DIR / "trained_morl_nw_final.pth"
    cos_model_path = MODELS_DIR / "trained_morl_cosine_final.pth"

    # --- Training ---
    if not args.evaluate_only:
        print(f"\nStart time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        nw_model_path, _ = train_agent("nw", TRAIN_CONFIG, env, specs, preferences)
        cos_model_path, _ = train_agent("cosine", TRAIN_CONFIG, env, specs, preferences)

        print(f"\nTraining complete: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # --- Evaluation ---
    if not args.train_only:
        if not nw_model_path.exists():
            print(f"ERROR: NW model not found: {nw_model_path}")
            sys.exit(1)
        if not cos_model_path.exists():
            print(f"ERROR: Cosine model not found: {cos_model_path}")
            sys.exit(1)

        nw_raw = evaluate_agent(nw_model_path, "nw", env, specs, EVAL_CONFIG)
        cos_raw = evaluate_agent(cos_model_path, "cosine", env, specs, EVAL_CONFIG)

        generate_comparison(nw_raw, cos_raw)

    print(f"\nDone: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
