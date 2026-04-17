"""
Train LLM-guided MO-DQN agents using Ollama for action masking.
Compares LLM-guided DDQN vs standard DDQN with both NW and cosine scalarization.

The LLM provides:
  1. Action masking — blocks clearly wrong actions based on circuit design knowledge
  2. Preference boosting — shifts weight toward the most-violated objective

Usage:
    python train_llm_ddqn.py                   # Train + evaluate both scalarizations
    python train_llm_ddqn.py --evaluate-only   # Evaluate existing models only
    python train_llm_ddqn.py --train-only      # Train only
"""

import os
import sys
import pickle
import numpy as np
import json
import torch
import gym
import random
from pathlib import Path
from datetime import datetime

# Add methodology to path
BASE_DIR = Path(__file__).parent
METHODOLOGY_DIR = BASE_DIR / "methodology"
sys.path.insert(0, str(METHODOLOGY_DIR))
sys.path.insert(0, str(BASE_DIR))

# Enable surrogate simulation
os.environ['AUTOCKT_USE_SURROGATE'] = 'true'

from autockt.envs.autockt_mo_env import AutoCktMOEnv
from autockt.agents.mo_agent import MO_DQN_LLM_Agent
from autockt.utils.mo_utils import generate_preference_vectors
from llm_constraint import LLMActionFilter

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR  = RESULTS_DIR / "models_llm_ddqn"
DATASET_PATH = BASE_DIR / "autockt" / "gen_specs" / "ngspice_specs_gen_two_stage_opamp"
LLM_CACHE_PATH = RESULTS_DIR / "llm_action_cache.json"

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

LLM_CONFIG = {
    'model': 'llama3.2:3b',
    'use_llm': True,          # Use actual Ollama LLM for action masking
    'use_preference_boost': True,
    'verbose': True,
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Helpers to extract specs from env for LLM queries
# ---------------------------------------------------------------------------
def _get_current_and_target_specs(env):
    """Extract current and target spec arrays from env for LLM query."""
    if not hasattr(env.base_env, 'cur_specs') or not hasattr(env.base_env, 'specs_id'):
        return None, None

    specs_id = env.base_env.specs_id
    cur = env.base_env.cur_specs.copy()
    tgt = env.base_env.specs_ideal.copy()

    spec_dict_cur = dict(zip(specs_id, cur))
    spec_dict_tgt = dict(zip(specs_id, tgt))

    # Order: [gain, ugbw, pm, ibias]
    current = [
        spec_dict_cur.get('gain', spec_dict_cur.get('gain_min', cur[0])),
        spec_dict_cur.get('ugbw', spec_dict_cur.get('ugbw_min', cur[1] if len(cur) > 1 else 0)),
        spec_dict_cur.get('phm',  spec_dict_cur.get('phm_min',  cur[2] if len(cur) > 2 else 0)),
        spec_dict_cur.get('ibias_max', cur[3] if len(cur) > 3 else 0),
    ]
    target = [
        spec_dict_tgt.get('gain', spec_dict_tgt.get('gain_min', tgt[0])),
        spec_dict_tgt.get('ugbw', spec_dict_tgt.get('ugbw_min', tgt[1] if len(tgt) > 1 else 0)),
        spec_dict_tgt.get('phm',  spec_dict_tgt.get('phm_min',  tgt[2] if len(tgt) > 2 else 0)),
        spec_dict_tgt.get('ibias_max', tgt[3] if len(tgt) > 3 else 0),
    ]
    return current, target


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_agent(scalarization, config, llm_config, env, specs, preferences):
    """Train a single MO_DQN_LLM_Agent with the given scalarization."""
    tag = f"LLM-{scalarization.upper()}"
    print(f"\n{'='*80}")
    print(f"  TRAINING LLM-GUIDED DDQN — {scalarization.upper()} SCALARIZATION")
    print(f"{'='*80}\n")

    state_dim = env.observation_space.shape[0] if hasattr(env.observation_space, 'shape') else 20
    if isinstance(env.action_space, gym.spaces.Tuple):
        action_dim = env.action_space.spaces[0].n
        num_params = len(env.action_space.spaces)
    else:
        action_dim = env.action_space.n if hasattr(env.action_space, 'n') else 7
        num_params = 1
    reward_dim = 4

    # Create LLM filter
    llm_filter = LLMActionFilter(
        model=llm_config['model'],
        use_llm=llm_config['use_llm'],
        cache_path=str(LLM_CACHE_PATH),
        verbose=llm_config['verbose'],
    )

    agent = MO_DQN_LLM_Agent(
        state_dim, action_dim, reward_dim,
        llm_filter=llm_filter,
        scalarization=scalarization,
        use_preference_boost=llm_config['use_preference_boost'],
    )
    print(f"  [{tag}] Agent created: state={state_dim}, action={action_dim}, reward={reward_dim}")

    training_specs = list(range(min(config['num_training_specs'], len(list(specs.values())[0]))))
    episode_count = 0
    history = {'episodes': [], 'rewards': [], 'scalarization': scalarization, 'llm_guided': True}
    mask_update_freq = 3  # Update LLM mask every N steps

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

                # Initial LLM mask query
                cur_specs, tgt_specs = _get_current_and_target_specs(env)
                if cur_specs is not None:
                    agent.update_llm_mask(cur_specs, tgt_specs)

                # Apply preference boost
                boosted_pref = agent.get_boosted_preference(preference)
                agent.set_preference(boosted_pref)

                for step in range(config['max_steps']):
                    epsilon = max(0.1, 1.0 - episode_count / 1000)
                    try:
                        action = agent.select_action(state, None, epsilon=epsilon)
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

                    # Store experience with original preference (not boosted)
                    if hasattr(agent, 'memory') and hasattr(agent.memory, 'push'):
                        agent.memory.push(state, action, reward_vec, next_state, done, preference)

                    if hasattr(agent, 'memory') and len(agent.memory) > 32 and episode_count % 4 == 0:
                        try:
                            agent.update(batch_size=32)
                        except Exception:
                            pass

                    state = next_state
                    episode_reward += float(np.sum(reward_vec)) if isinstance(reward_vec, (list, np.ndarray)) else float(reward_vec)

                    # Periodically update LLM mask during episode
                    if step % mask_update_freq == 0 and step > 0:
                        cur_specs, tgt_specs = _get_current_and_target_specs(env)
                        if cur_specs is not None:
                            agent.update_llm_mask(cur_specs, tgt_specs)

                    if done:
                        break

                # Reset preference back to original for next episode
                agent.set_preference(preference)

                episode_count += 1
                history['episodes'].append(episode_count)
                history['rewards'].append(episode_reward)

                if episode_count % config['save_frequency'] == 0:
                    _save_model(agent, MODELS_DIR / f"checkpoint_llm_{scalarization}_ep{episode_count}.pth", episode_count)

    # Save final model
    final_path = MODELS_DIR / f"trained_llm_{scalarization}_final.pth"
    _save_model(agent, final_path, episode_count)
    print(f"  [{tag}] Training complete — {episode_count} episodes. Model: {final_path}")

    # Save history
    hist_path = RESULTS_DIR / f"training_history_llm_{scalarization}.json"
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)

    # Print LLM stats
    llm_filter.print_stats()

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
        'llm_guided': True,
    }, path)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def check_target_reached(actual_specs, target_specs, specs_id):
    """Tolerance-based target check."""
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


def evaluate_agent(model_path, scalarization, llm_config, env, specs, config):
    """Evaluate a trained LLM-guided agent."""
    tag = f"LLM-{scalarization.upper()}"
    print(f"\n{'='*80}")
    print(f"  EVALUATING LLM-GUIDED DDQN — {scalarization.upper()}")
    print(f"{'='*80}\n")

    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    state_dim = checkpoint.get('state_dim', 64)
    action_dim = checkpoint.get('action_dim', 64)
    reward_dim = checkpoint.get('reward_dim', 4)

    llm_filter = LLMActionFilter(
        model=llm_config['model'],
        use_llm=llm_config['use_llm'],
        cache_path=str(LLM_CACHE_PATH),
        verbose=False,
    )

    agent = MO_DQN_LLM_Agent(
        state_dim, action_dim, reward_dim,
        llm_filter=llm_filter,
        scalarization=scalarization,
        use_preference_boost=llm_config['use_preference_boost'],
    )
    if 'agent_state_dict' in checkpoint and checkpoint['agent_state_dict'] is not None:
        agent.q_network.load_state_dict(checkpoint['agent_state_dict'])
    if 'preference_net_state_dict' in checkpoint and checkpoint['preference_net_state_dict'] is not None:
        agent.preference_net.load_state_dict(checkpoint['preference_net_state_dict'])
    agent.q_network.eval()
    print(f"  [{tag}] Model loaded from {model_path}")

    num_specs_total = len(list(specs.values())[0])
    num_targets = min(config['num_targets'], num_specs_total)

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
    mask_update_freq = 5  # Update mask every N steps during eval

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

                # Initial LLM mask
                cur_specs, tgt_specs = _get_current_and_target_specs(env)
                if cur_specs is not None:
                    agent.update_llm_mask(cur_specs, tgt_specs)

                # Apply preference boost
                boosted_pref = agent.get_boosted_preference(preference)
                agent.set_preference(boosted_pref)

                while not done and steps < config['max_steps']:
                    action = agent.select_action(state, None)
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

                    # Periodically update mask
                    if steps % mask_update_freq == 0:
                        cur_specs, tgt_specs = _get_current_and_target_specs(env)
                        if cur_specs is not None:
                            agent.update_llm_mask(cur_specs, tgt_specs)

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

    raw_path = RESULTS_DIR / f"morl_raw_llm_{scalarization}_agent.json"
    with open(raw_path, 'w') as f:
        json.dump(results, f, indent=2, default=_convert)
    print(f"  [{tag}] Saved {len(results['all_solutions'])} solutions to {raw_path}")
    print(f"  [{tag}] Reached: {results['reached_count']}/{results['total_evaluated']}")

    llm_filter.print_stats()
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
# Comparison: LLM-DDQN vs Standard DDQN
# ---------------------------------------------------------------------------
def generate_comparison(llm_raw_paths, std_raw_paths):
    """Compare LLM-guided vs standard DDQN (best-of-both scalarizations)."""
    import pandas as pd

    print(f"\n{'='*80}")
    print("  LLM-GUIDED DDQN vs STANDARD DDQN")
    print(f"{'='*80}\n")

    def _fom(s):
        gt = max(float(s.get('target_gain_linear', 1e-9)), 1e-9)
        ut = max(float(s.get('target_ugbw_mhz', 1e-9)), 1e-9)
        pt = max(float(s.get('target_pm_deg', 1e-9)), 1e-9)
        it = max(float(s.get('target_ibias_ma', 1e-9)), 1e-9)
        g = float(s['output_gain_linear'])
        u = float(s['output_ugbw_mhz'])
        p = float(s['output_pm_deg'])
        i = float(s['output_ibias_ma'])
        return (g-gt)/gt + (u-ut)/ut + (p-pt)/pt - (i-it)/it

    def best_per_spec(all_solutions):
        by_spec = {}
        for sol in all_solutions:
            f = _fom(sol)
            sol['fom'] = f
            spec = sol['spec']
            if spec not in by_spec or f > by_spec[spec]['fom']:
                by_spec[spec] = sol
        return by_spec

    # Merge solutions from both scalarizations for each approach
    llm_all = []
    for p in llm_raw_paths:
        with open(p) as f:
            llm_all.extend(json.load(f)['all_solutions'])
    std_all = []
    for p in std_raw_paths:
        with open(p) as f:
            std_all.extend(json.load(f)['all_solutions'])

    llm_best = best_per_spec(llm_all)
    std_best = best_per_spec(std_all)

    all_specs = sorted(set(llm_best.keys()) | set(std_best.keys()))
    rows = []
    wins_llm, wins_std, ties = 0, 0, 0

    for spec in all_specs:
        ls, ss = llm_best.get(spec), std_best.get(spec)
        if ls is None or ss is None:
            continue
        lf, sf = ls['fom'], ss['fom']
        if abs(lf - sf) < 1e-9:
            winner = 'tie'; ties += 1
        elif lf > sf:
            winner = 'llm_ddqn'; wins_llm += 1
        else:
            winner = 'standard_ddqn'; wins_std += 1
        rows.append({
            'spec': spec, 'fom_llm_ddqn': lf, 'fom_standard_ddqn': sf, 'fom_winner': winner,
        })

    df = pd.DataFrame(rows)
    llm_mean = df['fom_llm_ddqn'].mean()
    std_mean = df['fom_standard_ddqn'].mean()

    out_path = RESULTS_DIR / "morl_compare_llm_vs_ddqn.csv"
    df.to_csv(out_path, index=False)

    print(f"  Standard DDQN wins: {wins_std} | Mean FOM: {std_mean:.4f}")
    print(f"  LLM-guided wins:    {wins_llm} | Mean FOM: {llm_mean:.4f}")
    print(f"  Ties:               {ties}")
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train & evaluate LLM-guided DDQN agents')
    parser.add_argument('--evaluate-only', action='store_true')
    parser.add_argument('--train-only', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if not DATASET_PATH.exists():
        print(f"ERROR: Dataset not found: {DATASET_PATH}")
        sys.exit(1)

    with open(DATASET_PATH, 'rb') as f:
        specs = pickle.load(f)
    print(f"Loaded {len(list(specs.values())[0])} specs from {DATASET_PATH}")

    env = AutoCktMOEnv(generalize=True, num_valid=len(list(specs.values())[0]), run_valid=True)
    print("Environment initialized")

    try:
        preferences = generate_preference_vectors(4, method='focused', num_vectors=TRAIN_CONFIG['num_preferences'])
        if len(preferences) > TRAIN_CONFIG['num_preferences']:
            preferences = preferences[:TRAIN_CONFIG['num_preferences']]
    except Exception:
        preferences = generate_preference_vectors(4, method='random', num_vectors=TRAIN_CONFIG['num_preferences'])
    print(f"Using {len(preferences)} preference vectors")

    cos_model = MODELS_DIR / "trained_llm_cosine_final.pth"
    nw_model  = MODELS_DIR / "trained_llm_nw_final.pth"

    # --- Training ---
    if not args.evaluate_only:
        print(f"\nStart: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        cos_model, _ = train_agent("cosine", TRAIN_CONFIG, LLM_CONFIG, env, specs, preferences)
        nw_model, _  = train_agent("nw",     TRAIN_CONFIG, LLM_CONFIG, env, specs, preferences)

        print(f"Training done: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # --- Evaluation ---
    if not args.train_only:
        if not cos_model.exists() if isinstance(cos_model, Path) else not Path(cos_model).exists():
            print(f"ERROR: Cosine model not found: {cos_model}")
            sys.exit(1)
        if not nw_model.exists() if isinstance(nw_model, Path) else not Path(nw_model).exists():
            print(f"ERROR: NW model not found: {nw_model}")
            sys.exit(1)

        cos_raw = evaluate_agent(cos_model, "cosine", LLM_CONFIG, env, specs, EVAL_CONFIG)
        nw_raw  = evaluate_agent(nw_model,  "nw",     LLM_CONFIG, env, specs, EVAL_CONFIG)

        # Compare against standard DDQN results
        std_nw_raw  = RESULTS_DIR / "morl_raw_nw_agent.json"
        std_cos_raw = RESULTS_DIR / "morl_raw_cosine_agent.json"

        if std_nw_raw.exists() and std_cos_raw.exists():
            generate_comparison(
                llm_raw_paths=[cos_raw, nw_raw],
                std_raw_paths=[std_nw_raw, std_cos_raw],
            )
        else:
            print("\n  WARNING: Standard DDQN results not found. Skipping comparison.")
            print(f"    Expected: {std_nw_raw} and {std_cos_raw}")

    print(f"\nAll done: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
