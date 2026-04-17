"""
Train LLM-guided DDQN (cosine) using the EXACT same config as train_nw_vs_cosine.py.
Then evaluate BOTH the original standard cosine model and the LLM cosine model
through the same evaluation pipeline to produce consistent, comparable FOMs.

Usage:
    python train_llm_cosine_original.py                # Train + evaluate + compare
    python train_llm_cosine_original.py --evaluate-only # Evaluate existing models only
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

# Add methodology to path
BASE_DIR = Path(__file__).parent
METHODOLOGY_DIR = BASE_DIR / "methodology"
sys.path.insert(0, str(METHODOLOGY_DIR))
sys.path.insert(0, str(BASE_DIR))

# Enable surrogate simulation
os.environ['AUTOCKT_USE_SURROGATE'] = 'true'

from autockt.envs.autockt_mo_env import AutoCktMOEnv
from autockt.agents.mo_agent import MO_DQN_Agent, MO_DQN_LLM_Agent
from autockt.utils.mo_utils import generate_preference_vectors
from llm_constraint import LLMActionFilter

# ---------------------------------------------------------------------------
# Configuration — IDENTICAL to train_nw_vs_cosine.py
# ---------------------------------------------------------------------------
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR  = RESULTS_DIR / "models_llm_original"
DATASET_PATH = BASE_DIR / "autockt" / "gen_specs" / "ngspice_specs_gen_two_stage_opamp"
LLM_CACHE_PATH = RESULTS_DIR / "llm_action_cache_original.json"

# Same training config as the original
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
# Helper: extract specs for LLM queries (same as train_llm_ddqn.py)
# ---------------------------------------------------------------------------
def _get_current_and_target_specs(env):
    if not hasattr(env.base_env, 'cur_specs') or not hasattr(env.base_env, 'specs_id'):
        return None, None
    specs_id = env.base_env.specs_id
    cur = env.base_env.cur_specs.copy()
    tgt = env.base_env.specs_ideal.copy()
    spec_dict_cur = dict(zip(specs_id, cur))
    spec_dict_tgt = dict(zip(specs_id, tgt))
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
# Training — same as train_nw_vs_cosine.py but with LLM masking added
# ---------------------------------------------------------------------------
def train_llm_agent(config, env, specs, preferences):
    """Train LLM-guided cosine DDQN using same config as original."""
    tag = "LLM-COSINE"
    print(f"\n{'='*80}")
    print(f"  TRAINING LLM-GUIDED AGENT — COSINE SCALARIZATION")
    print(f"  (Same config as original train_nw_vs_cosine.py)")
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
        model='llama3.2:3b',
        use_llm=True,
        cache_path=str(LLM_CACHE_PATH),
        verbose=True,
    )

    agent = MO_DQN_LLM_Agent(
        state_dim, action_dim, reward_dim,
        llm_filter=llm_filter,
        scalarization="cosine",
        use_preference_boost=True,
    )
    print(f"  [{tag}] Agent: state={state_dim}, action={action_dim}, reward={reward_dim}")

    training_specs = list(range(min(config['num_training_specs'], len(list(specs.values())[0]))))
    episode_count = 0
    history = {'episodes': [], 'rewards': [], 'scalarization': 'cosine', 'llm_guided': True}
    mask_update_freq = 3

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

                # LLM mask at episode start
                cur_specs, tgt_specs = _get_current_and_target_specs(env)
                if cur_specs is not None:
                    agent.update_llm_mask(cur_specs, tgt_specs)

                # Preference boost
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

                    if hasattr(agent, 'memory') and hasattr(agent.memory, 'push'):
                        agent.memory.push(state, action, reward_vec, next_state, done, preference)

                    if hasattr(agent, 'memory') and len(agent.memory) > 32 and episode_count % 4 == 0:
                        try:
                            agent.update(batch_size=32)
                        except Exception:
                            pass

                    state = next_state
                    episode_reward += float(np.sum(reward_vec)) if isinstance(reward_vec, (list, np.ndarray)) else float(reward_vec)

                    # Re-query LLM every 3 steps
                    if step % mask_update_freq == 0 and step > 0:
                        cur_specs, tgt_specs = _get_current_and_target_specs(env)
                        if cur_specs is not None:
                            agent.update_llm_mask(cur_specs, tgt_specs)

                    if done:
                        break

                # Reset preference
                agent.set_preference(preference)

                episode_count += 1
                history['episodes'].append(episode_count)
                history['rewards'].append(episode_reward)

                if episode_count % config['save_frequency'] == 0:
                    _save_model(agent, MODELS_DIR / f"checkpoint_llm_cosine_ep{episode_count}.pth", episode_count)

    # Save final model
    final_path = MODELS_DIR / "trained_llm_cosine_original_final.pth"
    _save_model(agent, final_path, episode_count)
    print(f"  [{tag}] Training complete — {episode_count} episodes. Model: {final_path}")

    hist_path = RESULTS_DIR / "training_history_llm_cosine_original.json"
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)

    llm_filter.print_stats()
    return final_path


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
# Evaluation — same code as evaluate.py / train_nw_vs_cosine.py
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


def evaluate_model(model_path, label, env, specs, config, use_llm=False):
    """Evaluate a model — same pipeline as evaluate.py."""
    print(f"\n{'='*80}")
    print(f"  EVALUATING: {label}")
    print(f"{'='*80}\n")

    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    state_dim = checkpoint.get('state_dim', 64)
    action_dim = checkpoint.get('action_dim', 64)
    reward_dim = checkpoint.get('reward_dim', 4)

    if use_llm:
        llm_filter = LLMActionFilter(
            model='llama3.2:3b', use_llm=True,
            cache_path=str(LLM_CACHE_PATH), verbose=False,
        )
        agent = MO_DQN_LLM_Agent(
            state_dim, action_dim, reward_dim,
            llm_filter=llm_filter,
            scalarization="cosine",
            use_preference_boost=True,
        )
    else:
        agent = MO_DQN_Agent(state_dim, action_dim, reward_dim, scalarization="cosine")

    if 'agent_state_dict' in checkpoint and checkpoint['agent_state_dict'] is not None:
        agent.q_network.load_state_dict(checkpoint['agent_state_dict'])
    if 'preference_net_state_dict' in checkpoint and checkpoint['preference_net_state_dict'] is not None:
        agent.preference_net.load_state_dict(checkpoint['preference_net_state_dict'])
    agent.q_network.eval()

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

    for target_idx in range(num_targets):
        if target_idx % 100 == 0 and target_idx > 0:
            print(f"  [{label}] Progress: {target_idx}/{num_targets}")

        try:
            env.base_env.obj_idx = target_idx
            env.reset()
            target_spec = env.base_env.specs_ideal.copy()
            specs_id = env.base_env.specs_id
        except Exception as e:
            print(f"  [{label}] Skip spec {target_idx}: {e}")
            continue

        reached = False
        for pref_idx, preference in enumerate(preferences):
            try:
                agent.set_preference(preference)
                state = env.reset()
                done = False
                steps = 0

                # LLM mask at start of evaluation episode
                if use_llm:
                    cur_specs, tgt_specs = _get_current_and_target_specs(env)
                    if cur_specs is not None:
                        agent.update_llm_mask(cur_specs, tgt_specs)
                    boosted = agent.get_boosted_preference(preference)
                    agent.set_preference(boosted)

                while not done and steps < config['max_steps']:
                    action = agent.select_action(state, preference if not use_llm else None)
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

                    # Re-query LLM every 3 steps during eval too
                    if use_llm and steps % 3 == 0:
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

                # Reset preference for next iteration
                if use_llm:
                    agent.set_preference(preference)

            except Exception as e:
                print(f"  [{label}] Error spec {target_idx} pref {pref_idx}: {e}")
                continue

        if reached:
            results['reached_count'] += 1
        results['total_evaluated'] += 1

    print(f"  [{label}] Done: {results['reached_count']}/{results['total_evaluated']} specs reached")
    print(f"  [{label}] Total solutions: {len(results['all_solutions'])}")
    return results


# ---------------------------------------------------------------------------
# FOM + CSV generation (same formula as main.py)
# ---------------------------------------------------------------------------
def _fom(g, u, p, i, gt, ut, pt, it):
    def s(x): return max(float(x), 1e-9) if x is not None else 1e-9
    gt, ut, pt, it = s(gt), s(ut), s(pt), s(it)
    return (float(g)-gt)/gt + (float(u)-ut)/ut + (float(p)-pt)/pt - (float(i)-it)/it


def results_to_csv(results, out_csv, label):
    """Convert results to CSV in same format as morl_autockt_results_original_morl_cosine.csv."""
    rows = []
    for sol in results['all_solutions']:
        g, u, p, i = sol['output_gain_linear'], sol['output_ugbw_mhz'], sol['output_pm_deg'], sol['output_ibias_ma']
        gt, ut, pt, it = sol['target_gain_linear'], sol['target_ugbw_mhz'], sol['target_pm_deg'], sol['target_ibias_ma']
        f = _fom(g, u, p, i, gt, ut, pt, it)
        g_ok = g >= gt if gt else True
        u_ok = u >= ut if ut else True
        p_ok = p >= pt if pt else True
        i_ok = i <= it if it else True
        rows.append({
            'spec': sol['spec'], 'solution': sol['solution'],
            'target_gain_linear': gt, 'target_ugbw_mhz': ut, 'target_pm_deg': pt, 'target_ibias_ma': it,
            'output_gain_linear': g, 'output_gain_db': sol.get('output_gain_db', 0),
            'output_ugbw_mhz': u, 'output_pm_deg': p, 'output_ibias_ma': i,
            'fom': round(f, 6),
            'gain_pass': 'Yes' if g_ok else 'No', 'ugbw_pass': 'Yes' if u_ok else 'No',
            'pm_pass': 'Yes' if p_ok else 'No', 'ibias_pass': 'Yes' if i_ok else 'No',
            'complete_pass': 'Yes' if (g_ok and u_ok and p_ok and i_ok) else 'No',
        })

    df = pd.DataFrame(rows)
    n = len(df)
    ns = df['spec'].nunique()
    yes = (df['complete_pass'] == 'Yes').sum()
    avg_fom = df['fom'].mean()
    best_per_spec = df.groupby('spec')['fom'].max()
    avg_best = best_per_spec.mean()
    top20 = best_per_spec.nlargest(20).mean()

    summaries = pd.DataFrame([
        {'spec': 'summary', 'complete_pass': f'{yes}/{n} solutions; {ns}/{ns} specs'},
        {'spec': f'summary_avg_fom_{n}_solutions', 'fom': round(avg_fom, 6)},
        {'spec': f'summary_avg_best_fom_{ns}_specs', 'fom': round(avg_best, 6)},
        {'spec': 'summary_avg_top20_best_fom', 'fom': round(top20, 6)},
    ])
    df = pd.concat([df, summaries], ignore_index=True)
    df.to_csv(out_csv, index=False)
    print(f"  {label}: {yes}/{n} pass, avg FOM={avg_fom:.4f}, avg best={avg_best:.4f}, top20={top20:.4f}")
    return avg_fom, avg_best


def generate_comparison(std_results, llm_results):
    """Generate per-spec comparison CSV."""
    def best_per_spec(solutions):
        by_spec = {}
        for sol in solutions:
            spec = sol['spec']
            f = _fom(sol['output_gain_linear'], sol['output_ugbw_mhz'],
                     sol['output_pm_deg'], sol['output_ibias_ma'],
                     sol['target_gain_linear'], sol['target_ugbw_mhz'],
                     sol['target_pm_deg'], sol['target_ibias_ma'])
            sol['_fom'] = f
            if spec not in by_spec or f > by_spec[spec]['_fom']:
                by_spec[spec] = sol
        return by_spec

    std_best = best_per_spec(std_results['all_solutions'])
    llm_best = best_per_spec(llm_results['all_solutions'])
    all_specs = sorted(set(std_best.keys()) & set(llm_best.keys()))

    rows = []; wins_std = wins_llm = ties = 0
    std_foms = []; llm_foms = []

    for spec in all_specs:
        ss, ls = std_best[spec], llm_best[spec]
        std_foms.append(ss['_fom']); llm_foms.append(ls['_fom'])
        if abs(ss['_fom'] - ls['_fom']) < 1e-9: w = 'tie'; ties += 1
        elif ls['_fom'] > ss['_fom']: w = 'llm_ddqn'; wins_llm += 1
        else: w = 'standard_ddqn'; wins_std += 1
        rows.append({
            'spec': spec,
            'fom_standard_ddqn': round(ss['_fom'], 6),
            'fom_llm_ddqn': round(ls['_fom'], 6),
            'fom_winner': w,
        })

    df = pd.DataFrame(rows)
    summaries = pd.DataFrame([
        {'spec': 'summary_specs', 'fom_standard_ddqn': len(rows)},
        {'spec': 'summary_wins_standard', 'fom_standard_ddqn': wins_std},
        {'spec': 'summary_wins_llm', 'fom_standard_ddqn': wins_llm},
        {'spec': 'summary_ties', 'fom_standard_ddqn': ties},
        {'spec': 'summary_avg_fom_standard', 'fom_standard_ddqn': round(np.mean(std_foms), 6)},
        {'spec': 'summary_avg_fom_llm', 'fom_llm_ddqn': round(np.mean(llm_foms), 6)},
    ])
    df = pd.concat([df, summaries], ignore_index=True)
    out = RESULTS_DIR / "morl_compare_original_std_vs_llm_cosine.csv"
    df.to_csv(out, index=False)
    print(f"\n  Comparison saved: {out}")
    print(f"  Standard wins: {wins_std}, LLM wins: {wins_llm}, Ties: {ties}")
    print(f"  Standard avg best FOM: {np.mean(std_foms):.4f}")
    print(f"  LLM avg best FOM:     {np.mean(llm_foms):.4f}")
    print(f"  Improvement:          {np.mean(llm_foms) - np.mean(std_foms):.4f}")


# ===========================================================================
# MAIN
# ===========================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluate-only', action='store_true')
    args = parser.parse_args()

    start = datetime.now()
    print(f"\n{'#'*80}")
    print(f"  ORIGINAL-CONFIG LLM vs STANDARD DDQN COMPARISON")
    print(f"  Start: {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*80}")

    # Load dataset
    with open(DATASET_PATH, 'rb') as f:
        specs = pickle.load(f)
    print(f"\nDataset: {len(list(specs.values())[0])} specs")

    # Setup env
    env = AutoCktMOEnv(generalize=True, num_valid=len(list(specs.values())[0]), run_valid=True)

    # Preferences
    try:
        preferences = generate_preference_vectors(4, method='focused', num_vectors=TRAIN_CONFIG['num_preferences'])
        if len(preferences) > TRAIN_CONFIG['num_preferences']:
            preferences = preferences[:TRAIN_CONFIG['num_preferences']]
    except Exception:
        preferences = generate_preference_vectors(4, method='random', num_vectors=TRAIN_CONFIG['num_preferences'])

    # Paths
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    original_model = RESULTS_DIR / "models_nw_vs_cosine" / "trained_morl_cosine_final.pth"
    llm_model = MODELS_DIR / "trained_llm_cosine_original_final.pth"

    # ---- TRAINING ----
    if not args.evaluate_only:
        if not original_model.exists():
            print(f"\nERROR: Original model not found: {original_model}")
            print("Run train_nw_vs_cosine.py first.")
            sys.exit(1)

        print(f"\nOriginal model exists: {original_model}")
        print("Training LLM-guided version with SAME config...")
        llm_model = train_llm_agent(TRAIN_CONFIG, env, specs, preferences)

    # ---- EVALUATION ----
    print(f"\n{'#'*80}")
    print("  EVALUATION PHASE")
    print(f"{'#'*80}")

    if not original_model.exists():
        print(f"ERROR: {original_model} not found"); sys.exit(1)
    if not Path(llm_model).exists():
        print(f"ERROR: {llm_model} not found"); sys.exit(1)

    # Evaluate standard model (no LLM)
    std_results = evaluate_model(original_model, "Standard Cosine DDQN", env, specs, EVAL_CONFIG, use_llm=False)

    # Evaluate LLM model (with LLM masking during eval)
    llm_results = evaluate_model(llm_model, "LLM Cosine DDQN", env, specs, EVAL_CONFIG, use_llm=True)

    # ---- CSV GENERATION ----
    print(f"\n{'#'*80}")
    print("  CSV GENERATION")
    print(f"{'#'*80}")

    results_to_csv(std_results, RESULTS_DIR / "morl_original_standard_cosine.csv", "Standard Cosine")
    results_to_csv(llm_results, RESULTS_DIR / "morl_original_llm_cosine.csv", "LLM Cosine")
    generate_comparison(std_results, llm_results)

    elapsed = datetime.now() - start
    print(f"\nTotal time: {elapsed}")
    print("DONE")
