"""
Train LLM-masked DDQN agent in the EXACT same environment that produced the cosine 1.488 FOM.
Same config, surrogate, rewards, dataset. Uses cosine scalarization + LLM action masking.
Then evaluate on 1000 specs (10 prefs, correct epsilon) and compare against cosine 1.488.
"""
import os
import sys
import pickle
import numpy as np
import json
import torch
import gym
import pandas as pd
import random
import shutil
from pathlib import Path
from datetime import datetime
import time

BASE_DIR = Path(__file__).parent
METHODOLOGY_DIR = BASE_DIR / "methodology"
sys.path.insert(0, str(METHODOLOGY_DIR))
sys.path.insert(0, str(BASE_DIR))

os.environ['AUTOCKT_USE_SURROGATE'] = 'true'

from autockt.envs.autockt_mo_env import AutoCktMOEnv
from autockt.agents.mo_agent import MO_DQN_LLM_Agent
from autockt.utils.mo_utils import generate_preference_vectors
from llm_constraint import LLMActionFilter

DATASET_PATH = BASE_DIR / "dataset" / "ngspice_specs_gen_two_stage_opamp"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = RESULTS_DIR / "models_llm"
LLM_CACHE_PATH = RESULTS_DIR / "llm_action_cache.json"

WITH_15_WITH_20_RESULTS_DIR = Path(
    r"C:\Users\kobeo\OneDrive\Desktop\trail\with_15%_with_20\morl_experiments\morl_autockt\results"
)

# EXACT same training config as original train_and_save_model.py
TRAIN_CONFIG = {
    'num_training_specs': 50,
    'num_preferences': 10,
    'max_steps': 30,
    'training_episodes': 100,
    'save_frequency': 10,
}

EVAL_CONFIG = {
    'num_targets': 1000,
    'num_preferences': 10,
    'max_steps': 120,
}


def _compute_eval_metrics(results):
    if not results:
        return {
            'avg_best_fom': float('nan'),
            'top20_fom': float('nan'),
            'spec_pass_rate': float('nan'),
            'num_specs': 0,
            'num_solutions': 0,
        }
    df = pd.DataFrame(results)
    df['fom'] = df['fom'].astype(float)
    by_spec = df.groupby('spec')
    best = by_spec['fom'].max()
    avg_best = float(best.mean()) if len(best) else float('nan')
    top20 = float(best.nlargest(min(20, len(best))).mean()) if len(best) else float('nan')
    return {
        'avg_best_fom': avg_best,
        'top20_fom': top20,
        'spec_pass_rate': float('nan'),
        'num_specs': int(best.shape[0]),
        'num_solutions': int(df.shape[0]),
    }


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _fom(g, u, p, i, gt, ut, pt, it):
    def s(x): return max(float(x), 1e-9)
    gt, ut, pt, it = s(gt), s(ut), s(pt), s(it)
    return (float(g)-gt)/gt + (float(u)-ut)/ut + (float(p)-pt)/pt - (float(i)-it)/it


def _get_current_and_target_specs(env):
    """Extract current and target spec arrays from env for LLM query."""
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


def train_llm_agent(env, specs, preferences, num_params, scalarization, time_to_target=None):
    """Train LLM-masked DDQN agent with action masking + chosen scalarization."""
    print(f"\n{'='*70}")
    print(f"  TRAINING LLM-MASKED DDQN ({scalarization} scalarization)")
    print(f"{'='*70}\n")

    state_dim = env.observation_space.shape[0] if hasattr(env.observation_space, 'shape') else 20
    if isinstance(env.action_space, gym.spaces.Tuple):
        action_dim = env.action_space.spaces[0].n
    else:
        action_dim = env.action_space.n if hasattr(env.action_space, 'n') else 7
    reward_dim = 4

    llm_filter = LLMActionFilter(
        model='llama3.2:1b',
        use_llm=True,
        cache_path=str(LLM_CACHE_PATH),
        verbose=False,
    )

    agent = MO_DQN_LLM_Agent(
        state_dim, action_dim, reward_dim,
        llm_filter=llm_filter,
        scalarization=scalarization,
        use_preference_boost=True,
    )
    print(f"  Agent: state={state_dim}, action={action_dim}, reward={reward_dim}")
    print(f"  LLM available: {llm_filter.llm_available}")

    num_specs_total = len(list(specs.values())[0])
    training_specs = list(range(min(TRAIN_CONFIG['num_training_specs'], num_specs_total)))
    episode_count = 0
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    mask_update_freq = 10
    ttt_reached = False
    ttt_minutes = None
    ttt_metrics = None
    ttt_start = time.perf_counter()

    for spec_idx, spec_num in enumerate(training_specs):
        if spec_idx % 10 == 0:
            print(f"  [LLM] Spec {spec_idx+1}/{len(training_specs)}...")

        try:
            env.base_env.obj_idx = spec_num
            env.reset()
        except Exception:
            continue

        for pref_idx, preference in enumerate(preferences):
            agent.set_preference(preference)
            episodes_per_pref = max(1, TRAIN_CONFIG['training_episodes'] // len(preferences))

            for episode in range(episodes_per_pref):
                state = env.reset()

                # Initial LLM mask
                cur_specs, tgt_specs = _get_current_and_target_specs(env)
                if cur_specs is not None:
                    agent.update_llm_mask(cur_specs, tgt_specs)

                boosted_pref = agent.get_boosted_preference(preference)
                agent.set_preference(boosted_pref)

                for step in range(TRAIN_CONFIG['max_steps']):
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

                    # Periodically update LLM mask
                    if step % mask_update_freq == 0 and step > 0:
                        cur_specs, tgt_specs = _get_current_and_target_specs(env)
                        if cur_specs is not None:
                            agent.update_llm_mask(cur_specs, tgt_specs)

                    if done:
                        break

                # Reset preference
                agent.set_preference(preference)
                episode_count += 1

        if time_to_target and time_to_target.get('enabled'):
            every = int(time_to_target.get('eval_every_specs', 5))
            if every > 0 and ((spec_idx + 1) % every == 0 or (spec_idx + 1) == len(training_specs)):
                tmp_path = MODELS_DIR / f"llm_{scalarization}_ttt_tmp.pth"
                torch.save({
                    'agent_state_dict': agent.q_network.state_dict(),
                    'preference_net_state_dict': agent.preference_net.state_dict(),
                    'optimizer_state_dict': agent.optimizer.state_dict(),
                    'state_dim': state_dim,
                    'action_dim': action_dim,
                    'reward_dim': reward_dim,
                    'scalarization': scalarization,
                    'episode_count': episode_count,
                    'llm_guided': True,
                }, tmp_path)

                # Run a small evaluation subset with current weights
                metrics_eval_cfg = dict(EVAL_CONFIG)
                metrics_eval_cfg['num_targets'] = int(time_to_target.get('eval_subset', 50))
                llm_eval_res, _, _ = evaluate_llm_agent(
                    agent,
                    llm_filter,
                    env,
                    specs,
                    preferences,
                    scalarization,
                    track_mask_time=False,
                    num_targets_override=metrics_eval_cfg['num_targets'],
                )
                metrics = _compute_eval_metrics(llm_eval_res)
                print(
                    f"  [LLM] TTT eval on {metrics['num_specs']} specs: "
                    f"avg_best={metrics['avg_best_fom']:.4f}, top20={metrics['top20_fom']:.4f}"
                )

                ok = True
                if time_to_target.get('target_avg_best_fom') is not None:
                    ok = ok and (metrics['avg_best_fom'] >= float(time_to_target['target_avg_best_fom']))
                if time_to_target.get('target_top20_fom') is not None:
                    ok = ok and (metrics['top20_fom'] >= float(time_to_target['target_top20_fom']))

                if ok:
                    ttt_reached = True
                    ttt_minutes = (time.perf_counter() - ttt_start) / 60.0
                    ttt_metrics = metrics
                    break

        if (spec_idx + 1) % TRAIN_CONFIG['save_frequency'] == 0:
            cp = MODELS_DIR / f"llm_checkpoint_ep{episode_count}.pth"
            torch.save({
                'agent_state_dict': agent.q_network.state_dict(),
                'preference_net_state_dict': agent.preference_net.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'state_dim': state_dim, 'action_dim': action_dim, 'reward_dim': reward_dim,
                'scalarization': 'cosine', 'episode_count': episode_count, 'llm_guided': True,
            }, cp)

    final = MODELS_DIR / "llm_cosine_final.pth"
    torch.save({
        'agent_state_dict': agent.q_network.state_dict(),
        'preference_net_state_dict': agent.preference_net.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'state_dim': state_dim, 'action_dim': action_dim, 'reward_dim': reward_dim,
        'scalarization': 'cosine', 'episode_count': episode_count, 'llm_guided': True,
    }, final)
    print(f"  [LLM] Done — {episode_count} episodes. Model: {final}")
    llm_filter.print_stats()
    return agent, llm_filter, {
        'reached': bool(ttt_reached),
        'minutes': ttt_minutes,
        'metrics': ttt_metrics,
    }


def build_llm_agent(env, num_params, scalarization):
    """Build agent + LLM filter without training (for evaluate-only)."""
    state_dim = env.observation_space.shape[0] if hasattr(env.observation_space, 'shape') else 20
    if isinstance(env.action_space, gym.spaces.Tuple):
        action_dim = env.action_space.spaces[0].n
    else:
        action_dim = env.action_space.n if hasattr(env.action_space, 'n') else 7
    reward_dim = 4

    llm_filter = LLMActionFilter(
        model='llama3.2:1b',
        use_llm=True,
        cache_path=str(LLM_CACHE_PATH),
        verbose=False,
    )

    agent = MO_DQN_LLM_Agent(
        state_dim, action_dim, reward_dim,
        llm_filter=llm_filter,
        scalarization=scalarization,
        use_preference_boost=True,
    )
    return agent, llm_filter


def load_agent_weights(agent, model_path: Path):
    ckpt = torch.load(model_path, map_location='cpu')
    if 'agent_state_dict' in ckpt:
        agent.q_network.load_state_dict(ckpt['agent_state_dict'])
    if 'preference_net_state_dict' in ckpt and hasattr(agent, 'preference_net'):
        agent.preference_net.load_state_dict(ckpt['preference_net_state_dict'])
    return ckpt


def evaluate_llm_agent(agent, llm_filter, env, specs, preferences, scalarization, track_mask_time: bool = False, num_targets_override: int = None):
    """Evaluate with correct epsilon schedule and LLM masking."""
    print(f"\n{'='*70}")
    print(f"  EVALUATING LLM-MASKED DDQN ({scalarization})")
    print(f"  epsilon = max(0.05, 0.15*(1-step/120))")
    print(f"{'='*70}\n")

    agent.q_network.eval()
    max_steps = EVAL_CONFIG['max_steps']
    num_specs_total = len(list(specs.values())[0])
    if num_targets_override is None:
        num_targets = min(EVAL_CONFIG['num_targets'], num_specs_total)
    else:
        num_targets = min(int(num_targets_override), num_specs_total)
    mask_update_freq = 20

    target_specs_json = {}
    for i in range(num_specs_total):
        target_specs_json[str(i)] = {
            'target_gain_linear': float(specs['gain_min'][i]),
            'target_ugbw_mhz': float(specs['ugbw_min'][i]) / 1e6,
            'target_pm_deg': float(specs['phm_min'][i]),
            'target_ibias_ma': float(specs['ibias_max'][i]) * 1000.0,
        }

    num_params = len(env.action_space.spaces) if isinstance(env.action_space, gym.spaces.Tuple) else 1
    all_solutions = []
    mask_time_s = 0.0
    steps_to_done = []

    for target_idx in range(num_targets):
        if target_idx % 100 == 0 and target_idx > 0:
            print(f"  [LLM] Progress: {target_idx}/{num_targets}")

        try:
            env.base_env.obj_idx = target_idx
            env.reset()
            env_specs_id = env.base_env.specs_id
        except Exception:
            continue

        # Evaluate ALL 10 preferences (no early stop) = 10000 total solutions
        for pref_idx, preference in enumerate(preferences):
            try:
                agent.set_preference(preference)
                state = env.reset()
                done = False
                steps_taken = 0

                # Initial LLM mask
                cur_specs, tgt_specs = _get_current_and_target_specs(env)
                if cur_specs is not None:
                    if track_mask_time:
                        t0 = time.perf_counter()
                        agent.update_llm_mask(cur_specs, tgt_specs)
                        mask_time_s += (time.perf_counter() - t0)
                    else:
                        agent.update_llm_mask(cur_specs, tgt_specs)
                boosted_pref = agent.get_boosted_preference(preference)
                agent.set_preference(boosted_pref)

                for step in range(max_steps):
                    # SAME epsilon schedule as original evaluate_with_saved_model.py
                    epsilon = max(0.05, 0.15 * (1 - step / max_steps))
                    action = agent.select_action(state, None, epsilon=epsilon)

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
                    steps_taken += 1

                    if step % mask_update_freq == 0 and step > 0:
                        cur_specs, tgt_specs = _get_current_and_target_specs(env)
                        if cur_specs is not None:
                            if track_mask_time:
                                t0 = time.perf_counter()
                                agent.update_llm_mask(cur_specs, tgt_specs)
                                mask_time_s += (time.perf_counter() - t0)
                            else:
                                agent.update_llm_mask(cur_specs, tgt_specs)

                    if done:
                        break

                steps_to_done.append(steps_taken)

                # Reset preference
                agent.set_preference(preference)

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
                        'scalarization': f'{scalarization}+llm',
                    })
            except Exception:
                continue

    if len(steps_to_done) > 0:
        arr = np.array(steps_to_done, dtype=float)
        steps_stats = {
            'mean': float(arr.mean()),
            'p50': float(np.percentile(arr, 50)),
            'p90': float(np.percentile(arr, 90)),
            'p99': float(np.percentile(arr, 99)),
        }
        print(
            f"  [LLM] Steps-to-done: mean={steps_stats['mean']:.2f}, "
            f"p50={steps_stats['p50']:.0f}, p90={steps_stats['p90']:.0f}, p99={steps_stats['p99']:.0f}"
        )
    else:
        steps_stats = None

    return all_solutions, mask_time_s, steps_stats


def compare_with_cosine_1488(llm_results):
    """Compare LLM-masked results against original cosine 1.488."""
    orig_raw = Path(r"C:\Users\kobeo\OneDrive\Desktop\trail\with_15%_with_20\with_15%\morl_autockt\results\morl_autockt_results_raw.json")
    if not orig_raw.exists():
        print(f"  WARNING: Original cosine raw JSON not found: {orig_raw}")
        return

    with open(orig_raw) as f:
        cos_data = json.load(f)

    cos_best = {}
    for sol in cos_data['all_solutions']:
        sp = sol['spec']
        f = _fom(sol['output_gain_linear'], sol['output_ugbw_mhz'], sol['output_pm_deg'],
                 sol['output_ibias_ma'], sol['target_gain_linear'], sol['target_ugbw_mhz'],
                 sol['target_pm_deg'], sol['target_ibias_ma'])
        if sp not in cos_best or f > cos_best[sp]:
            cos_best[sp] = f

    llm_best = {}
    for sol in llm_results:
        sp = sol['spec']
        f = sol['fom']
        if sp not in llm_best or f > llm_best[sp]:
            llm_best[sp] = f

    common = sorted(set(cos_best.keys()) & set(llm_best.keys()))
    rows = []
    wins_cos = wins_llm = ties = 0
    for sp in common:
        cf = cos_best[sp]; lf = llm_best[sp]
        if abs(cf - lf) < 1e-9: w = 'tie'; ties += 1
        elif cf > lf: w = 'cosine_1488'; wins_cos += 1
        else: w = 'llm_masked'; wins_llm += 1
        rows.append({'spec': sp, 'fom_cosine_1488': round(cf, 6), 'fom_llm_masked': round(lf, 6), 'winner': w})

    cos_avg = np.mean([cos_best[s] for s in common])
    llm_avg = np.mean([llm_best[s] for s in common])

    cmp = pd.DataFrame(rows)
    sums = pd.DataFrame([
        {'spec': 'summary_specs', 'fom_cosine_1488': len(common)},
        {'spec': 'summary_wins_cosine_1488', 'fom_cosine_1488': wins_cos},
        {'spec': 'summary_wins_llm_masked', 'fom_llm_masked': wins_llm},
        {'spec': 'summary_ties', 'fom_cosine_1488': ties},
        {'spec': 'summary_avg_best_fom_cosine_1488', 'fom_cosine_1488': round(cos_avg, 6)},
        {'spec': 'summary_avg_best_fom_llm_masked', 'fom_llm_masked': round(llm_avg, 6)},
    ])
    out = pd.concat([cmp, sums], ignore_index=True)
    path = RESULTS_DIR / "morl_compare_llm_masked_vs_cosine_1488.csv"
    out.to_csv(path, index=False)

    print(f"\n{'='*70}")
    print(f"  LLM-MASKED vs COSINE 1.488")
    print(f"{'='*70}")
    print(f"  Cosine 1.488 avg best FOM: {cos_avg:.6f}")
    print(f"  LLM-masked avg best FOM:   {llm_avg:.6f}")
    print(f"  Cosine wins: {wins_cos}")
    print(f"  LLM wins:    {wins_llm}")
    print(f"  Ties:        {ties}")
    winner = 'LLM-MASKED' if llm_avg > cos_avg else 'COSINE 1.488'
    print(f"  WINNER: {winner} (by {abs(llm_avg - cos_avg):.6f})")
    print(f"  Saved: {path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--scalarization', type=str, default='cosine', choices=['cosine', 'nw'])
    parser.add_argument('--evaluate-only', action='store_true')
    parser.add_argument('--model-path', type=str, default='')
    parser.add_argument('--ttt', action='store_true', help='Enable time-to-target early stopping during training')
    parser.add_argument('--ttt-eval-subset', type=int, default=50, help='Number of specs for periodic TTT evaluation')
    parser.add_argument('--ttt-eval-every-specs', type=int, default=5, help='Run a TTT evaluation every N training specs')
    parser.add_argument('--ttt-target-avg-best-fom', type=float, default=None)
    parser.add_argument('--ttt-target-top20-fom', type=float, default=None)
    args = parser.parse_args()

    set_seed(args.seed)

    start = datetime.now()
    print(f"\n{'#'*70}")
    print(f"  TRAIN LLM-MASKED DDQN — SAME ENV AS COSINE 1.488")
    print(f"  Start: {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*70}")

    with open(DATASET_PATH, 'rb') as f:
        specs = pickle.load(f)
    num_specs = len(list(specs.values())[0])
    print(f"\nDataset: {num_specs} specs")

    env = AutoCktMOEnv(generalize=True, num_valid=num_specs, run_valid=True)
    num_params = len(env.action_space.spaces) if isinstance(env.action_space, gym.spaces.Tuple) else 1

    try:
        preferences = generate_preference_vectors(4, method='focused', num_vectors=TRAIN_CONFIG['num_preferences'])
        if len(preferences) > TRAIN_CONFIG['num_preferences']:
            preferences = preferences[:TRAIN_CONFIG['num_preferences']]
    except Exception:
        preferences = generate_preference_vectors(4, method='random', num_vectors=TRAIN_CONFIG['num_preferences'])
    print(f"Preferences: {len(preferences)} vectors")

    # Train or load
    time_to_target = {
        'enabled': bool(args.ttt) and (not args.evaluate_only),
        'eval_subset': int(args.ttt_eval_subset),
        'eval_every_specs': int(args.ttt_eval_every_specs),
        'target_avg_best_fom': args.ttt_target_avg_best_fom,
        'target_top20_fom': args.ttt_target_top20_fom,
    }

    ttt = None
    if args.evaluate_only:
        agent, llm_filter = build_llm_agent(env, num_params, args.scalarization)
        default_model = MODELS_DIR / "llm_cosine_final.pth"
        model_path = Path(args.model_path) if args.model_path else default_model
        if not model_path.exists():
            raise FileNotFoundError(
                f"Evaluate-only requested but model not found: {model_path}. "
                f"Either run training once or pass --model-path to an existing .pth file."
            )
        load_agent_weights(agent, model_path)
        print(f"\n  Loaded model: {model_path}")
    else:
        agent, llm_filter, ttt = train_llm_agent(env, specs, preferences, num_params, args.scalarization, time_to_target=time_to_target)

    # Evaluate
    eval_t0 = time.perf_counter()
    llm_results, mask_time_s, steps_stats = evaluate_llm_agent(
        agent,
        llm_filter,
        env,
        specs,
        preferences,
        args.scalarization,
        track_mask_time=True,
    )
    eval_s = time.perf_counter() - eval_t0

    print("\nLLM masking overhead:")
    print(f"  Total eval time (s): {eval_s:.2f}")
    print(f"  update_llm_mask time (s): {mask_time_s:.2f}")
    if eval_s > 0:
        print(f"  Mask share: {100.0 * mask_time_s / eval_s:.1f}%")

    # Persist runtime info
    try:
        runtime_path = WITH_15_WITH_20_RESULTS_DIR / "runtime_summary.json"
        entry = {
            'script': 'train_llm_masked.py',
            'timestamp': datetime.now().isoformat(timespec='seconds'),
            'seed': int(args.seed),
            'scalarization': str(args.scalarization),
            'evaluate_only': bool(args.evaluate_only),
            'model_path': str(model_path) if args.evaluate_only else '',
            'seconds': {
                'eval_total': round(float(eval_s), 6),
                'mask_update_llm_mask': round(float(mask_time_s), 6),
            },
            'minutes': {
                'eval_total': round(float(eval_s) / 60.0, 6),
                'mask_update_llm_mask': round(float(mask_time_s) / 60.0, 6),
            },
            'mask_share_percent': round((100.0 * float(mask_time_s) / float(eval_s)) if eval_s > 0 else 0.0, 3),
            'steps_to_done': steps_stats,
            'time_to_target': time_to_target if time_to_target.get('enabled') else None,
            'ttt_results': ttt,
        }
        if runtime_path.exists():
            with open(runtime_path, 'r') as f:
                data = json.load(f)
        else:
            data = []
        if not isinstance(data, list):
            data = [data]
        data.append(entry)
        WITH_15_WITH_20_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(runtime_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved runtime: {runtime_path}")
    except Exception as e:
        print(f"WARNING: Could not save runtime_summary.json: {e}")

    # Save LLM CSV
    df = pd.DataFrame(llm_results)
    n = len(df); ns = df['spec'].nunique()
    avg_fom = df['fom'].astype(float).mean()
    best = df.groupby('spec')['fom'].max()
    avg_best = best.mean()
    sums = pd.DataFrame([
        {'spec': f'summary_avg_fom_{n}_solutions', 'fom': round(avg_fom, 6)},
        {'spec': f'summary_avg_best_fom_{ns}_specs', 'fom': round(avg_best, 6)},
    ])
    df_out = pd.concat([df, sums], ignore_index=True)
    csv_name = "morl_autockt_results_llm_masked.csv" if args.scalarization == 'cosine' else "morl_autockt_results_llm_masked_nw.csv"
    csv_path = RESULTS_DIR / csv_name
    df_out.to_csv(csv_path, index=False)
    print(f"\n  LLM CSV: {csv_path}")
    print(f"  LLM: {n} solutions, avg FOM={avg_fom:.6f}, avg best={avg_best:.6f}")
    print(f"  Gain mean: {df['output_gain_linear'].mean():.1f}")

    try:
        WITH_15_WITH_20_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        dst = WITH_15_WITH_20_RESULTS_DIR / ("morl_autockt_results_llm_masked.csv" if args.scalarization == 'cosine' else "morl_autockt_results_llm_masked_nw.csv")
        shutil.copyfile(csv_path, dst)
        print(f"  Copied to: {dst}")
    except Exception as e:
        print(f"  WARNING: Could not copy CSV into with_15%_with_20 results folder: {e}")

    # Compare against cosine 1.488
    if args.scalarization == 'cosine':
        compare_with_cosine_1488(llm_results)

    elapsed = datetime.now() - start
    print(f"\nTotal time: {elapsed}")
    print("DONE")
