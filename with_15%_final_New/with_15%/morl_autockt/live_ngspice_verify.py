"""
LIVE NGSpice verification of sample efficiency claim.
Trains a small MORL DDQN with live NGSpice and measures actual steps + time.

Comparison:
  - Single-preference (like Original AutoCkt): 1 attempt per spec
  - Multi-preference (MORL / Beyond AutoCkt): best of N preferences per spec
"""
import sys, pickle, time, json, random
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "methodology"))
sys.path.insert(0, str(BASE_DIR / "autockt"))

from methodology.autockt.envs.autockt_mo_env import AutoCktMOEnv
from autockt.agents.mo_agent import MO_DQN_Agent
from autockt.utils.mo_utils import generate_preference_vectors

RESULTS_DIR = BASE_DIR / "results"
TOL = 0.15

# ── Small-scale config (feasible with live NGSpice ~0.2s/step) ──
TRAIN_SPECS = 3
TRAIN_PREFS = 3
TRAIN_EPISODES = 30        # per spec-pref combo
TRAIN_MAX_STEPS = 30
EVAL_SPECS = 5
EVAL_MAX_STEPS = 60        # same as original AutoCkt traj_len
EVAL_PREFS = 10            # multi-preference

def check_reached(env):
    if not hasattr(env.base_env, 'cur_specs'):
        return False
    actual = env.base_env.cur_specs
    target = env.base_env.specs_ideal
    sid = env.base_env.specs_id
    a = dict(zip(sid, actual))
    t = dict(zip(sid, target))
    ga = a.get('gain_min', 0); gt = t.get('gain_min', 0)
    ua = a.get('ugbw_min', 0); ut = t.get('ugbw_min', 0)
    pa = a.get('phm_min', 0);  pt = t.get('phm_min', 0)
    ia = a.get('ibias_max', 0); it = t.get('ibias_max', 0)
    if ga < 100: ga = 10**(ga/20)
    if gt < 100: gt = 10**(gt/20)
    return (ga >= gt*(1-TOL) and ua >= ut*(1-TOL) and pa >= pt*(1-TOL) and ia <= it*(1+TOL))

def get_reward_vec(env):
    """4D reward: how close to each target (higher=better, 0=at target)."""
    if not hasattr(env.base_env, 'cur_specs'):
        return np.zeros(4)
    actual = env.base_env.cur_specs
    target = env.base_env.specs_ideal
    sid = env.base_env.specs_id
    a = dict(zip(sid, actual))
    t = dict(zip(sid, target))
    ga = a.get('gain_min', 0); gt = t.get('gain_min', 0)
    ua = a.get('ugbw_min', 0); ut = t.get('ugbw_min', 0)
    pa = a.get('phm_min', 0);  pt = t.get('phm_min', 0)
    ia = a.get('ibias_max', 1); it = t.get('ibias_max', 1)
    if ga < 100: ga = 10**(ga/20)
    if gt < 100: gt = 10**(gt/20)
    r_gain = min(ga / max(gt, 1e-9), 2.0) - 1.0
    r_ugbw = min(ua / max(ut, 1e-9), 2.0) - 1.0
    r_pm   = min(pa / max(pt, 1e-9), 2.0) - 1.0
    r_ibias = 1.0 - min(ia / max(it, 1e-9), 2.0)
    return np.array([r_gain, r_ugbw, r_pm, r_ibias], dtype=np.float32)

# =====================================================================
print("=" * 60)
print("  LIVE NGSPICE TRAINING + EVALUATION")
print("=" * 60)

env = AutoCktMOEnv()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n if hasattr(env.action_space, 'n') else 3
reward_dim = 4
print(f"  Env ready. state_dim={state_dim}, action_dim={action_dim}")

# Time a single step
env.base_env.obj_idx = 0
t0 = time.time(); env.reset(); t_reset = time.time() - t0
t0 = time.time(); env.step(1); t_step = time.time() - t0
print(f"  NGSpice per step: {t_step:.3f}s")

# Generate preferences
all_prefs = generate_preference_vectors(4, method='focused', num_vectors=EVAL_PREFS)[:EVAL_PREFS]
train_prefs = all_prefs[:TRAIN_PREFS]

# Create agent
agent = MO_DQN_Agent(state_dim, action_dim, reward_dim, scalarization='cosine')

# =====================================================================
# TRAINING with live NGSpice
# =====================================================================
total_train_steps = TRAIN_SPECS * TRAIN_PREFS * TRAIN_EPISODES * TRAIN_MAX_STEPS
est_time = total_train_steps * t_step
print(f"\n  Training: {TRAIN_SPECS} specs × {TRAIN_PREFS} prefs × {TRAIN_EPISODES} eps × {TRAIN_MAX_STEPS} steps")
print(f"  Max {total_train_steps} NGSpice calls (~{est_time/60:.1f} min)")
print(f"  Training...")

t_train_start = time.time()
ep_count = 0
total_steps_actual = 0
for spec_idx in range(TRAIN_SPECS):
    env.base_env.obj_idx = spec_idx
    for pref in train_prefs:
        agent.set_preference(pref)
        for ep in range(TRAIN_EPISODES):
            state = env.reset()
            ep_reward = 0
            for step in range(TRAIN_MAX_STEPS):
                action = agent.select_action(state, pref)
                res = env.step(action)
                next_state = res[0]
                done = res[2] if len(res) >= 3 else False
                if len(res) >= 5:
                    done = done or res[3]  # truncated
                reward_vec = get_reward_vec(env)
                agent.store_experience(state, action, reward_vec.tolist(), next_state, done, pref.tolist() if hasattr(pref, 'tolist') else list(pref))
                agent.update(batch_size=32)
                state = next_state
                total_steps_actual += 1
                if done:
                    break
            ep_count += 1
            if ep_count % 30 == 0:
                elapsed = time.time() - t_train_start
                print(f"    Episode {ep_count}/{TRAIN_SPECS*TRAIN_PREFS*TRAIN_EPISODES} ({elapsed:.0f}s)")

t_train = time.time() - t_train_start
print(f"  Training done: {ep_count} episodes, {total_steps_actual} steps, {t_train:.1f}s")
agent.epsilon = 0.0  # greedy for eval

# =====================================================================
# EVALUATION — Multi-preference (MORL / Beyond AutoCkt)
# =====================================================================
print(f"\n{'='*60}")
print(f"  EVAL: Multi-preference (MORL) — {EVAL_SPECS} specs × {EVAL_PREFS} prefs")
print(f"{'='*60}")

morl_data = []
t_eval_start = time.time()
for spec_idx in range(EVAL_SPECS):
    env.base_env.obj_idx = spec_idx
    best_steps = EVAL_MAX_STEPS
    best_time = 0
    found = False
    for pref in all_prefs:
        agent.set_preference(pref)
        t0 = time.time()
        state = env.reset()
        for step in range(1, EVAL_MAX_STEPS + 1):
            action = agent.select_action(state, pref, epsilon=0.0)
            res = env.step(action)
            state = res[0]; done = res[2]
            if check_reached(env):
                el = time.time() - t0
                if step < best_steps:
                    best_steps = step
                    best_time = el
                found = True
                break
            if done: break
        if not found:
            best_time = time.time() - t0
    status = "REACHED" if found else "FAILED"
    morl_data.append({'spec': spec_idx, 'steps': best_steps, 'time': best_time, 'reached': found})
    print(f"  Spec {spec_idx}: {best_steps:>3} steps, {best_time:.2f}s [{status}]")
t_morl_eval = time.time() - t_eval_start

# =====================================================================
# EVALUATION — Single-preference (like Original AutoCkt: 1 attempt)
# =====================================================================
print(f"\n{'='*60}")
print(f"  EVAL: Single-preference (Original AutoCkt-like) — {EVAL_SPECS} specs × 1 attempt")
print(f"{'='*60}")

single_pref = np.array([0.25, 0.25, 0.25, 0.25])  # equal weights (single-objective)
single_data = []
t_single_start = time.time()
for spec_idx in range(EVAL_SPECS):
    env.base_env.obj_idx = spec_idx
    agent.set_preference(single_pref)
    t0 = time.time()
    state = env.reset()
    found = False
    steps_taken = EVAL_MAX_STEPS
    for step in range(1, EVAL_MAX_STEPS + 1):
        action = agent.select_action(state, single_pref, epsilon=0.0)
        res = env.step(action)
        state = res[0]; done = res[2]
        if check_reached(env):
            steps_taken = step
            found = True
            break
        if done:
            steps_taken = step
            break
    elapsed = time.time() - t0
    status = "REACHED" if found else "FAILED"
    single_data.append({'spec': spec_idx, 'steps': steps_taken, 'time': elapsed, 'reached': found})
    print(f"  Spec {spec_idx}: {steps_taken:>3} steps, {elapsed:.2f}s [{status}]")
t_single_eval = time.time() - t_single_start

# =====================================================================
# SUMMARY
# =====================================================================
print(f"\n{'='*60}")
print("  RESULTS — LIVE NGSPICE MEASUREMENT")
print(f"{'='*60}")

print(f"\n  NGSpice time per step: {t_step:.3f}s")
print(f"  Training time: {t_train:.1f}s ({total_steps_actual} steps)")

morl_reached = [d for d in morl_data if d['reached']]
single_reached = [d for d in single_data if d['reached']]

print(f"\n  {'Approach':<35} {'Reached':>8} {'Avg Steps':>10} {'Avg Time':>10}")
print(f"  {'-'*63}")

if morl_reached:
    ms = np.mean([d['steps'] for d in morl_reached])
    mt = np.mean([d['time'] for d in morl_reached])
    print(f"  {'Multi-pref (MORL/Beyond AutoCkt)':<35} {len(morl_reached):>5}/{EVAL_SPECS} {ms:>10.1f} {mt:>9.2f}s")
else:
    ms = float('nan')
    print(f"  {'Multi-pref (MORL/Beyond AutoCkt)':<35} {0:>5}/{EVAL_SPECS}        N/A       N/A")

if single_reached:
    ss = np.mean([d['steps'] for d in single_reached])
    st = np.mean([d['time'] for d in single_reached])
    print(f"  {'Single-pref (Original AutoCkt-like)':<35} {len(single_reached):>5}/{EVAL_SPECS} {ss:>10.1f} {st:>9.2f}s")
else:
    ss = float('nan')
    print(f"  {'Single-pref (Original AutoCkt-like)':<35} {0:>5}/{EVAL_SPECS}        N/A       N/A")

# All data (including failed)
all_morl_steps = [d['steps'] for d in morl_data]
all_single_steps = [d['steps'] for d in single_data]
all_morl_time = [d['time'] for d in morl_data]
all_single_time = [d['time'] for d in single_data]

print(f"\n  Including failed specs (max {EVAL_MAX_STEPS} steps):")
print(f"  {'Multi-pref (all specs)':<35} avg={np.mean(all_morl_steps):.1f} steps, {np.mean(all_morl_time):.2f}s")
print(f"  {'Single-pref (all specs)':<35} avg={np.mean(all_single_steps):.1f} steps, {np.mean(all_single_time):.2f}s")

if not np.isnan(ms) and not np.isnan(ss):
    improvement = (ss - ms) / ss * 100
    print(f"\n  IMPROVEMENT: {ss:.1f} → {ms:.1f} steps ({improvement:.1f}% fewer steps)")
    print(f"  CLAIM:       27 → 12 steps (55.6% fewer steps)")

# Save
result = {
    'ngspice_step_time_seconds': t_step,
    'training': {'specs': TRAIN_SPECS, 'prefs': TRAIN_PREFS, 'episodes': TRAIN_EPISODES,
                 'total_steps': total_steps_actual, 'time_seconds': t_train},
    'multi_pref_morl': {'specs': EVAL_SPECS, 'prefs': EVAL_PREFS,
                         'reached': len(morl_reached), 'data': morl_data,
                         'avg_steps_reached': float(ms) if morl_reached else None,
                         'avg_steps_all': float(np.mean(all_morl_steps))},
    'single_pref_original': {'specs': EVAL_SPECS, 'prefs': 1,
                              'reached': len(single_reached), 'data': single_data,
                              'avg_steps_reached': float(ss) if single_reached else None,
                              'avg_steps_all': float(np.mean(all_single_steps))},
}
with open(RESULTS_DIR / "live_ngspice_verification.json", 'w') as f:
    json.dump(result, f, indent=2, default=str)
print(f"\n  Saved: results/live_ngspice_verification.json")
