"""
Live NGSpice measurement: trained MORL DDQN agent.
Single-preference (Original AutoCkt-like) vs Multi-preference (MORL).
"""
import sys, os, time, json
import numpy as np
import torch
from pathlib import Path

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "methodology"))
sys.path.insert(0, str(BASE_DIR / "autockt"))

from methodology.autockt.envs.autockt_mo_env import AutoCktMOEnv
from autockt.agents.mo_agent import MO_DQN_Agent
from autockt.utils.mo_utils import generate_preference_vectors

RESULTS_DIR = BASE_DIR / "results"
NUM_SPECS = 10
MAX_STEPS = 60   # same as Original AutoCkt eval traj_len

print("=" * 70)
print("  LIVE NGSPICE: TRAINED MORL DDQN AGENT")
print(f"  {NUM_SPECS} specs, max {MAX_STEPS} steps, NGSpice on every step")
print("=" * 70)

# ── Environment ──
env = AutoCktMOEnv()
env.max_episode_steps = MAX_STEPS
print(f"  NGSpice: {env.base_env.sim_env.ngspice_cmd}")

# Time single step
env.base_env.obj_idx = 0
t0 = time.time(); env.reset(); t_reset = time.time() - t0
t0 = time.time(); env.step(1); t_step = time.time() - t0
print(f"  Per-step NGSpice time: {t_step:.3f}s")

# ── Load trained model ──
model_path = RESULTS_DIR / "models_llm_ddqn" / "checkpoint_llm_cosine_ep2300.pth"
ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
agent = MO_DQN_Agent(
    state_dim=ckpt['state_dim'],   # 15
    action_dim=ckpt['action_dim'], # 3
    reward_dim=ckpt['reward_dim'], # 4
    scalarization=ckpt.get('scalarization', 'cosine'))
agent.q_network.load_state_dict(ckpt['agent_state_dict'])
agent.q_network.eval()
agent.epsilon = 0.0  # greedy
print(f"  Model: {model_path.name} (ep {ckpt.get('episode_count')})")

# ── Preferences ──
prefs = generate_preference_vectors(4, method='focused', num_vectors=10)[:10]
equal_pref = np.array([0.25, 0.25, 0.25, 0.25])

def run_episode(env, spec_idx, pref, max_steps):
    """Run 1 episode with live NGSpice, return (steps, reached, wall_time, final_reward)."""
    env.base_env.obj_idx = spec_idx
    agent.set_preference(pref)
    t0 = time.time()
    state = env.reset()
    
    for step in range(1, max_steps + 1):
        action = agent.select_action(state, pref, epsilon=0.0)
        res = env.step(action)
        state = res[0]
        # Check base env done (reward >= 10)
        base_reward = env.base_env.reward(env.base_env.cur_specs, env.base_env.specs_ideal)
        done = (base_reward >= 10)
        if done:
            return step, True, time.time() - t0, base_reward
        # Also check MO env done
        if res[2]:
            return step, False, time.time() - t0, base_reward
    
    return max_steps, False, time.time() - t0, base_reward


# =====================================================================
# 1. SINGLE-PREFERENCE (Original AutoCkt-like): 1 attempt, equal weights
# =====================================================================
print(f"\n{'='*70}")
print(f"  1. SINGLE-PREFERENCE (Original AutoCkt-like): 1 attempt per spec")
print(f"{'='*70}")

single_results = []
for idx in range(NUM_SPECS):
    steps, reached, wt, rew = run_episode(env, idx, equal_pref, MAX_STEPS)
    tag = "DONE" if reached else f"FAIL(r={rew:.2f})"
    single_results.append({'spec': idx, 'steps': steps, 'reached': reached, 'time': wt, 'reward': rew})
    print(f"  Spec {idx:>2}: {steps:>3} steps, {wt:.2f}s [{tag}]")

# =====================================================================
# 2. MULTI-PREFERENCE (MORL): 10 attempts, best of 10
# =====================================================================
print(f"\n{'='*70}")
print(f"  2. MULTI-PREFERENCE (MORL): best of 10 attempts per spec")
print(f"{'='*70}")

multi_results = []
for idx in range(NUM_SPECS):
    best_steps = MAX_STEPS
    best_time = 0
    best_rew = -999
    found = False
    for pi, pref in enumerate(prefs):
        steps, reached, wt, rew = run_episode(env, idx, pref, MAX_STEPS)
        if reached and steps < best_steps:
            best_steps = steps
            best_time = wt
            found = True
        if rew > best_rew:
            best_rew = rew
    tag = "DONE" if found else f"FAIL(r={best_rew:.2f})"
    multi_results.append({'spec': idx, 'steps': best_steps, 'reached': found, 'time': best_time, 'reward': best_rew})
    print(f"  Spec {idx:>2}: {best_steps:>3} steps (best of 10), {best_time:.2f}s [{tag}]")

# =====================================================================
# SUMMARY
# =====================================================================
print(f"\n{'='*70}")
print("  LIVE NGSPICE RESULTS")
print(f"{'='*70}")
print(f"\n  NGSpice per step: {t_step:.3f}s")

sr = [r for r in single_results if r['reached']]
mr = [r for r in multi_results if r['reached']]

print(f"\n  {'Approach':<40} {'Reached':>8} {'Avg Steps':>10} {'Avg Time':>10}")
print(f"  {'-'*68}")

s_avg = np.mean([r['steps'] for r in sr]) if sr else float('nan')
m_avg = np.mean([r['steps'] for r in mr]) if mr else float('nan')
s_time = np.mean([r['time'] for r in sr]) if sr else float('nan')
m_time = np.mean([r['time'] for r in mr]) if mr else float('nan')

print(f"  {'Single-pref (Orig AutoCkt-like)':<40} {len(sr):>5}/{NUM_SPECS} {s_avg:>10.1f} {s_time:>9.2f}s")
print(f"  {'Multi-pref (MORL / Beyond AutoCkt)':<40} {len(mr):>5}/{NUM_SPECS} {m_avg:>10.1f} {m_time:>9.2f}s")

# Show rewards for failed specs
print(f"\n  Reward distribution (all specs):")
s_rews = [f"{r['reward']:.2f}" for r in single_results]
m_rews = [f"{r['reward']:.2f}" for r in multi_results]
print(f"    Single-pref: {s_rews}")
print(f"    Multi-pref:  {m_rews}")

if sr and mr:
    imp = (s_avg - m_avg) / s_avg * 100
    print(f"\n  MEASURED:  {s_avg:.1f} → {m_avg:.1f} steps ({imp:.1f}% fewer)")
    print(f"  CLAIM:     27 → 12 steps (55.6% fewer)")

# Save
output = {
    'ngspice_step_time': t_step,
    'single_pref': {'reached': len(sr), 'avg_steps': float(s_avg), 'results': single_results},
    'multi_pref': {'reached': len(mr), 'avg_steps': float(m_avg), 'results': multi_results},
}
with open(RESULTS_DIR / "live_ngspice_verification.json", 'w') as f:
    json.dump(output, f, indent=2, default=str)
print(f"\n  Saved: results/live_ngspice_verification.json")
