"""
Live NGSpice measurement: actual steps + wall-clock time per solution.
MORL DDQN agent vs Random baseline on 5 specs.
"""
import sys, pickle, random, time, json
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
NUM_SPECS = 5
MAX_STEPS = 60  # same as original AutoCkt eval
TOL = 0.15

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

print("=" * 60)
print("  LIVE NGSPICE TIMING MEASUREMENT")
print(f"  {NUM_SPECS} specs, max {MAX_STEPS} steps each")
print("=" * 60)

# Create environment
env = AutoCktMOEnv()
print("  Environment created (NGSpice ready)")

# Time a single NGSpice simulation
print("\n  Timing single NGSpice simulation...")
t0 = time.time()
env.base_env.obj_idx = 0
env.reset()
single_sim_time = time.time() - t0
print(f"  Single sim (reset): {single_sim_time:.3f}s")

t0 = time.time()
env.step(1)  # KEEP action
single_step_time = time.time() - t0
print(f"  Single sim (step):  {single_step_time:.3f}s")

# Load MORL model (use LLM checkpoint as standard DDQN — no LLM mask applied)
model_path = RESULTS_DIR / "models_llm_ddqn" / "checkpoint_llm_cosine_ep2300.pth"
ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
agent = MO_DQN_Agent(
    ckpt.get('state_dim', 64), ckpt.get('action_dim', 64),
    ckpt.get('reward_dim', 4), scalarization='cosine')
agent.q_network.load_state_dict(ckpt['agent_state_dict'])
agent.q_network.eval()
prefs = generate_preference_vectors(4, method='focused', num_vectors=10)[:10]
print(f"  MORL DDQN model loaded")

# ── Measure MORL DDQN ──
print(f"\n{'='*60}")
print(f"  MORL DDQN (Beyond AutoCkt) — {NUM_SPECS} specs × 10 prefs")
print(f"{'='*60}")
morl_results = []
for idx in range(NUM_SPECS):
    env.base_env.obj_idx = idx
    best_steps = MAX_STEPS
    best_time = 0
    found = False
    for pref in prefs:
        agent.set_preference(pref)
        t_start = time.time()
        state = env.reset()
        for step in range(1, MAX_STEPS + 1):
            action = agent.select_action(state, None)
            res = env.step(action)
            state = res[0]
            done = res[2]
            if check_reached(env):
                elapsed = time.time() - t_start
                if step < best_steps:
                    best_steps = step
                    best_time = elapsed
                found = True
                break
            if done:
                break
        if not found:
            best_time = time.time() - t_start
    status = "REACHED" if found else "FAILED"
    morl_results.append({'spec': idx, 'steps': best_steps, 'time': best_time, 'reached': found})
    print(f"  Spec {idx}: {best_steps:>3} steps, {best_time:.2f}s  [{status}]")

# ── Measure Random baseline (proxy for Original AutoCkt difficulty) ──
print(f"\n{'='*60}")
print(f"  RANDOM BASELINE — {NUM_SPECS} specs × 1 attempt")
print(f"{'='*60}")
rand_results = []
for idx in range(NUM_SPECS):
    env.base_env.obj_idx = idx
    t_start = time.time()
    state = env.reset()
    found = False
    steps = 0
    for step in range(1, MAX_STEPS + 1):
        action = random.randrange(3)
        res = env.step(action)
        state = res[0]
        done = res[2]
        steps = step
        if check_reached(env):
            found = True
            break
        if done:
            break
    elapsed = time.time() - t_start
    status = "REACHED" if found else "FAILED"
    rand_results.append({'spec': idx, 'steps': steps, 'time': elapsed, 'reached': found})
    print(f"  Spec {idx}: {steps:>3} steps, {elapsed:.2f}s  [{status}]")

# ── Summary ──
print(f"\n{'='*60}")
print("  RESULTS SUMMARY")
print(f"{'='*60}")

morl_reached = [r for r in morl_results if r['reached']]
rand_reached = [r for r in rand_results if r['reached']]

print(f"\n  Single NGSpice sim time: {single_step_time:.3f}s")

print(f"\n  {'Agent':<25} {'Reached':>8} {'Avg Steps':>10} {'Avg Time':>10} {'Time/Step':>10}")
print(f"  {'-'*63}")

if morl_reached:
    m_steps = np.mean([r['steps'] for r in morl_reached])
    m_time = np.mean([r['time'] for r in morl_reached])
    print(f"  {'MORL DDQN':<25} {len(morl_reached):>5}/{NUM_SPECS} {m_steps:>10.1f} {m_time:>9.2f}s {m_time/m_steps:>9.3f}s")
else:
    print(f"  {'MORL DDQN':<25} {0:>5}/{NUM_SPECS}        -          -          -")

if rand_reached:
    r_steps = np.mean([r['steps'] for r in rand_reached])
    r_time = np.mean([r['time'] for r in rand_reached])
    print(f"  {'Random baseline':<25} {len(rand_reached):>5}/{NUM_SPECS} {r_steps:>10.1f} {r_time:>9.2f}s {r_time/r_steps:>9.3f}s")
else:
    print(f"  {'Random baseline':<25} {0:>5}/{NUM_SPECS}        -          -          -")

# Original AutoCkt estimate
orig_steps = 27
orig_time = orig_steps * single_step_time
print(f"  {'Orig AutoCkt (est.)':<25} {'':>8} {orig_steps:>10} {orig_time:>9.2f}s {single_step_time:>9.3f}s")

if morl_reached:
    print(f"\n  MORL DDQN avg: {m_steps:.1f} steps × {single_step_time:.3f}s = {m_steps*single_step_time:.2f}s per solution")
    print(f"  Orig AutoCkt:  {orig_steps} steps × {single_step_time:.3f}s = {orig_time:.2f}s per solution")
    time_saved = orig_time - m_steps * single_step_time
    print(f"  Time saved:    {time_saved:.2f}s per solution ({time_saved/orig_time*100:.1f}%)")

# Save
output = {
    'single_ngspice_sim_seconds': single_step_time,
    'morl_results': morl_results,
    'random_results': rand_results,
    'morl_avg_steps': float(np.mean([r['steps'] for r in morl_reached])) if morl_reached else None,
    'morl_avg_time': float(np.mean([r['time'] for r in morl_reached])) if morl_reached else None,
    'original_autockt_estimated_steps': orig_steps,
    'original_autockt_estimated_time': orig_time,
}
with open(RESULTS_DIR / "ngspice_timing.json", 'w') as f:
    json.dump(output, f, indent=2, default=str)
print(f"\n  Saved: results/ngspice_timing.json")
