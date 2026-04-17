"""
Train NW agent in the EXACT same environment as cosine 1.488.
Same code, surrogate, rewards, dataset, config. Only scalarization = NW.
Evaluate on 1000 specs × 10 prefs = 10000 solutions with STRICT pass checks.
"""
import os, sys, pickle, numpy as np, json, torch, gym, pandas as pd
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).parent
METHODOLOGY_DIR = BASE_DIR / "methodology"
sys.path.insert(0, str(METHODOLOGY_DIR))
os.environ['AUTOCKT_USE_SURROGATE'] = 'true'

from autockt.envs.autockt_mo_env import AutoCktMOEnv
from autockt.agents.mo_agent import MO_DQN_Agent
from autockt.utils.mo_utils import generate_preference_vectors

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

DATASET_PATH = BASE_DIR / "dataset" / "ngspice_specs_gen_two_stage_opamp"
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_DIR = Path(r"C:\Users\kobeo\OneDrive\Desktop\trail\with_15%_with_20\with_15%\morl_autockt\results")


def _fom(g, u, p, i, gt, ut, pt, it):
    def s(x): return max(float(x), 1e-9)
    gt, ut, pt, it = s(gt), s(ut), s(pt), s(it)
    return (float(g)-gt)/gt + (float(u)-ut)/ut + (float(p)-pt)/pt - (float(i)-it)/it


def train_nw(env, specs, preferences, num_params):
    print(f"\n{'='*70}")
    print(f"  TRAINING NW AGENT")
    print(f"{'='*70}\n")

    state_dim = env.observation_space.shape[0] if hasattr(env.observation_space, 'shape') else 20
    if isinstance(env.action_space, gym.spaces.Tuple):
        action_dim = env.action_space.spaces[0].n
    else:
        action_dim = env.action_space.n if hasattr(env.action_space, 'n') else 7
    reward_dim = 4

    agent = MO_DQN_Agent(state_dim, action_dim, reward_dim, scalarization='nw')
    print(f"  [NW] state={state_dim}, action={action_dim}, reward={reward_dim}")

    num_specs_total = len(list(specs.values())[0])
    training_specs = list(range(min(TRAIN_CONFIG['num_training_specs'], num_specs_total)))
    episode_count = 0

    models_dir = RESULTS_DIR / "models_nw"
    models_dir.mkdir(parents=True, exist_ok=True)

    for spec_idx, spec_num in enumerate(training_specs):
        if spec_idx % 10 == 0:
            print(f"  [NW] Spec {spec_idx+1}/{len(training_specs)}...")
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
                for step in range(TRAIN_CONFIG['max_steps']):
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
                    if done:
                        break
                episode_count += 1

        if (spec_idx + 1) % TRAIN_CONFIG['save_frequency'] == 0:
            cp_path = models_dir / f"model_nw_checkpoint_ep{episode_count}.pth"
            torch.save({
                'agent_state_dict': agent.q_network.state_dict(),
                'preference_net_state_dict': agent.preference_net.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'state_dim': state_dim, 'action_dim': action_dim, 'reward_dim': reward_dim,
                'scalarization': 'nw', 'episode_count': episode_count,
            }, cp_path)

    final_path = models_dir / "model_nw_final.pth"
    torch.save({
        'agent_state_dict': agent.q_network.state_dict(),
        'preference_net_state_dict': agent.preference_net.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'state_dim': state_dim, 'action_dim': action_dim, 'reward_dim': reward_dim,
        'scalarization': 'nw', 'episode_count': episode_count,
    }, final_path)
    print(f"  [NW] Done — {episode_count} episodes. Model: {final_path}")
    return agent


def evaluate_nw(agent, env, specs, preferences, num_params):
    print(f"\n{'='*70}")
    print(f"  EVALUATING NW AGENT (1000 specs × 10 prefs)")
    print(f"  epsilon = max(0.05, 0.15*(1-step/120))")
    print(f"{'='*70}\n")

    agent.q_network.eval()
    num_specs_total = len(list(specs.values())[0])
    num_targets = min(EVAL_CONFIG['num_targets'], num_specs_total)
    max_steps = EVAL_CONFIG['max_steps']

    target_specs_json = {}
    for i in range(num_specs_total):
        target_specs_json[str(i)] = {
            'target_gain_linear': float(specs['gain_min'][i]),
            'target_ugbw_mhz': float(specs['ugbw_min'][i]) / 1e6,
            'target_pm_deg': 75.0,
            'target_ibias_ma': float(specs['ibias_max'][i]) * 1000.0,
        }

    all_solutions = []

    for target_idx in range(num_targets):
        if target_idx % 100 == 0 and target_idx > 0:
            print(f"  [NW] Progress: {target_idx}/{num_targets}")

        try:
            env.base_env.obj_idx = target_idx
            env.reset()
            env_specs_id = env.base_env.specs_id
        except Exception:
            continue

        for pref_idx, preference in enumerate(preferences):
            try:
                agent.set_preference(preference)
                state = env.reset()
                done = False

                for step in range(max_steps):
                    # SAME epsilon schedule as cosine evaluation
                    epsilon = max(0.05, 0.15 * (1 - step / max_steps))
                    action = agent.select_action(state, preference, epsilon=epsilon)

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

                    # STRICT pass checks
                    gp = 'Yes' if go >= gt else 'No'
                    up = 'Yes' if uo >= ut else 'No'
                    pp = 'Yes' if po >= pt else 'No'
                    ip = 'Yes' if io <= it else 'No'
                    cp = 'Yes' if gp=='Yes' and up=='Yes' and pp=='Yes' and ip=='Yes' else 'No'

                    all_solutions.append({
                        'spec': target_idx + 1,
                        'solution': pref_idx + 1,
                        'target_gain_linear': gt, 'target_ugbw_mhz': ut,
                        'target_pm_deg': pt, 'target_ibias_ma': it,
                        'output_gain_linear': round(go, 6), 'output_gain_db': round(float(gd), 6),
                        'output_ugbw_mhz': round(uo, 6), 'output_pm_deg': round(po, 4),
                        'output_ibias_ma': round(io, 6),
                        'fom': round(f, 6),
                        'gain_pass': gp, 'ugbw_pass': up, 'pm_pass': pp,
                        'ibias_pass': ip, 'complete_pass': cp,
                        'scalarization': 'nw',
                    })
            except Exception:
                continue

    print(f"  [NW] Done: {len(all_solutions)} solutions")
    return all_solutions


if __name__ == "__main__":
    start = datetime.now()
    print(f"\n{'#'*70}")
    print(f"  TRAIN NW AGENT — SAME ENV AS COSINE 1.488")
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

    # Train NW
    nw_agent = train_nw(env, specs, preferences, num_params)

    # Evaluate NW with strict pass checks
    nw_results = evaluate_nw(nw_agent, env, specs, preferences, num_params)

    # Save NW CSV
    df = pd.DataFrame(nw_results)
    n = len(df)
    cp_count = (df['complete_pass'] == 'Yes').sum()
    avg_fom = df['fom'].astype(float).mean()
    avg_best = df.groupby('spec')['fom'].max().mean()
    top20 = df.groupby('spec')['fom'].max().nlargest(20).mean()

    sums = pd.DataFrame([
        {'spec': 'summary', 'fom': f'avg_fom={avg_fom:.6f}; avg_best={avg_best:.6f}; top20={top20:.6f}; pass={cp_count}/{n}'},
    ])
    out = pd.concat([df, sums], ignore_index=True)

    nw_csv_path = OUTPUT_DIR / "morl_autockt_results_nw.csv"
    out.to_csv(nw_csv_path, index=False)
    print(f"\n  NW CSV: {nw_csv_path}")
    print(f"  NW: {n} solutions, avg FOM={avg_fom:.6f}, avg best={avg_best:.6f}")
    print(f"  Strict pass: {cp_count}/{n}")

    elapsed = datetime.now() - start
    print(f"\nTotal time: {elapsed}")
    print("DONE")
