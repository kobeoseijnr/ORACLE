"""
Train BOTH standard cosine and LLM-masked agents in the SAME environment
(PM>=75 surrogate, PM=75 YAML target) for a fair same-scale comparison.
Then evaluate both on 1000 specs (10 prefs) and merge results into cosine CSV.
"""
import os
import sys
import pickle
import numpy as np
import json
import torch
import gym
import pandas as pd
import io
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).parent
METHODOLOGY_DIR = BASE_DIR / "methodology"
sys.path.insert(0, str(METHODOLOGY_DIR))
sys.path.insert(0, str(BASE_DIR))

os.environ['AUTOCKT_USE_SURROGATE'] = 'true'

from autockt.envs.autockt_mo_env import AutoCktMOEnv
from autockt.agents.mo_agent import MO_DQN_Agent, MO_DQN_LLM_Agent
from autockt.utils.mo_utils import generate_preference_vectors
from llm_constraint import LLMActionFilter

DATASET_PATH = BASE_DIR / "dataset" / "ngspice_specs_gen_two_stage_opamp"
RESULTS_DIR = BASE_DIR / "results"
LLM_CACHE_PATH = RESULTS_DIR / "llm_action_cache_fair.json"

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


def _fom(g, u, p, i, gt, ut, pt, it):
    def s(x): return max(float(x), 1e-9)
    gt, ut, pt, it = s(gt), s(ut), s(pt), s(it)
    return (float(g)-gt)/gt + (float(u)-ut)/ut + (float(p)-pt)/pt - (float(i)-it)/it


def _get_specs(env):
    if not hasattr(env.base_env, 'cur_specs') or not hasattr(env.base_env, 'specs_id'):
        return None, None
    sid = env.base_env.specs_id
    cur = env.base_env.cur_specs.copy()
    tgt = env.base_env.specs_ideal.copy()
    dc = dict(zip(sid, cur)); dt = dict(zip(sid, tgt))
    current = [dc.get('gain', dc.get('gain_min', cur[0])),
               dc.get('ugbw', dc.get('ugbw_min', cur[1] if len(cur)>1 else 0)),
               dc.get('phm', dc.get('phm_min', cur[2] if len(cur)>2 else 0)),
               dc.get('ibias_max', cur[3] if len(cur)>3 else 0)]
    target = [dt.get('gain', dt.get('gain_min', tgt[0])),
              dt.get('ugbw', dt.get('ugbw_min', tgt[1] if len(tgt)>1 else 0)),
              dt.get('phm', dt.get('phm_min', tgt[2] if len(tgt)>2 else 0)),
              dt.get('ibias_max', tgt[3] if len(tgt)>3 else 0)]
    return current, target


def train_standard(env, specs, preferences, num_params):
    tag = "COSINE"
    print(f"\n{'='*70}\n  TRAINING STANDARD COSINE\n{'='*70}\n")
    sd = env.observation_space.shape[0] if hasattr(env.observation_space, 'shape') else 20
    ad = env.action_space.spaces[0].n if isinstance(env.action_space, gym.spaces.Tuple) else (env.action_space.n if hasattr(env.action_space, 'n') else 7)
    agent = MO_DQN_Agent(sd, ad, 4, scalarization='cosine')
    print(f"  [{tag}] state={sd}, action={ad}, reward=4")
    training_specs = list(range(min(TRAIN_CONFIG['num_training_specs'], len(list(specs.values())[0]))))
    ec = 0
    for si, sn in enumerate(training_specs):
        if si % 10 == 0: print(f"  [{tag}] Spec {si+1}/{len(training_specs)}...")
        try: env.base_env.obj_idx = sn; env.reset()
        except: continue
        for pi, pref in enumerate(preferences):
            agent.set_preference(pref)
            for ep in range(max(1, TRAIN_CONFIG['training_episodes']//len(preferences))):
                state = env.reset()
                for step in range(TRAIN_CONFIG['max_steps']):
                    eps = max(0.1, 1.0 - ec/1000)
                    try:
                        action = agent.select_action(state, pref, epsilon=eps)
                        if isinstance(env.action_space, gym.spaces.Tuple):
                            if isinstance(action, (int, np.integer)): action = tuple([int(np.clip(action,0,2))]*num_params)
                            elif isinstance(action, (list, np.ndarray)):
                                action = tuple([int(np.clip(a,0,2)) for a in action[:num_params]])
                                while len(action)<num_params: action=action+(0,)
                    except:
                        action = tuple([np.random.randint(0,3) for _ in range(num_params)]) if isinstance(env.action_space, gym.spaces.Tuple) else env.action_space.sample()
                    sr = env.step(action)
                    if len(sr)==5: ns, rv, done, tr, info = sr; done = done or tr
                    elif len(sr)==4: ns, rv, done, info = sr
                    else: break
                    if hasattr(agent,'memory') and hasattr(agent.memory,'push'): agent.memory.push(state,action,rv,ns,done,pref)
                    if hasattr(agent,'memory') and len(agent.memory)>32 and ec%4==0:
                        try: agent.update(batch_size=32)
                        except: pass
                    state = ns
                    if done: break
                ec += 1
    print(f"  [{tag}] Done — {ec} episodes")
    return agent


def train_llm(env, specs, preferences, num_params):
    tag = "LLM"
    print(f"\n{'='*70}\n  TRAINING LLM-MASKED COSINE\n{'='*70}\n")
    sd = env.observation_space.shape[0] if hasattr(env.observation_space, 'shape') else 20
    ad = env.action_space.spaces[0].n if isinstance(env.action_space, gym.spaces.Tuple) else (env.action_space.n if hasattr(env.action_space, 'n') else 7)
    llm_filter = LLMActionFilter(model='llama3.2:1b', use_llm=True, cache_path=str(LLM_CACHE_PATH), verbose=True)
    agent = MO_DQN_LLM_Agent(sd, ad, 4, llm_filter=llm_filter, scalarization='cosine', use_preference_boost=True)
    print(f"  [{tag}] state={sd}, action={ad}, LLM available={llm_filter.llm_available}")
    training_specs = list(range(min(TRAIN_CONFIG['num_training_specs'], len(list(specs.values())[0]))))
    ec = 0; muf = 3
    for si, sn in enumerate(training_specs):
        if si % 10 == 0: print(f"  [{tag}] Spec {si+1}/{len(training_specs)}...")
        try: env.base_env.obj_idx = sn; env.reset()
        except: continue
        for pi, pref in enumerate(preferences):
            agent.set_preference(pref)
            for ep in range(max(1, TRAIN_CONFIG['training_episodes']//len(preferences))):
                state = env.reset()
                cs, ts = _get_specs(env)
                if cs: agent.update_llm_mask(cs, ts)
                bp = agent.get_boosted_preference(pref); agent.set_preference(bp)
                for step in range(TRAIN_CONFIG['max_steps']):
                    eps = max(0.1, 1.0 - ec/1000)
                    try:
                        action = agent.select_action(state, None, epsilon=eps)
                        if isinstance(env.action_space, gym.spaces.Tuple):
                            if isinstance(action, (int, np.integer)): action = tuple([int(np.clip(action,0,2))]*num_params)
                            elif isinstance(action, (list, np.ndarray)):
                                action = tuple([int(np.clip(a,0,2)) for a in action[:num_params]])
                                while len(action)<num_params: action=action+(0,)
                    except:
                        action = tuple([np.random.randint(0,3) for _ in range(num_params)]) if isinstance(env.action_space, gym.spaces.Tuple) else env.action_space.sample()
                    sr = env.step(action)
                    if len(sr)==5: ns, rv, done, tr, info = sr; done = done or tr
                    elif len(sr)==4: ns, rv, done, info = sr
                    else: break
                    if hasattr(agent,'memory') and hasattr(agent.memory,'push'): agent.memory.push(state,action,rv,ns,done,pref)
                    if hasattr(agent,'memory') and len(agent.memory)>32 and ec%4==0:
                        try: agent.update(batch_size=32)
                        except: pass
                    state = ns
                    if step%muf==0 and step>0:
                        cs, ts = _get_specs(env)
                        if cs: agent.update_llm_mask(cs, ts)
                    if done: break
                agent.set_preference(pref)
                ec += 1
    print(f"  [{tag}] Done — {ec} episodes")
    llm_filter.print_stats()
    return agent, llm_filter


def evaluate(agent, label, env, specs, preferences, llm_filter=None):
    tag = label.upper()
    print(f"\n{'='*70}\n  EVALUATING — {tag}\n{'='*70}\n")
    agent.q_network.eval()
    ms = EVAL_CONFIG['max_steps']
    nt = min(EVAL_CONFIG['num_targets'], len(list(specs.values())[0]))
    np_ = len(env.action_space.spaces) if isinstance(env.action_space, gym.spaces.Tuple) else 1
    is_llm = hasattr(agent, 'update_llm_mask')
    muf = 5

    tgt_json = {}
    for i in range(len(list(specs.values())[0])):
        tgt_json[str(i)] = {
            'tgl': float(specs['gain_min'][i]), 'tum': float(specs['ugbw_min'][i])/1e6,
            'tpd': float(specs['phm_min'][i]), 'tim': float(specs['ibias_max'][i])*1000.0,
        }

    solutions = []
    for ti in range(nt):
        if ti%100==0 and ti>0: print(f"  [{tag}] {ti}/{nt}")
        try: env.base_env.obj_idx=ti; env.reset(); esid=env.base_env.specs_id
        except: continue
        for pi, pref in enumerate(preferences):
            try:
                agent.set_preference(pref)
                state = env.reset()
                if is_llm:
                    cs, ts = _get_specs(env)
                    if cs: agent.update_llm_mask(cs, ts)
                    bp = agent.get_boosted_preference(pref); agent.set_preference(bp)
                done = False
                for step in range(ms):
                    eps = max(0.05, 0.15*(1-step/ms))
                    action = agent.select_action(state, None if is_llm else pref, epsilon=eps)
                    if isinstance(env.action_space, gym.spaces.Tuple):
                        if isinstance(action, (int, np.integer)): action = tuple([int(np.clip(action,0,2))]*np_)
                        elif isinstance(action, (list, np.ndarray)):
                            al=[int(np.clip(a,0,2)) for a in action[:np_]]
                            while len(al)<np_: al.append(0)
                            action=tuple(al)
                        elif not isinstance(action, tuple): action=tuple([int(np.clip(action,0,2))]*np_)
                    sr = env.step(action)
                    if len(sr)==5: ns,rw,done,tr,info=sr; done=done or tr
                    elif len(sr)==4: ns,rw,done,info=sr
                    else: break
                    state = ns
                    if is_llm and step%muf==0 and step>0:
                        cs, ts = _get_specs(env)
                        if cs: agent.update_llm_mask(cs, ts)
                    if done: break
                if is_llm: agent.set_preference(pref)
                if hasattr(env.base_env, 'cur_specs'):
                    actual = env.base_env.cur_specs.copy()
                    sd = dict(zip(esid, actual))
                    gv=sd.get('gain',sd.get('gain_min',actual[0]))
                    uv=sd.get('ugbw',sd.get('ugbw_min',actual[1] if len(actual)>1 else 0))
                    pv=sd.get('phm',sd.get('phm_min',actual[2] if len(actual)>2 else 0))
                    iv=sd.get('ibias_max',sd.get('ibias',actual[3] if len(actual)>3 else 0))
                    if gv<100: gl=10**(gv/20); gd=gv
                    else: gl=gv; gd=20*np.log10(gv) if gv>0 else 0
                    td=tgt_json.get(str(ti),{})
                    gt,ut,pt,it=td.get('tgl',1e-9),td.get('tum',1e-9),td.get('tpd',1e-9),td.get('tim',1e-9)
                    go,uo,po,io_=float(gl),float(uv/1e6),float(pv),float(iv*1000)
                    f=_fom(go,uo,po,io_,gt,ut,pt,it)
                    gp='Yes' if go>=gt else 'No'
                    up='Yes' if uo>=ut else 'No'
                    pp='Yes' if po>=pt else 'No'
                    ip='Yes' if io_<=it else 'No'
                    cp='Yes' if gp=='Yes' and up=='Yes' and pp=='Yes' and ip=='Yes' else 'No'
                    solutions.append({
                        'spec':ti+1,'solution':pi+1,
                        'target_gain_linear':gt,'target_ugbw_mhz':ut,'target_pm_deg':pt,'target_ibias_ma':it,
                        'output_gain_linear':round(go,6),'output_gain_db':round(float(gd),6),
                        'output_ugbw_mhz':round(uo,6),'output_pm_deg':round(po,4),'output_ibias_ma':round(io_,6),
                        'fom':round(f,6),
                        'gain_pass':gp,'ugbw_pass':up,'pm_pass':pp,'ibias_pass':ip,'complete_pass':cp,
                        'scalarization':label,
                    })
            except: continue
    print(f"  [{tag}] {len(solutions)} solutions")
    return solutions


if __name__ == "__main__":
    start = datetime.now()
    print(f"\n{'#'*70}")
    print(f"  FAIR COMPARISON: COSINE vs LLM-MASKED")
    print(f"  Same env, same surrogate (PM>=75), same config")
    print(f"  Start: {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*70}")

    with open(DATASET_PATH, 'rb') as f: specs = pickle.load(f)
    ns = len(list(specs.values())[0])
    print(f"\nDataset: {ns} specs")
    env = AutoCktMOEnv(generalize=True, num_valid=ns, run_valid=True)
    np_ = len(env.action_space.spaces) if isinstance(env.action_space, gym.spaces.Tuple) else 1
    try:
        prefs = generate_preference_vectors(4, method='focused', num_vectors=TRAIN_CONFIG['num_preferences'])
        if len(prefs)>TRAIN_CONFIG['num_preferences']: prefs=prefs[:TRAIN_CONFIG['num_preferences']]
    except:
        prefs = generate_preference_vectors(4, method='random', num_vectors=TRAIN_CONFIG['num_preferences'])
    print(f"Preferences: {len(prefs)} vectors")

    # Train both
    cos_agent = train_standard(env, specs, prefs, np_)
    llm_agent, llm_filter = train_llm(env, specs, prefs, np_)

    # Evaluate both
    cos_results = evaluate(cos_agent, 'cosine', env, specs, prefs)
    llm_results = evaluate(llm_agent, 'cosine+llm', env, specs, prefs, llm_filter)

    # Build merged CSV
    df_cos = pd.DataFrame(cos_results)
    df_llm = pd.DataFrame(llm_results)

    # Rename LLM columns
    llm_rename = {c: f'llm_{c}' for c in ['output_gain_linear','output_gain_db','output_ugbw_mhz',
                   'output_pm_deg','output_ibias_ma','fom','gain_pass','ugbw_pass','pm_pass',
                   'ibias_pass','complete_pass','scalarization']}
    df_llm_r = df_llm[['spec','solution']+list(llm_rename.keys())].rename(columns=llm_rename)
    merged = df_cos.merge(df_llm_r, on=['spec','solution'], how='left')
    merged['fom_winner'] = merged.apply(
        lambda r: 'tie' if abs(r['fom']-r['llm_fom'])<1e-9
        else ('cosine' if r['fom']>r['llm_fom'] else 'llm_masked'), axis=1)

    # Stats
    n=len(merged); nsp=merged['spec'].nunique()
    ca=merged['fom'].mean(); la=merged['llm_fom'].mean()
    cb=merged.groupby('spec')['fom'].max(); lb=merged.groupby('spec')['llm_fom'].max()
    cab=cb.mean(); lab=lb.mean()
    ct20=cb.nlargest(20).mean(); lt20=lb.nlargest(20).mean()
    cpass=(merged['complete_pass']=='Yes').sum(); lpass=(merged['llm_complete_pass']=='Yes').sum()
    bm=pd.DataFrame({'c':cb,'l':lb})
    wc=(bm['c']>bm['l']).sum(); wl=(bm['l']>bm['c']).sum(); tb=(abs(bm['c']-bm['l'])<1e-9).sum()
    rwc=(merged['fom_winner']=='cosine').sum(); rwl=(merged['fom_winner']=='llm_masked').sum()
    rwt=(merged['fom_winner']=='tie').sum()

    sums = pd.DataFrame([
        {'spec':'summary_cosine','fom':f'avg_fom={ca:.6f}; avg_best={cab:.6f}; top20={ct20:.6f}; pass={cpass}/{n}'},
        {'spec':'summary_llm_masked','llm_fom':f'avg_fom={la:.6f}; avg_best={lab:.6f}; top20={lt20:.6f}; pass={lpass}/{n}'},
        {'spec':'summary_best_per_spec_wins','fom':f'cosine={wc}','llm_fom':f'llm={wl}','fom_winner':f'ties={tb}'},
        {'spec':'summary_per_row_wins','fom':f'cosine={rwc}','llm_fom':f'llm={rwl}','fom_winner':f'ties={rwt}'},
        {'spec':'summary_avg_fom_cosine','fom':round(ca,6)},
        {'spec':'summary_avg_fom_llm','llm_fom':round(la,6)},
        {'spec':'summary_avg_best_fom_cosine','fom':round(cab,6)},
        {'spec':'summary_avg_best_fom_llm','llm_fom':round(lab,6)},
        {'spec':'summary_top20_best_fom_cosine','fom':round(ct20,6)},
        {'spec':'summary_top20_best_fom_llm','llm_fom':round(lt20,6)},
    ])
    out = pd.concat([merged, sums], ignore_index=True)
    out_path = RESULTS_DIR / "morl_cosine_vs_llm_fair_comparison.csv"
    out.to_csv(out_path, index=False)

    # Also save to with_15%_with_20 results
    out_path2 = Path(r"C:\Users\kobeo\OneDrive\Desktop\trail\with_15%_with_20\with_15%\morl_autockt\results\morl_autockt_results_original_cosine_with_llm.csv")
    out.to_csv(out_path2, index=False)

    print(f"\n{'#'*70}")
    print(f"  RESULTS — FAIR COMPARISON")
    print(f"{'#'*70}")
    print(f"\n  COSINE:      avg_fom={ca:.6f}  avg_best={cab:.6f}  top20={ct20:.6f}  pass={cpass}/{n}")
    print(f"  LLM-MASKED:  avg_fom={la:.6f}  avg_best={lab:.6f}  top20={lt20:.6f}  pass={lpass}/{n}")
    print(f"\n  Best-per-spec: Cosine={wc}, LLM={wl}, Ties={tb}")
    print(f"  Per-row:       Cosine={rwc}, LLM={rwl}, Ties={rwt}")
    winner='LLM-MASKED' if lab>cab else 'COSINE'
    print(f"  WINNER: {winner} (by {abs(lab-cab):.6f})")
    print(f"\n  Saved: {out_path}")
    print(f"  Also: {out_path2}")
    print(f"\n  Total time: {datetime.now()-start}")
    print("DONE")
