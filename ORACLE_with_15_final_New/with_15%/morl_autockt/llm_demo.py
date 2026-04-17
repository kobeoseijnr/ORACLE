"""
Live demo of LLM action masking for circuit optimization.
Shows the full prompt sent to Ollama, the raw LLM response, and the parsed mask.
"""
import sys, json, requests, numpy as np
sys.path.insert(0, '.')
from llm_constraint import (
    LLMActionFilter, SYSTEM_PROMPT, ACTION_PROMPT_TEMPLATE,
    _status_label, _discretize_state
)

ACTION_NAMES = {0: "DECREASE (-1)", 1: "KEEP (0)", 2: "INCREASE (+2)"}
OBJ_NAMES = ["gain", "ugbw", "phase_margin", "ibias"]

SCENARIOS = [
    {
        "name": "Scenario A: All specs far below target, ibias is fine",
        "current": [100, 2e6, 30, 0.001],
        "target":  [300, 5e6, 75, 0.003],
        "expect": "Block DECREASE → agent must INCREASE to reach targets",
    },
    {
        "name": "Scenario B: gain/ugbw/pm met, but ibias way over target",
        "current": [500, 10e6, 80, 0.005],
        "target":  [300, 5e6, 75, 0.003],
        "expect": "Block INCREASE → agent must DECREASE to reduce power",
    },
    {
        "name": "Scenario C: All specs met (optimal region)",
        "current": [400, 8e6, 80, 0.002],
        "target":  [300, 5e6, 75, 0.003],
        "expect": "Block DECREASE → agent should KEEP or carefully INCREASE",
    },
    {
        "name": "Scenario D: Mixed — specs below AND ibias over (conflict)",
        "current": [200, 3e6, 50, 0.004],
        "target":  [300, 5e6, 75, 0.003],
        "expect": "Only KEEP allowed — can't increase (ibias) or decrease (specs)",
    },
]

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2:3b"

def run_demo():
    print("=" * 80)
    print("  LLM ACTION MASKING — LIVE DEMO WITH OLLAMA")
    print("=" * 80)
    print(f"\n  Model: {MODEL} (local Ollama)")
    print(f"  Action space: 0=DECREASE, 1=KEEP, 2=INCREASE")
    print(f"  Objectives: gain↑, ugbw↑, phase_margin↑, ibias↓\n")

    # Also run rule-based for comparison
    rule_filter = LLMActionFilter(use_llm=False)

    for i, sc in enumerate(SCENARIOS):
        c, t = sc["current"], sc["target"]
        print(f"\n{'─'*80}")
        print(f"  {sc['name']}")
        print(f"{'─'*80}")
        print(f"  Expected behavior: {sc['expect']}")
        print()

        # Show spec ratios
        ratios = [c[j] / max(t[j], 1e-9) for j in range(4)]
        labels = []
        for j in range(4):
            hib = (j < 3)  # higher is better for gain/ugbw/pm
            label, _ = _status_label(ratios[j], 1.0, hib)
            labels.append(label)
        print(f"  Current specs:  gain={c[0]:.0f}  ugbw={c[1]/1e6:.1f}MHz  pm={c[2]:.0f}°  ibias={c[3]*1000:.1f}mA")
        print(f"  Target specs:   gain={t[0]:.0f}  ugbw={t[1]/1e6:.1f}MHz  pm={t[2]:.0f}°  ibias={t[3]*1000:.1f}mA")
        print(f"  Ratios:         gain={ratios[0]:.2f}x({labels[0]})  ugbw={ratios[1]:.2f}x({labels[1]})  pm={ratios[2]:.2f}x({labels[2]})  ibias={ratios[3]:.2f}x({labels[3]})")
        print()

        # Build the exact prompt sent to Ollama
        gain_s, _ = _status_label(ratios[0], 1.0, True)
        ugbw_s, _ = _status_label(ratios[1], 1.0, True)
        pm_s, _   = _status_label(ratios[2], 1.0, True)
        ibias_s, _ = _status_label(ratios[3], 1.0, False)
        prompt = ACTION_PROMPT_TEMPLATE.format(
            gain_status=gain_s, gain_ratio=ratios[0],
            ugbw_status=ugbw_s, ugbw_ratio=ratios[1],
            pm_status=pm_s, pm_ratio=ratios[2],
            ibias_status=ibias_s, ibias_ratio=ratios[3],
        )
        print(f"  ┌─── PROMPT SENT TO OLLAMA ───")
        for line in prompt.strip().split('\n'):
            print(f"  │ {line}")
        print(f"  └────────────────────────────\n")

        # Query Ollama
        try:
            resp = requests.post(OLLAMA_URL, json={
                "model": MODEL,
                "system": SYSTEM_PROMPT,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 100},
            }, timeout=30)
            llm_raw = resp.json().get("response", "(no response)")
        except Exception as e:
            llm_raw = f"(error: {e})"

        print(f"  ┌─── RAW LLM RESPONSE ───")
        for line in llm_raw.strip().split('\n'):
            print(f"  │ {line}")
        print(f"  └────────────────────────\n")

        # Parse LLM response
        try:
            start = llm_raw.index('{')
            end = llm_raw.rindex('}') + 1
            parsed = json.loads(llm_raw[start:end])
            llm_allowed = parsed.get("allowed", [0,1,2])
            llm_boost = parsed.get("preference_boost", "none")
        except:
            llm_allowed = "PARSE FAILED"
            llm_boost = "N/A"

        # Rule-based result
        rule_mask, rule_boost = rule_filter.get_action_mask_and_preference(c, t)
        rule_allowed = [j for j in range(3) if rule_mask[j]]

        print(f"  ┌─── RESULTS ───")
        print(f"  │ LLM response:   allowed={llm_allowed}, boost={llm_boost}")
        print(f"  │ Rule-based:     allowed={rule_allowed}, boost={rule_boost}")
        print(f"  │")
        print(f"  │ Rule-based action mask breakdown:")
        for a in range(3):
            status = "✓ ALLOWED" if rule_mask[a] else "✗ BLOCKED"
            print(f"  │   Action {a} ({ACTION_NAMES[a]}): {status}")
        print(f"  │")
        print(f"  │ Preference boost: '{rule_boost}'")
        if rule_boost != "none":
            pref = np.array([0.25, 0.25, 0.25, 0.25])
            boosted = LLMActionFilter(use_llm=False).apply_preference_boost(pref, rule_boost)
            print(f"  │   Before: {pref} → After: {boosted}")
        print(f"  └────────────────\n")

    # Cache stats
    cache_key_example = _discretize_state(0.33, 0.40, 0.40, 0.33)
    print(f"\n{'─'*80}")
    print(f"  CACHING STRATEGY")
    print(f"{'─'*80}")
    print(f"  Each spec ratio is discretized into 4 bins:")
    print(f"    higher-is-better: 'far' (<0.5), 'below' (0.5-0.8), 'close' (0.8-1.0), 'met' (>=1.0)")
    print(f"    lower-is-better:  'met' (<=1.0), 'close' (1.0-1.2), 'above' (1.2-1.5), 'far' (>1.5)")
    print(f"  Total possible states: 4^4 = 256")
    print(f"  Example cache key for Scenario A: '{cache_key_example}'")
    print(f"  → After first query, identical states hit cache instantly (0ms vs ~2s per LLM call)")

if __name__ == "__main__":
    run_demo()
