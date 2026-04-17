"""Live demo: send prompts to Ollama and show raw responses."""
import sys, json, requests
sys.path.insert(0, '.')
from llm_constraint import SYSTEM_PROMPT, ACTION_PROMPT_TEMPLATE, _status_label

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2:3b"
ACTION = {0: "DECREASE(-1)", 1: "KEEP(0)", 2: "INCREASE(+2)"}

CASES = [
    ("A: All specs FAR BELOW target, ibias fine",
     [100, 2e6, 30, 0.001], [300, 5e6, 75, 0.003]),
    ("B: All specs MET, ibias WAY OVER",
     [500, 10e6, 80, 0.005], [300, 5e6, 75, 0.003]),
    ("C: All specs MET, ibias fine (optimal)",
     [400, 8e6, 80, 0.002], [300, 5e6, 75, 0.003]),
    ("D: Specs below AND ibias over (conflict)",
     [200, 3e6, 50, 0.004], [300, 5e6, 75, 0.003]),
]

print("=" * 70)
print("  OLLAMA LIVE DEMO - llama3.2:3b action masking")
print("=" * 70)

for name, cur, tgt in CASES:
    ratios = [cur[i] / max(tgt[i], 1e-9) for i in range(4)]
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

    print(f"\n{'~' * 70}")
    print(f"  SCENARIO {name}")
    print(f"{'~' * 70}")
    print(f"  Current: gain={cur[0]:.0f} ugbw={cur[1]/1e6:.1f}MHz pm={cur[2]:.0f} ibias={cur[3]*1000:.1f}mA")
    print(f"  Target:  gain={tgt[0]:.0f} ugbw={tgt[1]/1e6:.1f}MHz pm={tgt[2]:.0f} ibias={tgt[3]*1000:.1f}mA")
    print(f"  Ratios:  gain={ratios[0]:.2f}x({gain_s}) ugbw={ratios[1]:.2f}x({ugbw_s}) pm={ratios[2]:.2f}x({pm_s}) ibias={ratios[3]:.2f}x({ibias_s})")
    print()
    print("  PROMPT SENT TO OLLAMA:")
    print("  " + "-" * 50)
    for line in prompt.strip().split('\n'):
        print(f"  | {line}")
    print("  " + "-" * 50)
    print()

    resp = requests.post(OLLAMA_URL, json={
        "model": MODEL, "system": SYSTEM_PROMPT, "prompt": prompt,
        "stream": False, "options": {"temperature": 0.1, "num_predict": 100},
    }, timeout=30)
    raw = resp.json().get("response", "")

    print(f"  OLLAMA RAW RESPONSE:")
    print(f"  >>> {raw.strip()}")
    print()

    try:
        start = raw.index('{')
        end = raw.rindex('}') + 1
        parsed = json.loads(raw[start:end])
        allowed = parsed.get("allowed", [])
        boost = parsed.get("preference_boost", "none")
        blocked = [i for i in [0,1,2] if i not in allowed]
        print(f"  PARSED:")
        print(f"    Allowed actions: {[ACTION[a] for a in allowed]}")
        print(f"    Blocked actions: {[ACTION[a] for a in blocked]}")
        print(f"    Preference boost: {boost}")
    except Exception as e:
        print(f"  PARSE ERROR: {e}")

print(f"\n{'=' * 70}")
print("  SYSTEM PROMPT (sent with every query):")
print("=" * 70)
for line in SYSTEM_PROMPT.strip().split('\n'):
    print(f"  {line}")
