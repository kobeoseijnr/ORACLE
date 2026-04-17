"""Single live call to Ollama with timing."""
import requests, json, time

URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2:3b"

SYSTEM = (
    "You are a circuit design expert advising an RL agent optimizing a two-stage operational amplifier.\n"
    "The agent controls 7 transistor parameters with ONE action per step:\n"
    "  Action 0: DECREASE (-1)\n  Action 1: KEEP (0)\n  Action 2: INCREASE (+2)\n"
    "4 objectives: gain (higher=better), ugbw (higher=better), pm (higher=better), ibias (LOWER=better).\n"
    "Larger transistors increase gain/ugbw but also increase ibias (power)."
)

PROMPT = (
    "Current circuit state relative to targets:\n"
    "  gain:  FAR_BELOW (ratio: 0.33x of target)\n"
    "  ugbw:  FAR_BELOW (ratio: 0.40x of target)\n"
    "  pm:    FAR_BELOW (ratio: 0.40x of target)\n"
    "  ibias: MET (ratio: 0.33x of target, lower is better)\n\n"
    'Which actions should be ALLOWED? Reply with ONLY a JSON object:\n'
    '{"allowed": [list of action indices from 0,1,2], "preference_boost": "gain" or "ugbw" or "pm" or "ibias" or "none"}\n\n'
    "Rules:\n"
    "- If ibias exceeds target, action 2 (INCREASE) should usually be blocked.\n"
    "- If gain/ugbw/pm are far below target, action 0 (DECREASE) should usually be blocked.\n"
    "- Always allow at least one action."
)

print("=" * 60)
print("  LIVE OLLAMA CALL")
print("=" * 60)
print(f"  Model: {MODEL}")
print(f"\n  Scenario: All specs FAR BELOW, ibias fine")
print(f"  gain=0.33x  ugbw=0.40x  pm=0.40x  ibias=0.33x")
print(f"\n  Sending to Ollama now...")

t0 = time.time()
resp = requests.post(URL, json={
    "model": MODEL,
    "system": SYSTEM,
    "prompt": PROMPT,
    "stream": False,
    "options": {"temperature": 0.1, "num_predict": 100},
}, timeout=30)
elapsed = time.time() - t0

raw = resp.json()["response"].strip()
print(f"  Response time: {elapsed:.2f} seconds")
print(f"\n  Ollama raw response:")
print(f"  >>> {raw}")

try:
    s = raw.index('{')
    e = raw.rindex('}') + 1
    parsed = json.loads(raw[s:e])
    allowed = parsed["allowed"]
    boost = parsed["preference_boost"]
    names = {0: "DECREASE", 1: "KEEP", 2: "INCREASE"}
    blocked = [i for i in [0,1,2] if i not in allowed]
    print(f"\n  Parsed result:")
    print(f"    Allowed: {[names[a] for a in allowed]}")
    print(f"    Blocked: {[names[a] for a in blocked]}")
    print(f"    Boost:   {boost}")
except Exception as ex:
    print(f"\n  Parse error: {ex}")
