"""
LLM-based Action Filter for Multi-Objective Circuit Optimization.
Uses a local Ollama model to provide circuit-design-aware action masking
and dynamic preference adjustment for the MO-DQN agent.

The LLM encodes domain knowledge about two-stage op-amp design:
  - Increasing transistor sizes generally increases gain/UGBW but also power (ibias)
  - Decreasing transistor sizes reduces power but may hurt gain/phase margin
  - When all specs are nearly met, conservative (keep) actions are preferred

Action space: [0=decrease_all(-1), 1=keep(0), 2=increase_all(+2)]
Objectives:   [gain, ugbw, phase_margin, ibias(lower=better)]
"""

import json
import hashlib
import os
import numpy as np

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3.2:1b"

# ── Prompt Template ──────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are a circuit design expert advising an RL agent optimizing a two-stage operational amplifier.

The agent controls 7 transistor parameters simultaneously with ONE action per step:
  Action 0: DECREASE all parameter indices by 1 (smaller transistors → less gain, less power)
  Action 1: KEEP all parameter indices the same
  Action 2: INCREASE all parameter indices by 2 (larger transistors → more gain, more power)

There are 4 design objectives:
  gain       – higher is better (must meet or exceed target)
  ugbw       – higher is better (must meet or exceed target)
  pm (phase margin) – higher is better (must meet or exceed target)
  ibias      – LOWER is better (must stay at or below target)

Circuit design knowledge:
  - Larger transistors increase gain and bandwidth but increase bias current (power).
  - Smaller transistors reduce power but may reduce gain and phase margin.
  - When gain/ugbw/pm are all met but ibias is too high, DECREASE is the right move.
  - When ibias is fine but gain/ugbw/pm are too low, INCREASE is the right move.
  - When specs are very close to targets, KEEP may be safest to avoid overshooting.
"""

ACTION_PROMPT_TEMPLATE = """\
Current circuit state relative to targets:
  gain:  {gain_status} (ratio: {gain_ratio:.2f}x of target)
  ugbw:  {ugbw_status} (ratio: {ugbw_ratio:.2f}x of target)
  pm:    {pm_status} (ratio: {pm_ratio:.2f}x of target)
  ibias: {ibias_status} (ratio: {ibias_ratio:.2f}x of target, lower is better)

Which actions should be ALLOWED? Reply with ONLY a JSON object:
{{"allowed": [list of allowed action indices from 0,1,2], "preference_boost": "gain" or "ugbw" or "pm" or "ibias" or "none"}}

Rules:
- If ibias exceeds target, action 2 (INCREASE) should usually be blocked.
- If gain/ugbw/pm are far below target, action 0 (DECREASE) should usually be blocked.
- Always allow at least one action.
- preference_boost: which objective needs the most attention right now, or "none" if balanced.
"""


def _status_label(current, target, higher_is_better=True):
    """Classify a spec as far_below / below / met / above target."""
    if target == 0:
        return "met", 1.0
    ratio = current / target
    if higher_is_better:
        if ratio >= 1.0:
            return "MET", ratio
        elif ratio >= 0.8:
            return "SLIGHTLY_BELOW", ratio
        else:
            return "FAR_BELOW", ratio
    else:  # lower is better (ibias)
        if ratio <= 1.0:
            return "MET", ratio
        elif ratio <= 1.2:
            return "SLIGHTLY_ABOVE", ratio
        else:
            return "FAR_ABOVE", ratio


def _discretize_state(gain_r, ugbw_r, pm_r, ibias_r):
    """Discretize spec ratios into bins for cache key."""
    def _bin(ratio, higher_is_better=True):
        if higher_is_better:
            if ratio >= 1.0: return "met"
            if ratio >= 0.8: return "close"
            if ratio >= 0.5: return "below"
            return "far"
        else:
            if ratio <= 1.0: return "met"
            if ratio <= 1.2: return "close"
            if ratio <= 1.5: return "above"
            return "far"
    return f"{_bin(gain_r)}|{_bin(ugbw_r)}|{_bin(pm_r)}|{_bin(ibias_r, False)}"


class LLMActionFilter:
    """
    Uses a local Ollama LLM to provide action masks and preference boosts
    for circuit optimization. Caches responses by discretized state.
    Falls back to rule-based heuristics if LLM is unavailable.
    """

    def __init__(self, model=DEFAULT_MODEL, use_llm=True, cache_path=None, verbose=False):
        self.model = model
        self.use_llm = use_llm and HAS_REQUESTS
        self.verbose = verbose
        self.cache = {}
        self.cache_path = cache_path
        self.llm_available = False
        self.stats = {"llm_calls": 0, "cache_hits": 0, "rule_fallbacks": 0}

        if self.cache_path and os.path.exists(self.cache_path):
            with open(self.cache_path, 'r') as f:
                self.cache = json.load(f)
            if self.verbose:
                print(f"  [LLM] Loaded {len(self.cache)} cached responses from {self.cache_path}")

        if self.use_llm:
            self._check_ollama()

    def _check_ollama(self):
        """Check if Ollama is running and model is available."""
        try:
            resp = requests.get("http://localhost:11434/api/tags", timeout=5)
            if resp.status_code == 200:
                models = [m['name'] for m in resp.json().get('models', [])]
                self.llm_available = any(self.model in m for m in models)
                if self.verbose:
                    if self.llm_available:
                        print(f"  [LLM] Ollama ready, model '{self.model}' available")
                    else:
                        print(f"  [LLM] Ollama running but model '{self.model}' not found. Available: {models}")
                        print(f"  [LLM] Will use rule-based fallback. Run: ollama pull {self.model}")
        except Exception as e:
            if self.verbose:
                print(f"  [LLM] Ollama not reachable ({e}). Using rule-based fallback.")
            self.llm_available = False

    def get_action_mask_and_preference(self, current_specs, target_specs):
        """
        Get action mask and preference boost given current vs target specs.

        Args:
            current_specs: dict or array [gain, ugbw, pm, ibias]
            target_specs:  dict or array [gain, ugbw, pm, ibias]

        Returns:
            action_mask: np.array of shape (3,) with True for allowed actions
            preference_boost: str — which objective to boost, or "none"
        """
        if isinstance(current_specs, dict):
            c = [current_specs.get('gain', 0), current_specs.get('ugbw', 0),
                 current_specs.get('pm', 0), current_specs.get('ibias', 0)]
            t = [target_specs.get('gain', 1), target_specs.get('ugbw', 1),
                 target_specs.get('pm', 1), target_specs.get('ibias', 1)]
        else:
            c, t = list(current_specs), list(target_specs)

        # Compute ratios
        gain_r = c[0] / max(t[0], 1e-9)
        ugbw_r = c[1] / max(t[1], 1e-9)
        pm_r   = c[2] / max(t[2], 1e-9)
        ibias_r = c[3] / max(t[3], 1e-9)

        # Check cache
        cache_key = _discretize_state(gain_r, ugbw_r, pm_r, ibias_r)
        if cache_key in self.cache:
            self.stats["cache_hits"] += 1
            entry = self.cache[cache_key]
            return np.array(entry["mask"], dtype=bool), entry["boost"]

        # Try LLM
        if self.llm_available:
            mask, boost = self._query_llm(gain_r, ugbw_r, pm_r, ibias_r)
            if mask is not None:
                self.stats["llm_calls"] += 1
                self.cache[cache_key] = {"mask": mask.tolist(), "boost": boost}
                self._save_cache()
                return mask, boost

        # Rule-based fallback
        self.stats["rule_fallbacks"] += 1
        mask, boost = self._rule_based(gain_r, ugbw_r, pm_r, ibias_r)
        self.cache[cache_key] = {"mask": mask.tolist(), "boost": boost}
        self._save_cache()
        return mask, boost

    def _query_llm(self, gain_r, ugbw_r, pm_r, ibias_r):
        """Query Ollama for action mask."""
        gain_s, _ = _status_label(gain_r, 1.0, True)
        ugbw_s, _ = _status_label(ugbw_r, 1.0, True)
        pm_s,   _ = _status_label(pm_r, 1.0, True)
        ibias_s, _ = _status_label(ibias_r, 1.0, False)

        prompt = ACTION_PROMPT_TEMPLATE.format(
            gain_status=gain_s, gain_ratio=gain_r,
            ugbw_status=ugbw_s, ugbw_ratio=ugbw_r,
            pm_status=pm_s, pm_ratio=pm_r,
            ibias_status=ibias_s, ibias_ratio=ibias_r,
        )

        try:
            resp = requests.post(OLLAMA_URL, json={
                "model": self.model,
                "system": SYSTEM_PROMPT,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 100},
            }, timeout=30)

            if resp.status_code != 200:
                return None, "none"

            text = resp.json().get("response", "")
            return self._parse_llm_response(text)
        except Exception as e:
            if self.verbose:
                print(f"  [LLM] Query failed: {e}")
            return None, "none"

    def _parse_llm_response(self, text):
        """Parse LLM JSON response into mask and boost."""
        try:
            # Find JSON in response
            start = text.index('{')
            end = text.rindex('}') + 1
            data = json.loads(text[start:end])

            allowed = data.get("allowed", [0, 1, 2])
            boost = data.get("preference_boost", "none")

            # Validate
            allowed = [a for a in allowed if a in [0, 1, 2]]
            if not allowed:
                allowed = [0, 1, 2]
            if boost not in ["gain", "ugbw", "pm", "ibias", "none"]:
                boost = "none"

            mask = np.array([i in allowed for i in range(3)], dtype=bool)
            return mask, boost
        except (ValueError, json.JSONDecodeError, KeyError):
            return None, "none"

    def _rule_based(self, gain_r, ugbw_r, pm_r, ibias_r):
        """
        Rule-based heuristic fallback encoding circuit design knowledge.
        Returns (action_mask, preference_boost).
        """
        mask = np.array([True, True, True], dtype=bool)  # [decrease, keep, increase]
        boost = "none"

        # Count how many "higher-is-better" specs are unmet
        unmet_hib = sum(1 for r in [gain_r, ugbw_r, pm_r] if r < 1.0)
        ibias_over = ibias_r > 1.0

        # If ibias exceeds target, block INCREASE (action 2)
        if ibias_over and ibias_r > 1.1:
            mask[2] = False

        # If most higher-is-better specs are far below, block DECREASE (action 0)
        if unmet_hib >= 2 and min(gain_r, ugbw_r, pm_r) < 0.7:
            mask[0] = False

        # If all specs are met, prefer KEEP
        if unmet_hib == 0 and not ibias_over:
            mask[0] = False  # don't decrease if everything is good

        # Determine which objective needs most attention
        if ibias_over and ibias_r > max(1.0/gain_r if gain_r > 0 else 0,
                                         1.0/ugbw_r if ugbw_r > 0 else 0,
                                         1.0/pm_r if pm_r > 0 else 0):
            boost = "ibias"
        elif gain_r < ugbw_r and gain_r < pm_r and gain_r < 1.0:
            boost = "gain"
        elif ugbw_r < gain_r and ugbw_r < pm_r and ugbw_r < 1.0:
            boost = "ugbw"
        elif pm_r < gain_r and pm_r < ugbw_r and pm_r < 1.0:
            boost = "pm"

        # Safety: always allow at least one action
        if not mask.any():
            mask[1] = True  # keep is always safe

        return mask, boost

    def apply_preference_boost(self, preference, boost, boost_strength=0.15):
        """
        Adjust preference vector based on LLM recommendation.
        Shifts weight toward the most-violated objective.

        Args:
            preference: np.array of shape (reward_dim,) — current preference
            boost: str — objective to boost
            boost_strength: float — how much to shift (0.0 to 0.5)

        Returns:
            adjusted_preference: np.array — re-normalized preference
        """
        if boost == "none":
            return preference

        pref = preference.copy().astype(np.float64)
        boost_map = {"gain": 0, "ugbw": 1, "pm": 2, "ibias": 3}
        idx = boost_map.get(boost)
        if idx is None or idx >= len(pref):
            return preference

        # Boost the target objective, reduce others proportionally
        pref[idx] += boost_strength
        total = pref.sum()
        if total > 0:
            pref = pref / total

        return pref.astype(np.float32)

    def _save_cache(self):
        """Persist cache to disk."""
        if self.cache_path:
            try:
                with open(self.cache_path, 'w') as f:
                    json.dump(self.cache, f, indent=2)
            except Exception:
                pass

    def print_stats(self):
        """Print usage statistics."""
        print(f"  [LLM] Stats: {self.stats['llm_calls']} LLM calls, "
              f"{self.stats['cache_hits']} cache hits, "
              f"{self.stats['rule_fallbacks']} rule fallbacks, "
              f"{len(self.cache)} cached states")
