"""Preflight: call every enabled model once so a wrong model id, a missing key or an exhausted quota
shows up in seconds, not two hours into the run.  A failing LOCAL model stops the run (exit 1); a failing
cloud model is written to .runlogs/disabled_models.txt and left out of this run."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from assistant import providers, registry  # noqa: E402

def gemini_fallback(spec):
    """The configured Gemini id is not served to this key: pick the cheapest listed flash-lite / flash model
    we have a price for, and record the swap in models.yaml so the change is visible."""
    import re
    import yaml
    avail = providers.gemini_models(spec)
    print("    Gemini models available to this key:", ", ".join(m for m in avail if "flash" in m)[:400])
    known = [m for m in avail if m in providers.GEMINI_PRICES and "preview" not in m and "exp" not in m]
    if not known:
        return None
    pick = min(known, key=lambda m: (0 if "lite" in m else 1, providers.GEMINI_PRICES[m][1]))
    pin, pout = providers.GEMINI_PRICES[pick]
    path = registry.config.REGISTRY_PATH
    raw = yaml.safe_load(path.read_text())
    for e in raw["models"]:
        if e["name"] == spec.name:
            e["model"], e["price_in"], e["price_out"] = pick, pin, pout
    text = path.read_text()
    text = re.sub(rf"(- name: {re.escape(spec.name)}\n(?:    .*\n)*?    model: )\S+", rf"\g<1>{pick}", text)
    text = re.sub(rf"(- name: {re.escape(spec.name)}\n(?:    .*\n)*?    price_in: )[\d.]+", rf"\g<1>{pin}", text)
    text = re.sub(rf"(- name: {re.escape(spec.name)}\n(?:    .*\n)*?    price_out: )[\d.]+", rf"\g<1>{pout}", text)
    path.write_text(text)
    print(f"    -> {spec.name} now uses {pick} (${pin}/${pout} per 1M); models.yaml updated")
    spec.model, spec.price_in, spec.price_out = pick, pin, pout
    return spec


bad, failed_cloud = [], []
for name, spec in registry.load().items():
    if not spec.local and not os.getenv(spec.api_key_env or ""):
        print(f"  SKIP {name:<20} {spec.api_key_env} not set (cloud model left out of this run)")
        continue
    g = providers.generate(spec, "You are terse.", [{"role": "user", "content": "Reply with the single word OK."}],
                           max_tokens=20, raise_errors=False)
    if g.error and spec.provider == "gemini" and " 404" in g.error:
        try:
            if gemini_fallback(spec):
                g = providers.generate(spec, "You are terse.", [{"role": "user", "content": "Reply with the single word OK."}],
                                       max_tokens=20, raise_errors=False)
        except Exception as e:  # noqa: BLE001
            print("    Gemini model lookup failed:", str(e)[:160])
    if g.error:
        (failed_cloud if not spec.local else bad).append(name)
        print(f"  FAIL {name:<20} {g.error[:160]}" + ("  -> left out of this run" if not spec.local else ""))
    else:
        print(f"  ok   {name:<20} {g.seconds:5.1f}s  {g.tokens_in:>4} in {g.tokens_out:>4} out  ${g.cost_usd:.6f}  -> {g.text[:30]!r}")
(ROOT / ".runlogs").mkdir(exist_ok=True)
(ROOT / ".runlogs" / "disabled_models.txt").write_text(",".join(failed_cloud))
sys.exit(1 if bad else 0)
