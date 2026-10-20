"""Model A, part 2: the Routing Utility Score (RUS).  Full definition: docs/SCORING.md

    RUS(m, i) = w_a * A(m, i)
              + w_c * (1 - min(C(m, i) / C_max, 1))
              + w_l * (1 - min(L(m, i) / L_max, 1))

    A(m, i)  lower bound of the 95% Wilson interval of m's measured accuracy on intent i
    C(m, i)  m's mean measured cost per query on intent i, USD (0 for local models)
    L(m, i)  m's median measured latency on intent i, seconds
    C_max    the most the user will pay for one query      (config.COST_BUDGET_USD)
    L_max    the longest the user will wait for one answer  (config.LATENCY_BUDGET_S)
    w_*      user weights, normalised to sum to 1

Every term is a fraction in [0, 1], so RUS is in [0, 1] and reads as "share of the ideal
model": a free, instant model that is always right scores 1.

Eligible models: enabled in the registry, have a scorecard for intent i (n > 0), accept images if
the query has one, and are local if the prompt carries private context and the policy is "local".
The model with the highest RUS answers; ties go to the cheaper, then faster model.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import config, registry, scorecard


@dataclass
class RouteDecision:
    intent: str
    chosen: str | None
    scores: dict[str, float]
    breakdown: dict[str, dict[str, float]] = field(default_factory=dict)
    inputs: dict[str, dict[str, float]] = field(default_factory=dict)  # A, C, L per model (measured)
    excluded: dict[str, str] = field(default_factory=dict)             # model -> reason
    weights: dict[str, float] = field(default_factory=dict)


def normalise(weights: dict | None) -> dict[str, float]:
    w = dict(weights or config.DEFAULT_WEIGHTS)
    s = sum(w.values()) or 1.0
    return {k: w[k] / s for k in ("accuracy", "cost", "latency")}


def rus(stats: dict, weights: dict, c_max: float, l_max: float) -> tuple[float, dict, dict]:
    a = stats["ci95"][0]
    c = stats["mean_cost_usd"]
    l = stats["median_latency_s"]
    parts = {
        "accuracy": weights["accuracy"] * a,
        "cost": weights["cost"] * (1 - min(c / c_max, 1.0)) if c_max > 0 else 0.0,
        "latency": weights["latency"] * (1 - min(l / l_max, 1.0)) if l_max > 0 else 0.0,
    }
    return round(sum(parts.values()), 4), {k: round(v, 4) for k, v in parts.items()}, {"A": a, "C_usd": c, "L_s": l}


def route(intent: str, weights=None, *, cards: dict | None = None, specs: dict | None = None,
          image: bool = False, private: bool = False, policy: str | None = None,
          c_max: float | None = None, l_max: float | None = None, pool: list[str] | None = None) -> RouteDecision:
    w = normalise(weights)
    cards = cards if cards is not None else scorecard.load_all()
    specs = specs if specs is not None else registry.load()
    policy = policy or config.PRIVATE_CONTEXT_POLICY
    c_max = config.COST_BUDGET_USD if c_max is None else c_max
    l_max = config.LATENCY_BUDGET_S if l_max is None else l_max

    scores, breakdown, inputs, excluded = {}, {}, {}, {}
    for name, spec in specs.items():
        if pool is not None and name not in pool:
            continue
        st = cards.get(name, {}).get("intents", {}).get(intent)
        if not st or not st.get("eligible"):
            excluded[name] = "not calibrated for this task type" if name not in cards else "cannot do this task type"
            continue
        if image and not spec.vision:
            excluded[name] = "no image input"
            continue
        if private and policy == "local" and not spec.local:
            excluded[name] = "private context stays on this computer"
            continue
        s, parts, inp = rus(st, w, c_max, l_max)
        scores[name], breakdown[name], inputs[name] = s, parts, inp
    chosen = max(scores, key=lambda m: (scores[m], -inputs[m]["C_usd"], -inputs[m]["L_s"])) if scores else None
    return RouteDecision(intent, chosen, scores, breakdown, inputs, excluded, w)
