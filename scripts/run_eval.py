"""Final evaluation: does routing on MEASURED scorecards beat single models, and what does it cost?

1. Every enabled model answers all 200 public-benchmark tasks (resumable; results/final/raw.jsonl).
2. Scorecards are computed from the CALIBRATION split only (results/final/scorecards/, copied to scorecards/).
3. The intent classifier labels every TEST prompt (it never saw these prompts; they were written by
   the dataset authors, not by this team).
4. Policies are compared on the TEST split only:
      always <model>                every fixed model
      routed: <preset>              classifier + RUS with the calibration scorecards
      oracle                        cheapest model that got the task right (upper bound, hindsight)
   For each: accuracy with 95% Wilson CI, accuracy per intent, measured cost per 1,000 queries
   (USD and INR), median latency, share of queries answered on-device.
5. Paired significance: exact McNemar test, routed vs each fixed model (same 100 test tasks).
6. Weight sweep: accuracy vs cost as the cost weight rises (Pareto frontier).

Usage
  python scripts/run_eval.py                  # all enabled models in models.yaml
  python scripts/run_eval.py --models a b     # subset
  python scripts/run_eval.py --provider mock  # offline smoke run (ASSISTANT_FORCE_PROVIDER=mock)
  python scripts/run_eval.py --analyse-only   # recompute everything from raw.jsonl, no generation
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="*", default=None)
ap.add_argument("--provider", default=None, help="mock = offline smoke run")
ap.add_argument("--out", default=str(ROOT / "results" / "final"))
ap.add_argument("--analyse-only", action="store_true")
ap.add_argument("--no-regrade", action="store_true", help="with --analyse-only: keep stored marks")
ap.add_argument("--no-copy", action="store_true", help="do not copy scorecards to scorecards/")
args = ap.parse_args()
if args.provider:
    os.environ["ASSISTANT_FORCE_PROVIDER"] = args.provider

from assistant import config, providers, registry  # noqa: E402
from assistant.bench import runner  # noqa: E402
from assistant.intent import IntentClassifier  # noqa: E402
from assistant.router import route  # noqa: E402
from assistant.scorecard import wilson  # noqa: E402

OUT = Path(args.out)
RAW = OUT / "raw.jsonl"
PRESETS = {
    "balanced": {"accuracy": 0.6, "cost": 0.2, "latency": 0.2},
    "quality-first": {"accuracy": 1.0, "cost": 0.0, "latency": 0.0},
    "cost-saver": {"accuracy": 0.4, "cost": 0.5, "latency": 0.1},
}


def mcnemar(a: list[bool], b: list[bool]) -> dict:
    """Exact two-sided McNemar test on paired outcomes."""
    b01 = sum(1 for x, y in zip(a, b) if x and not y)
    b10 = sum(1 for x, y in zip(a, b) if y and not x)
    n = b01 + b10
    if n == 0:
        return {"a_only": 0, "b_only": 0, "p_value": 1.0}
    k = min(b01, b10)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {"a_only": b01, "b_only": b10, "p_value": round(p, 4)}


def main():
    specs = registry.load()
    if args.models:
        specs = {k: v for k, v in specs.items() if k in args.models}
    if not config.PROVIDER_OVERRIDE:
        need_ollama = [s for s in specs.values() if s.provider == "ollama"]
        if need_ollama and not args.analyse_only:
            if not providers.ollama_available():
                sys.exit("Ollama not reachable: start the Ollama app or run `ollama serve`.")
            have = providers.ollama_models()
            missing = [s.model for s in need_ollama if s.model not in have and f"{s.model}:latest" not in have]
            if missing:
                sys.exit(f"pull these first: {' '.join('ollama pull ' + m for m in missing)}")
        for s in list(specs.values()):
            if not s.local and not os.getenv(s.api_key_env or ""):
                print(f"skipping {s.name}: {s.api_key_env} not set in .env")
                specs.pop(s.name)
    names = list(specs)
    print("models:", names)

    tasks = runner.load_tasks()
    calib = [t for t in tasks if t["split"] == "calibration"]
    test = [t for t in tasks if t["split"] == "test"]

    if args.analyse_only:
        cache = {}
        extra = [OUT.parent / "final_gemini" / "raw.jsonl", OUT.parent / "final_openai" / "raw.jsonl",
                 OUT.parent / "final_retry" / "raw.jsonl"]  # side runs and budget retries (retries come last and win)
        lines = RAW.read_text().splitlines()
        for f in extra:
            if f.exists() and f.resolve() != RAW.resolve():
                lines += f.read_text().splitlines()
        for ln in lines:
            r = json.loads(ln)
            if r["model"] in specs and (not r.get("error") or (r["task"], r["model"]) not in cache):
                cache[(r["task"], r["model"])] = r
        if not args.no_regrade:
            print("re-graded with current graders:", runner.regrade(cache, tasks), "marks changed")
    else:
        cache = runner.run(specs, tasks, RAW)

    # ------------------------------------------------ 2. scorecards from calibration only
    cards = runner.scorecards_from(cache, specs, calib)
    sc_dir = OUT / "scorecards"
    sc_dir.mkdir(parents=True, exist_ok=True)
    for name, card in cards.items():
        (sc_dir / f"{name}.json").write_text(json.dumps(card, indent=2))
    if not args.no_copy and not config.PROVIDER_OVERRIDE:
        config.REFERENCE_SCORECARDS.mkdir(exist_ok=True)
        for f in sc_dir.glob("*.json"):
            shutil.copy(f, config.REFERENCE_SCORECARDS / f.name)

    # ------------------------------------------------ 3. intent on unseen test prompts
    clf = IntentClassifier()
    pred = {}
    for t in test:
        p = clf.predict(t["prompt"])
        pred[t["id"]] = {"text_only": p.intent, "confidence": round(p.confidence, 3),
                         "pipeline": "multimodal" if t.get("image") else p.intent}
    intents = config.INTENTS
    conf = {a: {b: sum(1 for t in test if t["intent"] == a and pred[t["id"]]["text_only"] == b) for b in intents}
            for a in intents}
    text_tasks = [t for t in test if not t.get("image")]
    intent_eval = {
        "n": len(test),
        "pipeline_accuracy": _acc([pred[t["id"]]["pipeline"] == t["intent"] for t in test]),
        "text_only_accuracy_on_text_tasks": _acc([pred[t["id"]]["text_only"] == t["intent"] for t in text_tasks]),
        "by_intent": {it: _acc([pred[t["id"]]["pipeline"] == it for t in test if t["intent"] == it]) for it in intents},
        "confusion_text_only": conf,
        "misrouted": [{"task": t["id"], "intent": t["intent"], "predicted": pred[t["id"]]["pipeline"],
                       "prompt": t["prompt"][:160]} for t in test if pred[t["id"]]["pipeline"] != t["intent"]],
    }

    # ------------------------------------------------ 4. policies on the test split
    def outcome(t, m):
        r = cache[(t["id"], m)]
        return {"correct": bool(r["correct"]) and not r.get("ineligible"), "cost": r["cost_usd"],
                "seconds": r["seconds"], "local": specs[m].local, "model": m}

    policies: dict[str, list[dict]] = {}
    for m in names:
        policies[f"always {m}"] = [outcome(t, m) for t in test]
    route_choice = {}
    for pname, w in PRESETS.items():
        rows = []
        for t in test:
            d = route(pred[t["id"]]["pipeline"], w, cards=cards, specs=specs, image=bool(t.get("image")))
            route_choice.setdefault(pname, {})[t["id"]] = d.chosen
            rows.append(outcome(t, d.chosen or config.FIXED_MODEL))
        policies[f"routed: {pname}"] = rows
    oracle = []
    for t in test:
        right = [m for m in names if outcome(t, m)["correct"]]
        pick = min(right, key=lambda m: (outcome(t, m)["cost"], outcome(t, m)["seconds"])) if right else \
            min(names, key=lambda m: outcome(t, m)["cost"])
        oracle.append(outcome(t, pick))
    policies["oracle (hindsight)"] = oracle

    # local-first cascade: keep the local answer when a reference-free check passes, else escalate to
    # the balanced RUS choice.  Latency = local attempt(s) + escalation when it happens.
    from assistant import cascade
    if all(m in names for m in (cascade.CODER, *cascade.REASONERS)):
        rows, steps = [], {}
        for t in test:
            it = pred[t["id"]]["pipeline"]
            esc = route_choice["balanced"][t["id"]] or config.FIXED_MODEL
            step = None
            if it == "coding":
                step = cascade.check_code(t["prompt"], cache[(t["id"], cascade.CODER)]["answer"])
                tried = [cascade.CODER]
            elif it == "reasoning":
                step = cascade.check_agreement({m: cache[(t["id"], m)]["answer"] for m in cascade.REASONERS})
                tried = list(cascade.REASONERS)
            if step and step.kept_local:
                o = outcome(t, step.model)
                o["seconds"] = sum(cache[(t["id"], m)]["seconds"] for m in tried)
            else:
                o = outcome(t, esc)
                if step:
                    o["seconds"] += sum(cache[(t["id"], m)]["seconds"] for m in tried)
            steps[t["id"]] = {"intent": it, "kept_local": bool(step and step.kept_local),
                              "reason": step.reason if step else "routed (no cheap check for this task type)"}
            rows.append(o)
        policies["routed: local-first cascade"] = rows
        route_choice["cascade"] = steps
        PRESETS_EVAL = list(PRESETS) + ["local-first cascade"]
    else:
        PRESETS_EVAL = list(PRESETS)

    summary = {}
    for pname, rows in policies.items():
        summary[pname] = _summ(rows, test)

    cloud = [m for m in names if not specs[m].local]
    local = [m for m in names if specs[m].local]
    best_local = max(local, key=lambda m: summary[f"always {m}"]["accuracy"]) if local else None
    best_cloud = max(cloud, key=lambda m: summary[f"always {m}"]["accuracy"]) if cloud else None
    significance = {}
    for pname in PRESETS_EVAL:
        a = [r["correct"] for r in policies[f"routed: {pname}"]]
        significance[pname] = {m: mcnemar(a, [r["correct"] for r in policies[f"always {m}"]]) for m in names}
    savings = {}
    if best_cloud:
        base = summary[f"always {best_cloud}"]
        for pname in PRESETS_EVAL:
            s = summary[f"routed: {pname}"]
            savings[pname] = {
                "vs": best_cloud,
                "cost_reduction": round(1 - s["cost_per_1k_usd"] / base["cost_per_1k_usd"], 4) if base["cost_per_1k_usd"] else None,
                "accuracy_delta_pts": round(100 * (s["accuracy"] - base["accuracy"]), 1),
                "accuracy_retained": round(s["accuracy"] / base["accuracy"], 4) if base["accuracy"] else None,
                "share_on_device": s["share_local"],
            }

    # ------------------------------------------------ 6. weight sweep (cost weight 0 -> 0.9)
    sweep = []
    for k in range(10):
        wc = k / 10
        w = {"accuracy": round(0.9 - wc, 2) if wc <= 0.9 else 0, "cost": wc, "latency": 0.1}
        rows = []
        for t in test:
            d = route(pred[t["id"]]["pipeline"], w, cards=cards, specs=specs, image=bool(t.get("image")))
            rows.append(outcome(t, d.chosen or config.FIXED_MODEL))
        s = _summ(rows, test)
        sweep.append({"weights": w, "accuracy": s["accuracy"], "cost_per_1k_usd": s["cost_per_1k_usd"],
                      "share_local": s["share_local"]})

    result = {
        "models": {m: specs[m].to_dict() for m in names},
        "n_calibration": len(calib), "n_test": len(test), "presets": PRESETS,
        "budgets": {"C_max_usd": config.COST_BUDGET_USD, "L_max_s": config.LATENCY_BUDGET_S},
        "usd_to_inr": config.USD_TO_INR,
        "scorecards": {m: c["intents"] for m, c in cards.items()},
        "intent": intent_eval, "policies": summary, "route_choice": route_choice,
        "best_local": best_local, "best_cloud": best_cloud,
        "significance_vs_fixed": significance, "savings_vs_best_cloud": savings, "weight_sweep": sweep,
        "test_outcomes": {p: [dict(r, task=t["id"], intent=t["intent"]) for r, t in zip(rows, test)]
                          for p, rows in policies.items()},
    }
    (OUT / "eval.json").write_text(json.dumps(result, indent=2))
    _print(result)
    print(f"\nsaved -> {OUT / 'eval.json'}")


def _acc(bools):
    n = len(bools)
    c = sum(bools)
    lo, hi = wilson(c, n)
    return {"n": n, "correct": c, "accuracy": round(c / n, 4) if n else 0.0, "ci95": [round(lo, 4), round(hi, 4)]}


def _summ(rows, test):
    n = len(rows)
    a = _acc([r["correct"] for r in rows])
    cost = sum(r["cost"] for r in rows)
    by_int = defaultdict(list)
    for r, t in zip(rows, test):
        by_int[t["intent"]].append(r["correct"])
    return {**a, "by_intent": {it: _acc(v) for it, v in by_int.items()},
            "cost_total_usd": round(cost, 6), "cost_per_1k_usd": round(1000 * cost / n, 4),
            "cost_per_1k_inr": round(1000 * cost / n * config.USD_TO_INR, 2),
            "median_latency_s": round(statistics.median(r["seconds"] for r in rows), 2),
            "mean_latency_s": round(sum(r["seconds"] for r in rows) / n, 2),
            "share_local": round(sum(r["local"] for r in rows) / n, 4),
            "models_used": dict(sorted(defaultdict(int, {m: sum(1 for r in rows if r["model"] == m)
                                                          for m in {r["model"] for r in rows}}).items()))}


def _print(res):
    print("\n== intent on unseen test prompts:", res["intent"]["pipeline_accuracy"])
    print("\n== scorecards (calibration split): accuracy [95% CI]  median s  $/query")
    for m, ints in res["scorecards"].items():
        print(f"  {m}")
        for it, s in ints.items():
            if s.get("eligible"):
                print(f"    {it:<11} {s['accuracy']:.2f} [{s['ci95'][0]:.2f}, {s['ci95'][1]:.2f}]  "
                      f"{s['median_latency_s']:>6.1f}s  ${s['mean_cost_usd']:.6f}")
    print("\n== test split (n=%d): accuracy [95%% CI] | $ per 1k | on-device" % res["n_test"])
    for p, s in res["policies"].items():
        print(f"  {p:<34} {s['accuracy']:.3f} [{s['ci95'][0]:.2f}, {s['ci95'][1]:.2f}] | "
              f"${s['cost_per_1k_usd']:.3f} | {s['share_local']:.0%}")
    if res["savings_vs_best_cloud"]:
        print("\n== vs always", res["best_cloud"], json.dumps(res["savings_vs_best_cloud"], indent=1))


if __name__ == "__main__":
    main()
