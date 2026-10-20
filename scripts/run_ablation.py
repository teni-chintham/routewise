"""Ablation study: run Models baseline / A / B / C / D on two benchmarks, same code, only switches change.

  memory    facts stated over 4 sessions, then 30 recall questions in a fresh session
            recall accuracy = share of questions whose answer contains the gold fact
  rag       36 questions answerable only from data/corpus (fictional company,
            so no model can know the answers without retrieval)
            accuracy      = answer contains the gold fact
            abstention    = not correct AND explicitly says it does not know
            hallucination = neither correct nor an abstention (a wrong answer stated as fact)
            citation      = answer contains a [n] passage marker
  Every rate is reported with its 95% Wilson interval.

  Routing pool: local models only by default (the ablation isolates components; the cloud model's
  value is measured by scripts/run_eval.py).  --privacy-check additionally runs Model D on the memory
  benchmark with the FULL pool and counts how many prompts carrying personal context left the device.

Usage
  python scripts/run_ablation.py --privacy-check          # all variants + privacy check
  python scripts/run_ablation.py --provider mock --embedder tfidf   # offline smoke run
  python scripts/run_ablation.py --variants A D --limit 10
Outputs results/final/ablation.json and .csv (figures: scripts/make_figures.py)
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from assistant import config, providers, registry  # noqa: E402
from assistant.intent import METRICS_PATH  # noqa: E402
from assistant.pipeline import VARIANTS, Assistant  # noqa: E402
from assistant.scorecard import wilson  # noqa: E402


def ci(c: int, n: int) -> list[float]:
    lo, hi = wilson(c, n)
    return [round(lo, 4), round(hi, 4)]


POOL = None  # set in main()

EVAL = config.DATA_DIR / "eval"
ABSTAIN = re.compile(
    r"(don'?t know|do not know|don'?t have|do not have|not (?:in|from|part of|mentioned|contain|available|have)|no information|"
    r"cannot (?:find|determine|answer)|can'?t (?:find|determine|answer)|unable to|not provided|unknown to me)", re.I)


def contains_gold(answer: str, gold: list[str], any_: bool = True) -> bool:
    a = answer.lower()
    hits = [g.lower() in a for g in gold]
    return any(hits) if any_ else all(hits)


# --------------------------------------------------------------------------- benchmarks
def bench_memory(variant: str, embedder: str, limit: int | None, pool=None) -> dict:
    data = json.loads((EVAL / "memory_bench.json").read_text())
    with tempfile.TemporaryDirectory() as td:
        db = str(Path(td) / "mem.db")
        a = Assistant.variant(variant, embedder=embedder, memory_db=db, pool=pool or POOL, ledger=False)
        egress = {"calls": 0, "cloud_calls": 0, "cloud_calls_with_private_context": 0}
        # phase 1: the user talks across several sessions
        t0 = time.time()
        for sess in data["sessions"]:
            a.new_session()
            for msg in sess:
                _count(egress, a.chat(msg))
        # phase 2: brand-new session, ask what it remembers
        a.new_session()
        qs = data["questions"][:limit] if limit else data["questions"]
        rows, correct = [], 0
        for q in qs:
            r = a.chat(q["q"])
            _count(egress, r)
            ok = contains_gold(r.answer, q["gold"])
            correct += ok
            rows.append({"q": q["q"], "answer": r.answer[:200], "correct": ok, "n_hits": len(r.memory_hits),
                         "model": r.model})
        n = len(qs)
        stored = len(a.memory.all_facts()) if a.memory else 0
        return {"recall_accuracy": round(correct / n, 3), "recall_ci95": ci(correct, n), "correct": correct, "n": n,
                "facts_stored": stored, "egress": egress, "seconds": round(time.time() - t0, 1), "rows": rows}


def _count(egress, r):
    egress["calls"] += 1
    egress["cloud_calls"] += r.left_device
    egress["cloud_calls_with_private_context"] += r.left_device and r.private_context


def bench_rag(variant: str, embedder: str, limit: int | None) -> dict:
    data = json.loads((EVAL / "rag_qa.json").read_text())
    qs = data[:limit] if limit else data
    a = Assistant.variant(variant, embedder=embedder, memory_db=str(Path(tempfile.mkdtemp()) / "m.db"),
                          pool=POOL, ledger=False)
    t0 = time.time()
    rows, correct, abstain, cited, by_model = [], 0, 0, 0, 0
    for q in qs:
        r = a.chat(q["q"])
        ok = contains_gold(r.answer, q["gold"], q.get("any", True))
        ab = (not ok) and bool(ABSTAIN.search(r.answer))
        cit = bool(re.search(r"\[\d+\]", r.answer))        # answer points to a source (model or pipeline)
        own = cit and not r.sources_added                     # the model wrote the [n] marker itself
        correct += ok; abstain += ab; cited += cit; by_model += own
        rows.append({"q": q["q"], "answer": r.answer[:200], "correct": ok, "abstained": ab,
                     "cited": cit, "cited_by_model": own, "n_passages": len(r.passages), "model": r.model})
    n = len(qs)
    halluc = n - correct - abstain
    return {"accuracy": round(correct / n, 3), "accuracy_ci95": ci(correct, n),
            "abstention_rate": round(abstain / n, 3), "abstention_ci95": ci(abstain, n),
            "hallucination_rate": round(halluc / n, 3), "hallucination_ci95": ci(halluc, n),
            "citation_rate": round(cited / n, 3), "citation_ci95": ci(cited, n),
            "model_citation_rate": round(by_model / n, 3),
            "counts": {"correct": correct, "abstained": abstain, "hallucinated": halluc, "cited": cited,
                       "cited_by_model": by_model},
            "n": n, "seconds": round(time.time() - t0, 1), "rows": rows}


def bench_routing() -> dict:
    if not METRICS_PATH.exists():
        raise SystemExit("run scripts/train_intent.py first")
    m = json.loads(METRICS_PATH.read_text())
    c = round(m["accuracy"] * m["n_test"])
    return {"intent_accuracy": round(m["accuracy"], 3), "intent_ci95": ci(c, m["n_test"]),
            "macro_f1": round(m["macro_f1"], 3), "n_test": m["n_test"], "n_total": m["n_total"],
            "per_class": {k: round(v["f1-score"], 3) for k, v in m["per_class"].items()}}


# --------------------------------------------------------------------------- main
def main():
    global POOL
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["baseline", "A", "B", "C", "D"])
    ap.add_argument("--provider", default=None, help="mock = offline smoke run")
    ap.add_argument("--embedder", default=config.EMBEDDER)
    ap.add_argument("--limit", type=int, default=None, help="questions per benchmark (quick runs)")
    ap.add_argument("--skip", nargs="*", default=[], choices=["memory", "rag"])
    ap.add_argument("--pool", default="local", choices=["local", "all"], help="models routing may pick")
    ap.add_argument("--privacy-check", action="store_true")
    ap.add_argument("--out", default=str(config.RESULTS_DIR / "final" / "ablation.json"))
    args = ap.parse_args()
    if args.provider:
        import os
        os.environ["ASSISTANT_FORCE_PROVIDER"] = args.provider
        config.PROVIDER_OVERRIDE = args.provider

    specs = registry.load()
    POOL = [m for m, s in specs.items() if s.local] if args.pool == "local" else list(specs)
    if not config.PROVIDER_OVERRIDE and not providers.ollama_available():
        sys.exit("Ollama not reachable. Start the Ollama app, or use --provider mock --embedder tfidf")

    results = {"provider": config.PROVIDER_OVERRIDE or "registry", "embedder": args.embedder,
               "fixed_model": config.FIXED_MODEL, "pool": POOL, "routing": bench_routing(), "variants": {}}
    print(f"pool {POOL}; fixed model {config.FIXED_MODEL}")
    for v in args.variants:
        flags = VARIANTS[v]
        print(f"\n== Model {v}  {flags}")
        entry = {"flags": flags}
        if "memory" not in args.skip:
            entry["memory"] = bench_memory(v, args.embedder, args.limit)
            m = entry["memory"]
            print(f"   memory recall  {m['recall_accuracy']:.3f} {m['recall_ci95']}  ({m['seconds']}s)")
        if "rag" not in args.skip:
            entry["rag"] = bench_rag(v, args.embedder, args.limit)
            r = entry["rag"]
            print(f"   rag accuracy   {r['accuracy']:.3f}  hallucination {r['hallucination_rate']:.3f}  "
                  f"abstain {r['abstention_rate']:.3f}  cited {r['citation_rate']:.3f}  ({r['seconds']}s)")
        results["variants"][v] = entry

    if args.privacy_check:
        full = list(specs)
        print(f"\n== privacy check: Model D, memory benchmark, full pool {full}")
        m = bench_memory("D", args.embedder, args.limit, pool=full)
        results["privacy_check"] = {"pool": full, "policy": config.PRIVATE_CONTEXT_POLICY,
                                    "recall_accuracy": m["recall_accuracy"], "recall_ci95": m["recall_ci95"],
                                    "egress": m["egress"]}
        print("   ", results["privacy_check"])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    write_csv(results, out.with_suffix(".csv"))
    print(f"\nsaved -> {out}")


def write_csv(results: dict, path: Path):
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "routing", "memory", "rag", "memory_recall", "rag_accuracy", "abstention_rate",
                    "hallucination_rate", "citation_rate"])
        for v, e in results["variants"].items():
            fl = e["flags"]
            m, r = e.get("memory", {}), e.get("rag", {})
            w.writerow([v, fl["routing"], fl["memory"], fl["rag"], m.get("recall_accuracy", ""),
                        r.get("accuracy", ""), r.get("abstention_rate", ""), r.get("hallucination_rate", ""),
                        r.get("citation_rate", "")])


if __name__ == "__main__":
    main()
