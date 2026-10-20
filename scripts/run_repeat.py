"""Stability check: re-run every cloud model on the test split and compare with the first run.

Local models are deterministic (temperature 0, fixed seed) so they are not repeated. Hosted models can
drift between calls even at temperature 0; this measures by how much.

For each cloud model: accuracy in run 1 and run 2 (with 95% Wilson intervals), the share of tasks whose
mark is the same in both runs, and an exact McNemar test of whether the two runs differ.
Writes results/final/repeat.json (answers in results/final/repeat_raw.jsonl).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from assistant import registry  # noqa: E402
from assistant.bench import runner  # noqa: E402
from assistant.scorecard import wilson  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "results" / "final"


def mcnemar(a, b):
    import math
    b01 = sum(1 for x, y in zip(a, b) if x and not y)
    b10 = sum(1 for x, y in zip(a, b) if y and not x)
    n = b01 + b10
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b01, b10) + 1)) / 2 ** n)
    return {"run1_only": b01, "run2_only": b10, "p_value": round(p, 4)}


def main():
    import os
    specs = {n: s for n, s in registry.load().items() if not s.local and os.getenv(s.api_key_env or "")}
    tasks = runner.load_tasks("test")
    first = {}
    for f in [OUT / "raw.jsonl", OUT.parent / "final_gemini" / "raw.jsonl", OUT.parent / "final_openai" / "raw.jsonl",
              OUT.parent / "final_retry" / "raw.jsonl"]:  # retries of cut-off answers come last and win
        if not f.exists():
            continue
        for ln in f.read_text().splitlines():
            r = json.loads(ln)
            if not r.get("error"):
                first[(r["task"], r["model"])] = r
    second = runner.run(specs, tasks, OUT / "repeat_raw.jsonl")
    res = {}
    for m in specs:
        pairs = [(first[(t["id"], m)], second[(t["id"], m)]) for t in tasks
                 if (t["id"], m) in first and (t["id"], m) in second and not second[(t["id"], m)].get("error")
                 and not first[(t["id"], m)].get("ineligible")]
        if not pairs:
            continue
        a = [bool(x["correct"]) for x, _ in pairs]
        b = [bool(y["correct"]) for _, y in pairs]
        n = len(pairs)
        res[m] = {"n": n,
                  "run1": {"correct": sum(a), "accuracy": round(sum(a) / n, 4), "ci95": [round(v, 4) for v in wilson(sum(a), n)]},
                  "run2": {"correct": sum(b), "accuracy": round(sum(b) / n, 4), "ci95": [round(v, 4) for v in wilson(sum(b), n)]},
                  "same_mark": round(sum(x == y for x, y in zip(a, b)) / n, 4),
                  "same_text": round(sum(x["answer"].strip() == y["answer"].strip() for x, y in pairs) / n, 4),
                  "mcnemar": mcnemar(a, b)}
        print(f"{m:<20} run1 {sum(a)}/{n}  run2 {sum(b)}/{n}  same mark {res[m]['same_mark']:.1%}  "
              f"identical text {res[m]['same_text']:.1%}  p={res[m]['mcnemar']['p_value']}")
    (OUT / "repeat.json").write_text(json.dumps(res, indent=2))
    print("saved ->", OUT / "repeat.json")


if __name__ == "__main__":
    main()
