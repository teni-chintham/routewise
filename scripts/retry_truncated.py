"""Re-ask answers that were cut off by the output-token budget and graded wrong.

Rule (applied to every model alike): if an answer used the whole output budget for its task type AND was
graded wrong, the cut-off may be the reason, so the task is asked once more with the model's current
budget (models that think before answering get at least `min_output_tokens`). Retried answers are written
to results/final_retry/raw.jsonl; scripts/run_eval.py --analyse-only uses them in place of the originals.

Usage: python scripts/retry_truncated.py [--models a b] [--raw results/final/raw.jsonl ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from assistant import registry  # noqa: E402
from assistant.bench import runner  # noqa: E402
from assistant.bench.graders import grade  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="*")
ap.add_argument("--raw", nargs="*", default=[str(ROOT / "results/final/raw.jsonl"), str(ROOT / "results/final_gemini/raw.jsonl")])
ap.add_argument("--out", default=str(ROOT / "results/final_retry/raw.jsonl"))
a = ap.parse_args()

specs = registry.load()
tasks = {t["id"]: t for t in runner.load_tasks()}
out = Path(a.out)
out.parent.mkdir(parents=True, exist_ok=True)
done = set()
if out.exists():
    done = {(json.loads(l)["task"], json.loads(l)["model"]) for l in out.read_text().splitlines()}
todo = []
for f in a.raw:
    if not Path(f).exists():
        continue
    for ln in Path(f).read_text().splitlines():
        r = json.loads(ln)
        if r.get("error") or r.get("ineligible") or r["model"] not in specs or (a.models and r["model"] not in a.models):
            continue
        if r["tokens_out"] < runner.MAX_TOKENS[r["intent"]]:
            continue
        ok, _ = grade(r["answer"], runner.check_for(tasks[r["task"]]))
        if not ok and (r["task"], r["model"]) not in done:
            todo.append((r["task"], r["model"]))
todo = sorted(set(todo))
print(f"{len(todo)} cut-off wrong answers to re-ask")
with out.open("a") as fh:
    for k, (tid, m) in enumerate(todo, 1):
        r = runner.run_one(specs[m], tasks[tid])
        r["retry_of_truncated"] = True
        fh.write(json.dumps(r) + "\n")
        fh.flush()
        print(f"[{k}/{len(todo)}] {tid:<13} {m:<18} {'PASS' if r['correct'] else 'fail'} {r['tokens_out']:>5} tok  {(r['note'] or r['error'] or '')[:50]}")
