"""Run models on the public benchmark, grade every answer, and turn calibration results into scorecards."""
from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

from .. import config, providers, scorecard
from ..pipeline import BASE_SYSTEM
from ..registry import ModelSpec
from .graders import grade

BENCH = config.DATA_DIR / "eval" / "public_bench.json"
MAX_TOKENS = {"writing": 1024, "coding": 700, "reasoning": 700, "search": 200, "multimodal": 100}


def check_for(task: dict) -> dict:
    """The grading spec for a task; factual-answer graders also see the question (to ignore echoed words)."""
    return {**task["check"], "question": task["prompt"]} if task["check"]["type"] == "alias" else task["check"]


def load_tasks(split: str | None = None, path: Path = BENCH) -> list[dict]:
    tasks = json.loads(Path(path).read_text())["tasks"]
    return [t for t in tasks if split is None or t["split"] == split]


def run_one(spec: ModelSpec, task: dict) -> dict:
    image = str(config.DATA_DIR / "eval" / task["image"]) if task.get("image") else None
    row = {"task": task["id"], "intent": task["intent"], "split": task["split"], "model": spec.name,
           "provider": config.PROVIDER_OVERRIDE or spec.provider, "backend_model": spec.model}
    if image and not spec.vision:
        return {**row, "ineligible": True, "correct": False, "note": "no image input", "seconds": 0.0,
                "cost_usd": 0.0, "tokens_in": 0, "tokens_out": 0, "answer": "", "error": None}
    g = providers.generate(spec, BASE_SYSTEM, [{"role": "user", "content": task["prompt"] + task.get("suffix", "")}],
                           image=image, max_tokens=MAX_TOKENS[task["intent"]], raise_errors=False)
    ok, note = grade(g.text, check_for(task)) if not g.error else (False, "error")
    return {**row, "ineligible": False, "correct": ok, "note": note, "seconds": g.seconds, "cost_usd": g.cost_usd,
            "tokens_in": g.tokens_in, "tokens_out": g.tokens_out, "answer": g.text[:8000], "error": g.error}


def run(specs: dict[str, ModelSpec], tasks: list[dict], raw_path: Path, log=print) -> dict[tuple, dict]:
    """Every model answers every task.  Resumable: rows already in raw_path are reused.
    A row that errored (timeout, rate limit, quota) is retried on the next run.

    Local models run one after another (an 8 GB machine holds one model in memory at a time);
    each cloud model runs in its own thread alongside them, since it only waits on the network."""
    import threading
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    cache: dict[tuple, dict] = {}
    if raw_path.exists():
        for ln in raw_path.read_text().splitlines():
            r = json.loads(ln)
            if not r.get("error"):
                cache[(r["task"], r["model"])] = r
    lanes = {"local": [s for s in specs.values() if s.local]}
    for s in specs.values():
        if not s.local:
            lanes[s.name] = [s]
    todo = {lane: [(t, s) for s in ss for t in tasks if (t["id"], s.name) not in cache] for lane, ss in lanes.items()}
    total = sum(len(v) for v in todo.values())
    log(f"{len(cache)} cached answers, {total} to run: " + ", ".join(f"{k} {len(v)}" for k, v in todo.items()))
    lock = threading.Lock()
    t0 = time.time()
    done = {lane: 0 for lane in todo}
    fh = raw_path.open("a")

    def work(lane):
        for t, s in todo[lane]:
            r = run_one(s, t)
            with lock:
                fh.write(json.dumps(r) + "\n")
                fh.flush()
                if not r.get("error"):
                    cache[(t["id"], s.name)] = r
                else:
                    cache.setdefault((t["id"], s.name), r)
                done[lane] += 1
                k, n = done[lane], len(todo[lane])
                eta = (time.time() - t0) / k * (n - k)
                log(f"[{lane} {k}/{n}] {t['id']:<13} {s.name:<18} {'PASS' if r['correct'] else 'fail'} "
                    f"{r['seconds']:>6.1f}s  {(r['note'] or r['error'] or '')[:50]}  lane eta {eta / 60:.0f} min")

    threads = [threading.Thread(target=work, args=(lane,), daemon=True) for lane in todo if todo[lane]]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    fh.close()
    return cache


def scorecards_from(cache: dict[tuple, dict], specs: dict[str, ModelSpec], tasks: list[dict],
                    benchmark: str = "public_bench calibration split") -> dict[str, dict]:
    cards = {}
    for name, spec in specs.items():
        by_intent = defaultdict(list)
        for t in tasks:
            r = cache.get((t["id"], name))
            if r is not None:
                by_intent[t["intent"]].append(r)
        cards[name] = scorecard.build(spec, by_intent, benchmark)
    return cards


def calibrate(spec: ModelSpec, raw_path: Path | None = None, out_dir: Path | None = None, log=print) -> Path:
    """Measure one model on the calibration split and save its scorecard (this machine)."""
    tasks = load_tasks("calibration")
    raw_path = raw_path or config.ensure_home() / "calibration_raw.jsonl"
    cache = run({spec.name: spec}, tasks, raw_path, log)
    card = scorecards_from(cache, {spec.name: spec}, tasks)[spec.name]
    return scorecard.save(card, out_dir)


def regrade(cache: dict[tuple, dict], tasks: list[dict]) -> int:
    """Re-mark every stored answer with the current graders (no generation). Returns how many marks changed."""
    by_id = {t["id"]: t for t in tasks}
    changed = 0
    for key, r in cache.items():
        t = by_id.get(r["task"])
        if t is None or r.get("ineligible") or r.get("error"):
            continue
        ok, note = grade(r["answer"], check_for(t))
        changed += ok != r["correct"]
        r["correct"], r["note"] = ok, note
    return changed
