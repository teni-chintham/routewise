"""Scorecards: what a model was MEASURED to do, per task type, on this machine.

A scorecard is produced by `python -m assistant calibrate`.  For every intent it stores

    n, correct          calibration tasks attempted / graded correct
    accuracy            correct / n
    ci95                95% Wilson score interval for the accuracy
    median_latency_s    median wall-clock seconds per task (this machine, this network)
    mean_cost_usd       mean measured cost per task = (tokens_in*price_in + tokens_out*price_out)/1e6
    mean_tokens_in/out  measured from the provider's usage field
    eligible            false when the model cannot take the task (e.g. no image input)

The router reads only these numbers.
"""
from __future__ import annotations

import json
import math
import platform
import statistics
import time
from pathlib import Path

from . import config


def wilson(correct: int, n: int, z: float = config.CONFIDENCE_Z) -> tuple[float, float]:
    """95% Wilson score interval.  Unlike p +/- 1.96*sqrt(p(1-p)/n) it stays inside [0,1] and does
    not collapse to zero width at 0/n or n/n, so 10/10 is reported as [0.72, 1.00], not 'exactly 100%'."""
    if n == 0:
        return 0.0, 0.0
    p = correct / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def summarise(rows: list[dict]) -> dict:
    """rows: graded attempts for one model and one intent, each with correct, seconds, cost_usd, tokens."""
    rows = [r for r in rows if not r.get("ineligible")]
    n = len(rows)
    if n == 0:
        return {"eligible": False, "n": 0}
    c = sum(bool(r["correct"]) for r in rows)
    lo, hi = wilson(c, n)
    lat = [r["seconds"] for r in rows if not r.get("error")] or [r["seconds"] for r in rows]
    return {
        "eligible": True, "n": n, "correct": c, "accuracy": round(c / n, 4), "ci95": [round(lo, 4), round(hi, 4)],
        "median_latency_s": round(statistics.median(lat), 3),
        "p90_latency_s": round(sorted(lat)[max(0, math.ceil(0.9 * len(lat)) - 1)], 3),
        "mean_cost_usd": round(sum(r.get("cost_usd", 0) for r in rows) / n, 8),
        "mean_tokens_in": round(sum(r.get("tokens_in", 0) for r in rows) / n, 1),
        "mean_tokens_out": round(sum(r.get("tokens_out", 0) for r in rows) / n, 1),
        "errors": sum(1 for r in rows if r.get("error")),
    }


def build(spec, rows_by_intent: dict[str, list[dict]], benchmark: str) -> dict:
    return {
        "model": spec.name, "spec": spec.to_dict(), "benchmark": benchmark,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "machine": f"{platform.system()} {platform.machine()} {platform.node()}",
        "intents": {it: summarise(rows_by_intent.get(it, [])) for it in config.INTENTS},
    }


def save(card: dict, directory: Path | None = None) -> Path:
    directory = Path(directory or config.SCORECARD_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    p = directory / f"{card['model']}.json"
    p.write_text(json.dumps(card, indent=2))
    return p


def load_all(directory: Path | None = None) -> dict[str, dict]:
    """Scorecards from `directory` (default: this machine's, falling back to the shipped reference ones)."""
    dirs = [Path(directory)] if directory else [config.SCORECARD_DIR, config.REFERENCE_SCORECARDS]
    cards: dict[str, dict] = {}
    for d in dirs:
        if not d.exists():
            continue
        for f in sorted(d.glob("*.json")):
            card = json.loads(f.read_text())
            cards.setdefault(card["model"], card)  # this machine's card wins over the reference one
    return cards
