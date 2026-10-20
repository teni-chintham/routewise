"""Phrasing robustness of the intent classifier (no generation, runs in seconds).

The 50 team-written tasks from Review 2, each phrased three ways (direct / casual / verbose), are
classified.  Because these prompts were written by the same team that wrote the training data, this
is a robustness check across phrasing styles, NOT the headline accuracy (that is the public test split
in scripts/run_eval.py).  Every rate is reported with n and its 95% Wilson interval.

Writes results/final/phrasing.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from assistant import config  # noqa: E402
from assistant.intent import IntentClassifier  # noqa: E402
from assistant.scorecard import wilson  # noqa: E402

STYLES = ["direct", "casual", "verbose"]


def rate(bools):
    n, c = len(bools), sum(bools)
    lo, hi = wilson(c, n)
    return {"n": n, "correct": c, "accuracy": round(c / n, 4), "ci95": [round(lo, 4), round(hi, 4)]}


def main():
    tasks = json.loads((config.DATA_DIR / "eval" / "team_prompts_150.json").read_text())
    clf = IntentClassifier()
    rows = []
    for t in tasks:
        for style, prompt in zip(STYLES, t["prompts"]):
            p = clf.predict(prompt)
            pred = "multimodal" if t.get("image") else p.intent
            rows.append({"task": t["id"], "intent": t["intent"], "style": style, "pred_text_only": p.intent,
                         "pred": pred, "confidence": round(p.confidence, 3)})
    out = {
        "note": "team-written prompts; robustness across phrasing, not the headline accuracy",
        "all_text_only": rate([r["pred_text_only"] == r["intent"] for r in rows]),
        "all_with_image_rule": rate([r["pred"] == r["intent"] for r in rows]),
        "by_style": {s: rate([r["pred"] == r["intent"] for r in rows if r["style"] == s]) for s in STYLES},
        "same_route_all_3": rate([len({r["pred"] for r in rows if r["task"] == t["id"]}) == 1 for t in tasks]),
        "rows": rows,
    }
    dst = config.RESULTS_DIR / "final" / "phrasing.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=2))
    for k, v in out["by_style"].items():
        print(f"{k:<8} {v['correct']}/{v['n']} = {v['accuracy']:.1%}  95% CI [{v['ci95'][0]:.1%}, {v['ci95'][1]:.1%}]")
    print("saved ->", dst)


if __name__ == "__main__":
    main()
