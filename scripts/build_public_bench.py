"""Build the public benchmark: 40 tasks per intent, written by people outside this team.

    coding      MBPP (Austin et al. 2021, CC-BY-4.0)             graded by the dataset's own unit tests
    reasoning   GSM8K test set (Cobbe et al. 2021, MIT)           graded by the exact final number
    search      Natural Questions open, dev (Lee et al. 2019,     graded by gold answer aliases
                CC-BY-SA-3.0)
    writing     IFEval (Zhou et al. 2023, Apache-2.0)             graded by verifiable instructions (strict)
    multimodal  generated images (charts, counting, colours,      graded by known ground truth
                shapes, text), seed-fixed

Each intent is shuffled with a fixed seed and split in half (40 / 40 at the default 80 per type):
    calibration  used ONLY to measure each model's scorecard
    test         used ONLY to evaluate routing (never seen by the scorecards)

Usage:  python scripts/build_public_bench.py [--per-intent 40] [--src DIR]
Writes data/eval/public_bench.json and data/eval/vision/*.png
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from assistant.bench.graders import IFEVAL_SUPPORTED  # noqa: E402

RAW = "https://raw.githubusercontent.com"
SOURCES = {
    "gsm8k_test.jsonl": f"{RAW}/openai/grade-school-math/master/grade_school_math/data/test.jsonl",
    "mbpp.jsonl": f"{RAW}/google-research/google-research/master/mbpp/mbpp.jsonl",
    "ifeval.jsonl": f"{RAW}/google-research/google-research/master/instruction_following_eval/data/input_data.jsonl",
    "nq_dev.jsonl": f"{RAW}/google-research-datasets/natural-questions/master/nq_open/NQ-open.dev.jsonl",
}
OUT = ROOT / "data" / "eval" / "public_bench.json"
VISION_DIR = ROOT / "data" / "eval" / "vision"
SEED = 2026

CODE_SUFFIX = "\n\nReturn only the Python code in a single ```python code block."
REASON_SUFFIX = "\n\nThink step by step, then end your reply with a final line of the form 'Answer: <number>'."
SEARCH_SUFFIX = "\n\nAnswer in one short sentence."


def load(src: Path, name: str) -> list[dict]:
    p = src / name
    if not p.exists():
        src.mkdir(parents=True, exist_ok=True)
        print("downloading", SOURCES[name])
        urllib.request.urlretrieve(SOURCES[name], p)
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def coding(src, rng, n):
    rows = [r for r in load(src, "mbpp.jsonl") if 11 <= r["task_id"] <= 510]  # official MBPP test range
    out = []
    for r in rng.sample(rows, n):
        prompt = f"{r['text']} Your code should pass this test:\n{r['test_list'][0]}"
        out.append({"id": f"mbpp-{r['task_id']}", "intent": "coding", "source": "MBPP", "prompt": prompt,
                    "suffix": CODE_SUFFIX,
                    "check": {"type": "code", "setup": r["test_setup_code"], "tests": r["test_list"]}})
    return out


def reasoning(src, rng, n):
    rows = load(src, "gsm8k_test.jsonl")
    idx = rng.sample(range(len(rows)), n)
    out = []
    for i in idx:
        r = rows[i]
        gold = float(r["answer"].split("####")[-1].strip().replace(",", ""))
        out.append({"id": f"gsm8k-{i}", "intent": "reasoning", "source": "GSM8K", "prompt": r["question"],
                    "suffix": REASON_SUFFIX, "check": {"type": "number", "value": gold}})
    return out


def search(src, rng, n):
    rows = load(src, "nq_dev.jsonl")
    idx = rng.sample(range(len(rows)), n)
    out = []
    for i in idx:
        r = rows[i]
        q = r["question"].strip()
        q = q[0].upper() + q[1:] + ("" if q.endswith("?") else "?")
        out.append({"id": f"nq-{i}", "intent": "search", "source": "NQ-open", "prompt": q, "suffix": SEARCH_SUFFIX,
                    "check": {"type": "alias", "answers": r["answer"]}})
    return out


def writing(src, rng, n):
    rows = []
    for r in load(src, "ifeval.jsonl"):
        if not set(r["instruction_id_list"]) <= IFEVAL_SUPPORTED:
            continue
        # keep generation time bounded on an 8 GB laptop: skip essays longer than 400 words
        if any(k and k.get("relation") == "at least" and (k.get("num_words") or 0) > 400 for k in r["kwargs"]):
            continue
        rows.append(r)
    out = []
    for r in rng.sample(rows, n):
        out.append({"id": f"ifeval-{r['key']}", "intent": "writing", "source": "IFEval", "prompt": r["prompt"],
                    "suffix": "", "check": {"type": "ifeval", "instructions": r["instruction_id_list"],
                                            "kwargs": r["kwargs"]}})
    return out, len(rows)


# --------------------------------------------------------------------------- vision
COLORS = {"red": (220, 40, 40), "green": (40, 160, 70), "blue": (40, 80, 220), "yellow": (240, 200, 30),
          "orange": (245, 130, 20), "purple": (140, 60, 180)}
WORDS = ["APPLE", "RIVER", "PLANET", "CASTLE", "GARDEN", "ROCKET", "SILVER", "TIGER", "WINDOW", "MARKET"]


def _font(size):
    from PIL import ImageFont
    for f in ("/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/Library/Fonts/Arial.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def vision(rng, n):
    from PIL import Image, ImageDraw
    VISION_DIR.mkdir(parents=True, exist_ok=True)
    kinds = ["bars", "count", "color", "shape", "word"]
    out = []
    for i in range(n):
        kind = kinds[i % len(kinds)]
        img = Image.new("RGB", (448, 448), "white")
        d = ImageDraw.Draw(img)
        name = f"vis-{i + 1:02d}"
        if kind == "bars":
            labels = rng.sample(list("ABCDEFGH"), 4)
            heights = rng.sample(range(80, 360, 20), 4)
            for k, (lab, h) in enumerate(zip(labels, heights)):
                x = 40 + k * 100
                d.rectangle([x, 400 - h, x + 60, 400], fill=(60, 90, 160))
                d.text((x + 20, 408), lab, fill="black", font=_font(28))
            ans = labels[heights.index(max(heights))]
            q = "Which bar in this chart is the tallest? Reply with its letter label only."
            chk = {"type": "choice", "answer": ans, "options": labels}
        elif kind == "count":
            k = rng.randint(2, 7)
            cells = rng.sample([(x, y) for x in range(4) for y in range(4)], k)
            for cx, cy in cells:
                x, y = 30 + cx * 105, 30 + cy * 105
                d.ellipse([x, y, x + 70, y + 70], fill=(40, 80, 220))
            q = "How many circles are in this image? Reply with a number only."
            chk = {"type": "number", "value": k}
        elif kind == "color":
            c = rng.choice(list(COLORS))
            d.rectangle([90, 90, 358, 358], fill=COLORS[c])
            q = "What colour is the square in this image? Reply with one word."
            chk = {"type": "choice", "answer": c, "options": list(COLORS)}
        elif kind == "shape":
            s = rng.choice(["circle", "square", "triangle"])
            if s == "circle":
                d.ellipse([90, 90, 358, 358], fill=(30, 30, 30))
            elif s == "square":
                d.rectangle([90, 90, 358, 358], fill=(30, 30, 30))
            else:
                d.polygon([(224, 70), (380, 370), (68, 370)], fill=(30, 30, 30))
            q = "What shape is shown in this image? Reply with one word."
            chk = {"type": "choice", "answer": s, "options": ["circle", "square", "triangle"]}
        else:
            w = rng.choice(WORDS)
            d.text((40, 180), w, fill="black", font=_font(72))
            q = "What word is written in this image? Reply with the word only."
            chk = {"type": "alias", "answers": [w]}
        img.save(VISION_DIR / f"{name}.png")
        out.append({"id": name, "intent": "multimodal", "source": "generated", "prompt": q, "suffix": "",
                    "image": f"vision/{name}.png", "check": chk})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-intent", type=int, default=80)
    ap.add_argument("--src", default=str(ROOT / "data" / "eval" / "_downloads"))
    args = ap.parse_args()
    n, src = args.per_intent, Path(args.src)

    tasks, notes = [], {}
    for intent, fn in [("coding", coding), ("reasoning", reasoning), ("search", search), ("writing", writing)]:
        rng = random.Random(f"{SEED}-{intent}")
        res = fn(src, rng, n)
        if isinstance(res, tuple):
            res, notes["ifeval_eligible_prompts"] = res
        tasks += res
    tasks += vision(random.Random(f"{SEED}-multimodal"), n)

    # 50 / 50 calibration / test split inside each intent, seed-fixed
    for intent in sorted({t["intent"] for t in tasks}):
        group = [t for t in tasks if t["intent"] == intent]
        random.Random(f"{SEED}-split-{intent}").shuffle(group)
        for k, t in enumerate(group):
            t["split"] = "calibration" if k < len(group) // 2 else "test"

    meta = {"seed": SEED, "per_intent": n, "split": "50/50 calibration/test within each intent",
            "sources": {"coding": "MBPP test range 11-510", "reasoning": "GSM8K test",
                        "search": "NQ-open dev", "writing": "IFEval (supported instruction types)",
                        "multimodal": "generated, scripts/build_public_bench.py"}, **notes}
    OUT.write_text(json.dumps({"meta": meta, "tasks": tasks}, indent=1))
    by = {}
    for t in tasks:
        by.setdefault((t["intent"], t["split"]), 0)
        by[(t["intent"], t["split"])] += 1
    print(json.dumps({f"{k[0]}/{k[1]}": v for k, v in sorted(by.items())}, indent=1))
    print(f"wrote {len(tasks)} tasks -> {OUT}")


if __name__ == "__main__":
    main()
