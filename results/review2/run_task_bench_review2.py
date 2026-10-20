"""Per-task benchmark: does each kind of task get routed to the right model, and is the answer correct?

50 tasks (10 per intent), each written three ways (direct / casual / verbose).

  Routing   the intent classifier + router are run on all 150 phrasings.
  Answers   every task's direct phrasing is sent to EVERY model in the pool
            (temperature 0), and each answer is graded automatically:
              coding      code is extracted and run against unit tests
              reasoning   final number must match
              search      answer must contain the fact
              writing     instruction constraints are checked (line count, words, JSON keys...)
              multimodal  generated test image + keyword check
  Because generation is deterministic and routing only chooses WHICH model answers,
  the routed system's result on a task is the chosen model's graded answer.

Usage
  python scripts/run_task_bench.py                    # Ollama, all models
  python scripts/run_task_bench.py --provider mock    # offline smoke test
  python scripts/run_task_bench.py --limit 2          # 2 tasks per intent
Outputs results/task_bench.json, results/task_bench_raw.jsonl (resumable), results/tb_*.png
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import subprocess
import sys
import textwrap
import time
from collections import defaultdict
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from assistant import config, providers  # noqa: E402
from assistant.intent import IntentClassifier  # noqa: E402
from assistant.pipeline import BASE_SYSTEM  # noqa: E402
from assistant.router import route  # noqa: E402

EVAL = config.DATA_DIR / "eval"
RAW = config.RESULTS_DIR / "task_bench_raw.jsonl"
OUT = config.RESULTS_DIR / "task_bench.json"
STYLES = ["direct", "casual", "verbose"]
POOL = list(config.MODEL_POOL)

CODE_SUFFIX = "\n\nReturn only the Python code in a single ```python code block."
REASON_SUFFIX = "\n\nThink step by step, then end your reply with a final line of the form 'Answer: <number>'."


# ------------------------------------------------------------------ generation
def gen_ollama(model_key: str, prompt: str, image: str | None) -> str:
    model = config.MODEL_POOL[model_key]["ollama"]
    msg = {"role": "user", "content": prompt}
    if image:
        msg["images"] = [base64.b64encode(Path(image).read_bytes()).decode()]
    r = requests.post(f"{config.OLLAMA_HOST}/api/chat", timeout=600, json={
        "model": model, "stream": False,
        "messages": [{"role": "system", "content": BASE_SYSTEM}, msg],
        "options": {"temperature": 0, "seed": 7, "num_predict": 700}})
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
    return r.json()["message"]["content"].strip()


def generate(provider, model_key, prompt, image):
    t0 = time.time()
    try:
        if provider == "ollama":
            text = gen_ollama(model_key, prompt, image)
        else:
            text, _ = providers.generate(model_key, BASE_SYSTEM, [{"role": "user", "content": prompt}],
                                         image=image, backend=provider)
        err = None
    except Exception as e:  # a model that cannot take images, timeouts...
        text, err = "", str(e)[:200]
    return text, round(time.time() - t0, 2), err


# ------------------------------------------------------------------ grading
def extract_code(text: str) -> str:
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    return max(blocks, key=len) if blocks else text


def grade_code(text, chk):
    code = extract_code(text)
    harness = code + "\n\nimport json as _j\n_res = []\n"
    for args, exp in chk["tests"]:
        harness += f"try:\n    _res.append({chk['func']}(*_j.loads({json.dumps(json.dumps(args))})) == _j.loads({json.dumps(json.dumps(exp))}))\nexcept Exception:\n    _res.append(False)\n"
    harness += "print('RESULT', _j.dumps(_res))\n"
    try:
        p = subprocess.run([sys.executable, "-c", harness], capture_output=True, text=True, timeout=10)
        m = re.search(r"RESULT (\[.*\])", p.stdout)
        res = json.loads(m.group(1)) if m else []
    except subprocess.TimeoutExpired:
        res = []
    ok = bool(res) and all(res)
    return ok, f"{sum(res)}/{len(chk['tests'])} tests passed" if res else "code did not run"


def grade_number(text, chk):
    m = re.search(r"answer\s*[:=]\s*\**\s*(-?[\d,]*\.?\d+)", text, re.I)
    nums = [m.group(1)] if m else re.findall(r"-?\d[\d,]*\.?\d*", text)
    if not nums:
        return False, "no number"
    try:
        v = float(nums[-1].replace(",", "").rstrip("."))
    except ValueError:
        return False, "unparsable"
    return abs(v - chk["value"]) < 1e-6 * max(1, abs(chk["value"])) + 1e-9, f"got {v:g}"


def grade_keywords(text, chk):
    t = text.lower()
    has = (lambda w: re.search(rf"\b{re.escape(w)}\b", t) is not None) if chk.get("word") else (lambda w: w in t)
    if any(has(w) for w in chk.get("exclude_words", [])):
        return False, "contains excluded answer"
    if "all" in chk:
        return all(has(w) for w in chk["all"]), ""
    return any(has(w) for w in chk["any"]), ""


def grade_ab(text, chk):
    """Which of two labels (A/B) the answer picks: a lone letter, or 'A is taller' / 'bar A'."""
    want, other = chk["answer"], "B" if chk["answer"] == "A" else "A"
    t = text.strip().strip(".*\"'")
    if t in ("A", "B"):
        return t == want, ""
    pick = re.search(r"\b(?:bar|label(?:led)?|column)\s+([AB])\b|\b([AB])\b\s+(?:is|appears|looks)\s+taller|taller[^.]*?\b([AB])\b", text)
    if pick:
        got = next(g for g in pick.groups() if g)
        return got == want, f"picked {got}"
    return False, "no clear choice"


def content_lines(text):
    return [ln for ln in text.strip().splitlines() if ln.strip() and not ln.strip().startswith("```")]


def grade_writing(text, chk):
    t = text.strip()
    k = chk["type"]
    if k == "lines":
        n = len(content_lines(t)); return n == chk["n"], f"{n} lines"
    if k == "bullets":
        n = sum(1 for ln in content_lines(t) if re.match(r"^\s*[-*•]\s+", ln)); other = len(content_lines(t)) - n
        return n == chk["n"] and other == 0, f"{n} bullets, {other} other lines"
    if k == "words":
        n = len(re.findall(r"\b[\w'-]+\b", t))
        return chk["min"] <= n <= chk["max"] and chk["include"] in t.lower(), f"{n} words"
    if k == "subject":
        first = content_lines(t)[0] if content_lines(t) else ""
        first = re.sub(r"^(subject\s*:\s*)", "", first.strip().strip('"*'), flags=re.I)
        n = len(first.split())
        return len(content_lines(t)) == 1 and n <= chk["max_words"] and any(w in first.lower() for w in chk["any"]), f"{n} words"
    if k == "lowercase":
        return t == t.lower() and chk["include"] in t, "has capitals" if t != t.lower() else ""
    if k == "tweet":
        return len(t) <= chk["max_chars"] and chk["include"].lower() in t.lower(), f"{len(t)} chars"
    if k == "json":
        m = re.search(r"\{.*\}", t, re.S)
        try:
            obj = json.loads(m.group(0)) if m else None
        except Exception:
            obj = None
        return isinstance(obj, dict) and sorted(obj) == sorted(chk["keys"]), "invalid json" if obj is None else ""
    if k == "tagline":
        line = content_lines(t)[0].strip().strip('"*') if content_lines(t) else ""
        n = len(line.split())
        return len(content_lines(t)) == 1 and n <= chk["max_words"] and not re.search(r"\bbikes?\b", t.lower()), f"{n} words"
    if k == "starts":
        return t.lower().lstrip('"*').startswith(chk["prefix"]), ""
    raise ValueError(k)


def grade(text, chk):
    if not text:
        return False, "no answer"
    k = chk["type"]
    if k == "code":
        return grade_code(text, chk)
    if k == "number":
        return grade_number(text, chk)
    if k == "keywords":
        return grade_keywords(text, chk)
    if k == "ab_choice":
        return grade_ab(text, chk)
    return grade_writing(text, chk)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default=config.PROVIDER)
    ap.add_argument("--limit", type=int, default=None, help="tasks per intent")
    ap.add_argument("--fresh", action="store_true", help="ignore cached answers in task_bench_raw.jsonl")
    args = ap.parse_args()
    if args.provider == "ollama" and not providers.ollama_available():
        sys.exit("Ollama not reachable. Run `ollama serve` or use --provider mock")

    tasks = json.loads((EVAL / "task_bench.json").read_text())
    if args.limit:
        per = defaultdict(int); keep = []
        for t in tasks:
            if per[t["intent"]] < args.limit:
                keep.append(t); per[t["intent"]] += 1
        tasks = keep
    img_dir = EVAL / "images"
    if any(t.get("image") for t in tasks) and not (img_dir / "mm-01.png").exists():
        subprocess.run([sys.executable, str(ROOT / "scripts/make_task_images.py")], check=True)

    # ---------------- 1. routing on every phrasing
    clf = IntentClassifier()
    routing_rows = []
    for t in tasks:
        for style, prompt in zip(STYLES, t["prompts"]):
            p = clf.predict(prompt)
            pred_img = "multimodal" if t.get("image") else p.intent  # pipeline rule: image attached => multimodal
            routing_rows.append({"task": t["id"], "intent": t["intent"], "style": style, "prompt": prompt,
                                 "pred_text_only": p.intent, "confidence": round(p.confidence, 3),
                                 "pred_pipeline": pred_img,
                                 "model_text_only": route(p.intent).chosen,
                                 "model_pipeline": route(pred_img).chosen})
    print(f"routing: {len(routing_rows)} prompts classified")

    # ---------------- 2. every model answers every task (cached / resumable)
    cache = {}
    if RAW.exists() and not args.fresh:
        for ln in RAW.read_text().splitlines():
            r = json.loads(ln)
            if r["provider"] == args.provider:
                cache[(r["task"], r["model"])] = r
    t0 = time.time()
    with RAW.open("a") as fh:
        for i, t in enumerate(tasks):
            prompt = t["prompts"][0] + (CODE_SUFFIX if t["intent"] == "coding" else REASON_SUFFIX if t["intent"] == "reasoning" else "")
            image = str(img_dir / t["image"]) if t.get("image") else None
            for m in POOL:
                if (t["id"], m) in cache:
                    continue
                text, secs, err = generate(args.provider, m, prompt, image)
                ok, note = grade(text, t["check"])
                row = {"provider": args.provider, "task": t["id"], "intent": t["intent"], "model": m,
                       "backend_model": config.MODEL_POOL[m].get(args.provider, args.provider),
                       "answer": text[:1500], "correct": ok, "note": note, "seconds": secs, "error": err}
                cache[(t["id"], m)] = row
                fh.write(json.dumps(row) + "\n"); fh.flush()
                print(f"[{i + 1}/{len(tasks)}] {t['id']:<10} {m:<7} {'PASS' if ok else 'fail'}  {secs:>6.1f}s  {note or err or ''}"[:120])
    print(f"generation done in {time.time() - t0:.0f}s")

    summary = summarise(tasks, routing_rows, cache, args.provider)
    OUT.write_text(json.dumps(summary, indent=2))
    try:
        figures(summary)
    except Exception as e:
        print("figures skipped:", e)
    print_summary(summary)
    print(f"saved -> {OUT}")


def summarise(tasks, routing_rows, cache, provider):
    intents = config.INTENTS
    grid = {(t["id"], m): cache[(t["id"], m)]["correct"] for t in tasks for m in POOL}
    by_intent = defaultdict(list)
    for t in tasks:
        by_intent[t["intent"]].append(t["id"])
    acc = {m: {it: sum(grid[(tid, m)] for tid in by_intent[it]) / len(by_intent[it]) for it in intents if by_intent[it]}
           for m in POOL}
    overall = {m: sum(grid[(t["id"], m)] for t in tasks) / len(tasks) for m in POOL}
    lat = {m: sum(cache[(t["id"], m)]["seconds"] for t in tasks) / len(tasks) for m in POOL}
    errors = {m: sum(1 for t in tasks if cache[(t["id"], m)]["error"]) for m in POOL}

    # routing accuracy
    rr = routing_rows
    def racc(rows, key="pred_text_only"):
        return sum(r[key] == r["intent"] for r in rows) / len(rows) if rows else 0
    routing = {
        "text_only_accuracy": racc(rr),
        "pipeline_accuracy": racc(rr, "pred_pipeline"),
        "by_style": {s: racc([r for r in rr if r["style"] == s]) for s in STYLES},
        "by_intent": {it: racc([r for r in rr if r["intent"] == it]) for it in intents},
        "consistency": sum(len({r["pred_pipeline"] for r in rr if r["task"] == t["id"]}) == 1 for t in tasks) / len(tasks),
        "confusion": {it: {p: sum(1 for r in rr if r["intent"] == it and r["pred_text_only"] == p) for p in intents} for it in intents},
        "misrouted": [r for r in rr if r["pred_pipeline"] != r["intent"]],
    }

    # system-level answer accuracy
    def routed_acc(style_idx, key="model_pipeline"):
        rows = {r["task"]: r for r in rr if r["style"] == STYLES[style_idx]}
        return sum(grid[(t["id"], rows[t["id"]][key])] for t in tasks) / len(tasks)
    true_route = {it: route(it).chosen for it in intents}
    best_per_intent = {it: max(POOL, key=lambda m: (acc[m].get(it, 0), -lat[m])) for it in intents if by_intent[it]}
    systems = {
        "routed (classifier + router)": routed_acc(0),
        "routed, casual phrasing": routed_acc(1),
        "routed, verbose phrasing": routed_acc(2),
        "router with true intent": sum(grid[(t["id"], true_route[t["intent"]])] for t in tasks) / len(tasks),
        "best model per intent (hindsight)": sum(grid[(t["id"], best_per_intent[t["intent"]])] for t in tasks) / len(tasks),
        "any model correct (upper bound)": sum(any(grid[(t["id"], m)] for m in POOL) for t in tasks) / len(tasks),
        **{f"always {m}": overall[m] for m in POOL},
    }
    rows_direct = {r["task"]: r for r in rr if r["style"] == "direct"}
    routed_by_intent = {it: sum(grid[(tid, rows_direct[tid]["model_pipeline"])] for tid in by_intent[it]) / len(by_intent[it])
                        for it in intents if by_intent[it]}
    routed_latency = sum(cache[(t["id"], rows_direct[t["id"]]["model_pipeline"])]["seconds"] for t in tasks) / len(tasks)

    # routing with MEASURED accuracy + latency instead of priors
    maxlat = max(lat.values()) or 1
    measured_pool = {m: {**config.MODEL_POOL[m], "accuracy": {it: round(acc[m].get(it, 0), 3) for it in intents},
                         "latency": round(lat[m] / maxlat, 3)} for m in POOL}
    measured_route = {it: route(it, pool=measured_pool).chosen for it in intents}
    systems["router with measured priors"] = sum(grid[(t["id"], measured_route[rows_direct[t["id"]]["pred_pipeline"]])]
                                                 for t in tasks) / len(tasks)

    return {"provider": provider, "n_tasks": len(tasks), "n_prompts": len(rr),
            "models": {m: config.MODEL_POOL[m].get(provider, provider) for m in POOL},
            "accuracy_by_model_intent": acc, "accuracy_by_model": overall, "mean_latency_s": lat,
            "errors_by_model": errors, "routing": routing, "systems": systems,
            "routed_by_intent": routed_by_intent, "routed_mean_latency_s": routed_latency,
            "prior_route": true_route, "measured_route": measured_route, "best_per_intent": best_per_intent,
            "measured_pool": {m: {"accuracy": measured_pool[m]["accuracy"], "latency": measured_pool[m]["latency"]} for m in POOL},
            "tasks": [{"id": t["id"], "intent": t["intent"], "routed_model": rows_direct[t["id"]]["model_pipeline"],
                       "results": {m: {k: cache[(t["id"], m)][k] for k in ("correct", "note", "seconds", "error", "answer")}
                                   for m in POOL}} for t in tasks]}


def print_summary(s):
    print("\n== routing")
    r = s["routing"]
    print(f"  intent accuracy (text only) {r['text_only_accuracy']:.3f}   with image rule {r['pipeline_accuracy']:.3f}")
    print("  by phrasing:", {k: round(v, 3) for k, v in r["by_style"].items()}, f" consistency {r['consistency']:.3f}")
    print("\n== answer accuracy by model x intent")
    for m, row in s["accuracy_by_model_intent"].items():
        print(f"  {m:<7}", "  ".join(f"{k[:6]} {v:.1f}" for k, v in row.items()), f"| overall {s['accuracy_by_model'][m]:.3f}  lat {s['mean_latency_s'][m]:.1f}s")
    print("\n== systems")
    for k, v in s["systems"].items():
        print(f"  {k:<36} {v:.3f}")


def figures(s):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    intents = config.INTENTS
    R = config.RESULTS_DIR

    M = np.array([[s["accuracy_by_model_intent"][m].get(it, 0) for it in intents] for m in POOL])
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(intents))); ax.set_xticklabels(intents)
    ax.set_yticks(range(len(POOL))); ax.set_yticklabels([f"{m}\n({s['models'][m]})" for m in POOL], fontsize=8)
    for i in range(len(POOL)):
        for j in range(len(intents)):
            ax.text(j, i, f"{M[i, j]:.0%}", ha="center", va="center", color="white" if M[i, j] > .55 else "black", fontsize=10)
    ax.set_title("Measured answer accuracy: model x task type"); fig.colorbar(im, ax=ax, fraction=.03)
    fig.tight_layout(); fig.savefig(R / "tb_accuracy_heatmap.png", dpi=170); plt.close(fig)

    items = [(k, v) for k, v in s["systems"].items() if "phrasing" not in k]
    items.sort(key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(7.5, 4))
    cols = ["#1F4E79" if k.startswith("routed") else "#C0691E" if "always" in k else "#8FAADC" for k, _ in items]
    ax.barh([k for k, _ in items], [v for _, v in items], color=cols)
    for i, (_, v) in enumerate(items):
        ax.text(v + .01, i, f"{v:.0%}", va="center", fontsize=9)
    ax.set_xlim(0, 1.1); ax.set_xlabel("tasks answered correctly (50 tasks)"); ax.set_title("Routed system vs single fixed models")
    fig.tight_layout(); fig.savefig(R / "tb_systems.png", dpi=170); plt.close(fig)

    C = np.array([[s["routing"]["confusion"][a][b] for b in intents] for a in intents])
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.imshow(C, cmap="Greens")
    ax.set_xticks(range(5)); ax.set_xticklabels(intents, rotation=30); ax.set_yticks(range(5)); ax.set_yticklabels(intents)
    ax.set_xlabel("predicted intent"); ax.set_ylabel("true intent")
    for i in range(5):
        for j in range(5):
            ax.text(j, i, C[i, j], ha="center", va="center", color="white" if C[i, j] > C.max() / 2 else "black")
    ax.set_title("Intent routing on 150 unseen prompts (text only)")
    fig.tight_layout(); fig.savefig(R / "tb_confusion.png", dpi=170); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.4))
    st = s["routing"]["by_style"]
    ax.bar(list(st), list(st.values()), color="#1F4E79")
    for i, v in enumerate(st.values()):
        ax.text(i, v + .02, f"{v:.0%}", ha="center")
    ax.set_ylim(0, 1.1); ax.set_ylabel("intent accuracy"); ax.set_title("Routing accuracy by prompt phrasing")
    fig.tight_layout(); fig.savefig(R / "tb_phrasing.png", dpi=170); plt.close(fig)


if __name__ == "__main__":
    main()
