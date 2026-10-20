"""Every results chart in the report and deck, generated from results/final/*.json.

No number is typed in here: each figure reads the JSON written by run_eval.py, run_ablation.py and
run_phrasing_study.py, and its caption footer names that file and the sample size.

Usage: python scripts/make_figures.py [--src results/final] [--out results/final/figures]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# reference palette (dataviz skill), light mode; text never wears a series colour
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#ffffff"
ROUTED, CLOUD, LOCAL, ORACLE = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
SEQ = matplotlib.colors.LinearSegmentedColormap.from_list("seq", ["#f2f6fc", "#9cc1ee", "#2a78d6", "#0b3d7a"])
INTENTS = ["coding", "writing", "search", "reasoning", "multimodal"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.grid": False, "legend.frameon": False,
})


def footer(fig, text):
    fig.text(0.01, -0.02, text, fontsize=7.5, color=MUTED, ha="left", va="top")


def save(fig, out, name):
    fig.savefig(out / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  ", name)


def colour_for(policy, specs):
    if policy.startswith("routed"):
        return ROUTED
    if policy.startswith("oracle"):
        return ORACLE
    m = policy.removeprefix("always ")
    return LOCAL if specs.get(m, {}).get("local") else CLOUD


def pct(x):
    return f"{100 * x:.0f}%"


def pct1(x):
    v = f"{100 * x:.1f}"
    return (v[:-2] if v.endswith(".0") else v) + "%"


SHORT = {"qwen2.5-coder-3b": "qwen-coder 3B", "llama3.2-3b": "llama 3B", "llama3.2-1b": "llama 1B",
         "moondream-1.8b": "moondream", "gemini-3.1-flash-lite": "Gemini",
         "claude-haiku-5.5": "Haiku", "claude-sonnet-5.5": "Sonnet", "gpt-5-mini": "GPT-5 mini"}


# --------------------------------------------------------------------------- eval figures
def fig_scorecards(ev, out):
    models = list(ev["scorecards"])
    A = np.full((len(models), len(INTENTS)), np.nan)
    fig, ax = plt.subplots(figsize=(8.6, 0.62 * len(models) + 1.6))
    for i, m in enumerate(models):
        for j, it in enumerate(INTENTS):
            s = ev["scorecards"][m][it]
            if s.get("eligible"):
                A[i, j] = s["accuracy"]
    ax.imshow(np.ma.masked_invalid(A), cmap=SEQ, vmin=0, vmax=1, aspect="auto")
    for i, m in enumerate(models):
        for j, it in enumerate(INTENTS):
            s = ev["scorecards"][m][it]
            if not s.get("eligible"):
                ax.text(j, i, "no image\ninput", ha="center", va="center", fontsize=7.5, color=MUTED)
                continue
            dark = s["accuracy"] > 0.55
            ax.text(j, i - 0.12, f"{pct(s['accuracy'])}  ({s['correct']}/{s['n']})", ha="center", va="center",
                    fontsize=8.5, fontweight="bold", color="white" if dark else INK)
            ax.text(j, i + 0.2, f"CI {pct(s['ci95'][0])}-{pct(s['ci95'][1])}", ha="center", va="center",
                    fontsize=7, color="white" if dark else INK2)
    ax.set_xticks(range(len(INTENTS)), INTENTS)
    ax.set_yticks(range(len(models)), [f"{m}\n({'local' if ev['models'][m].get('local') else 'cloud'})" for m in models])
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("Measured scorecards: accuracy per task type (calibration split)", loc="left")
    ns = sorted({s["n"] for c in ev["scorecards"].values() for s in c.values() if s.get("eligible")})
    footer(fig, f"n = {'/'.join(map(str, ns))} tasks per cell (MBPP, IFEval, NQ-open, GSM8K, generated images). 95% Wilson intervals. "
                f"Source: results/final/eval.json")
    save(fig, out, "fig_scorecards")


def fig_policies(ev, out):
    pol = ev["policies"]
    names = sorted(pol, key=lambda p: pol[p]["accuracy"])
    fig, ax = plt.subplots(figsize=(8.6, 0.42 * len(names) + 1.4))
    for k, p in enumerate(names):
        s = pol[p]
        c = colour_for(p, ev["models"])
        ax.plot([s["ci95"][0], s["ci95"][1]], [k, k], color=c, lw=2, solid_capstyle="round")
        ax.plot(s["accuracy"], k, "o", ms=8, color=c, mec=SURFACE, mew=2)
        ax.text(s["ci95"][1] + 0.012, k, f"{pct1(s['accuracy'])}  ({s['correct']}/{s['n']})", va="center", fontsize=8.5,
                color=INK)
    ax.set_yticks(range(len(names)), names)
    ax.set_xlim(0, 1.12)
    ax.set_xticks(np.linspace(0, 1, 6), [pct(x) for x in np.linspace(0, 1, 6)])
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("tasks answered correctly (dot) with 95% Wilson interval (line)")
    ax.set_title(f"Routing vs single models on {ev['n_test']} unseen test tasks", loc="left")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=l) for c, l in
               [(ROUTED, "routed (RUS)"), (LOCAL, "always a local model"), (CLOUD, "always the cloud model"),
                (ORACLE, "oracle, hindsight upper bound")]]
    ax.legend(handles=handles, loc="lower right", fontsize=8)
    footer(fig, "Test tasks were never used to build the scorecards. Source: results/final/eval.json")
    save(fig, out, "fig_policies")


def fig_cost_accuracy(ev, out):
    """Full view plus a zoom on the cluster of good, cheap policies (they sit within a few rupees)."""
    pol, rate = ev["policies"], ev["usd_to_inr"]
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    sw = sorted(ev["weight_sweep"], key=lambda s: (s["cost_per_1k_usd"], s["accuracy"]))
    xs = [s["cost_per_1k_usd"] * rate for s in sw]
    ys = [s["accuracy"] for s in sw]
    pts = {p: (s["cost_per_1k_usd"] * rate, s["accuracy"]) for p, s in pol.items()}
    near = {p for p, (x, y) in pts.items() if y >= 0.8 and x <= 30}
    zx = (min(pts[p][0] for p in near) - 1.5, max(pts[p][0] for p in near) + 2.5)
    zy = (min(pts[p][1] for p in near) - 0.02, max(pts[p][1] for p in near) + 0.02)

    def draw(a, which, fs):
        a.plot(xs, ys, "-", color=ROUTED, lw=2, alpha=0.35, zorder=1)
        texts, seen = [], {}
        for p in which:
            x, y = pts[p]
            twin = seen.get((round(x, 3), round(y, 4)))
            if twin is not None:  # identical outcome: one label for both
                twin.set_text(twin.get_text().replace(" (routed)", "") + " = " + p.removeprefix("routed: ") + " (routed)")
                continue
            c = colour_for(p, ev["models"])
            a.plot(x, y, "D" if p.startswith("routed") else "o", ms=8, color=c, mec=SURFACE, mew=1.5, zorder=3)
            texts.append(a.text(x, y, p.removeprefix("always ").removeprefix("routed: ") + ("" if not p.startswith("routed") else " (routed)"),
                                fontsize=fs, color=INK))
            seen[(round(x, 3), round(y, 4))] = texts[-1]
        return texts

    t_main = draw(ax, [p for p in pol if p not in near], 8)
    ax.add_patch(plt.Rectangle((zx[0], zy[0]), zx[1] - zx[0], zy[1] - zy[0], fill=False, ec=INK2, lw=0.8, ls="--"))
    ins = ax.inset_axes([0.36, 0.1, 0.62, 0.55])
    t_ins = draw(ins, sorted(near), 7.5)
    ins.set_xlim(*zx)
    ins.set_ylim(*zy)
    ins.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: pct(v)))
    ins.tick_params(labelsize=7)
    ins.set_facecolor("#fbfbfa")
    for sp in ins.spines.values():
        sp.set_visible(True)
        sp.set_color(INK2)
    ins.set_title("zoom: the cheap, accurate cluster (dashed box)", fontsize=8, loc="left", color=INK2,
                  fontweight="normal")
    ax.annotate("zoomed\nbelow", (zx[1], (zy[0] + zy[1]) / 2), xytext=(zx[1] + 8, 0.72), fontsize=7.5, color=INK2,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    ins.set_xlabel("₹ per 1,000 queries", fontsize=7, color=INK2)
    try:
        from adjustText import adjust_text
        adjust_text(t_main, ax=ax, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6), expand=(1.3, 1.6))
        adjust_text(t_ins, ax=ins, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6), expand=(1.4, 1.8))
    except ImportError:
        pass
    ax.set_xlabel("measured cost per 1,000 queries (₹)")
    ax.set_ylabel("test accuracy")
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_ylim(0, 1)
    ax.set_yticks(np.linspace(0, 1, 6), [pct(y) for y in np.linspace(0, 1, 6)])
    ax.set_title("Accuracy vs cost: what each policy buys", loc="left")
    footer(fig, f"Diamonds: routed policies; dots: one model always. Faint line: the router as the cost weight rises 0 to 0.9.\n"
                f"Cost = provider token counts x list price; local models cost ₹0; USD→INR {rate}. Source: results/final/eval.json")
    save(fig, out, "fig_cost_accuracy")


def fig_by_intent(ev, out):
    pol = ev["policies"]
    bl, bc = ev.get("best_local"), ev.get("best_cloud")
    series = [("routed: balanced", ROUTED, "routed (balanced)")]
    if bl:
        series.append((f"always {bl}", LOCAL, f"always {bl}"))
    if bc:
        series.append((f"always {bc}", CLOUD, f"always {bc}"))
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    w = 0.8 / len(series)
    for k, (p, c, lab) in enumerate(series):
        for j, it in enumerate(INTENTS):
            s = pol[p]["by_intent"].get(it)
            if not s:
                continue
            x = j - 0.4 + w * (k + 0.5)
            ax.bar(x, s["accuracy"], w - 0.03, color=c, label=lab if j == 0 else None)
            ax.plot([x, x], s["ci95"], color=INK2, lw=1)
    ax.set_xticks(range(len(INTENTS)), INTENTS)
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.linspace(0, 1, 6), [pct(y) for y in np.linspace(0, 1, 6)])
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    ax.set_title("Test accuracy per task type (thin line = 95% interval)", loc="left")
    footer(fig, f"n = {ev['n_test'] // 5} test tasks per task type. Source: results/final/eval.json")
    save(fig, out, "fig_by_intent")


def fig_traffic(ev, out):
    pres = [p for p in ev["policies"] if p.startswith("routed")]
    models = list(ev["models"])
    cols = [LOCAL if ev["models"][m].get("local") else CLOUD for m in models]
    fig, ax = plt.subplots(figsize=(8.6, 0.6 * len(pres) + 1.4))
    order = sorted(pres, key=lambda p: ("cascade" in p, p != "routed: balanced", p))[::-1]
    for k, p in enumerate(order):
        used, left = ev["policies"][p]["models_used"], 0
        for m in models:
            v = used.get(m, 0) / ev["n_test"]
            if v <= 0:
                continue
            ax.barh(k, v, left=left, color=LOCAL if ev["models"][m].get("local") else CLOUD, edgecolor=SURFACE, lw=2,
                    height=0.6)
            if v >= 0.05:
                name = m if v >= 0.25 else SHORT.get(m, m)
                ax.text(left + v / 2, k, f"{name}\n{pct(v)}", ha="center", va="center", fontsize=7, color="white")
            left += v
    ax.set_yticks(range(len(order)), order)
    ax.set_xlim(0, 1)
    ax.set_xticks(np.linspace(0, 1, 6), [pct(x) for x in np.linspace(0, 1, 6)])
    ax.set_title("Where the router sent the test queries", loc="left")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=LOCAL, label="local model (free, data stays here)"),
                       plt.Rectangle((0, 0), 1, 1, color=CLOUD, label="cloud model (paid)")], fontsize=8,
              loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    footer(fig, f"n = {ev['n_test']} test queries. Source: results/final/eval.json")
    save(fig, out, "fig_traffic")


def fig_confusion(ev, out):
    C = np.array([[ev["intent"]["confusion_text_only"][a][b] for b in INTENTS] for a in INTENTS])
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    ax.imshow(C, cmap=SEQ)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, C[i, j], ha="center", va="center", color="white" if C[i, j] > C.max() / 2 else INK)
    ax.set_xticks(range(5), INTENTS, rotation=30)
    ax.set_yticks(range(5), INTENTS)
    ax.set_xlabel("predicted task type")
    ax.set_ylabel("true task type")
    for sp in ax.spines.values():
        sp.set_visible(False)
    a = ev["intent"]["text_only_accuracy_on_text_tasks"]
    ax.set_title("Intent classifier on unseen public prompts", loc="left", fontsize=11)
    footer(fig, f"Text only (no image rule). Text tasks: {a['correct']}/{a['n']} = {pct(a['accuracy'])}, "
                f"95% CI {pct(a['ci95'][0])}-{pct(a['ci95'][1])}. Source: results/final/eval.json")
    save(fig, out, "fig_confusion")


# --------------------------------------------------------------------------- other studies
def fig_phrasing(ph, out):
    st = ph["by_style"]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for k, (name, s) in enumerate(st.items()):
        ax.bar(k, s["accuracy"], 0.55, color=ROUTED)
        ax.plot([k, k], s["ci95"], color=INK2, lw=1.2)
        ax.text(k, s["ci95"][1] + 0.02, f"{s['correct']}/{s['n']}\nCI {pct(s['ci95'][0])}-{pct(s['ci95'][1])}",
                ha="center", fontsize=8, color=INK)
    ax.set_xticks(range(len(st)), list(st))
    ax.set_ylim(0, 1.18)
    ax.set_yticks(np.linspace(0, 1, 6), [pct(y) for y in np.linspace(0, 1, 6)])
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title("Intent accuracy by phrasing style (team-written prompts)", loc="left", fontsize=11)
    footer(fig, "Robustness check only: same authors as the training data. Source: results/final/phrasing.json")
    save(fig, out, "fig_phrasing")


def fig_ablation(ab, out):
    vs = [v for v in ["baseline", "A", "B", "C", "D"] if v in ab["variants"]]
    labels = ["Baseline" if v == "baseline" else f"Model {v}" for v in vs]
    panels = [("Memory recall", "memory", "recall_accuracy", "recall_ci95", "higher is better"),
              ("RAG accuracy", "rag", "accuracy", "accuracy_ci95", "higher is better"),
              ("Hallucination", "rag", "hallucination_rate", "hallucination_ci95", "lower is better")]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), sharey=True)
    for ax, (title, b, key, cik, hint) in zip(axes, panels):
        for k, v in enumerate(vs):
            e = ab["variants"][v][b]
            on = (b == "memory" and ab["variants"][v]["flags"]["memory"]) or (b == "rag" and ab["variants"][v]["flags"]["rag"])
            ax.bar(k, e[key], 0.6, color=ROUTED if on else "#b9c9de")
            ax.plot([k, k], e[cik], color=INK2, lw=1)
            if b == "memory":
                num = e["correct"]
            else:
                num = e["counts"]["correct" if key == "accuracy" else "hallucinated"]
            ax.text(k, e[cik][1] + 0.03, f"{num}/{e['n']}", ha="center", fontsize=8, color=INK)
        ax.set_xticks(range(len(vs)), labels, rotation=30, fontsize=8)
        ax.set_title(f"{title}\n", loc="left", fontsize=10.5)
        ax.text(0, 1.02, hint, transform=ax.transAxes, fontsize=7.5, color=INK2)
        ax.set_ylim(0, 1.12)
        ax.yaxis.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
    axes[0].set_yticks(np.linspace(0, 1, 6), [pct(y) for y in np.linspace(0, 1, 6)])
    n_m = ab["variants"][vs[0]]["memory"]["n"]
    n_r = ab["variants"][vs[0]]["rag"]["n"]
    footer(fig, f"Dark bars: the build has the component for that benchmark. labels: answered / asked; n = {n_m} memory / {n_r} RAG questions per build, "
                f"95% Wilson intervals. Local models only. Source: results/final/ablation.json")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, out, "fig_ablation")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "results" / "final"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    src = Path(a.src)
    out = Path(a.out) if a.out else src / "figures"
    out.mkdir(parents=True, exist_ok=True)
    if (src / "eval.json").exists():
        ev = json.loads((src / "eval.json").read_text())
        for f in (fig_scorecards, fig_policies, fig_cost_accuracy, fig_by_intent, fig_traffic, fig_confusion):
            f(ev, out)
    if (src / "phrasing.json").exists():
        fig_phrasing(json.loads((src / "phrasing.json").read_text()), out)
    if (src / "ablation.json").exists():
        fig_ablation(json.loads((src / "ablation.json").read_text()), out)
    print("figures ->", out)


if __name__ == "__main__":
    main()
