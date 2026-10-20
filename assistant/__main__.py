"""Command line.

    python -m assistant models                       list registered models and whether each is calibrated
    python -m assistant add-model NAME --provider ollama --model qwen2.5:7b --local
    python -m assistant calibrate [--model NAME]     measure model(s) on the calibration split -> scorecard
    python -m assistant route "your question" [--image x.png] [--private]   show the RUS table, no generation
    python -m assistant chat [--variant D]           interactive chat in the terminal
    python -m assistant where                        where your data is stored
"""
from __future__ import annotations

import argparse
import sys

from . import config, registry, scorecard


def _table(rows, head):
    w = [max(len(str(r[i])) for r in rows + [head]) for i in range(len(head))]
    print("  ".join(str(h).ljust(w[i]) for i, h in enumerate(head)))
    for r in rows:
        print("  ".join(str(c).ljust(w[i]) for i, c in enumerate(r)))


def cmd_models(a):
    specs, cards = registry.load(include_disabled=True), scorecard.load_all()
    rows = []
    for n, s in specs.items():
        st = cards.get(n)
        rows.append([n, s.provider, s.model, "local" if s.local else "cloud", "yes" if s.vision else "no",
                     f"${s.price_in}/{s.price_out}", "yes" if st else "NO - run calibrate", "" if s.enabled else "disabled"])
    _table(rows, ["name", "provider", "model", "where", "vision", "USD/1M in/out", "calibrated", ""])


def cmd_add(a):
    spec = registry.ModelSpec(name=a.name, provider=a.provider, model=a.model, local=a.local, vision=a.vision,
                              price_in=a.price_in, price_out=a.price_out, base_url=a.base_url,
                              api_key_env=a.api_key_env, rpm=a.rpm)
    registry.add(spec)
    print(f"added {a.name}. Next: python -m assistant calibrate --model {a.name}")


def cmd_calibrate(a):
    from .bench.runner import calibrate
    specs = registry.load()
    names = [a.model] if a.model else list(specs)
    for n in names:
        print(f"== calibrating {n} on the calibration split (200 tasks)")
        print("scorecard ->", calibrate(specs[n]))


def cmd_route(a):
    from .intent import IntentClassifier
    from .router import route
    it = IntentClassifier().predict(a.query)
    intent = "multimodal" if a.image else it.intent
    d = route(intent, image=bool(a.image), private=a.private)
    print(f"intent: {intent} (classifier confidence {it.confidence:.2f})   weights {d.weights}")
    rows = []
    for m in sorted(d.scores, key=d.scores.get, reverse=True):
        i, b = d.inputs[m], d.breakdown[m]
        rows.append([("> " if m == d.chosen else "  ") + m, f"{i['A']:.3f}", f"${i['C_usd']:.6f}", f"{i['L_s']:.1f}s",
                     f"{b['accuracy']:.3f}", f"{b['cost']:.3f}", f"{b['latency']:.3f}", f"{d.scores[m]:.3f}"])
    _table(rows, ["model", "A (CI low)", "C per query", "L median", "w_a*A", "w_c*(1-C/Cmax)", "w_l*(1-L/Lmax)", "RUS"])
    for m, why in d.excluded.items():
        print(f"  excluded {m}: {why}")


def cmd_chat(a):
    from .pipeline import Assistant
    bot = Assistant.variant(a.variant, cascade=not a.no_local_first)
    print(f"Model {a.variant}. Data in {config.HOME}. Ctrl+C to quit.")
    while True:
        try:
            q = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            return
        if q:
            r = bot.chat(q)
            where = "cloud" if r.left_device else "on this computer"
            note = f" | {r.cascade_note}" if r.cascade_note else ""
            print(f"\n[{r.model}, {where}, {r.latency_s:.1f}s, ${r.cost_usd:.6f}{note}]\n{r.answer}")


def cmd_where(a):
    print(f"personal data   {config.HOME}")
    print(f"  memory        {config.MEMORY_DB}")
    print(f"  scorecards    {config.SCORECARD_DIR}  (fallback: {config.REFERENCE_SCORECARDS})")
    print(f"  cost ledger   {config.LEDGER}")
    print(f"  egress audit  {config.EGRESS_LOG}")
    print(f"documents       {config.CORPUS_DIR}  (index {config.INDEX_PATH})")


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m assistant")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("models").set_defaults(fn=cmd_models)
    s = sub.add_parser("add-model")
    s.add_argument("name"); s.add_argument("--provider", required=True); s.add_argument("--model", required=True)
    s.add_argument("--local", action="store_true"); s.add_argument("--vision", action="store_true")
    s.add_argument("--price-in", type=float, default=0.0); s.add_argument("--price-out", type=float, default=0.0)
    s.add_argument("--base-url"); s.add_argument("--api-key-env"); s.add_argument("--rpm", type=int)
    s.set_defaults(fn=cmd_add)
    s = sub.add_parser("calibrate"); s.add_argument("--model"); s.set_defaults(fn=cmd_calibrate)
    s = sub.add_parser("route"); s.add_argument("query"); s.add_argument("--image")
    s.add_argument("--private", action="store_true"); s.set_defaults(fn=cmd_route)
    s = sub.add_parser("chat"); s.add_argument("--variant", default="D", choices=["baseline", "A", "B", "C", "D"])
    s.add_argument("--no-local-first", action="store_true", help="skip the local-first check-then-escalate step")
    s.set_defaults(fn=cmd_chat)
    sub.add_parser("where").set_defaults(fn=cmd_where)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
