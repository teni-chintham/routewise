"""Streamlit demo.

Tabs
  Chat               talk to any build (baseline, A-D), inspect every decision: intent, the RUS table, memory, passages
  Compare models     one prompt -> every registered model side by side; the router's pick is highlighted
  Compare A/B/C/D    one question -> baseline, A, B, C, D (shared memory profile + document index)
  Scorecards         what each model was measured to do (n, accuracy, 95% CI, latency, cost) and how RUS uses it
  Results            final evaluation and ablation, every number with its sample size and 95% CI
  Privacy & cost     where data lives, the egress audit log and the cost ledger
"""
import json
import sys
import tempfile
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from assistant import config, providers, registry, scorecard  # noqa: E402
from assistant.intent import IntentClassifier  # noqa: E402
from assistant.pipeline import BASE_SYSTEM, VARIANTS, Assistant  # noqa: E402
from assistant.router import route  # noqa: E402

st.set_page_config(page_title="Context-Aware AI Assistant", layout="wide")
st.title("Context-Aware Personal AI Assistant")
st.caption("Routes each query to the model measured best for it. Memory and documents stay on this computer.")

VARIANT_NAMES = {"A": "Model A: routing only", "B": "Model B: memory only", "C": "Model C: RAG only",
                 "D": "Model D: full system", "baseline": "Baseline: fixed model, nothing else"}
SPECS = registry.load()
CARDS = scorecard.load_all()
FINAL = config.RESULTS_DIR / "final"

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Routing weights")
    qp = st.query_params  # ?wa=0.3&wc=0.6&wl=0.1 presets the sliders (used by the screenshot script)
    wa = st.slider("accuracy", 0.0, 1.0, float(qp.get("wa", config.DEFAULT_WEIGHTS["accuracy"])), 0.05)
    wc = st.slider("cost", 0.0, 1.0, float(qp.get("wc", config.DEFAULT_WEIGHTS["cost"])), 0.05)
    wl = st.slider("latency", 0.0, 1.0, float(qp.get("wl", config.DEFAULT_WEIGHTS["latency"])), 0.05)
    weights = {"accuracy": wa, "cost": wc, "latency": wl}
    st.caption(f"C_max = ${config.COST_BUDGET_USD}/query, L_max = {config.LATENCY_BUDGET_S:.0f}s (set in .env)")
    local_first = st.checkbox("Local first: try on this computer, check the answer, escalate only if the check fails",
                              value=True)
    policy = st.radio("Private context (memory, documents)", ["local", "allow"],
                      format_func={"local": "stays on this computer", "allow": "may go to cloud models"}.get)
    embedder = st.selectbox("Embedder (always local)", ["ollama", "tfidf"], index=["ollama", "tfidf"].index(config.EMBEDDER))
    image = st.file_uploader("Image (multimodal)", type=["png", "jpg", "jpeg"])
    st.write("Ollama:", "running" if providers.ollama_available() else "NOT running")
    st.caption("Models: " + ", ".join(f"{n} ({'local' if s.local else 'cloud'})" for n, s in SPECS.items()))
    missing = [n for n in SPECS if n not in CARDS]
    if missing:
        st.warning("Not calibrated (router will skip them): " + ", ".join(missing))


def save_upload():
    if not image:
        return None
    config.ensure_home()
    p = config.HOME / f"upload_{image.name}"
    p.write_bytes(image.getvalue())
    return str(p)


@st.cache_resource
def classifier():
    return IntentClassifier()


def rus_table(d):
    rows = []
    for m in sorted(d.scores, key=d.scores.get, reverse=True):
        i, b = d.inputs[m], d.breakdown[m]
        rows.append({"model": ("→ " if m == d.chosen else "") + m, "A": f"{i['A']:.3f}",
                     "C $/query": f"{i['C_usd']:.6f}", "L s": f"{i['L_s']:.1f}",
                     "accuracy term": f"{b['accuracy']:.3f}", "cost term": f"{b['cost']:.3f}",
                     "latency term": f"{b['latency']:.3f}", "RUS": f"{d.scores[m]:.3f}"})
    st.dataframe(rows, hide_index=True, use_container_width=True)
    st.caption("A = accuracy lower 95% bound on this task type · C = measured cost per query · L = median latency")
    for m, why in d.excluded.items():
        st.caption(f"excluded {m}: {why}")


tab_chat, tab_models, tab_variants, tab_cards, tab_results, tab_priv = st.tabs(
    ["Chat", "Compare models", "Compare A/B/C/D", "Scorecards", "Results", "Privacy & cost"])

# ================================================================ Chat
with tab_chat:
    variant = st.radio("Build", ["D", "A", "B", "C", "baseline"], format_func=VARIANT_NAMES.get, horizontal=True)
    c1, c2 = st.columns([1, 1])
    if c1.button("New session"):
        st.session_state.pop("assistant", None)
        st.session_state.history = []
    if c2.button("Wipe memory"):
        Assistant.variant("B", embedder=embedder).memory.clear()
        st.success("memory cleared")
    key = (variant, embedder, policy, local_first, json.dumps(weights))
    if st.session_state.get("key") != key or "assistant" not in st.session_state:
        st.session_state.assistant = Assistant.variant(variant, embedder=embedder, weights=weights, policy=policy,
                                                       cascade=local_first)
        st.session_state.key = key
        st.session_state.history = []
    a: Assistant = st.session_state.assistant

    for m in st.session_state.history:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
    with st.form("chat", clear_on_submit=True):
        q = st.text_input("Ask something", key="chat_q")
        sent = st.form_submit_button("Send")
    if sent and q:
        with st.spinner("thinking"):
            r = a.chat(q, image=save_upload())
        where = "cloud" if r.left_device else "on this computer"
        st.session_state.history += [{"role": "user", "content": q},
                                     {"role": "assistant", "content": f"{r.answer}\n\n*{r.model} · {where} · "
                                                                      f"{r.latency_s}s · ${r.cost_usd:.6f}*"}]
        st.session_state.last = r
        st.rerun()

    st.subheader("Inspector")
    r = st.session_state.get("last")
    if not r:
        st.info("Ask something to see the routing, memory and retrieval decisions.")
    else:
        left, right = st.columns([1, 2.6])
        with left:
            if r.intent:
                st.markdown(f"**Intent:** `{r.intent.intent}` ({r.intent.confidence:.2f})")
                st.bar_chart(r.intent.probs, height=170)
            st.markdown(f"**This answer:** {r.tokens_in} tokens in, {r.tokens_out} out, ${r.cost_usd:.6f}, "
                        f"{'sent to cloud' if r.left_device else 'never left this computer'}")
        with right:
            if r.cascade_note:
                st.markdown(f"**Local first:** {r.cascade_note}")
            if r.decision:
                st.markdown(f"**Answered by:** `{r.model}`  ·  **RUS pick:** `{r.decision.chosen}`" +
                            ("  · private context, local models only" if r.private_context and policy == "local" else ""))
                st.latex(r"RUS = w_a A + w_c\left(1-\min\left(\tfrac{C}{C_{max}},1\right)\right) + "
                         r"w_l\left(1-\min\left(\tfrac{L}{L_{max}},1\right)\right)")
                rus_table(r.decision)
            else:
                st.markdown(f"**Fixed model:** `{r.model}` (routing off)")
        if a.use_memory:
            st.markdown(f"**Memory hits:** {len(r.memory_hits)}")
            for h in r.memory_hits:
                st.write(f"- ({h.kind}, {h.score:.2f}) {h.text}")
            with st.expander("All stored facts"):
                st.write(a.memory.all_facts())
        if a.use_rag:
            st.markdown(f"**Retrieved passages:** {len(r.passages)}")
            for k, p in enumerate(r.passages, 1):
                with st.expander(f"[{k}] {p.doc}  sim={p.score:.2f}", expanded=k == 1):
                    st.markdown(p.text.replace("#", "\\#"))  # escape: a passage may start with a markdown heading
        with st.expander("System prompt sent to the model"):
            st.code(r.system_prompt)

# ================================================================ Compare models
with tab_models:
    st.markdown("Send **one prompt to every registered model**. The router's choice is highlighted, with the table that picked it.")
    examples = {"Coding": "Write a Python function is_prime(n) that returns True if n is prime.",
                "Writing": "Write a haiku about monsoon rain. Output only the three lines.",
                "Search": "Who wrote the novel Midnight's Children?",
                "Reasoning": "A shirt costs 800 rupees after a 20% discount. What was the original price? End with 'Answer: <number>'.",
                "Multimodal": "Describe what is in this image."}
    ex = st.selectbox("Example prompt", ["(type your own)"] + list(examples))
    prompt = st.text_area("Prompt", value=examples.get(ex, ""), height=90, key=f"cmp_prompt_{ex}")
    if st.button("Run on all models", type="primary") and prompt:
        img = save_upload()
        p = classifier().predict(prompt)
        intent = "multimodal" if img else p.intent
        d = route(intent, weights, cards=CARDS, specs=SPECS, image=bool(img))
        st.markdown(f"**Intent:** `{intent}` ({p.confidence:.2f}) → **router picks `{d.chosen}`**")
        rus_table(d)
        cols = st.columns(len(SPECS))
        for col, (m, spec) in zip(cols, SPECS.items()):
            with col:
                picked = m == d.chosen
                st.markdown(f"#### {'✅ ' if picked else ''}{m}")
                st.caption(f"{spec.model} · {'local' if spec.local else 'cloud'}")
                with st.spinner("generating"):
                    g = providers.generate(spec, BASE_SYSTEM, [{"role": "user", "content": prompt}], image=img,
                                           raise_errors=False)
                ans = g.text if not g.error else f"⚠️ {g.error}"
                (st.success if picked else st.markdown)(ans)
                st.caption(f"{g.seconds}s · ${g.cost_usd:.6f}")

# ================================================================ Compare variants
DEFAULT_PROFILE = ("My name is Arjun and I work at PayLoop as a data engineer.\n"
                   "I live in Koramangala, Bangalore.\nI am allergic to peanuts.\nMy favourite language is Rust.")
with tab_variants:
    st.markdown("Ask **one question through all five builds**. Every build shares the same user profile "
                "(stored in memory for the builds that have memory) and the same document index (used by the builds with RAG).")
    profile = st.text_area("What the user told the assistant in earlier sessions", DEFAULT_PROFILE, height=110)
    vq_examples = ["Where do I work?", "Who chairs the Helix board?", "What is the rated payload of the Orbit 2?",
                   "Write a Python function that reverses a string."]
    vq = st.selectbox("Question", vq_examples + ["(type your own)"])
    if vq == "(type your own)":
        vq = st.text_input("Your question", key="vq_own")
    if st.button("Run through baseline, A, B, C, D", type="primary") and vq:
        db = str(Path(tempfile.mkdtemp()) / "cmp.db")
        cols = st.columns(5)
        for col, v in zip(cols, ["baseline", "A", "B", "C", "D"]):
            with col:
                fl = VARIANTS[v]
                st.markdown(f"#### {VARIANT_NAMES[v].split(':')[0]}")
                st.caption(" · ".join(f"{k} {'on' if fl[k] else 'off'}" for k in ("routing", "memory", "rag")))
                with st.spinner("running"):
                    av = Assistant.variant(v, embedder=embedder, weights=weights, memory_db=db, user_id=f"cmp-{v}",
                                           policy=policy, ledger=False)
                    if av.memory:
                        av.memory.clear()
                        for line in profile.splitlines():
                            if line.strip():
                                av.memory.add_turn("profile", "user", line.strip())
                        av.new_session()
                    t0 = time.time()
                    r = av.chat(vq)
                    st.markdown(r.answer)
                    st.caption(f"{r.model} · {'cloud' if r.left_device else 'local'} · {len(r.memory_hits)} memory hits · "
                               f"{len(r.passages)} passages · {time.time() - t0:.1f}s")

# ================================================================ Scorecards
with tab_cards:
    st.markdown("A **scorecard** is what a model was *measured* to do on the 200-task calibration split "
                "(MBPP, GSM8K, NQ-open, IFEval, generated images). The router uses nothing else. "
                "Accuracy is shown with its 95% Wilson interval; RUS uses the **lower** end, so a model is "
                "never credited with more accuracy than the sample supports.")
    if not CARDS:
        st.info("No scorecards yet. Run: python -m assistant calibrate")
    for m, card in CARDS.items():
        st.markdown(f"**{m}**  ·  {card['spec'].get('model')}  ·  measured on {card.get('machine', '?')}, {card.get('created', '')[:10]}")
        st.dataframe([{"task type": it, "n": str(s.get("n", 0)), "correct": str(s.get("correct", "-")),
                       "accuracy": f"{s['accuracy']:.0%}" if s.get("eligible") else "-",
                       "95% CI": f"{s['ci95'][0]:.0%} to {s['ci95'][1]:.0%}" if s.get("eligible") else "cannot do this",
                       "median s": f"{s['median_latency_s']:.1f}" if s.get("eligible") else "-",
                       "$ per query": f"{s['mean_cost_usd']:.6f}" if s.get("eligible") else "-"}
                      for it, s in card["intents"].items()], hide_index=True)

# ================================================================ Results
with tab_results:
    ev_p, ab_p = FINAL / "eval.json", FINAL / "ablation.json"
    if ev_p.exists():
        ev = json.loads(ev_p.read_text())
        st.subheader(f"Routing vs single models: {ev['n_test']} unseen test tasks")
        st.caption("Test tasks were never used to build the scorecards. ± is the 95% Wilson interval.")
        st.dataframe([{"policy": p, "accuracy": f"{s['accuracy']:.1%}", "95% CI": f"{s['ci95'][0]:.1%} to {s['ci95'][1]:.1%}",
                          "n": s["n"], "$ per 1,000 queries": f"{s['cost_per_1k_usd']:.3f}", "₹ per 1,000": f"{s['cost_per_1k_inr']:.2f}",
                          "median s": s["median_latency_s"], "on this computer": f"{s['share_local']:.0%}"}
                      for p, s in ev["policies"].items()], hide_index=True)
        i = ev["intent"]["pipeline_accuracy"]
        st.markdown(f"Intent classifier on the {i['n']} unseen public prompts: **{i['accuracy']:.1%}** "
                    f"(95% CI {i['ci95'][0]:.1%} to {i['ci95'][1]:.1%})")
    if ab_p.exists():
        ab = json.loads(ab_p.read_text())
        st.subheader("Ablation (local models)")
        st.dataframe([{"build": VARIANT_NAMES[k],
            "memory recall": f"{e['memory']['recall_accuracy']:.1%} [{e['memory']['recall_ci95'][0]:.0%}, {e['memory']['recall_ci95'][1]:.0%}]",
            "RAG accuracy": f"{e['rag']['accuracy']:.1%} [{e['rag']['accuracy_ci95'][0]:.0%}, {e['rag']['accuracy_ci95'][1]:.0%}]",
            "hallucination": f"{e['rag']['hallucination_rate']:.1%}", "abstain": f"{e['rag']['abstention_rate']:.1%}",
            "n": f"{e['memory']['n']} / {e['rag']['n']}"} for k, e in ab["variants"].items() if "memory" in e and "rag" in e],
            hide_index=True)
    figs = sorted((FINAL / "figures").glob("*.png")) if (FINAL / "figures").exists() else []
    for f in figs:
        st.image(str(f), caption=f.stem)
    if not ev_p.exists():
        st.info("Run ./run_final.command (or scripts/run_eval.py then scripts/run_ablation.py) to fill this tab.")

# ================================================================ Privacy & cost
with tab_priv:
    st.markdown(f"""
**Everything personal stays in `{config.HOME}` on this computer:** memory database, document index,
scorecards, cost ledger, egress log. Embeddings are computed locally. A cloud model only ever sees the
prompt for one query, and by default never a prompt that carries your memory or your documents.
""")
    for title, path in [("Egress audit (every request that left this computer)", config.EGRESS_LOG),
                        ("Cost ledger (every answer)", config.LEDGER)]:
        st.subheader(title)
        if path.exists():
            rows = [json.loads(l) for l in path.read_text().splitlines()[-200:]]
            st.dataframe(rows[::-1])
            if path == config.LEDGER and rows:
                tot = sum(r["cost_usd"] for r in rows)
                local = sum(1 for r in rows if not r["left_device"])
                st.metric("Spend over these answers", f"${tot:.4f}  (₹{tot * config.USD_TO_INR:.2f})",
                          f"{local}/{len(rows)} answered on this computer")
        else:
            st.caption("empty")
