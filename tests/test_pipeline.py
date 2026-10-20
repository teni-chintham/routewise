"""Offline tests: mock provider + tfidf embedder, no Ollama, no API keys.
Run: pytest -q
"""
import json
import os
import sys
import tempfile
from pathlib import Path

_tmp = tempfile.mkdtemp()
os.environ["ASSISTANT_FORCE_PROVIDER"] = "mock"
os.environ["ASSISTANT_EMBEDDER"] = "tfidf"
os.environ["ASSISTANT_HOME"] = _tmp
os.environ["ASSISTANT_INDEX"] = str(Path(_tmp) / "rag_index.pkl")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

from assistant import config, intent, rag, registry  # noqa: E402
from assistant.bench.graders import grade  # noqa: E402
from assistant.memory import Memory, extract_facts  # noqa: E402
from assistant.pipeline import Assistant  # noqa: E402
from assistant.router import route  # noqa: E402
from assistant.scorecard import summarise, wilson  # noqa: E402

SPECS = {
    "local-small": registry.ModelSpec("local-small", "mock", "m", local=True),
    "local-vision": registry.ModelSpec("local-vision", "mock", "m", local=True, vision=True),
    "cloud-big": registry.ModelSpec("cloud-big", "mock", "m", local=False, vision=True, price_in=0.3, price_out=2.5),
}


def _st(acc_correct, n, cost, lat):
    rows = [{"correct": k < acc_correct, "seconds": lat, "cost_usd": cost} for k in range(n)]
    return summarise(rows)


CARDS = {
    "local-small": {"model": "local-small", "intents": {
        "coding": _st(16, 20, 0, 4), "search": _st(4, 20, 0, 2), "writing": _st(10, 20, 0, 6),
        "reasoning": _st(12, 20, 0, 8), "multimodal": {"eligible": False, "n": 0}}},
    "local-vision": {"model": "local-vision", "intents": {
        "coding": _st(2, 20, 0, 3), "search": _st(2, 20, 0, 2), "writing": _st(2, 20, 0, 3),
        "reasoning": _st(0, 20, 0, 3), "multimodal": _st(9, 20, 0, 3)}},
    "cloud-big": {"model": "cloud-big", "intents": {
        "coding": _st(18, 20, 0.0008, 6), "search": _st(16, 20, 0.0002, 3), "writing": _st(17, 20, 0.0012, 8),
        "reasoning": _st(19, 20, 0.0009, 7), "multimodal": _st(18, 20, 0.0003, 4)}},
}


@pytest.fixture(scope="session", autouse=True)
def artifacts():
    if not intent.MODEL_PATH.exists():
        intent.train()
    rag.build_index(embedder="tfidf")


# ---------------------------------------------------------------- scoring
def test_wilson_never_reports_certainty_from_small_samples():
    lo, hi = wilson(10, 10)
    assert hi == 1.0 and 0.70 < lo < 0.73          # 10/10 -> [0.72, 1.00]
    lo, hi = wilson(0, 10)
    assert lo == 0.0 and 0.27 < hi < 0.29
    lo, hi = wilson(50, 100)
    assert 0.40 < lo < 0.41 and 0.59 < hi < 0.60


def test_rus_breakdown_adds_up_and_is_in_unit_interval():
    d = route("coding", cards=CARDS, specs=SPECS)
    for m, s in d.scores.items():
        assert abs(sum(d.breakdown[m].values()) - s) < 1e-3
        assert 0 <= s <= 1
    assert d.inputs["local-small"]["A"] == CARDS["local-small"]["intents"]["coding"]["ci95"][0]


def test_router_uses_measured_numbers():
    assert route("search", {"accuracy": 1, "cost": 0, "latency": 0}, cards=CARDS, specs=SPECS).chosen == "cloud-big"
    assert route("coding", cards=CARDS, specs=SPECS).chosen == "local-small"  # good enough locally and free


def test_cost_weight_moves_traffic_to_free_models():
    q = route("reasoning", {"accuracy": 1, "cost": 0, "latency": 0}, cards=CARDS, specs=SPECS)
    c = route("reasoning", {"accuracy": 0.3, "cost": 0.7, "latency": 0}, cards=CARDS, specs=SPECS)
    assert q.chosen == "cloud-big" and c.chosen == "local-small"


def test_private_context_stays_local():
    d = route("search", {"accuracy": 1, "cost": 0, "latency": 0}, cards=CARDS, specs=SPECS, private=True, policy="local")
    assert d.chosen != "cloud-big" and "cloud-big" in d.excluded
    d2 = route("search", {"accuracy": 1, "cost": 0, "latency": 0}, cards=CARDS, specs=SPECS, private=True, policy="allow")
    assert d2.chosen == "cloud-big"


def test_images_only_go_to_vision_models():
    d = route("multimodal", cards=CARDS, specs=SPECS, image=True)
    assert "local-small" not in d.scores and d.chosen in ("local-vision", "cloud-big")


def test_uncalibrated_model_is_never_picked():
    specs = dict(SPECS, newcomer=registry.ModelSpec("newcomer", "mock", "m", local=True))
    d = route("coding", cards=CARDS, specs=specs)
    assert "newcomer" in d.excluded and d.chosen != "newcomer"


def test_registry_loads_and_validates(tmp_path):
    specs = registry.load()
    assert config.FIXED_MODEL in specs and all(s.price_in == 0 for s in specs.values() if s.local)
    bad = tmp_path / "m.yaml"
    bad.write_text("models:\n  - {name: x, provider: ollama, model: y, local: true, price_in: 1}\n")
    with pytest.raises(ValueError):
        registry.load(bad)


# ---------------------------------------------------------------- graders
def test_graders():
    code = "```python\ndef add(a, b):\n    return a + b\n```"
    assert grade(code, {"type": "code", "setup": "", "tests": ["assert add(1, 2) == 3"]})[0]
    assert not grade("def add(a,b): return a-b", {"type": "code", "setup": "", "tests": ["assert add(1, 2) == 3"]})[0]
    assert grade("so 12,100 in total.\nAnswer: 12,100", {"type": "number", "value": 12100})[0]
    assert grade("It was written by Salman Rushdie.", {"type": "alias", "answers": ["Salman Rushdie"]})[0]
    assert not grade("B or C", {"type": "choice", "answer": "B", "options": ["A", "B", "C"]})[0]
    chk = {"type": "ifeval", "instructions": ["punctuation:no_comma", "change_case:english_lowercase"], "kwargs": [{}, {}]}
    assert grade("all lower and no commas here", chk)[0] and not grade("Has, Commas", chk)[0]


def test_public_bench_is_split_and_disjoint():
    d = json.loads((config.DATA_DIR / "eval" / "public_bench.json").read_text())["tasks"]
    cal = {t["id"] for t in d if t["split"] == "calibration"}
    test = {t["id"] for t in d if t["split"] == "test"}
    assert len(cal) == len(test) and len(cal) >= 100 and not cal & test


# ---------------------------------------------------------------- memory, rag, pipeline
def test_intent_accuracy():
    m = intent.train()
    assert m["accuracy"] >= 0.80


def test_fact_extraction():
    facts = extract_facts("Hi! My name is Arjun. What is 2+2? I use Neovim.")
    assert any("Arjun" in f for f in facts) and any("Neovim" in f for f in facts)
    assert not any("2+2" in f for f in facts)


def test_memory_recall_across_sessions(tmp_path):
    m = Memory(str(tmp_path / "m.db"), embedder="tfidf")
    m.add_turn("s1", "user", "My dog is called Bruno and he is 4.")
    m.add_turn("s2", "user", "I drive a blue Honda City.")
    hits = m.recall("What is my dog's name?", top_k=1)
    assert hits and "Bruno" in hits[0].text


def test_rag_retrieval():
    ps = rag.Retriever().retrieve("What is the rated payload of the Orbit 2?")
    assert ps and any("600 kg" in p.text for p in ps)


def test_model_d_end_to_end_keeps_memory_local(tmp_path):
    a = Assistant.variant("D", embedder="tfidf", memory_db=str(tmp_path / "d.db"), specs=SPECS, cards=CARDS,
                          fixed_model="local-small", ledger=False)
    a.chat("My name is Arjun and I live in Bangalore.")
    a.new_session()
    r = a.chat("What is my name?")
    assert "Arjun" in r.answer and r.decision is not None
    assert r.private_context and not r.left_device
    r2 = a.chat("Who chairs the Helix board?")
    assert r2.passages and any("Meera Iyer" in p.text for p in r2.passages)


def test_baseline_cannot_answer_corpus(tmp_path):
    a = Assistant.variant("baseline", memory_db=str(tmp_path / "b.db"), specs=SPECS, cards=CARDS,
                          fixed_model="local-small", ledger=False)
    assert "Meera" not in a.chat("Who chairs the Helix board?").answer


def test_cascade_checks_need_no_reference_answer():
    from assistant import cascade
    prompt = "Write a function add(a, b). Your code should pass this test:\nassert add(1, 2) == 3"
    assert cascade.check_code(prompt, "```python\ndef add(a, b):\n    return a + b\n```").kept_local
    assert not cascade.check_code(prompt, "```python\ndef add(a, b):\n    return a - b\n```").kept_local
    assert not cascade.check_code("Write add(a, b).", "def add(a,b): return a+b").kept_local  # nothing to check
    assert cascade.check_agreement({"m1": "so Answer: 12", "m2": "Answer: 12.0"}).kept_local
    assert not cascade.check_agreement({"m1": "Answer: 12", "m2": "Answer: 13"}).kept_local


def test_attach_sources_only_when_answer_uses_a_passage():
    from assistant.pipeline import attach_sources
    from assistant.rag import Passage
    ps = [Passage("company.md", 0, "Helix Robotics was founded in Pune in March 2019.", 0.6),
          Passage("board.md", 1, "Meera Iyer chairs the board.", 0.5)]
    out, added = attach_sources("Helix Robotics was founded in Pune in 2019.", ps)
    assert added and out.endswith("Sources: [1] company.md")
    assert attach_sources("I don't know.", ps) == ("I don't know.", False)
    assert attach_sources("It was 2019 [1].", ps) == ("It was 2019 [1].", False)


def test_classifier_training_never_reads_the_public_test_split():
    from assistant.intent import public_calibration_prompts
    import json
    from assistant.intent import PUBLIC_BENCH
    df = public_calibration_prompts()
    assert len(df) > 0
    cal_ids = {g.removeprefix("public-") for g in df["group"]}
    test_ids = {t["id"] for t in json.loads(PUBLIC_BENCH.read_text())["tasks"] if t["split"] == "test"}
    assert not cal_ids & test_ids
