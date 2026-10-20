"""Deterministic graders.  Every answer is marked correct or wrong by code, never by a model.

    code        extract the Python block, run the dataset's own unit tests in a subprocess
    number      last number in the reply (or 'Answer: <n>') must equal the gold number
    alias       reply must contain one of the gold answers after SQuAD-style normalisation
    ifeval      every verifiable instruction in the prompt must hold (IFEval, strict mode)
    choice      reply must name the right option (vision tasks)

Each grader returns (correct: bool, note: str) so a human can audit any single mark.
"""
from __future__ import annotations

import json
import re
import string
import os
import subprocess
import sys


# --------------------------------------------------------------------------- code (MBPP + team tasks)
def extract_code(text: str) -> str:
    blocks = re.findall(r"```(?:python|py)?[ \t]*\n(.*?)```", text, re.S)
    return max(blocks, key=len) if blocks else text


def grade_code(text: str, chk: dict) -> tuple[bool, str]:
    """The model's code is executed first; if its own example lines crash (a failing assert, input(), a
    print of an undefined name), the functions it defined before the crash are kept and still tested,
    so a correct function is not failed because of a broken usage example."""
    code = extract_code(text)
    harness = [
        "import sys, io",
        "sys.stdin = io.StringIO('')",
        "_ns = {'__name__': '__solution__'}",
        "_setup = " + json.dumps(chk.get("setup", "")),
        "_code = " + json.dumps(code),
        "try:\n    exec(compile(_setup, 'setup', 'exec'), _ns)\nexcept BaseException:\n    pass",
        "try:\n    exec(compile(_code, 'solution', 'exec'), _ns)\nexcept SyntaxError:\n    print('SYNTAX')\nexcept BaseException:\n    pass",
        "_res = []",
    ]
    for t in chk["tests"]:
        harness.append("try:\n    exec(" + json.dumps(t) + ", _ns)\n    _res.append(True)\nexcept BaseException:\n    _res.append(False)")
    harness.append("print('RESULT', sum(_res), len(_res))")
    try:
        # fixed hash seed: code that iterates over a set (e.g. max(set(s), key=s.count)) breaks ties
        # by hash order, so without it the same answer could pass on one run and fail on the next
        env = {**os.environ, "PYTHONHASHSEED": "0"}
        p = subprocess.run([sys.executable, "-c", "\n".join(harness)], capture_output=True, text=True, timeout=15,
                           env=env)
    except subprocess.TimeoutExpired:
        return False, "timed out"
    if "SYNTAX" in p.stdout:
        return False, "syntax error in code"
    m = re.search(r"RESULT (\d+) (\d+)", p.stdout)
    if not m:
        return False, "code did not run"
    passed, total = int(m.group(1)), int(m.group(2))
    return passed == total, f"{passed}/{total} tests passed"


# --------------------------------------------------------------------------- number (GSM8K)
def grade_number(text: str, chk: dict) -> tuple[bool, str]:
    m = re.search(r"answer\s*(?:is)?\s*[:=]?\s*\**\s*\$?\s*(-?[\d,]*\.?\d+)", text, re.I)
    nums = [m.group(1)] if m else re.findall(r"-?\d[\d,]*\.?\d*", text)
    if not nums:
        return False, "no number"
    try:
        v = float(nums[-1].replace(",", "").rstrip("."))
    except ValueError:
        return False, "unparsable"
    gold = float(chk["value"])
    return abs(v - gold) < 1e-6 * max(1.0, abs(gold)), f"got {v:g}, want {gold:g}"


# --------------------------------------------------------------------------- alias (NQ-open)
def normalize(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


_ALIAS_STOP = {"the", "a", "an", "of", "in", "on", "at", "to", "and", "or", "by", "for", "is", "was", "de"}


def _alias_tokens(s: str) -> list[str]:
    s = re.sub(r"\([^)]*\)", " ", s)  # drop parentheticals in gold answers, e.g. "total cost (TC)"
    return [t for t in normalize(s).split() if t not in _ALIAS_STOP]


def grade_alias(text: str, chk: dict) -> tuple[bool, str]:
    """Correct if a gold answer appears in the reply after normalisation (strict, the NQ convention), or,
    unless ASSISTANT_STRICT_ALIAS=1, by word overlap: every number in the gold answer must appear, and
    one-word golds must appear, two-word golds need their head (last) word, longer golds two thirds of their
    words; words already in the question are ignored (so "Missouri River" is not matched by "Mississippi River").  So "May 10, 1940" matches "10 May 1940", "Rachel Bilson" matches "Rachel Sarah Bilson", "Germany"
    matches "Nazi Germany", but "2020" never matches "2010" and "United States" never matches "United Kingdom"."""
    import os
    t = f" {normalize(text)} "
    for g in chk["answers"]:
        if normalize(g) and f" {normalize(g)} " in t:
            return True, f"found '{g}'"
    if os.getenv("ASSISTANT_STRICT_ALIAS") == "1":
        return False, "no gold answer in reply"
    words = set(t.split())
    asked = {w.rstrip("s") for w in normalize(chk.get("question", "")).split()}  # words the question contains prove nothing
    for g in chk["answers"]:
        toks = [x for x in _alias_tokens(g) if x.rstrip("s") not in asked or any(c.isdigit() for c in x)]
        if not toks:
            continue
        nums = [x for x in toks if any(c.isdigit() for c in x)]
        if nums and not all(x in words for x in nums):
            continue
        hit = sum(x in words for x in toks)
        if len(toks) == 1:
            ok = hit == 1
        elif len(toks) == 2:  # e.g. "Nazi Germany": the head word (last) must be there
            ok = toks[-1] in words
        else:                 # e.g. "Rachel Sarah Bilson": at least two thirds of the words
            ok = hit / len(toks) >= 2 / 3
        if ok:
            return True, f"matched {hit}/{len(toks)} words of '{g}'"
    return False, "no gold answer in reply"


# --------------------------------------------------------------------------- choice (vision)
def grade_choice(text: str, chk: dict) -> tuple[bool, str]:
    opts = [str(o) for o in chk.get("options", [])]
    if all(len(o) == 1 for o in opts):  # letter labels: match the capital letter itself, case-sensitive
        t, want = text, str(chk["answer"])
        others = [o for o in opts if o != want]
        has = lambda w: re.search(rf"(?<![A-Za-z]){re.escape(w)}(?![A-Za-z])", t) is not None  # noqa: E731
    else:
        t = " ".join("".join(ch for ch in text.lower() if ch not in string.punctuation).split())
        want = str(chk["answer"]).lower()
        others = [o.lower() for o in opts if o.lower() != want]
        has = lambda w: re.search(rf"\b{re.escape(w)}\b", t) is not None  # noqa: E731
    if not has(want):
        return False, "right option not named"
    named_others = [o for o in others if has(o)]
    if named_others:  # hedging across options is not an answer
        return False, f"also names {named_others[0]}"
    return True, ""


# --------------------------------------------------------------------------- IFEval (strict)
# Re-implementation of the official IFEval checks (google-research/instruction_following_eval,
# Apache-2.0) for the instruction types we sample.  Types needing a language detector or a
# sentence tokenizer are excluded from the benchmark instead of being approximated.
def _rel(n: int, relation: str, target: int) -> bool:
    return n < target if relation == "less than" else n >= target


def _words(t: str) -> list[str]:
    return re.findall(r"\w+", t)


def _check(iid: str, kw: dict, resp: str, prompt: str) -> bool:
    kw = {k: v for k, v in (kw or {}).items() if v is not None}
    r = resp
    if iid == "punctuation:no_comma":
        return "," not in r
    if iid == "length_constraints:number_words":
        return _rel(len(_words(r)), kw["relation"], kw["num_words"])
    if iid == "keywords:forbidden_words":
        return not any(re.search(r"\b" + re.escape(w) + r"\b", r, re.I) for w in kw["forbidden_words"])
    if iid == "detectable_format:number_highlighted_sections":
        n = sum(1 for h in re.findall(r"\*[^\n\*]*\*", r) if h.strip("*").strip())
        n += sum(1 for h in re.findall(r"\*\*[^\n\*]*\*\*", r) if h.removeprefix("**").removesuffix("**").strip())
        return n >= kw["num_highlights"]
    if iid == "keywords:frequency":
        return _rel(len(re.findall(re.escape(kw["keyword"]), r, re.I)), kw["relation"], kw["frequency"])
    if iid == "startend:quotation":
        s = r.strip()
        return len(s) > 1 and s[0] == '"' and s[-1] == '"'
    if iid == "change_case:english_lowercase":
        return r == r.lower()
    if iid == "change_case:english_capital":
        return r == r.upper() and any(c.isalpha() for c in r)
    if iid == "keywords:existence":
        return all(re.search(re.escape(k), r, re.I) for k in kw["keywords"])
    if iid == "detectable_format:title":
        return any(t.strip("<>").strip() for t in re.findall(r"<<[^\n]+>>", r))
    if iid == "detectable_format:number_bullet_lists":
        n = len(re.findall(r"^\s*\*[^\*].*$", r, re.M)) + len(re.findall(r"^\s*-.*$", r, re.M))
        return n == kw["num_bullets"]
    if iid == "detectable_content:number_placeholders":
        return len(re.findall(r"\[.*?\]", r)) >= kw["num_placeholders"]
    if iid == "detectable_content:postscript":
        mk = kw["postscript_marker"].strip()
        low = r.lower()
        if mk.upper() == "P.P.S":
            pat = r"\s*p\.\s?p\.\s?s.*$"
        elif mk.upper() == "P.S.":
            pat = r"\s*p\.\s?s\..*$"
        else:
            pat = r"\s*" + re.escape(mk.lower()) + r".*$"
        return bool(re.findall(pat, low, re.M))
    if iid == "startend:end_checker":
        return r.strip().strip('"').lower().endswith(kw["end_phrase"].strip().lower())
    if iid == "detectable_format:json_format":
        s = r.strip()
        s = re.sub(r"^```(?:json|Json|JSON)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s).strip()
        try:
            json.loads(s)
            return True
        except ValueError:
            return False
    if iid == "keywords:letter_frequency":
        return _rel(r.lower().count(kw["letter"].lower()), kw["let_relation"], kw["let_frequency"])
    if iid == "length_constraints:number_paragraphs":
        paras = re.split(r"\s?\*\*\*\s?", r)
        n = len(paras)
        for i, p in enumerate(paras):
            if not p.strip():
                if i in (0, len(paras) - 1):
                    n -= 1
                else:
                    return False
        return n == kw["num_paragraphs"]
    if iid == "change_case:capital_word_frequency":
        caps = [w for w in _words(r) if w.isupper()]
        return _rel(len(caps), kw["capital_relation"], kw["capital_frequency"])
    if iid == "detectable_format:multiple_sections":
        parts = re.split(r"\s?" + re.escape(kw["section_spliter"]) + r"\s?\d+\s?", r)
        return len(parts) - 1 >= kw["num_sections"]
    if iid == "detectable_format:constrained_response":
        return any(o in r for o in ("My answer is yes.", "My answer is no.", "My answer is maybe."))
    if iid == "combination:repeat_prompt":
        return r.strip().lower().startswith(kw["prompt_to_repeat"].strip().lower())
    if iid == "combination:two_responses":
        parts = [p.strip() for p in r.split("******")]
        parts = [p for p in parts if p]
        return len(parts) == 2 and parts[0] != parts[1]
    raise KeyError(iid)


IFEVAL_SUPPORTED = {
    "punctuation:no_comma", "length_constraints:number_words", "keywords:forbidden_words",
    "detectable_format:number_highlighted_sections", "keywords:frequency", "startend:quotation",
    "change_case:english_lowercase", "change_case:english_capital", "keywords:existence",
    "detectable_format:title", "detectable_format:number_bullet_lists",
    "detectable_content:number_placeholders", "detectable_content:postscript", "startend:end_checker",
    "detectable_format:json_format", "keywords:letter_frequency", "length_constraints:number_paragraphs",
    "change_case:capital_word_frequency", "detectable_format:multiple_sections",
    "detectable_format:constrained_response", "combination:repeat_prompt", "combination:two_responses",
}


def grade_ifeval(text: str, chk: dict) -> tuple[bool, str]:
    failed = [iid for iid, kw in zip(chk["instructions"], chk["kwargs"]) if not _check(iid, kw, text, "")]
    n = len(chk["instructions"])
    return not failed, f"{n - len(failed)}/{n} instructions met" + (f"; failed {failed[0]}" if failed else "")


# --------------------------------------------------------------------------- dispatch
GRADERS = {"code": grade_code, "number": grade_number, "alias": grade_alias,
           "ifeval": grade_ifeval, "choice": grade_choice}


def grade(text: str, chk: dict) -> tuple[bool, str]:
    if not text or not text.strip():
        return False, "no answer"
    return GRADERS[chk["type"]](text, chk)
