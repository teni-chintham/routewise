"""Local-first cascade: answer on this computer, check the answer, escalate to the routed model only if
the check fails.  The checks use only what the user gave and what the local models said, never a
reference answer, so they work on any real query:

    coding     the local coder's code is run against the example `assert` lines in the user's own
               prompt; it is kept only if there is at least one such example and every one passes
    reasoning  two different local models answer; the answer is kept only if both give the same
               final number
    other      no cheap check exists, so the query is routed normally (RUS)

Escalation means the query goes to the model the Routing Utility Score picks; private context still
keeps it on local models.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .bench.graders import grade

CODER = "qwen2.5-coder-3b"
REASONERS = ("llama3.2-3b", "qwen2.5-coder-3b")


@dataclass
class CascadeStep:
    kept_local: bool
    model: str | None
    reason: str


def prompt_asserts(prompt: str) -> list[str]:
    """`assert ...` lines the user wrote in the prompt (their own examples)."""
    return [m.group(0).strip() for m in re.finditer(r"\bassert\s[^\n]+", prompt)]


def final_number(text: str) -> float | None:
    m = re.search(r"answer\s*(?:is)?\s*[:=]?\s*\**\s*\$?\s*(-?[\d,]*\.?\d+)", text, re.I)
    nums = [m.group(1)] if m else re.findall(r"-?\d[\d,]*\.?\d*", text)
    try:
        return float(nums[-1].replace(",", "").rstrip(".")) if nums else None
    except ValueError:
        return None


def check_code(prompt: str, answer: str) -> CascadeStep:
    tests = prompt_asserts(prompt)
    if not tests:
        return CascadeStep(False, None, "no example test in the prompt to check against")
    ok, note = grade(answer, {"type": "code", "setup": "", "tests": tests})
    return CascadeStep(ok, CODER if ok else None, f"local code vs the prompt's example: {note}")


def check_agreement(answers: dict[str, str]) -> CascadeStep:
    nums = {m: final_number(a) for m, a in answers.items()}
    vals = list(nums.values())
    if None not in vals and len(set(vals)) == 1:
        return CascadeStep(True, REASONERS[0], f"local models agree on {vals[0]:g}")
    return CascadeStep(False, None, "local models disagree: " + ", ".join(f"{m}={v}" for m, v in nums.items()))
