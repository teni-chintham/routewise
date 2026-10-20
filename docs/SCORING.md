# How models are scored and how the router decides

This document defines every number the system reports. Nothing in the router is typed in by hand:
each input is measured, and each measurement can be traced to a saved answer in `results/final/raw.jsonl`.

## 1. The benchmark

400 tasks, 80 per task type, all written by people outside this team and graded by code, not by a model.

| Task type | Source | How an answer is marked correct |
|---|---|---|
| coding | MBPP (Austin et al., 2021), test range | the dataset's own `assert` tests run against the extracted code; all must pass |
| reasoning | GSM8K test set (Cobbe et al., 2021) | the final number equals the gold number |
| search (factual) | Natural Questions open, dev (Lee et al., 2019) | the reply contains one of the gold answers after normalisation (lower case, no punctuation or articles) |
| writing | IFEval (Zhou et al., 2023) | every verifiable instruction in the prompt holds (strict mode: word limits, no commas, bullet counts, JSON, end phrase ...) |
| multimodal | 40 generated images (bar charts, counting, colours, shapes, words) | the reply names the true answer and no other option |

Each type is shuffled with a fixed seed (2026) and split in half:

* **calibration split** (200 tasks, 40 per type): used only to measure each model's scorecard;
* **test split** (200 tasks, 40 per type): used only to evaluate routing.

The router is therefore never graded on the data its numbers came from.

## 2. The scorecard (per model, per task type)

| Field | Definition | Unit |
|---|---|---|
| `n` | calibration tasks attempted | count |
| `correct` | tasks graded correct | count |
| `accuracy` | correct / n | fraction |
| `ci95` | 95% Wilson score interval for accuracy | [low, high] |
| `median_latency_s` | median wall-clock seconds per task, on the machine that ran it | seconds |
| `mean_cost_usd` | mean of (tokens_in x price_in + tokens_out x price_out) / 10^6, token counts from the provider's own usage report | USD per query |
| `eligible` | false when the model cannot take the task (e.g. no image input) | |

### Why the Wilson interval

With 20 tasks, a model that gets 20/20 has not been shown to be perfect. The Wilson interval

    centre = (p + z^2/2n) / (1 + z^2/n)
    half   = z * sqrt(p(1-p)/n + z^2/4n^2) / (1 + z^2/n)        z = 1.96

gives the range of true accuracies consistent with the sample. Examples:
10/10 -> [0.72, 1.00]; 20/20 -> [0.84, 1.00]; 50/100 -> [0.40, 0.60].
Every rate in the report is printed with its n and this interval.

## 3. The Routing Utility Score (RUS)

For a query of task type *i*, every eligible model *m* gets

    RUS(m, i) = w_a * A(m, i)
              + w_c * (1 - min(C(m, i) / C_max, 1))
              + w_l * (1 - min(L(m, i) / L_max, 1))

| Symbol | Meaning | Default |
|---|---|---|
| A(m, i) | **lower bound** of the 95% Wilson interval of m's accuracy on i | measured |
| C(m, i) | m's mean measured cost per query on i (0 for local models) | measured |
| L(m, i) | m's median measured latency on i | measured |
| C_max | the most the user will pay for one query | $0.002 |
| L_max | the longest the user will wait | 30 s |
| w_a, w_c, w_l | user weights, normalised to sum to 1 | 0.6, 0.2, 0.2 |

**Range and meaning.** Each bracket is a fraction in [0, 1], so RUS is in [0, 1]. It reads as
"share of an ideal model": a model that is certainly always right, free and instant scores 1.
A model that costs at least C_max and takes at least L_max can only earn the accuracy term.

**Why the lower bound.** Using the point estimate would let a model with a lucky small sample win.
Using the lower bound means a model is never credited with more accuracy than its evidence supports,
and the credit grows as more calibration data comes in.

**Eligibility.** A model is scored only if it is enabled in `models.yaml`, has a scorecard for this
task type, accepts an image when the query has one, and, when the prompt carries personal memory or
private documents and the privacy policy is `local` (default), runs on this computer.

**Decision.** The highest RUS wins; ties go to the cheaper, then the faster model. Every decision
returns the full table (A, C, L and each weighted term per model, plus the reason any model was
excluded), shown live in the app and by `python -m assistant route "..."`.

### Worked example

Filled in from the final scorecards in `results/final/eval.json` (see the report, Table "RUS worked example").

## 4. How routing is evaluated

On the 100 test tasks, the intent classifier labels each prompt (it has never seen these prompts),
the router picks a model with the calibration scorecards, and that model's graded answer on the task
is the routed system's answer. Because every model answered every test task at temperature 0, any
routing policy can be evaluated on the same answers:

| Policy | What it does |
|---|---|
| always *m* | every query to model *m* |
| routed: balanced | RUS with weights 0.6 / 0.2 / 0.2 |
| routed: quality-first | RUS with weights 1 / 0 / 0 |
| routed: cost-saver | RUS with weights 0.4 / 0.5 / 0.1 |
| oracle (hindsight) | for each task, the cheapest model that got it right; an upper bound no real router can reach |

Reported for each policy: accuracy with 95% Wilson interval, accuracy per task type, total measured cost
per 1,000 queries in USD and INR, median latency, and share of queries answered on the device.

**Significance.** Routed and fixed policies answer the same 200 tasks, so they are compared with the
exact McNemar test on the tasks where exactly one of them is right. "Routing beats model X" is claimed
only when p < 0.05; otherwise the report says the difference is not significant.

**Cost reduction** = 1 - (cost per 1,000 queries, routed) / (cost per 1,000 queries, always the best cloud model),
reported together with the accuracy difference, so a saving is never shown without what it cost in quality.

## 4b. Local-first cascade (assistant/cascade.py)

RUS alone rarely picks a 1-3B local model, because cheap cloud models are far more accurate for almost no
cost. The cascade keeps a local answer only when it can be checked without a reference answer:

| Task type | Check | If it fails |
|---|---|---|
| coding | the local coder's code passes every `assert` line written in the user's own prompt (no assert: skip) | escalate to the RUS choice |
| reasoning | llama3.2-3b and qwen2.5-coder-3b give the same final number | escalate to the RUS choice |
| other | none | RUS choice |

On the test split it kept 57 answers local (54 correct), 28.5% of all queries, at 88.0% overall accuracy.

## 5. Ablation metrics (memory and retrieval)

| Metric | Definition |
|---|---|
| memory recall | share of the 30 recall questions whose answer contains the fact the simulated user stated in an earlier session |
| RAG accuracy | share of the 36 closed-domain questions whose answer contains the gold fact |
| abstention | not correct AND explicitly says it does not know |
| hallucination | neither correct nor an abstention: a wrong answer stated as fact |
| citation rate | answer contains a `[n]` passage marker |
| private egress | prompts carrying memory or private passages that were sent to a cloud model (target 0) |

All with n and 95% Wilson intervals.

## 6. What changed since Review 2

| Review 2 | Now |
|---|---|
| "accuracy" per model was a hand-entered prior (0.95, 0.90 ...) | measured on 200 public calibration tasks (40 per type), lower bound of 95% CI |
| cost and latency were unit-less guesses in [0, 1] | USD per query from token counts; median seconds on this machine |
| "routing gain +0.083" compared the router's score with itself | routing graded on a disjoint test split, against every fixed model, with McNemar tests |
| 50 team-written test tasks, same style as the training data, 10 per type, no intervals ("100%") | 200 public tasks written by others; every rate with n and 95% CI; team prompts kept only as a phrasing robustness check |
| model "roles" named GPT / Claude / Gemini / Sonar played by small local models | real model names; any model added through `models.yaml` |
