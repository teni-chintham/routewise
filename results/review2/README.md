# Review 2 results (archived, superseded)

These are the Review 2 numbers, kept for the record. They are superseded by `results/final/` because:

- the router used hand-entered accuracy "priors" (0.95, 0.90 ...) and unit-less cost/latency guesses,
  so the "routing gain" was the router grading itself on its own guesses;
- the 50 test tasks were written by the team, in the same style as the classifier's training data,
  10 per task type, and no confidence intervals were reported (10/10 shows as 100% but its 95% CI is 72-100%).

The final evaluation uses public benchmarks written by other people, measured scorecards, a disjoint
calibration/test split, and 95% Wilson intervals on every rate. See docs/SCORING.md.
