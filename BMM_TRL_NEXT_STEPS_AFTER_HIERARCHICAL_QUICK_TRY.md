# BMM-TRL next steps after hierarchical pivot quick try

Date: 2026-06-11

This plan follows `BMM_TRL_HIERARCHICAL_PIVOT_QUICK_TRY_20260611_173408.md`.

## Executive decision

The quick hierarchical pivot did **not** clear the continue gate.

My recommendation is:

```text
Pause active policy-facing experimentation now.
Do not run more BMM policy/controller sweeps.
Do not move to larger OGBench tasks.
Do not keep tuning BMM losses, witnesses, Q/V action extraction, or low-level BC.
```

The project is not a complete failure, but the current implementation is not on a credible path to a strong long-horizon offline RL policy result without becoming a much larger hierarchical RL project.

The best next step is to package the work as a diagnostic/research-note result, discuss with your advisor, and then decide whether to:

```text
A. write up the reachability/subgoal findings;
B. formally pivot into hierarchical RL with a serious low-level controller;
C. pause and move to a more promising project.
```

My recommendation is **A or C**, unless your advisor specifically wants a hierarchical RL project.

## Why this decision is justified

The go/no-go criterion from the previous plan was:

```text
If BMM_V > geometric_midpoint: continue hierarchical pivot.
If BMM_V <= geometric_midpoint: pause the project.
```

The new run gives:

```text
BMM_V final_d == geometric_midpoint final_d
BMM_V improve == geometric_midpoint improve
BMM_V does not beat geometric_midpoint
```

So the hierarchical pivot did not pass its own fast decision criterion.

## What remains positive

Do not throw away the whole story. Several components worked:

```text
1. Logged-offset high-budget labels were correctly diagnosed as behavior-time labels.
2. Clean geodesic V and Q reachability critics learned well.
3. Q/V max-min transitive budget-holdout improved heldout parent-budget classification.
4. The V-next distillation control did not explain the budget-holdout improvement.
5. Dataset-support graph labels were learnable and mildly positive.
6. BMM/V subgoal selection is meaningfully better than random in diagnostics.
```

These support a narrower claim:

```text
BMM-style max-min reachability can help long-budget reachability bootstrapping and subgoal diagnostics.
```

## What failed or remains too weak

The policy-facing route did not become robust:

```text
1. Flat Q action ranking was weak.
2. Q/V joint action-subgoal selection was mixed.
3. Better candidate action coverage did not rescue the Q/V action path.
4. A tiny BC low-level controller did not make BMM/V beat geometric midpoint.
5. A stronger 5k-step, larger BC controller still did not make BMM/V beat geometric midpoint.
6. No policy smoke solved tasks.
```

The latest controller result is especially important:

```text
geometric_midpoint final_d: 106.8972
BMM_V final_d:             106.8972

geometric_midpoint improve: 67.3057
BMM_V improve:             67.3057
```

That means the remaining hierarchical direction is not clearly better than a simple baseline under the tested controller.

## Current research status

The viable contribution is not:

```text
BMM-TRL is an end-to-end long-horizon offline RL policy algorithm.
```

The viable contribution is closer to:

```text
BMM-TRL identifies a max-min budgeted reachability target that can bootstrap heldout longer budgets on clean reachability/support-graph diagnostics, but converting that critic improvement into policy improvement requires a separate high-level/low-level architecture.
```

This is a useful finding, but it is likely not enough by itself for a strong policy-performance paper.

## Recommended action: stop experiments and prepare advisor summary

Before doing any more coding or training, create:

```text
BMM_TRL_ADVISOR_SUMMARY_AND_DECISION.md
```

The summary should be 1-2 pages and include:

```text
1. Original hypothesis.
2. What was implemented.
3. Key positive critic-level results.
4. Key negative policy-facing results.
5. Why logged-offset labels failed.
6. Why grid/geodesic and graph-support labels were used.
7. Current decision: pause policy-facing work unless reframed as hierarchical RL.
8. Options A/B/C.
```

This is more valuable than another experiment.

## Option A: write up as a diagnostic / methods note

If you want to salvage a research artifact, write it as:

```text
Budgeted max-min reachability diagnostics for offline GCRL.
```

Possible contribution:

```text
1. TRL product backup is additive in distance error.
2. Budgeted max-min reachability gives a non-expansive target in score space.
3. Logged offset is a bad high-budget reachability target.
4. Support/geodesic reachability targets are learnable.
5. Budget-holdout results show max-min Q/V transitive improves heldout parent-budget classification.
6. Policy extraction remains unresolved.
```

Do not claim:

```text
SOTA policy performance.
End-to-end long-horizon offline RL solved.
BMM improves flat action ranking.
```

This could still be useful for an internal report, workshop note, or future project foundation.

## Option B: formal hierarchical RL pivot

Only choose this if you are willing to start a new project.

A real hierarchical pivot would need:

```text
1. a strong low-level goal-conditioned controller;
2. careful subgoal proposal over support states;
3. high-level replanning;
4. baselines such as geometric midpoint, graph shortest path, HIQL/HIGL-style planners;
5. real environment rollouts.
```

This is no longer just BMM critic research. It is a hierarchical RL system project.

### Minimum go/no-go if choosing Option B

Run exactly one serious low-level-controller experiment:

```text
controller = strong pretrained GCBC / HIQL / GCIQL / existing actor
selectors = random, geometric, BMM_V, oracle
```

Continue only if:

```text
BMM_V > geometric under the same strong controller.
```

If not, stop.

## Option C: pause and move on

This is reasonable if your goal is a high-impact offline RL policy result.

Reasons:

```text
1. The policy-facing signal is not robust.
2. The method now requires a separate controller.
3. The core BMM contribution is critic-level, not policy-level.
4. Further work likely needs a new hierarchical RL stack.
5. The opportunity cost is high.
```

Pausing does not mean the work was wasted. It produced a clear set of lessons and reusable diagnostics.

## What not to do next

Do not run:

```text
more BC controller sweeps;
more Q/V action extraction;
more action-ranking diagnostics;
more sparse-Q tables;
more loss or witness sweeps;
PointMaze large;
full OGBench policy benchmarks;
longer training hoping it fixes the issue.
```

The current evidence is enough to stop those directions.

## If you insist on one final experiment

Only one final experiment is defensible:

```text
Use an existing strong goal-conditioned controller, if one is already available.
```

No training from scratch. No sweeps.

Compare:

```text
random subgoal
geometric midpoint
BMM_V subgoal
oracle midpoint
```

Decision:

```text
If BMM_V > geometric: discuss a hierarchical pivot.
If BMM_V <= geometric: pause the project.
```

Given the 5k BC result, my expectation is that this may still be hard.

## Advisor discussion recommendation

Ask your advisor this concrete question:

```text
Given that BMM gives critic-level budget-holdout gains but not robust policy gains,
should we write it up as a diagnostic methods note, pivot to hierarchical RL,
or pause and move to a more promising project?
```

Bring three tables:

```text
1. Budget-holdout positive table.
2. Action/QV negative table.
3. Hierarchical controller tie/failure table.
```

## Bottom line

You made real scientific progress, but the go/no-go decision is now clear enough:

```text
Stop active BMM policy experimentation.
Package the findings.
Only continue if the project is explicitly reframed as hierarchical RL with a serious low-level controller.
```

My recommendation: **pause active experimentation and prepare an advisor-facing summary**.
