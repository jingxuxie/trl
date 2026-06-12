# BMM-TRL next steps after controller decision diagnostics

Date: 2026-06-11

This plan follows `BMM_TRL_VALUE_SUBGOAL_CONTROLLER_RESULTS_20260611_152903.md`.

## Executive decision

The latest result is a useful decision point.

My recommendation is:

```text
Stop active broad experimentation now.
Do not run larger OGBench policy benchmarks.
Do not return to flat Q or Q/V action extraction.
Pause the project unless you and your advisor explicitly want to reframe it.
```

The only viable continuation is a reframed project:

```text
BMM as a high-level reachability / subgoal-selection method,
paired with a separately strong low-level controller.
```

The current prototype does **not** support the stronger claim:

```text
BMM-TRL trains end-to-end into a strong long-horizon offline RL policy.
```

It does support a narrower critic/subgoal claim:

```text
BMM max-min reachability can improve heldout long-budget reachability
and can select useful subgoals in some diagnostics.
```

## Why this is the right decision

### Positive evidence

The project found real signal:

```text
1. Logged-offset targets were diagnosed as behavior-time labels and replaced.
2. Geodesic and support-graph reachability labels are learnable.
3. Budget-holdout Q/V transitive improves heldout parent-budget reachability.
4. V-next distillation does not explain the budget-holdout gain.
5. BMM/V subgoal selection beats random and geometric in the strict NN-controller smoke.
6. Tiny policy smoke with NN control makes progress toward goals.
```

These are not trivial results. They support the max-min reachability bootstrapping idea at the critic/subgoal level.

### Negative evidence

The policy-facing story is not robust:

```text
1. Flat Q action ranking is weak.
2. Q/V joint action-subgoal selection is mixed and not robust.
3. Better candidate actions do not rescue Q/V extraction.
4. The tiny BC controller does not make BMM/V clearly better than geometric.
5. No method solves the policy task in the current smoke tests.
6. BMM/V policy gains are sensitive to the low-level controller.
```

The most important new negative result is the BC-controller smoke:

```text
geometric_midpoint final_d: 102.9381, improve: 71.2648
BMM_V final_d:             106.8972, improve: 67.3057
```

With that controller, BMM/V beats random but not the simple geometric baseline. That fails the stronger policy-facing criterion.

## What this means for the research goal

The original goal was to reduce long-horizon error compounding with budgeted max-min reachability. The critic-level experiments are aligned with that goal.

But the policy goal needs an additional ingredient:

```text
a reliable low-level controller that can exploit selected subgoals.
```

Without that, the project becomes a hierarchical RL project, not just a BMM critic project.

So the right interpretation is:

```text
BMM may be useful for high-level reachability and subgoal planning.
The current implementation is not enough for end-to-end policy improvement.
```

## Recommended action now

Do **not** spend more compute immediately.

Instead, prepare a short advisor-facing decision memo with three options:

```text
Option A: write up the critic/subgoal diagnostics as a research note.
Option B: pivot to hierarchical RL with a separately trained low-level controller.
Option C: pause the project and move to a more promising direction.
```

My recommendation is Option A or C unless the advisor is specifically interested in hierarchical RL.

## Option A: critic / reachability / subgoal diagnostic project

This is the lowest-compute continuation.

### Claim

```text
Max-min budgeted reachability gives useful long-budget bootstrapping on clean support-reachability targets.
```

### Evidence to package

```text
1. Tabular/log-depth sanity.
2. Failure of logged-offset high-budget labels.
3. Geodesic V/Q learning success.
4. Budget-holdout BMM gains over A/F controls.
5. Dataset-support graph budget-holdout mild positive result.
6. Value-level subgoal selection better than random/geometric in strict NN smoke.
```

### What not to claim

```text
Do not claim policy SOTA.
Do not claim end-to-end long-horizon offline RL solved.
Do not claim flat action Q is strong.
```

### Next task

Create:

```text
BMM_TRL_PROJECT_SUMMARY_FOR_ADVISOR.md
```

with:

```text
one-page narrative;
key positive tables;
key negative tables;
recommended pivot/stop decision;
compute spent and remaining risks.
```

This can be done without new training.

## Option B: hierarchical RL pivot

Only choose this if you want a new project around low-level control.

### New project framing

```text
High-level BMM selects subgoals.
Low-level controller reaches selected subgoals.
```

### Minimum next experiment

Use an already-strong low-level controller, not the tiny 1k-step BC controller.

Possible controllers:

```text
1. existing goal-conditioned actor from a strong baseline;
2. GCBC trained longer and specifically for local subgoal reaching;
3. IQL/GCIQL/HIQL-style low-level controller;
4. oracle local controller only as an upper-bound diagnostic.
```

### Minimal go/no-go test

Compare selectors with the same low-level controller:

```text
random subgoal
geometric midpoint
BMM/V subgoal
oracle midpoint
```

Metrics:

```text
final distance to goal
goal distance improvement
subgoal reach rate
success rate as secondary
```

Run:

```text
one seed
one environment
few tasks
small max steps
```

### Continue only if

```text
BMM/V beats random and geometric under the same stronger low-level controller.
```

If not, stop.

## Option C: pause project

This is reasonable if your main goal is a strong OGBench policy result.

Reasons to pause:

```text
1. The policy interface is not robust.
2. The method now depends on a separate low-level controller.
3. General offline target construction remains nontrivial.
4. Further progress likely requires a new hierarchical RL implementation.
5. The current evidence is not enough for a long-horizon offline RL policy claim.
```

The project has still produced useful lessons and artifacts, but it may not be the best use of more time if you need a high-impact policy result quickly.

## One final cheap test, if you want a last signal

If you want one last fast test before pausing, do this only:

```text
selector comparison with a stronger low-level controller
```

Do not train BMM more.

### Setup

```text
selectors = random, geometric_midpoint, BMM_V, oracle_midpoint
controller = strongest existing goal-conditioned controller available
num_tasks = 5
episodes_per_task = 1
max_steps = 100
one seed
```

### Decision

```text
If BMM_V > geometric_midpoint: continue hierarchical pivot.
If BMM_V <= geometric_midpoint: pause the project.
```

This is the cleanest remaining go/no-go test.

## What to stop doing

Stop all of the following unless the project is formally reframed:

```text
flat Q extraction
Q/V action-selection sweeps
sparse-Q tables
loss-mode sweeps
witness sweeps
large-maze runs
full OGBench policy benchmarks
more tiny BC controller tuning
```

These have already given enough signal.

## Immediate task list

1. Create advisor summary document.
2. Decide with advisor between:
   ```text
   A. critic/subgoal diagnostic writeup
   B. hierarchical RL pivot
   C. pause project
   ```
3. If choosing B, run exactly one stronger-low-level-controller selector comparison.
4. If BMM/V does not beat geometric under that controller, pause.
5. If BMM/V does beat geometric, then and only then design a small hierarchical policy experiment.

## Bottom line

You are making progress scientifically, but the project has narrowed substantially.

The evidence supports:

```text
BMM as a reachability/subgoal bootstrapping idea.
```

The evidence does not yet support:

```text
BMM as an end-to-end offline RL policy method.
```

If your goal is a policy-performance project, pause or pivot. If your goal is a reachability/subgoal-planning project, continue only with a stronger low-level controller and very small, decision-oriented experiments.
