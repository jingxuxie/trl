# BMM-TRL next steps after value-subgoal controller diagnostics

Date: 2026-06-11

This plan follows `BMM_TRL_VALUE_SUBGOAL_CONTROLLER_RESULTS_20260611_152903.md`.

## Executive decision

The current evidence supports a narrow continuation, not a broad continuation.

```text
Continue flat Q extraction: no.
Continue Q/V joint action-subgoal extraction as the main path: no.
Run broad policy benchmarks now: no.
Stop the whole project immediately: no.
Continue only as high-level BMM subgoal planning + separate low-level controller: yes.
```

The project should now be framed as:

```text
BMM improves budgeted reachability / subgoal planning,
not flat neural action extraction.
```

The remaining question is whether the high-level BMM subgoal signal can be paired with a competent low-level controller to produce meaningful policy progress.

## Why this is the right decision

The original BMM-TRL prototype plan explicitly warned that large-budget critics can be too coarse for action ranking and suggested budget scan plus HIGL-style high-level subgoal selection as longer-term extensions. The current evidence matches that warning.

The path that has repeatedly worked is:

```text
budgeted reachability -> transitive bootstrapping -> subgoal/reachability selection
```

The path that has repeatedly been weak is:

```text
flat or joint neural Q action extraction
```

## What the latest result says

### Positive value-subgoal signal

The value-only subgoal diagnostic is clearly positive.

At 512 queries with same-cell controller hops 0:

```text
random state-valid:       0.0059
BMM/V state-valid:        0.1680
oracle midpoint valid:    0.3672

random path stretch:      52.8145
BMM/V path stretch:       1.0826
oracle path stretch:      0.2320

random midpoint error:    110.9444
BMM/V midpoint error:     24.6779
oracle midpoint error:    13.3493
```

This is much stronger than the Q/V action-selection diagnostics. It says the value-level BMM/reachability signal can choose useful intermediate subgoals.

### Low-level controller signal

A same-cell nearest-neighbor logged-action controller can reduce distance to BMM-selected subgoals:

```text
BMM/V nn query improve: 19.7958
BMM/V nn query reduce:  1.0000
nn source-to-query:     0.0000
```

This is important because the action source is local, not coming from far-away neighboring cells.

### Tiny policy smoke signal

The tiny policy smoke does not solve tasks, but it makes consistent progress:

```text
one-task smoke:
start goal distance:       197.9579
final goal distance:        98.9789
goal distance improvement:  98.9789
success:                     0.0000

three-task smoke:
start goal distance:       171.5635
final goal distance:        98.9789
goal distance improvement:  72.5845
success:                     0.0000
```

This is a weak-positive policy-facing signal. It justifies one more focused high-level-controller phase, not a benchmark.

## What this rules out

Stop pursuing these paths for now:

```text
flat Q_H(s,a,g) action extraction;
Q/V neural joint action-subgoal extraction as the main policy path;
large hyperparameter sweeps;
full OGBench policy benchmarks;
more logged-offset target work.
```

The evidence is now consistent across multiple diagnostics: BMM has value at the reachability/subgoal level, while action extraction is the weak link.

## What remains viable

The viable version is hierarchical:

```text
High level:
    choose subgoal w = argmax_w min(V_h(s,w), V_{H-h}(w,g))

Low level:
    use a separate controller to reach w
```

This is aligned with the original BMM-TRL long-term extension plan, especially budget scan and HIGL-style high-level subgoal selection.

## Main next question

Do not ask:

```text
Can BMM alone solve PointMaze policy performance?
```

Ask:

```text
Can BMM-selected subgoals improve a simple hierarchical controller over random/geometric/oracle-subgoal baselines?
```

This can be tested quickly.

## Runtime rule from now on

Use this strict rule:

```text
No training run longer than 30 minutes unless a cached diagnostic is positive.
One seed only.
Tiny policy smoke only.
No broad benchmark.
```

Prefer evaluation-only scripts and existing checkpoints.

## Milestone 1: compare subgoal selectors under the same low-level controller

The current smoke only tests BMM/V. Next compare subgoal selectors using the same controller.

### Subgoal selectors

```text
random subgoal
geometric midpoint / graph midpoint, if available
BMM/V value score
oracle midpoint
```

Use:

```text
score_BMM(w) = min(V_h(s,w), V_{H-h}(w,g))
```

### Controller

Use the current same-cell nearest-neighbor controller first:

```text
controller_hops = 0
replan every step
```

Then try controller hops 1 or 2 only if hops 0 is positive.

### Metrics

```text
final distance to goal
goal distance improvement
mean step goal improvement
subgoal valid fraction
subgoal reach / reduce fraction
failure examples
```

### Decision

Continue if:

```text
BMM/V subgoals beat random and simple geometric midpoint,
and approach oracle midpoint progress.
```

Pause if:

```text
oracle midpoint works but BMM/V does not;
or BMM/V is not better than random/geometric midpoint.
```

## Milestone 2: improve the low-level controller only if selector comparison is positive

The current nearest-neighbor controller is intentionally simple. If BMM/V subgoals are good, test better low-level control.

### Candidate controllers

```text
same-cell nearest neighbor    # current strict local baseline
neighbor-cell nearest neighbor
local dataset action that maximizes one-step progress to w
goal-conditioned BC toward w
existing repo actor toward w, if easy
```

### Fast diagnostic before environment rollouts

For a set of selected subgoals, report:

```text
one-step distance reduction to w
fraction reducing distance to w
next distance to w
action source distance from current state
```

Only run environment smoke if the one-step diagnostic improves.

## Milestone 3: deployable subgoal candidate proposal

Current subgoal candidates may use dense dataset/grid sampling. For a deployable offline method, add a non-oracle candidate proposal.

Options:

```text
dataset states sampled from support graph
nearest graph nodes around budget midpoint
farthest / diverse graph nodes
learned V-filtered candidates
```

Avoid relying on true grid oracle midpoints for policy claims.

The diagnostic should report:

```text
candidate coverage
oracle valid subgoal exists fraction
selected valid fraction
path stretch
```

## Milestone 4: dataset-support graph version

The final general-offline target should not rely on grid layout.

Repeat the value-subgoal diagnostic with:

```text
reachability_label_type = graph
score(w) = min(V_h^graph(s,w), V_{H-h}^graph(w,g))
```

Use one seed and existing graph infrastructure.

Continue only if:

```text
graph-based BMM/V subgoals are better than random and simple graph baselines.
```

This is important for the general offline RL story.

## Milestone 5: tiny hierarchical policy smoke

Only after Milestones 1-3 are positive.

### Policy

```text
for each step or every K steps:
    scan budgets to estimate H_hat(s,g)
    choose h = H_hat / 2
    choose w = argmax_w min(V_h(s,w), V_{H-h}(w,g))
    use low-level controller toward w
    replan
```

### Scope

```text
one environment
one seed
small number of tasks/episodes
no benchmark claim
```

### Compare only

```text
random subgoal + controller
BMM/V subgoal + same controller
oracle midpoint + same controller
```

This isolates the high-level planner.

## Stop / continue criteria

### Continue the project if

```text
BMM/V subgoals beat random/geometric baselines under the same controller;
low-level controller can exploit selected subgoals;
tiny policy smoke shows better progress than random subgoals.
```

### Continue but reframe if

```text
BMM/V subgoals are useful but low-level control is weak.
```

Then the project becomes:

```text
BMM high-level planning + separate low-level policy learning.
```

### Stop or pause if

```text
BMM/V subgoals do not beat simple baselines;
or BMM/V subgoals cannot be exploited by any cheap local controller;
or graph-based subgoal diagnostics fail and only grid-oracle results remain positive.
```

## What to avoid next

Do not run:

```text
more Q/V action-selection sweeps;
more sparse-Q tables;
large-maze benchmarks;
full policy comparisons;
new representation learning;
long training runs.
```

These are not justified until the high-level subgoal-controller pipeline is clearly useful.

## Recommended immediate task list

1. Add a subgoal-selector comparison mode to the policy smoke:
   ```text
   random / geometric / BMM-V / oracle midpoint
   ```
2. Run 128-query or tiny-task evaluation with the same-cell NN controller.
3. If BMM-V beats random/geometric, run the same with 512 queries or a few more tasks.
4. Add a one-step low-level controller comparison:
   ```text
   same-cell NN vs neighbor-cell NN vs local progress-max action
   ```
5. Repeat the value-subgoal diagnostic with dataset-support graph labels.
6. Only if these are positive, run one tiny hierarchical policy smoke.

## Bottom line

You should not stop immediately. You should stop pursuing flat Q / QV action extraction.

The remaining promising project is:

```text
BMM for high-level reachability and subgoal planning,
paired with a separate low-level controller.
```

Run one final focused high-level-controller phase. If it does not beat simple subgoal baselines or does not transfer to dataset-support graph labels, then it is reasonable to pause the project and move on.
