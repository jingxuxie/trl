# BMM-TRL next steps after value-subgoal next-step results

Date: 2026-06-11

This plan follows `BMM_TRL_VALUE_SUBGOAL_NEXT_STEPS_RESULTS_20260611_163357.md`.

## Executive decision

You are still on track **only under the reframed hierarchical direction**:

```text
High level: BMM budgeted value selects subgoals.
Low level: a separate controller reaches those subgoals.
```

Do **not** return to:

```text
flat Q action extraction
Q/V joint action-subgoal extraction as the main path
large policy benchmarks
broad sweeps
more critic/transitive tuning
```

The latest results are the strongest support so far for the hierarchical pivot:

```text
BMM/V subgoal selector beats random and geometric midpoint in tiny policy smoke.
Same-cell NN controller is the best cheap controller tested.
Dataset-support graph BMM/V subgoal selection is positive.
```

The next bottleneck is no longer the BMM critic. The bottleneck is the **low-level controller** and the deployability of the subgoal candidate proposal.

## What the latest result means

### 1. High-level BMM/V subgoals are useful

In the selector comparison with the same-cell NN controller, BMM/V produced the best final distance and goal-distance improvement among non-oracle selectors:

```text
random final_d:              118.7747
geometric_midpoint final_d:  112.1761
BMM_V final_d:               105.5775
oracle_midpoint final_d:     118.7747

random improve:              52.7888
geometric_midpoint improve:  59.3874
BMM_V improve:               65.9860
oracle_midpoint improve:     52.7888
```

This passes the narrow continue condition: BMM/V beats random and geometric under the same controller.

### 2. The cheap low-level controller is the current bottleneck

For BMM/V, same-cell NN controller hops 0 worked best:

```text
hops=0 final_d: 105.5775, improve: 65.9860
hops=1 final_d: 151.7677, improve: 19.7958
hops=2 final_d: 145.1691, improve: 26.3944
```

Looser nearest-neighbor action borrowing hurts. The next low-level controller should not be a wider nearest-neighbor heuristic. It should be a real goal-conditioned local controller or a better local progress-max controller.

### 3. Dataset-support graph subgoals work

The graph subgoal diagnostic is important for the general offline story:

```text
random state_valid:          0.7422
BMM_V_graph state_valid:     0.9902
random path_stretch:         29.0078
BMM_V_graph path_stretch:     4.7734
```

BMM/V graph subgoals clearly beat random and improve graph path stretch relative to Euclidean midpoint. This weakens the concern that the method only works with grid-oracle labels.

## Current research status

The viable claim is now:

```text
BMM-style budgeted reachability improves long-budget/subgoal bootstrapping,
and is useful as a high-level subgoal planner.
```

The non-viable claim, for this prototype, is:

```text
BMM directly produces a strong flat action-value policy.
```

This is consistent with the original project risk note that large-budget critics can be too coarse for action ranking and may require budget scan / high-level subgoal selection.

## Recommended next milestone

The next useful milestone is:

```text
BMM subgoal selector + better low-level controller
```

not more BMM critic tuning.

## Milestone 1: one-step low-level controller comparison

Before training any new controller, add a fast evaluation-only controller comparison.

### Controllers

Compare:

```text
same_cell_NN                 # current best cheap controller
local_progress_max           # among same-cell transitions, choose action minimizing d(next,w)
direct_to_goal_same_cell_NN  # same controller but goal is final goal g, not subgoal w
random_same_cell_action
oracle_progress_action       # diagnostic upper bound if available
```

### Inputs

Use selected subgoals from:

```text
random
geometric_midpoint
BMM_V
oracle_midpoint
```

### Metrics

Report:

```text
one_step_distance_to_subgoal_before
one_step_distance_to_subgoal_after
mean subgoal distance reduction
fraction reducing distance to subgoal
one_step_distance_to_final_goal_before/after
fraction reducing distance to final goal
action source distance from current state
```

### Decision

Continue only if:

```text
local_progress_max or same_cell_NN reliably reduces distance to BMM_V subgoals,
and BMM_V subgoals produce better one-step final-goal progress than random/geometric subgoals.
```

This should be evaluation-only and fast.

## Milestone 2: tiny goal-conditioned BC controller

Only if Milestone 1 is positive, train a tiny low-level controller.

### Training data

Use short-horizon subgoal reaching examples from the offline dataset:

```text
input:  (s_t, w)
target: a_t
where w = s_{t+k}, k in {1,2,4,8} or within one/two cells
```

Start with a small model:

```text
hidden_dims=(256,256)
steps=5k or less
batch_size=256
```

This should be treated as a low-level BC baseline, not a new BMM contribution.

### Fast validation

Before any environment rollout, validate one-step and short-rollout behavior:

```text
BC action reduces distance to selected subgoal
BC beats same_cell_NN on one-step reduction
BC does not produce out-of-support actions
```

### Decision

Run environment smoke only if:

```text
BC improves subgoal progress over same_cell_NN in offline diagnostics.
```

## Milestone 3: tiny hierarchical policy smoke with selector comparison

Run only after Milestone 1 or 2 passes.

### Policy skeleton

```text
for each episode:
    choose H_hat(s,g) by V budget scan or fixed H=160 first
    set h = H_hat / 2 or h=80 for the fixed smoke
    choose subgoal w using selector
    run the same low-level controller toward w for K steps
    replan
```

### Compare only

```text
random subgoal + same controller
geometric midpoint + same controller
BMM_V subgoal + same controller
oracle midpoint + same controller
```

Do not compare against TRL/GCIQL yet.

### Metrics

```text
final distance to goal
goal distance improvement
subgoal distance improvement
subgoal reduce fraction
success rate, secondary only
failure cases
```

### Scope

```text
one env
one seed
3-5 tasks
short max_steps
```

## Milestone 4: dataset-support graph version of the smoke

The grid/geodesic smoke is still partly oracle. The general offline story needs graph support.

Repeat the subgoal-selector diagnostic with:

```text
reachability_label_type=graph
score(w)=min(V_h^graph(s,w), V_{H-h}^graph(w,g))
```

Use graph candidate subgoals:

```text
support graph nodes/states
nearest graph nodes around budget midpoint
farthest/diverse graph nodes
```

Do not use grid oracle midpoint for claims in this setting.

### Decision

Continue toward a general offline paper only if:

```text
graph BMM_V subgoals beat random and simple graph baselines,
and the low-level controller can exploit them.
```

## Milestone 5: stop/pause criteria

Stop or pause the project if any of the following happens:

```text
BMM_V subgoals do not beat random/geometric under the same controller;
local controller cannot make progress toward BMM_V subgoals;
graph-support subgoal diagnostics fail and only grid-oracle results remain positive;
tiny hierarchical smoke does not beat random/geometric subgoal baselines.
```

At that point, the critic-level budget-holdout result can still be written up as an internal lesson or a small methods note, but it would not justify more OGBench policy work.

## Runtime policy

Use this strict schedule:

```text
<5 min: one-step controller diagnostics
<15 min: 128-query subgoal/controller eval
<30 min: tiny 3-task smoke
>30 min: only after a positive smoke and advisor approval
```

Do not run:

```text
large-maze experiments
broad policy benchmarks
more Q/V action-selection sweeps
more loss/witness sweeps
more sparse-Q tables
```

## Immediate task list

1. Add `local_progress_max` and `direct_to_goal_same_cell_NN` controller diagnostics.
2. Compare random/geometric/BMM_V/oracle subgoals under the same controller.
3. If BMM_V still wins, run a tiny hierarchical smoke with the selector comparison.
4. If the smoke is positive, train a tiny goal-conditioned BC low-level controller.
5. Repeat the value-subgoal diagnostic with graph-support labels and graph candidates.
6. Decide continue/pause based on whether graph BMM_V + low-level control works.

## Bottom line

You are still making progress, but only after narrowing the project.

The current best direction is:

```text
BMM as high-level reachability/subgoal planner
+
separate low-level controller
```

Do not spend more time on flat Q or Q/V action extraction. The next decisive experiment is whether a better local controller can exploit BMM-selected subgoals better than random/geometric subgoals.
