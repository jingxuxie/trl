# BMM-TRL fast decision plan

Date: 2026-06-11

This plan follows `BMM_TRL_ACTION_RANKING_FOLLOWUP_RESULTS_20260611_103956.md` and the concern that the project needs much faster iteration and a clearer go / pivot / stop decision.

## Executive take

The project is not dead, but it should stop broad experimentation now.

Current evidence:

```text
Positive:
  BMM Q/V transitive improves heldout parent-budget reachability in budget-holdout tests.
  The effect replicated over three seeds for grid-cell H8 and env-step H160.

Negative / unresolved:
  The improvement has not translated into the current flat same-cell action-ranking diagnostic.
  Grid/geodesic labels are oracle labels; the general offline version needs dataset-support graph targets.
  Runtime is too high for continued broad sweeps.
```

The next phase should be a **fast kill-or-pivot protocol**, not another long research sweep.

The key questions are:

```text
Q1. Is the action-ranking diagnostic valid?
Q2. Does BMM work with offline-support graph labels, not grid oracle labels?
Q3. If flat action ranking fails, does BMM help high-level subgoal selection?
```

Answer these with cached evaluation and one-seed A/B/F tests only.

## Code/experiment review: likely issues

### Issue 1: current action-ranking candidates are not true counterfactual actions

The current action-ranking sampler chooses a logged source transition, then takes other actions from transitions in the same grid cell. It repeats the logged source observation for all candidate actions, but labels each candidate by the next cell of the transition the action came from.

This means the diagnostic scores:

```text
Q(s_logged, a_candidate_from_other_state, g)
```

but the label is based on:

```text
s_candidate_other -> s_next_candidate_other
```

That is not a valid counterfactual unless all states in the grid cell are dynamically interchangeable.

This can easily destroy action-ranking signal even if the critic is valid for logged transitions.

### Issue 2: same-cell candidates may be too local or too noisy

The current diagnostic samples actions from the same grid cell. If those actions induce similar next-state distances, the task is weak. If they come from positions too far within the cell, labels are noisy. Both can happen.

Add these diagnostics before interpreting action-ranking failures:

```text
oracle_distance_score AUC
oracle selected distance
candidate next-distance spread
candidate source-position spread
V_next teacher action-ranking AUC
```

If the oracle score is not strong, the candidate set is not useful. If V_next is strong but Q is weak, Q training/action conditioning is the bottleneck. If both V_next and Q are weak, flat one-step action ranking may be the wrong use case.

### Issue 3: grid labels are not the final offline target

Grid labels were necessary to debug the idea, but they do not solve general offline RL. The next target for generality should be:

```text
offline-support graph reachability
```

not logged offset and not environment grid oracle.

Define:

```text
R_H^D(s,g) = 1[d_D(phi(s), phi(g)) <= H]
```

where `d_D` is shortest-path distance in a graph built from offline trajectories. Same-trajectory offset should be treated as positive evidence only, not as hard high-budget negatives.

### Issue 4: no obvious Q/V transitive implementation bug

The frozen-teacher Q/V transitive code structure is conceptually correct:

```text
first branch:  Q_h(s,a,w)
second branch: V_{H-h}(w,g)
y = max_w min(first, second)
```

The loss-mode and target-vs-parent diagnostics already caught the main conceptual issue: sampled max-min is often a lower-bound target, not an equality target. Using `bce_lower_bound` as default is reasonable.

## Runtime policy from now on

Use this rule:

```text
No new >30 minute training run unless a cached/cheap diagnostic is positive.
```

Default comparison set:

```text
A = supervised/no transitive
B = Q/V transitive
F = V-next distillation
```

Default seeds:

```text
one seed only until B beats A and F is near A.
```

Default saved checkpoints:

```text
reuse existing A/B/F checkpoints whenever possible.
```

Do not run:

```text
full A-G matrices
loss sweeps
uniform sparse sweeps
3-seed sweeps before one-seed signal
policy benchmarks before action/subgoal diagnostics
```

## 48-hour fast decision protocol

## Test 1: validate the action-ranking evaluator

This is evaluation-only. It should be the fastest next step.

### Add to `scripts/eval_bmm_action_ranking.py`

Add score modes:

```text
oracle_distance_score = -d_grid(s_next_candidate, g)
random_score
V_next_teacher_score = V_{H-1}(s_next_candidate, g)
Q_candidate_own_state = Q_H(s_candidate, a_candidate, g)
Q_repeated_source     = Q_H(s_logged, a_candidate, g)   # current behavior
```

Add query cache:

```text
--save_query_cache
--load_query_cache
--query_cache_path
```

Add candidate diagnostics:

```text
candidate_next_distance_spread_mean
candidate_source_position_spread_mean
candidate_unique_next_cell_count
oracle_best_distance
logged_distance
random_distance
```

### Interpret

```text
If oracle_distance_score AUC < 0.85:
    action-ranking candidate set is not informative. Fix evaluator before training.

If oracle is strong and V_next is strong but Q is weak:
    Q action-conditioning/training is the bottleneck.

If oracle is strong but V_next and Q are weak:
    flat one-step action ranking is likely not the right use of BMM.

If Q_candidate_own_state is strong but Q_repeated_source is weak:
    the current repeated-source counterfactual diagnostic is invalid/noisy.
```

### Go/no-go for flat action ranking

Continue flat Q policy extraction only if:

```text
B > A and B > F on a diagnostic where oracle and V_next are strong.
```

Otherwise pivot to high-level subgoal planning.

## Test 2: offline-support graph budget-holdout

This addresses the most important generalization concern: grid labels do not generalize to arbitrary offline datasets.

Run the same budget-holdout test using dataset-support graph distance.

### Minimal setup

Use one seed only:

```text
label = graph
geodesic/grid oracle = disabled
comparison = A/B/F only
budget-holdout parent = graph parent budget chosen from graph diameter/class balance
steps = 500 first, 1000 only if needed
```

Use a train-only graph if feasible:

```text
graph_scope = train_only
val states mapped to nearest train node/bin
report val coverage and nearest distance
```

If train-only mapping is not ready, use the current train+val oracle graph for a quick feasibility test, but label it clearly as `oracle-support-graph diagnostic`.

### Success criterion

Continue BMM as a general offline method if:

```text
B > A on heldout graph parent budget
F ≈ A
```

Pivot if:

```text
B does not beat A on graph labels,
but grid labels still work.
```

That would mean the current method depends too much on oracle geometric labels.

## Test 3: high-level subgoal selection

If flat action ranking is weak, test BMM where it naturally belongs: selecting a witness/subgoal.

Create:

```text
scripts/eval_bmm_subgoal_selection.py
```

For each `(s,g,H)`:

1. sample candidate subgoals `w` from dataset states or graph nodes;
2. score:

```text
score(w) = min(V_h(s,w), V_{H-h}(w,g))
```

or if using Q first branch:

```text
score(a,w) = min(Q_h(s,a,w), V_{H-h}(w,g))
```

3. compare selected `w` to oracle/graph midpoint quality:

```text
branch_valid = 1[d(s,w)<=h and d(w,g)<=H-h]
path_stretch = d(s,w) + d(w,g) - d(s,g)
midpoint_error = abs(d(s,w) - h)
```

Compare A/B/F or supervised-only vs BMM-transitive V/Q checkpoints.

### Interpret

```text
If B improves subgoal validity/path stretch but not flat action ranking:
    pivot project toward BMM as high-level subgoal planner.

If B does not improve subgoal selection either:
    reconsider/pause the neural BMM direction.
```

## Decision tree

### Continue current direction if

```text
offline-support graph budget-holdout is positive
and either action-ranking or subgoal-selection diagnostic is positive.
```

Next: medium policy smoke or PointMaze-large scaling.

### Pivot if

```text
critic budget-holdout remains positive,
flat action ranking is negative,
subgoal selection is positive.
```

New framing:

```text
BMM is a reachability/subgoal-planning module, not a flat Q action scorer.
```

This is still aligned with the original longer-term plan: budget scan and HIGL-style high-level subgoal selection.

### Pause or move to another project if

```text
graph-label budget-holdout fails,
action-ranking fails under oracle-validated candidates,
subgoal selection fails,
and only grid-oracle critic diagnostics remain positive.
```

Then the method is probably too dependent on oracle labels and not worth more OGBench policy time in its current form.

## Recommended immediate commands/tasks

### Task A: make action ranking fast and valid

1. Add query caching.
2. Add oracle and V-next scores.
3. Add `Q_candidate_own_state` vs `Q_repeated_source` comparison.
4. Run A/B/F on the existing checkpoints.

No new training.

### Task B: graph budget-holdout A/B/F

Run one seed:

```text
label=graph
variants=A,B,F
steps=500
```

Only run 1000 steps or more seeds if B beats A and F is near A.

### Task C: subgoal selection diagnostic

Implement the simplest V-only subgoal diagnostic first:

```text
score(w)=min(V_h(s,w), V_{H-h}(w,g))
```

No actor, no environment rollout, no new training.

## How to make tests faster

### Cache everything evaluation-side

```text
query sets
graph distances
cell distances
candidate actions/subgoals
critic scores if possible
```

### Use fewer eval samples first

For triage:

```text
num_queries = 128
candidate_count = 8
```

Only scale to 512 queries after a positive signal.

### Use 500 training steps for go/no-go

For graph holdout:

```text
steps = 500
```

Only use 1000 if the 500-step result is close or promising.

### Use one seed

Only replicate after a positive result.

## What this means for the research goal

The original goal was to get logarithmic-depth error behavior through max-min reachability composition. The value-level evidence now supports the core compositional bootstrapping story on clean budget-holdout tasks. The unsolved question is control usage:

```text
Does BMM help action selection directly,
or should it be used for subgoal planning / graph search?
```

The project should not be judged solely by the current flat action-ranking failure, because the original spec already identified actor signal from large budgets as a known risk and suggested budget scan / high-level subgoal selection as longer-term extensions.

## Bottom line

You should not give up yet, but you should stop open-ended training.

Run three fast diagnostics:

```text
1. validate action-ranking with oracle/V-next/cached queries;
2. test budget-holdout on dataset-support graph labels;
3. test BMM subgoal selection.
```

These should give a real go / pivot / stop signal without another week of long sweeps.
