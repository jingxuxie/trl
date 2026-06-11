# BMM-TRL next steps after action-ranking diagnostics

Date: 2026-06-11

This plan follows `BMM_TRL_ACTION_RANKING_RESULTS_20260611_022139.md`.

## Executive conclusion

You are still on the correct research track, but the latest action-ranking result is **not supportive of flat per-action policy extraction yet**.

The current status is:

```text
Critic-level budget-holdout result: positive and replicated.
Offline flat action-ranking result: mixed / negative for BMM versus A/F.
```

This means the BMM critic improvement is real at the heldout parent-budget reachability-classifier level, but it has not yet translated into better local action ranking under the current diagnostic.

That is not the same as saying the idea failed. It means the next question is:

```text
Is BMM better used as a high-level reachability/subgoal planner than as a flat one-step action scorer?
```

This is consistent with the original prototype risk note: a large-budget critic can be too coarse to rank actions, and action selection should use the smallest reachable budget or a higher-level subgoal mechanism.

## Why the current action-ranking result is mixed

The A/B/F critic rerun confirmed the critic-side result:

```text
B - A at H160: +0.0400 AUC, +0.0640 gap, -0.5976 BCE, -0.0398 ECE
F - A: much smaller
```

So BMM Q/V transitive still improves the heldout parent-budget critic metric.

But the action-ranking diagnostic shows:

```text
A parent action AUC: 0.6004
B parent action AUC: 0.5848
F parent action AUC: 0.6009
```

and B does not improve selected action distance or selected success. B has a larger parent-score gap, but that score gap is not ranking better actions in this same-cell candidate diagnostic.

## Possible explanations

### 1. The diagnostic may be too local/noisy

The current action-ranking sampler uses candidate actions from logged transitions in the same grid cell. Those actions are then scored at a repeated observation from one logged source state.

This is a reasonable first diagnostic, but it is approximate:

```text
a_candidate came from another state in the same cell,
not necessarily the exact same continuous state.
```

If same-cell actions are not meaningfully different, or if the candidate next-state distances are too close, the diagnostic may not reveal useful ranking differences.

### 2. H160 may be too coarse for one-step action ranking

Reachability within `H=160` is a long-horizon predicate on PointMaze medium. Many candidate actions from the same cell may all be reachable or all near-reachable. A large-budget Q can be good for parent reachability classification but too coarse for action discrimination.

This was a known risk in the original plan.

### 3. BMM may be more useful for high-level subgoal selection

BMM's max-min structure naturally says:

```text
choose a witness/subgoal w such that R_h(s,w) and R_{H-h}(w,g) are high
```

That is a high-level planning/subgoal operation, not necessarily a flat one-step action-ranking operation. If flat action ranking remains weak, pivoting to subgoal planning is not a failure; it may be the right use of the method.

## Runtime rule from now on

Use fewer runs.

Default rule:

```text
No new broad sweep.
One seed.
A/B/F only.
Use existing checkpoints whenever possible.
Only expand if the cheap diagnostic is positive.
```

The default comparison remains:

```text
A: no parent labels, no transitive
B: Q/V transitive
F: V-next distillation control
```

Do not rerun A/B/F training unless the question requires new checkpoints. Most next diagnostics should reuse the saved critics in:

```text
exp/bmm_qv_budget_holdout_20260611_021352
```

## Fast path: fix and stress-test action-ranking before more training

The next step should be evaluation-only, not training.

## Milestone 1: add oracle and teacher sanity baselines to action-ranking

Before judging BMM on the current action-ranking diagnostic, confirm that the diagnostic itself can distinguish good actions.

Add to `scripts/eval_bmm_action_ranking.py`:

```text
oracle_distance_score = -d_grid(s_next_candidate, g)
source_distance_score = -d_grid(s_source, g)      # should be weak or constant per query
random_score
V_next_teacher_score = V_{H-1}(s_next_candidate, g)
```

Report these in the same table as A/B/F.

Expected:

```text
oracle_distance_score should have very high action AUC and select lower-distance actions.
V_next_teacher_score should be meaningfully above A/B/F if the V teacher is useful for action ranking.
```

Interpretation:

- If oracle score barely beats A/B/F, the candidate sets are not informative.
- If oracle score is strong but V_next is weak, teacher/value calibration is the issue.
- If V_next is strong but Q critics are weak, the Q action-conditioned training is the issue.
- If V_next and Q are both weak, flat one-step action ranking may be the wrong use case.

This is cheap and should be run before any new training.

## Milestone 2: cache the action-ranking query set

To make iteration fast, add:

```text
--query_cache_path
--save_query_cache
--load_query_cache
```

Cache:

```text
observations
actions
goals
labels
distances
source_distances
source_cells
goal_cells
candidate_source_idxs
budget
remaining_budget
```

Then all A/B/F/oracle/V-next comparisons use the exact same cached query set without resampling.

This should make action-ranking experiments evaluation-only and much faster.

## Milestone 3: structured candidate sets

If oracle distance score is strong on the current candidate set, keep it. If not, build more informative candidate sets.

Add candidate modes:

```text
--candidate_mode=same_cell_random|cell_directional|neighbor_cell_transitions|oracle_diverse
```

### same_cell_random

Current behavior. Good as a baseline, but may be noisy.

### cell_directional

For each source cell, pick candidate transitions that move toward distinct neighboring cells/directions when possible:

```text
north/south/east/west/stay/diagonal-ish bins
```

This makes candidate actions cover different one-step outcomes.

### neighbor_cell_transitions

Allow candidates from adjacent cells as well as the same cell, but evaluate them from the source cell only as a diagnostic approximation. This increases diversity.

### oracle_diverse

Choose candidate transitions whose next-state distances to the sampled goal span a wide range. This is not a policy-realistic sampler, but it is useful to test whether Q scores contain any action-ranking signal when action quality varies.

Use `oracle_diverse` as a diagnostic only.

## Milestone 4: budget scan action scoring

The current action-ranking evaluates the parent budget `H160` and short-budget interpolation `max(H40,H80)`. Add the actual planned policy score:

```text
H_hat(s,g) = smallest H with V_H(s,g) >= tau
H_action = clamp(H_hat, min_budget, max_budget)
score = Q_{H_action}(s,a,g)
```

Also try:

```text
H_action = max(min_budget, H_hat - one_budget_step)
H_action = next_larger_budget(H_hat)
```

Why: long-budget Q may be too coarse. Budget scan may produce sharper action scores.

This should use the frozen V teacher and existing A/B/F Q critics. No new training.

## Milestone 5: high-level subgoal diagnostic

If flat action ranking remains weak after oracle/structured/budget-scan checks, pivot to BMM's natural use: subgoal selection.

Add:

```text
scripts/eval_bmm_subgoal_selection.py
```

For each `(s,g,H)`:

1. Sample candidate subgoal cells/states `w`.
2. Score:

```text
score(w) = min(V_h(s,w), V_{H-h}(w,g))
```

or for action-conditioned first branch:

```text
score(w,a) = min(Q_h(s,a,w), V_{H-h}(w,g))
```

3. Compare selected `w` against oracle geodesic midpoint quality:

```text
|d(s,w) - h|
|d(w,g) - (H-h)|
d(s,w) + d(w,g)
progress toward goal after choosing w
```

Report:

```text
subgoal oracle-valid rate
mean path stretch
midpoint distance error
selected subgoal diversity
B vs A/F difference
```

This diagnostic is more aligned with max-min BMM than one-step action ranking.

## Milestone 6: policy smoke only if a cheap diagnostic is positive

Run policy smoke only if one of these is true:

```text
1. B beats A/F in structured or budget-scan action ranking; or
2. B beats A/F in subgoal-selection quality.
```

If neither is true, policy evaluation will probably be expensive and inconclusive.

## Runtime-saving plan

### Stop doing

```text
full A-G matrices
3-seed training before one-seed evaluation success
loss sweeps
uniform sparse sweeps
new training for every diagnostic
```

### Start doing

```text
cache eval query sets
reuse saved checkpoints
run A/B/F only
run oracle sanity before neural comparisons
use one seed until a diagnostic is positive
write compact summaries instead of long per-run docs
```

### Approximate runtime ladder

Use this decision ladder:

```text
<5 min: cached eval with oracle/A/B/F/V-next
<15 min: new structured query cache + cached eval
<30 min: one new training row only if eval says it is needed
>1 hr: only for final replication after a positive result
```

## Recommended immediate task list

1. Extend `eval_bmm_action_ranking.py` with oracle distance, random, and V-next teacher scoring.
2. Add query caching.
3. Re-run action-ranking on the same A/B/F checkpoints with cached queries.
4. If oracle score is weak, implement `oracle_diverse` or `cell_directional` candidates.
5. Add budget-scan action scoring using the V teacher.
6. If B still does not beat A/F, implement subgoal-selection diagnostic.
7. Only then decide whether to run one policy smoke.

## Go / no-go criteria

### Continue flat policy path if

```text
B > A and B > F on structured/budget-scan action ranking,
and oracle/V-next baselines confirm the diagnostic is meaningful.
```

### Pivot to high-level BMM if

```text
oracle/V-next ranking is meaningful,
but B does not improve flat action ranking.
```

Then use BMM for:

```text
budget scan
subgoal selection
high-level planning
```

rather than as a flat Q scorer.

### Revisit critic training only if

```text
oracle ranking is strong,
V-next ranking is strong,
A/B/F Q ranking is weak.
```

Then the issue is Q action-conditioning/training rather than evaluation.

## Research status

The research idea is still alive, but the likely contribution may shift:

```text
BMM improves long-budget reachability bootstrapping on clean targets.
Flat one-step action ranking is not automatically improved.
BMM may be more naturally useful as a high-level subgoal/planning primitive.
```

This is still aligned with the original longer-term plan, which included budget scan and HIGL-style high-level subgoal selection.

## Bottom line

The new result is not supportive of immediate policy evaluation, but it is very informative. It says:

```text
Do not assume critic-level reachability gains imply flat action-ranking gains.
Test whether the action-ranking diagnostic is meaningful.
If flat ranking remains weak, pivot to subgoal selection, where max-min BMM is more naturally suited.
```

The fastest next progress is evaluation-only: oracle/V-next baselines, cached query sets, structured candidates, and budget-scan scoring. No more hour-long training runs until one of those cheap diagnostics is positive.
