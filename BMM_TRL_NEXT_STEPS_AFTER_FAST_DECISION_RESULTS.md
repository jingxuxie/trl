# BMM-TRL next steps after fast-decision results

Date: 2026-06-11

This plan follows `BMM_TRL_FAST_DECISION_RESULTS_20260611_124920.md`.

## Executive decision

The latest results give a **clear enough decision**:

```text
Continue flat one-step Q policy extraction: no.
Stop the whole project: no.
Pivot to BMM as reachability / subgoal-planning: yes.
```

The critic-side budget-holdout story remains positive, including on offline-support graph labels. But the current flat action-ranking story is not positive enough to justify policy benchmarks.

The next stage should be a short, evaluation-heavy pivot test:

```text
Can BMM improve joint action-subgoal selection?
```

If yes, continue as a high-level planning/subgoal project. If no, pause or move to a different project.

## What the fast-decision results say

### 1. Flat action ranking is not the right next policy path

The action-ranking evaluator is now much more credible than the earlier version:

- oracle distance score is perfect;
- V-next teacher is meaningfully above random;
- using candidate actions with their own logged source state improves learned critics;
- the repeated-source counterfactual issue was real and has been diagnosed.

However, BMM does not beat A/F on learned Q action AUC. B has slightly better selected distance/success in own-state mode, but not enough to justify policy evaluation.

Conclusion:

```text
Do not run flat Q policy benchmarks yet.
```

### 2. Offline-support graph labels are not a blocker

The graph-label budget-holdout result is mildly positive:

```text
H120 graph holdout:
A supervised-only AUC: 0.9783
B Q/V transitive AUC: 0.9849
F V-next distill AUC: 0.9780
```

The graph sampler also looks healthy:

```text
acceptance = 1.0
effective unique witnesses = 4.0
replacement = 0.0
oracle branch labels = 1.0
```

This is important because it weakens the concern that the method only works with oracle grid labels. The effect is small, but it exists in a dataset-support graph setting.

Caveat: if `exp/bmm_pointmaze_graph.npz` was built from train+val, then this is still an oracle-support diagnostic, not a strict offline train-only claim. A strict train-only graph check remains useful later.

### 3. Subgoal selection is the most promising pivot

Existing subgoal-selection results show:

```text
B Q/V > A/F on state-valid fraction, action-valid fraction, path stretch, and midpoint error.
```

But absolute quality is still much worse than V/V teacher. This suggests the current BMM signal is more naturally high-level than flat action-level, but the subgoal diagnostic needs a joint `(a,w)` version before policy.

## Why this does not rule out the research idea

The original BMM goal was max-min reachability composition, not necessarily flat one-step Q ranking. The original prototype spec explicitly listed budget scan and HIGL-style high-level subgoal selection as longer-term extensions, and it warned that a large-budget critic can be too coarse for action ranking. The current results fit that warning.

So the right interpretation is:

```text
BMM reachability bootstrapping works at the critic level.
Flat action ranking is weak.
High-level subgoal planning is now the likely path.
```

## What the results do and do not rule out

### Ruled out or weakened

```text
The original logged-offset target was adequate.        Ruled out.
Grid oracle is the only possible target.               Weakened by graph result.
The action-ranking failure was only repeated-source bug. Weakened; own-state still not enough.
Flat max-budget Q scoring is ready for policy.          Ruled out for now.
```

### Not ruled out

```text
BMM as a high-level subgoal planner.
BMM on strict train-only support graphs.
BMM on larger mazes with more meaningful long budgets.
BMM combined with a separate low-level controller.
```

### Still concerning

```text
The effect sizes are modest.
Flat action ranking did not improve.
Subgoal absolute quality is still poor.
General offline graph construction is unresolved.
```

## Runtime policy from now on

Use this strict rule:

```text
No new >30 minute training run until a cached evaluation diagnostic is positive.
```

Default experimental unit:

```text
one seed
A/B/F only
cached query sets
128 queries first, then 512 only if positive
reuse existing checkpoints
```

Do not run:

```text
broad sweeps
full A-G matrices
three seeds before one-seed signal
large policy benchmarks
flat Q policy training
```

## Next fast test: joint `(a,w)` selection

This is the highest-value next diagnostic.

### Motivation

Current diagnostics separately test:

```text
action ranking: choose a
subgoal selection: choose w using logged action
```

But BMM's natural policy object is joint:

```text
choose (a,w) maximizing min(Q_h(s,a,w), V_{H-h}(w,g))
```

This combines action and subgoal. It is much closer to the actual max-min idea than flat `Q_H(s,a,g)` ranking.

### Script to add

```text
scripts/eval_bmm_joint_action_subgoal.py
```

### Inputs

Reuse cached action-ranking queries:

```text
source state s
candidate actions a_i with candidate next states s'_i
goal g
budget H=160, split h=80/80
```

For each query, sample candidate subgoals `w_j`.

### Score

For each candidate pair `(a_i, w_j)`:

```text
score(a_i,w_j) = min(Q_h(s_i, a_i, w_j), V_{H-h}(w_j, g))
```

Use two state choices for Q branch:

```text
own_state mode: Q_h(s_i_candidate, a_i, w_j)
source_state mode: Q_h(s_logged, a_i, w_j)
```

The own-state mode is diagnostic; source-state mode is closer to policy but noisier.

### Oracle labels

For action-conditioned validity:

```text
action_valid(a_i,w_j) = 1[d(s'_i,w_j) <= h-1 and d(w_j,g) <= H-h]
```

For state-only validity:

```text
state_valid(w_j) = 1[d(s,w_j) <= h and d(w_j,g) <= H-h]
```

### Metrics

Report:

```text
selected action_valid fraction
selected state_valid fraction
selected next path stretch
selected source path stretch
selected midpoint error
selected action midpoint error
selected next distance to subgoal
selected final right distance
selected unique subgoal cells
selected action diversity
```

Also report oracle ceilings:

```text
oracle_best_action_subgoal
V/V teacher subgoal
random
```

### Compare

Use existing checkpoints:

```text
A: no transitive
B: Q/V transitive
F: V-next distillation
V/V teacher baseline
oracle baseline
random
```

No new training.

### Go / pivot / stop criterion

Continue toward BMM high-level planning if:

```text
B > A and B > F on action_valid or path stretch,
and oracle/V teacher baselines show the candidate set is meaningful.
```

Pause or pivot away if:

```text
oracle and V teacher are strong,
but B does not beat A/F in joint action-subgoal selection.
```

## Secondary fast test: strict train-only graph target

The graph result is important, but the strict general-offline version should avoid validation edges.

Add or run:

```text
graph_scope = train_only
val_mapping = nearest_train_bin_or_node
```

Report:

```text
val coverage
nearest train distance distribution
unmapped val fraction
```

Run one seed A/B/F with 500 steps only if the joint diagnostic is positive or if advisor discussion needs a general-offline target result.

Success:

```text
B > A on heldout parent budget
F near A
```

Failure interpretation:

```text
If train-only graph fails but train+val graph works, graph construction/generalization is the bottleneck.
```

## If joint diagnostic is positive

Run a tiny high-level policy smoke, not a benchmark.

### Policy sketch

At each decision:

```text
1. Use V to choose subgoal w = argmax_w min(V_h(s,w), V_{H-h}(w,g)).
2. Use Q to choose action a = argmax_a Q_h(s,a,w).
3. Execute for a short duration or until near w.
4. Replan.
```

Candidate actions can come from:

```text
FRS samples
nearest-neighbor dataset actions
behavior actor candidates
```

First smoke metrics:

```text
first-step progress to subgoal
subgoal reach rate
final distance to goal
success rate, but only as secondary
failure examples
```

Run one seed only.

## If joint diagnostic is negative

Do not run policy.

Decision options:

### Pivot to graph/planning BMM

Use BMM to learn/cache reachability over a support graph and do explicit graph search/subgoal selection. This is still a coherent project, but it is less of a flat neural RL method.

### Pause this project

If:

```text
flat action ranking negative
joint action-subgoal negative
strict graph target weak
```

then the current neural BMM direction is not promising enough for more OGBench time. Package the diagnostics as lessons learned and move on.

## Minimal next task list

1. Implement `scripts/eval_bmm_joint_action_subgoal.py`.
2. Use existing cached H160 action queries and existing A/B/F checkpoints.
3. Run 128-query joint diagnostic first.
4. If B > A/F, rerun 512 queries and write a short summary.
5. If positive, run one high-level policy smoke.
6. If negative, run only the strict train-only graph target if needed for final decision.
7. Decide continue / pivot / pause.

## Bottom line

The fast-decision results provide a clear direction:

```text
Do not continue flat Q extraction.
Do not stop the whole project yet.
Pivot to high-level BMM subgoal/action selection and validate it with one fast joint diagnostic.
```

If that joint diagnostic is positive, the project has a viable path. If it is negative under strong oracle/V-teacher baselines, it is reasonable to stop or substantially reframe the project.
