# BMM-TRL next steps after geodesic Q and first transitive diagnostics

Date: 2026-06-10

This plan follows `BMM_TRL_GEODESIC_TRANSITIVE_RESULTS_20260610_192601.md`.

## Executive conclusion

Yes, this is real progress.

You have now passed three important gates:

1. clean state-only geodesic `V_H(s,g)` works on PointMaze medium;
2. clean action-conditioned geodesic `Q_H(s,a,g)=1[d(s_next,g)<=H-1]` works;
3. the first state-only max-min transitive consistency run does not break the geodesic value diagnostic.

The mixed sparse-label transitive result does **not** look like a clear bug yet. It looks like an under-instrumented first transitive experiment with a one-witness target and already-strong supervised baselines.

The best next step is **not** policy evaluation yet and **not** PointMaze large yet. The best next step is:

```text
Instrument state-only V transitive, run a small controlled lambda/witness sweep, then implement Q/V transitive with a V teacher/checkpoint.
```

## What the current transitive result means

### What is good

The abundant-label comparison is a successful no-degradation check:

```text
H=64:  tied AUC, slightly smaller gap
H=128: better AUC and gap
monotonicity: 0.0000
```

This means the basic transitive plumbing is probably sane enough to continue.

### What is not proven yet

The sparse-label comparison does not prove a BMM label-efficiency gain:

```text
H=64:  gap improves, AUC dips slightly
H=128: AUC and gap are worse
```

But this is only one sparse setting, one transitive weight, one seed, one witness, and only two budgets. Also, the sparse supervised-only run is already very strong, so there is little room for transitive consistency to help.

### Why this does not imply a bug

The current transitive loss uses a soft bootstrapped target:

```text
y_trans = min(V_h(s,w), V_{H-h}(w,g))
```

Even when the witness is geodesically valid, the branch predictions can be biased or under-confident. With one sampled witness, the transitive target can act like noisy positive regularization. In sparse settings this can improve score scale while slightly hurting ranking, which matches the observed `H=64` pattern.

A real bug would be more likely if:

```text
abundant-label transitive caused large degradation,
monotonicity collapsed,
transitive targets were mostly invalid,
or y_trans was uncorrelated with oracle-positive parent labels.
```

That is not what the current results show.

## Priority order

Do these in order:

1. **Instrument state-only V transitive coverage and targets.**
2. **Run a small lambda/witness sweep on V transitive.**
3. **Run a sparse-label ablation only after instrumentation is clean.**
4. **Implement Q/V transitive with a state-only V teacher.**
5. **Only then consider PointMaze large or policy evaluation.**

## Milestone 1: instrument state-only V transitive

Before more sweeps, add diagnostics to `scripts/train_bmm_geodesic_value.py`.

### Add sampling metrics

For every transitive batch, log:

```text
trans_sample_acceptance_rate
trans_attempts_per_sample
trans_budget_count/H=64
trans_budget_count/H=128
trans_parent_distance_mean/H
trans_left_distance_mean/H
trans_right_distance_mean/H
trans_left_slack_mean/H     = h - d(s,w)
trans_right_slack_mean/H    = H-h - d(w,g)
trans_witness_cell_count_mean/H
trans_unique_witness_frac/H
trans_zero_left_frac/H      = fraction d(s,w)=0
trans_zero_right_frac/H     = fraction d(w,g)=0
```

These reveal whether witnesses are degenerate or badly distributed.

### Add target metrics

In `BMMTRLAgent.critic_loss` or the training script logs, report by budget:

```text
y_trans_mean/H
first_r_mean/H
second_r_mean/H
parent_r_mean/H
loss_trans/H
loss_sup/H
loss_trans_over_sup
```

Also report:

```text
trans_parent_oracle_label = 1[d(s,g)<=H]
trans_branch_oracle_valid = 1[d(s,w)<=h and d(w,g)<=H-h]
```

For the geodesic-valid sampler, these should be almost always one. If not, debug sampling.

### Add witness histograms to output JSON

Store coarse histograms:

```text
parent distance / H
left distance / h
right distance / (H-h)
left slack
right slack
```

This is more informative than only final AUC/gap.

## Milestone 2: quick V-transitive sweep

After instrumentation, run a small sweep on the medium maze.

### Budgets

Use:

```text
budgets=(64,128)
```

This is reasonable because the half splits are `(32,64)`, and `H=32 -> h=16` is below one calibrated grid cell on PointMaze medium.

Optionally add a cell-aligned stress test:

```text
budgets=(40,80,160)
```

because `steps_per_cell ~= 19.8`, so these roughly correspond to `2,4,8` grid cells. Use this only as a diagnostic, not as the main BMM benchmark, because `H=160` has limited negatives.

### Lambda sweep

Run:

```text
lambda_trans in {0.0, 0.01, 0.025, 0.05}
```

For each setting, record:

```text
heldout AUC/gap by H
BCE/calibration by H
monotonicity violation
loss_trans_over_sup
witness histograms
```

### No-degradation gate

Treat a setting as safe if:

```text
AUC drop <= 0.005 to 0.01 versus supervised-only
score gap drop <= 0.03
monotonicity remains <= 0.01
```

The likely safe default is:

```text
lambda_trans = 0.01 or 0.025
```

Do not use `0.05` unless it is clearly no-degradation.

## Milestone 3: meaningful sparse-label ablation

The current sparse test used `batch_size=64`, but supervised-only was still very strong. To test whether BMM helps, make supervision genuinely scarce.

### Sparse settings

Use fixed training steps and compare:

```text
batch_size=256  # abundant
batch_size=64   # 4x fewer labels
batch_size=32   # 8x fewer labels
batch_size=16   # 16x fewer labels, if stable
```

Alternatively keep batch size fixed and reduce direct supervised pairs while keeping transitive pairs fixed:

```text
sup_pairs_per_budget in {256,64,32,16}
trans_pairs_per_update fixed at 256
```

This is cleaner because it directly tests whether transitive consistency supplies extra structure when labels are scarce.

### Seeds

Use at least:

```text
seeds = 0, 1, 2
```

Do not overinterpret a single sparse run.

### Success criterion

The meaningful claim is not just higher final AUC once. The useful claim is:

```text
transitive reaches the threshold in fewer updates,
or keeps AUC/gap higher under 8x/16x fewer direct labels,
or improves monotonicity/calibration without hurting AUC.
```

### Suggested table

| setting | seed | lambda_trans | labels/update | H | AUC | gap | BCE | mono | steps_to_threshold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| supervised | 0 | 0 | 64 | 64 | | | | | |
| transitive | 0 | 0.01 | 64 | 64 | | | | | |

## Milestone 4: improve witness sampling if needed

If instrumentation shows poor witness coverage, fix the sampler before running Q/V transitive.

### Current likely issues to check

1. **One-witness variance.** A single witness may be too noisy.
2. **Degenerate witnesses.** The sampler may overuse `w=s` or very near-source witnesses.
3. **Slack imbalance.** One branch may be easy while the other is near the budget boundary.
4. **Parent distribution mismatch.** Transitive parents may not match supervised eval pairs.
5. **Branch budget calibration.** Step budgets and cell distances may not align smoothly because `steps_per_cell ~= 19.8`.

### Fixes

Add:

```text
num_trans_witnesses = K in {1,4,8}
```

Use:

```text
y_trans = max_w min(V_h(s,w), V_{H-h}(w,g))
```

Try witness sampling modes:

```text
uniform_valid
boundary_balanced       # left/right distances near branch budgets
slack_balanced          # avoid all-easy witnesses
multi_witness_max       # K witnesses then max-min
```

Start with `K=4`. If `K=4` helps sparse results, that is a strong sign the mixed sparse result was sampling variance, not a conceptual issue.

## Milestone 5: implement Q/V transitive next

Once V transitive is instrumented and no-degradation is confirmed, implement Q/V transitive.

Do **not** wait for a perfect sparse-label win before starting Q/V. The action-conditioned Q path is required for policy, and Q itself already passes cleanly.

### Correct Q/V target

For `(s,a,g,H)` and witness `w`:

```text
y_trans = max_w min(Q_h(s,a,w), V_{H-h}(w,g))
```

This is the action-conditioned version of the BMM decomposition. The first branch consumes the action. The second branch is state-only.

### Use a V teacher first

Do not try to learn both branches from one action-conditioned critic immediately.

Use one of these:

1. restore a passed state-only V checkpoint as a frozen teacher;
2. train a small V target in the same script but stop-gradient it for Q transitive;
3. use oracle geodesic labels for the second branch only as a debugging target.

The cleanest first diagnostic is:

```text
Q supervised labels + Q/V transitive where V branch is a frozen passed V checkpoint.
```

Your `train_bmm_geodesic_q.py` already has value-checkpoint flags for Q-V_next consistency. Extend that mechanism to score `V_{H-h}(w,g)` inside the transitive target.

### Q/V transitive experiment matrix

Run:

```text
Q supervised only
Q supervised + Q/V transitive, lambda_trans=0.01
Q supervised + Q/V transitive, lambda_trans=0.025
```

Use budgets:

```text
(64,128)
```

then stress:

```text
(32,64,128)
```

### Q/V success criterion

First gate:

```text
no degradation versus Q supervised-only
```

Second gate:

```text
improved label efficiency in sparse Q labels
```

Do not require policy improvement yet.

## Milestone 6: move to PointMaze large only after medium transitive story is stable

PointMaze large is useful for testing `H=256/512`, but it should not be used to debug transitive basics.

Before moving:

```bash
conda run -n bmm-trl python scripts/inspect_pointmaze_grid_bfs.py \
  --env_name=pointmaze-large-navigate-v0 \
  --budgets=64,128,256,512
```

Proceed only if class coverage is healthy for the budgets you care about.

Large-maze goal:

```text
Does the medium-maze transitive pattern reproduce at larger diameter?
```

not:

```text
Debug sampler, target, Q/V decomposition, and policy all at once.
```

## Milestone 7: policy evaluation comes after Q/V transitive

Policy evaluation should wait until:

1. `V_H(s,g)` geodesic supervised passes;
2. `Q_H(s,a,g)` geodesic supervised passes;
3. V transitive is no-degradation and instrumented;
4. Q/V transitive is no-degradation and instrumented.

Then run policy smoke tests using budget scan:

```text
H_hat(s,g) = smallest H with V_H(s,g) >= tau
H_action = clamp(H_hat, min_budget, max_budget)
a = argmax_a Q_H_action(s,a,g)
```

Do not always score actions at max budget. The original project spec already warned that a large-budget critic can be too coarse for action ranking; use the smallest reachable budget for sharper action discrimination.

## Direct answer to the options

### 1. Sweep state-only transitive settings first?

Yes, but keep it small and add instrumentation first.

Recommended:

```text
lambda_trans={0.0,0.01,0.025,0.05}
budgets=(64,128)
seeds={0,1,2} if affordable
```

### 2. Improve witness sampling or budget calibration first?

Improve **diagnostics** first, then sampling if diagnostics show a problem.

Add witness distance/slack histograms and acceptance rates. If they show degeneracy, add `K=4` multi-witness max and boundary-balanced witnesses.

### 3. Implement action-conditioned Q/V transitive next?

Yes, after the small V-transitive instrumentation/sweep. Do not wait for a perfect sparse V result.

Q/V transitive is the policy-relevant BMM target.

### 4. Run fuller sparse-label ablation table?

Yes, but only after the transitive sampler is instrumented. Use stronger scarcity than the current `batch_size=64` test, and use multiple seeds.

### 5. Move to PointMaze large?

Not yet. Medium still has the right next questions: instrumented transitive and Q/V decomposition. Move to large when you need larger horizon coverage, not while basic transitive questions remain unresolved.

## Immediate task list for Codex

1. Add transitive witness diagnostics to `train_bmm_geodesic_value.py`.
2. Add output JSON fields for witness histograms and transitive target means.
3. Run small V transitive sweep: `lambda_trans={0,0.01,0.025,0.05}` on `(64,128)`.
4. Add `num_trans_witnesses` and implement `K`-witness max-min; start with `K=4`.
5. Run sparse-label ablation with `sup_pairs_per_budget={256,64,32,16}` or equivalent.
6. Extend `train_bmm_geodesic_q.py` with Q/V transitive using a frozen V teacher checkpoint.
7. Run Q supervised-only vs Q+Q/V-transitive at `lambda_trans={0.01,0.025}`.
8. Only after Q/V no-degradation, inspect PointMaze large diameter and start larger-horizon diagnostics.

## Bottom line

You are making progress. The mixed sparse-label result is not a reason to stop or jump to policy; it is a reason to instrument the transitive sampler and run a controlled sweep.

The highest-probability path to progress is:

```text
instrument V transitive -> small lambda/K-witness sweep -> Q/V transitive with V teacher -> sparse-label table -> larger maze -> policy smoke
```

This keeps the research question focused: whether max-min transitive consistency improves sample efficiency or budget consistency on a clean reachability target.
