# BMM-TRL next steps after budget-holdout results

Date: 2026-06-11

This plan follows `BMM_TRL_BUDGET_HOLDOUT_RESULTS_20260611_003902.md`.

## Executive conclusion

You are making real progress.

This is the first result that directly tests the core BMM idea in the right way:

```text
Can shorter-budget Q/V knowledge help a longer parent budget when parent labels are missing or scarce?
```

The answer is now **tentatively yes** on PointMaze-medium grid-cell budget holdout.

Across seeds 0 and 1:

```text
No-parent H8: Q/V transitive improves H8 AUC and gap over no-transitive.
Few-parent H8: Q/V transitive improves H8 AUC and gap over few-label supervised.
V-next distillation: nearly matches no-transitive, so it does not explain the gain.
```

This is stronger than the earlier uniform sparse-Q table because it matches the theory: BMM should help compose short-horizon reachability into a longer-horizon parent, not necessarily improve every budget when all labels are uniformly reduced.

The result is still modest and not final, but it is the first genuinely BMM-specific positive signal.

## Why this took so long

The project has spent a lot of time in prototype mode because several early assumptions were wrong or underspecified:

1. **Logged offset was the wrong high-budget target.** It behaved like behavior time, not reachability.
2. **PointMaze medium H=256/512 were not meaningful binary geodesic tests.** They are above the calibrated maze diameter.
3. **Uniform sparse-label reduction was not the right BMM test.** It reduced all budgets together instead of testing long-horizon bootstrapping.
4. **Q/V transitive needed the correct action-conditioned semantics.** The useful target is `Q_H(s,a,g)=1[d(s_next,g)<=H-1]`, not logged future offset.
5. **Witness geometry mattered.** Some budget choices had too few nondegenerate witnesses.

This is not wasted effort. It narrowed the question from:

```text
Does this prototype work at all?
```

to:

```text
Under what target and supervision regime does max-min transitive composition add value?
```

That is the right research question.

## What the current result does and does not prove

### It supports

```text
BMM transitive can improve heldout parent-budget classification when shorter-budget labels are available.
```

The key deltas are consistent across seeds:

```text
Seed 0, no H8 parent labels:
B - A = +0.0175 AUC, +0.0703 gap, -0.0277 Q-V abs diff

Seed 1, no H8 parent labels:
B - A = +0.0172 AUC, +0.0571 gap, -0.0176 Q-V abs diff

Seed 0, 16 H8 labels:
D - C = +0.0082 AUC, +0.0533 gap

Seed 1, 16 H8 labels:
D - C = +0.0089 AUC, +0.0217 gap
```

### It does not yet prove

```text
BMM improves final policy performance.
BMM scales to large tasks.
BMM beats all simpler ways to train a Q critic.
The effect is large enough for a paper claim.
```

Those are later milestones.

## Immediate next decision

Do **not** go to full policy evaluation yet.

Do **not** return to broad uniform sparse sweeps.

The immediate next step is to confirm and package the budget-holdout signal with a minimal, decisive set of runs.

## Milestone 1: finish the budget-holdout replication

### Required runs

Run seed 2 only for the informative rows:

```text
A. no H8 labels, no transitive
B. no H8 labels + Q/V transitive
C. 16 H8 labels, no transitive
D. 16 H8 labels + Q/V transitive
F. no H8 labels + V-next distillation
```

Skip the full A-G matrix unless there is a specific reason. The current evidence says the key comparisons are:

```text
B vs A
D vs C
F vs A
```

### Optional seed 1 oracle control

Finish seed 1 G only if convenient:

```text
G. no H8 labels + oracle Q/V transitive
```

This is useful, but not as urgent as seed 2 A/B/C/D/F.

### Reporting

Report heldout-parent metrics separately from full gate pass:

```text
H8 AUC
H8 gap
H8 BCE
H8 ECE
ensemble-min H8 AUC/gap
Q-V abs diff
Q-V rank correlation
```

The full all-budget gate can remain in the report, but it should not decide whether the budget-holdout hypothesis passed. Some rows fail only because H2 is near threshold, while H8 is strong.

### Success criterion

If seed 2 shows:

```text
B - A improves H8 AUC and gap
D - C improves H8 AUC and gap
F - A does not explain the gain
```

then you have a real medium-grid-cell BMM budget-holdout result.

## Milestone 2: aggregate the result cleanly

Add a summary script:

```text
scripts/summarize_bmm_budget_holdout.py
```

It should aggregate runs by comparison:

```text
B - A
D - C
F - A
G - B, if oracle rows exist
```

Report:

```text
mean delta AUC
mean delta gap
mean delta BCE
mean delta ECE
mean delta Q-V abs
per-seed deltas
bootstrap or simple standard error if easy
```

Produce one compact table:

| comparison | seeds | delta H8 AUC | delta H8 gap | delta Q-V abs | interpretation |
|---|---:|---:|---:|---:|---|
| B-A | 0,1,2 | | | | no-parent BMM effect |
| D-C | 0,1,2 | | | | few-parent BMM effect |
| F-A | 0,1,2 | | | | V-next distill control |

This table is more valuable than another long diagnostic markdown.

## Milestone 3: env-step holdout after grid-cell replication

If seed 2 confirms the trend, run the env-step version:

```text
geodesic_budget_unit = env_steps
eval_budgets = (40,80,160)
supervised_budgets = (40,80)
trans_budgets = (160)
heldout_parent_budget = 160
```

Use the same comparisons:

```text
A/B/C/D/F
```

Start with seed 0. If positive, repeat seeds 1 and 2.

This tests whether the BMM signal survives the policy-facing budget scale.

## Milestone 4: compare against a non-BMM compositional baseline

You already have V-next distillation, which is good. Add one more cheap control before policy:

### Parent pseudo-label from oracle distance threshold

This is not a learnable baseline, but it tells the ceiling:

```text
oracle H8 upper bound
full supervised H8 upper bound
```

You already have E full supervised. Keep it in the summary.

### Optional: monotone budget interpolation baseline

Train only H2/H4 and predict H8 by monotonic extrapolation or max-budget bias. This can be very simple:

```text
score_H8_baseline = max(score_H2, score_H4)
```

If this weak baseline is near A and below B, it strengthens the BMM-specific story.

Do not spend too much time here unless BMM gains hold over 3 seeds.

## Milestone 5: decide whether to start policy smoke

Policy smoke becomes reasonable after one of these is true:

```text
A. grid-cell and env-step budget-holdout both show positive BMM deltas; or
B. grid-cell holdout is robust and you explicitly frame policy as exploratory.
```

The first policy smoke should not be a full benchmark comparison. It should answer:

```text
Does budget-conditioned Q action scoring produce sensible behavior at all?
```

Use:

```text
H_hat(s,g) = smallest H where V_H(s,g) >= tau
H_action = clamp(H_hat, min_budget, max_budget)
a = argmax_a Q_H_action(s,a,g)
```

Do not score actions only at max budget. The original risk remains: max-budget reachability can be too coarse for action ranking.

## Milestone 6: when to move to PointMaze large

Move to large after the medium budget-holdout story is summarized.

Before training:

```bash
conda run -n bmm-trl python scripts/inspect_pointmaze_grid_bfs.py \
  --env_name=pointmaze-large-navigate-v0 \
  --budgets=64,128,256,512
```

Choose budgets from calibrated diameter and class balance. Do not force 256/512 unless they are meaningful binary tests.

Large-maze goal:

```text
Does the budget-holdout BMM effect persist at larger diameter?
```

not:

```text
Debug policy and transitive training at the same time.
```

## If seed 2 fails

If seed 2 does not confirm B-A and D-C, do not immediately abandon the idea.

First inspect:

```text
H8 parent label distribution
witness effective K
qv target-parent stats
teacher V-next AUC on H8
whether F still matches A
```

Then run only the oracle branch row:

```text
G. no H8 labels + oracle Q/V transitive
```

Interpretation:

- If G helps but B does not, learned branch quality is the bottleneck.
- If G also does not help, the current parent objective or architecture is the bottleneck.
- If F helps as much as B, the gain is teacher distillation, not BMM.

This gives a clean pivot decision.

## Research story update

The story should now be:

```text
BMM-TRL requires clean reachability targets.
Logged behavior-time labels fail at high budgets.
Grid/geodesic labels give a clean testbed.
Uniform sparse labels are not the right stress test.
Budget-holdout is the right test of max-min transitive bootstrapping.
Initial budget-holdout results are modest but consistently positive across two seeds.
```

This is a much stronger story than the earlier broad sparse-Q result.

## Why this does not mean the method is hopeless

Prototype difficulty here came mostly from target definition, not from the max-min idea alone.

You had to discover:

```text
same-trajectory offset != reachability
PointMaze-medium 256/512 are one-class under geodesic distance
Q labels need next-state semantics
sampled max-min is better viewed as lower-bound consistency
uniform sparse labels do not test long-horizon bootstrapping well
```

Those discoveries are painful but normal for offline RL research. The latest budget-holdout result is the first experiment that actually matches the core hypothesis.

## What to stop doing now

Stop:

```text
broad uniform sparse-Q sweeps
loss-mode tuning in abundant labels
policy evaluation before budget-holdout replication
large-maze runs before medium summary
ranking/monotonicity additions
logged-offset hard negatives
```

## Immediate task list

1. Run seed 2 A/B/C/D/F for grid-cell budget holdout.
2. Optionally run seed 1 G oracle control.
3. Add `scripts/summarize_bmm_budget_holdout.py`.
4. Produce a 3-seed B-A / D-C / F-A delta table.
5. If positive, run env-step holdout `(40,80)->160` seed 0.
6. If env-step seed 0 is positive, repeat seeds 1 and 2.
7. Only then decide between policy smoke and PointMaze-large scaling.

## Bottom line

You are no longer just tuning a fragile prototype. You finally have a focused result that matches the BMM research claim.

The next step is not more open-ended prototyping. It is replication and summarization of the budget-holdout effect.

If the 3-seed budget-holdout table holds, you have a credible proof-of-concept for BMM-style transitive bootstrapping on clean reachability labels.
