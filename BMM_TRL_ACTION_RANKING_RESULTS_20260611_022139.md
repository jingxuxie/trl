# BMM-TRL Action-Ranking Results

Date: 2026-06-11

## What Ran

Following `BMM_TRL_NEXT_STEPS_AFTER_HOLDOUT_REPLICATION.md`, I stopped broad sweeps and ran only the important A/B/F comparison for the env-step budget holdout:

```text
env: pointmaze-medium-navigate-v0
budgets: 40, 80, 160
supervised budgets: 40, 80
heldout parent budget: 160
seed: 0
steps: 1000
checkpoint saving: enabled
run dir: exp/bmm_qv_budget_holdout_20260611_021352
```

Variants:

```text
A: no parent labels, no transitive loss
B: no parent labels, Q/V transitive lower-bound loss, lambda_qv_trans=0.01
F: no parent labels, V-next distillation control, lambda_vnext_distill=0.01
```

## Critic Holdout Check

The saved-checkpoint rerun reproduced the earlier seed-0 critic result at H160.

| comparison | delta AUC | delta gap | delta BCE | delta ECE | gate |
|---|---:|---:|---:|---:|---|
| B-A | +0.0400 | +0.0640 | -0.5976 | -0.0398 | B passes, A fails |
| F-A | +0.0035 | +0.0067 | -0.0774 | -0.0038 | F fails |

This confirms the critic-side story: BMM Q/V transitive bootstrapping improves the heldout parent-budget reachability metric more than the V-next control.

## Offline Action-Ranking Diagnostic

I added `scripts/eval_bmm_action_ranking.py` and evaluated the saved A/B/F critics on the same heldout action-ranking set.

Diagnostic setup:

```text
budget: H160
queries: 512
candidates per query: 8
candidate source: logged transitions from the same grid cell
oracle label: d_grid(s_next_candidate, g) <= H - 1
positive fraction: 0.4082
source cells: 20
goal cells: 16
```

Main parent-budget scores:

| critic | pairwise accuracy | action AUC | score gap | selected distance | selected improves | selected success |
|---|---:|---:|---:|---:|---:|---:|
| A parent | 0.6974 | 0.6004 | 0.0020 | 167.8775 | 0.1641 | 0.5078 |
| B parent | 0.6955 | 0.5848 | 0.0063 | 167.9935 | 0.1621 | 0.5000 |
| F parent | 0.6988 | 0.6009 | 0.0022 | 167.8775 | 0.1641 | 0.5078 |

Short-budget interpolation baseline, `max(score_H40, score_H80)`:

| critic | pairwise accuracy | action AUC | score gap | selected distance | selected improves | selected success |
|---|---:|---:|---:|---:|---:|---:|
| A interp | 0.6911 | 0.5995 | 0.0006 | 167.9549 | 0.1621 | 0.5039 |
| B interp | 0.6928 | 0.5773 | 0.0015 | 168.0322 | 0.1641 | 0.4980 |
| F interp | 0.6948 | 0.5994 | 0.0007 | 167.9162 | 0.1621 | 0.5078 |

## Interpretation

This is a mixed result:

```text
Positive: BMM improves the heldout parent-budget critic metric.
Negative: that improvement does not translate into better offline action ranking in this diagnostic.
```

B has the largest parent-score gap, but lower action AUC and slightly worse selected-action distance than A/F. Under the plan's go/no-go rule, this is not enough evidence to start a policy smoke.

The immediate next question should be whether the action-ranking diagnostic is too local/noisy or whether the learned BMM signal is better suited to high-level graph/subgoal planning than flat action selection.

## Recommended Next Step

Do not run a policy benchmark yet.

Recommended next diagnostics:

1. Stress-test the action-ranking evaluator with an oracle distance score and with the frozen V teacher to confirm the sampled candidate sets can distinguish useful actions.
2. Try a more structured candidate set: actions from neighboring cells or actions that intentionally cover different outgoing grid directions, not only arbitrary same-cell logged transitions.
3. If B still does not beat A/F on structured action ranking, pivot BMM toward high-level graph/subgoal selection rather than flat per-action Q ranking.

## Artifacts

```text
exp/bmm_qv_budget_holdout_20260611_021352/summary.json
exp/bmm_qv_budget_holdout_20260611_021352/action_ranking_abf_h160.json
exp/bmm_qv_budget_holdout_20260611_021352/action_ranking_abf_h160.md
scripts/eval_bmm_action_ranking.py
```
