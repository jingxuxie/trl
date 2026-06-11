# BMM-TRL Budget-Holdout Results

Date: 2026-06-11

This note summarizes the first budget-holdout bootstrapping diagnostic from
`BMM_TRL_NEXT_STEPS_AFTER_SPARSE_Q.md`.

## Implementation

Added budget-holdout support to `scripts/train_bmm_geodesic_q.py`:

- `--eval_budgets`
- `--supervised_budgets`
- `--parent_label_budget_frac`
- `--parent_label_pairs_per_budget`
- `--lambda_vnext_distill`
- `--vnext_distill_loss_type`
- `--vnext_distill_bce_margin`
- `--qv_branch_mode`

The Q/V transitive sampler now records oracle parent, left-branch, and
right-branch labels. The update path supports:

- learned Q + frozen V transitive target,
- oracle Q + frozen V target,
- oracle Q + oracle V target,
- direct V-next distillation control.

Added `scripts/run_bmm_qv_budget_holdout.py` for the A-G budget-holdout matrix
and `scripts/test_bmm_qv_budget_holdout.py` for runner/summary checks.

## Setup

Grid-cell PointMaze medium budget holdout:

```text
geodesic_budget_unit = grid_cells
eval_budgets = (2, 4, 8)
supervised_budgets = (2, 4)
trans_budgets = (8)
heldout_parent_budget = 8
value_teacher = exp/bmm_grid_cells_value_teacher_2_4_8/params_1000.pkl
steps = 1000
batch_size = 256
sup_pairs_per_budget = 256
trans_pairs_per_update = 256
num_trans_witnesses = 4
trans_witness_mode = slack_balanced
qv_trans_loss_type = bce_lower_bound
lambda_qv_trans = 0.01
lambda_vnext_distill = 0.01 for the V-next control only
```

Run directories:

```text
seed 0 full A-G: exp/bmm_qv_budget_holdout_20260611_000227
seed 1 A-F only: exp/bmm_qv_budget_holdout_20260611_001949
```

Seed 1 was stopped before seed 1 G and seed 2 because the partial repeat was
already taking significant time.

## Seed 0 H8 Parent Budget

| Variant | H8 AUC | H8 gap | min AUC | min gap | Q-V abs diff | Pass |
|---|---:|---:|---:|---:|---:|---|
| A no H8 labels, no transitive | 0.9230 | 0.5278 | 0.9218 | 0.5136 | 0.1843 | false |
| B no H8 labels + Q/V transitive | 0.9405 | 0.5981 | 0.9407 | 0.5850 | 0.1566 | false |
| C 16 H8 labels, no transitive | 0.9578 | 0.6405 | 0.9557 | 0.6260 | 0.1588 | true |
| D 16 H8 labels + Q/V transitive | 0.9660 | 0.6938 | 0.9654 | 0.6914 | 0.1509 | true |
| E full supervised upper bound | 0.9913 | 0.8002 | 0.9910 | 0.8024 | 0.1564 | true |
| F no H8 labels + V-next distill | 0.9232 | 0.5291 | 0.9224 | 0.5150 | 0.1837 | false |
| G no H8 labels + oracle Q/V transitive | 0.9419 | 0.6119 | 0.9407 | 0.5981 | 0.1529 | false |

Seed 0 deltas:

```text
B - A: +0.0175 H8 AUC, +0.0703 H8 gap, -0.0277 Q-V abs diff
D - C: +0.0082 H8 AUC, +0.0533 H8 gap, -0.0078 Q-V abs diff
F - A: +0.0002 H8 AUC, +0.0014 H8 gap, -0.0006 Q-V abs diff
G - B: +0.0014 H8 AUC, +0.0139 H8 gap, -0.0036 Q-V abs diff
```

## Seed 1 H8 Parent Budget

| Variant | H8 AUC | H8 gap | min AUC | min gap | Q-V abs diff | Pass |
|---|---:|---:|---:|---:|---:|---|
| A no H8 labels, no transitive | 0.9199 | 0.5186 | 0.9182 | 0.5085 | 0.1930 | false |
| B no H8 labels + Q/V transitive | 0.9371 | 0.5757 | 0.9365 | 0.5634 | 0.1753 | false |
| C 16 H8 labels, no transitive | 0.9462 | 0.6146 | 0.9445 | 0.6073 | 0.1685 | false |
| D 16 H8 labels + Q/V transitive | 0.9550 | 0.6363 | 0.9538 | 0.6345 | 0.1697 | true |
| E full supervised upper bound | 0.9868 | 0.7675 | 0.9869 | 0.7674 | 0.1567 | false |
| F no H8 labels + V-next distill | 0.9203 | 0.5198 | 0.9186 | 0.5096 | 0.1927 | false |

Seed 1 deltas:

```text
B - A: +0.0172 H8 AUC, +0.0571 H8 gap, -0.0176 Q-V abs diff
D - C: +0.0089 H8 AUC, +0.0217 H8 gap, +0.0012 Q-V abs diff
F - A: +0.0004 H8 AUC, +0.0012 H8 gap, -0.0002 Q-V abs diff
```

Run-level pass/fail is stricter than the heldout H8 question. For example,
seed 1 E has strong H8 but fails because H2 ensemble-min AUC is slightly below
the `0.90` gate. Seed 1 C similarly misses the full gate by a near-threshold H2
row, while seed 1 D passes.

## Interpretation

The core budget-holdout signal is positive across seeds 0 and 1:

```text
No-parent H8: Q/V transitive improves AUC and gap over the no-transitive baseline.
Few-parent H8: Q/V transitive improves AUC and gap over the few-label baseline.
V-next distillation: essentially matches the no-transitive baseline, so it does
not explain the Q/V transitive improvement.
```

This is a stronger BMM-specific result than the earlier uniform sparse-Q table.
The gain is modest but consistent on the heldout parent budget.

## Current Recommendation

Do not run policy evaluation yet.

Next useful runs:

```text
1. Finish only the missing oracle control for seed 1 if needed.
2. Run seed 2 only for A/B/C/D/F, not the full A-G matrix.
3. If seed 2 repeats the same H8 trend, run the env-step holdout (40,80)->160.
4. Keep reporting heldout-parent H8 metrics separately from all-budget gate pass.
```

The immediate evidence supports continuing the budget-holdout line rather than
returning to broad uniform sparse-label sweeps.
