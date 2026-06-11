# BMM-TRL Q/V Transitive Loss Ablation Results

Date: 2026-06-10

This follows `BMM_TRL_NEXT_STEPS_AFTER_QV_TRANSITIVE.md`. The goal was to test whether the sampled max-min Q/V target should be treated as an equality target or as a lower-bound consistency target before moving to sparse-Q label-efficiency experiments.

## Implementation

- Added `--qv_trans_loss_type` to `scripts/train_bmm_geodesic_q.py`:
  - `bce_equal`: existing equality BCE behavior.
  - `prob_hinge`: squared probability hinge on `max(0, y_trans - parent_r)`.
  - `bce_lower_bound`: BCE only when `y_trans > parent_r + margin`.
- Added `--qv_trans_bce_margin`, default `0.0`.
- Added Q/V target-vs-parent diagnostics:
  - `critic/qv_parent_r_mean`
  - `critic/qv_y_trans_mean`
  - `critic/qv_target_minus_parent_mean`
  - `critic/qv_frac_y_trans_gt_parent`
  - `critic/qv_frac_y_trans_lt_parent`
  - `critic/qv_first_q_mean`
  - `critic/qv_second_v_mean`
  - per-budget loss, parent, target, first-Q, and second-V means.
- Extended `scripts/test_bmm_qv_transitive_shapes.py` to run all three loss modes and require the new diagnostics.

## Verification

Passed:

```bash
python -m py_compile scripts/train_bmm_geodesic_q.py scripts/test_bmm_qv_transitive_shapes.py
conda run -n bmm-trl python scripts/test_bmm_qv_transitive_shapes.py
conda run -n bmm-trl python scripts/test_bmm_transitive_sampler.py
conda run -n bmm-trl python scripts/test_bmm_agent_shapes.py
conda run -n bmm-trl python scripts/test_pointmaze_grid_bfs.py
conda run -n bmm-trl python scripts/test_bmm_dataset_shapes.py
conda run -n bmm-trl python scripts/test_bmm_supervised_shapes.py
conda run -n bmm-trl python scripts/test_bmm_reachability_gate.py
conda run -n bmm-trl python scripts/test_bmm_hard_neg_shapes.py
conda run -n bmm-trl python scripts/test_bmm_tabular.py
```

Some CPU-only tests print the known JAX CUDA plugin warning before passing.

## Experiment Setup

Common setup:

```text
env=pointmaze-medium-navigate-v0
label=grid_geodesic
geodesic_budget_unit=env_steps
budgets=(40,80,160)
trans_budgets=(80,160)
steps=1000
batch_size=256
sup_pairs_per_budget=256
trans_pairs_per_update=256
eval_pairs=512
num_trans_witnesses=4
trans_witness_mode=slack_balanced
lambda_qv_trans=0.01 for Q/V transitive rows
frozen V teacher=exp/bmm_grid_value_qv_teacher_40_80_160/params_1000.pkl
```

JSON artifacts:

- `exp/bmm_grid_q_qv_loss_ablate_sup_40_80_160.json`
- `exp/bmm_grid_q_qv_loss_ablate_bce_equal_40_80_160.json`
- `exp/bmm_grid_q_qv_loss_ablate_prob_hinge_40_80_160.json`
- `exp/bmm_grid_q_qv_loss_ablate_bce_lower_bound_40_80_160.json`

## Final Eval Metrics

All rows passed the existing heldout threshold gate. Monotonicity violation was `0.0` for all final evals.

| mode | H40 AUC | H40 gap | H80 AUC | H80 gap | H160 AUC | H160 gap | Q-V next abs diff | Q-V next rank corr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| supervised only | 0.9563 | 0.5359 | 0.9786 | 0.5138 | 0.9875 | 0.6421 | 0.2036 | 0.9362 |
| bce_equal | 0.9565 | 0.5063 | 0.9763 | 0.4674 | 0.9857 | 0.6712 | 0.1345 | 0.9308 |
| prob_hinge | 0.9565 | 0.5084 | 0.9762 | 0.4692 | 0.9857 | 0.6719 | 0.1351 | 0.9304 |
| bce_lower_bound | 0.9564 | 0.5058 | 0.9765 | 0.4677 | 0.9857 | 0.6728 | 0.1327 | 0.9308 |

## Q/V Target Calibration

Final update diagnostics:

| mode | qv loss | parent mean | target mean | target-parent | frac target > parent | frac target < parent | first Q mean | second V mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bce_equal | 0.5050 | 0.7690 | 0.8197 | 0.0507 | 0.5918 | 0.4082 | 0.7590 | 0.8876 |
| prob_hinge | 0.0197 | 0.7715 | 0.8202 | 0.0487 | 0.5820 | 0.4180 | 0.7592 | 0.8876 |
| bce_lower_bound | 0.4990 | 0.7728 | 0.8202 | 0.0474 | 0.5840 | 0.4160 | 0.7597 | 0.8876 |

Per-budget final update snapshot:

| mode | H80 loss | H80 target | H80 parent | H160 loss | H160 target | H160 parent |
|---|---:|---:|---:|---:|---:|---:|
| bce_equal | 0.4916 | 0.8403 | 0.7822 | 0.5220 | 0.7937 | 0.7522 |
| prob_hinge | 0.0224 | 0.8409 | 0.7843 | 0.0164 | 0.7939 | 0.7552 |
| bce_lower_bound | 0.4904 | 0.8405 | 0.7858 | 0.5207 | 0.7946 | 0.7563 |

Witness diversity at the final update stayed the same across Q/V modes because the sampler and seed are shared:

```text
H80:  effective K = 1.0559, replacement frac = 1.0000
H160: effective K = 2.3805, replacement frac = 0.6726
```

## Interpretation

The new diagnostics confirm the conceptual concern but do not show a clear abundant-label win from lower-bound loss modes at `lambda_qv_trans=0.01`.

- `bce_equal` is applying downward pressure on about `40.8%` of valid parent predictions at the final update.
- Lower-bound modes remove or gate that downward pressure by construction, but the final heldout metrics are nearly identical to equality BCE in this abundant-label setting.
- All Q/V modes substantially improve Q-V-next absolute probability consistency relative to supervised-only, from `0.2036` to about `0.133-0.135`.
- All Q/V modes preserve H40 AUC, slightly reduce H80 AUC/gap, and improve H160 gap.
- `bce_lower_bound` gives the best Q-V-next absolute difference and H160 gap by a small margin, but the difference is too small to call decisive.

## Recommended Next Step

Move to the sparse-Q label-efficiency table before policy evaluation.

Use `bce_lower_bound` as the default candidate because it is theoretically aligned with the lower-bound interpretation and gave the best Q-V-next absolute difference here, but include `prob_hinge` in one small sparse sanity row if time permits.

Suggested first sparse single-seed table:

```text
sup_pairs_per_budget in {256,64,32,16}
lambda_qv_trans in {0.0,0.01,0.025}
qv_trans_loss_type=bce_lower_bound
trans_pairs_per_update=256
num_trans_witnesses=4
trans_budgets=(80,160)
seed=0
```

If that shows a useful BMM signal, repeat with seeds `{0,1,2}`. If it is mixed, run the clean grid-cell algebra diagnostic before main-path integration.
