# BMM-TRL Geodesic Transitive Diagnostic Results

Date: 2026-06-10

## Summary

Implemented the first BMM-specific follow-up after the action-conditioned
geodesic Q diagnostic: optional geodesic-valid max-min transitive consistency
inside `scripts/train_bmm_geodesic_value.py`.

This is still a standalone diagnostic path. It does not modify `agents/trl.py`,
OGBench, or the main policy-training data path.

## Code Added

`scripts/train_bmm_geodesic_value.py` now supports:

```text
--lambda_trans=<weight>
--trans_pos_boundary_frac=<frac>
```

When `--lambda_trans > 0`, each training batch overwrites the standard BMM
transitive fields with grid-geodesic valid state-only witnesses:

```text
y_trans = min(V_h(s,w), V_{H-h}(w,g))
```

The implementation currently supports grid-geodesic labels only. It was tested
with budgets `(64,128)` because smaller half-splits can be below one calibrated
PointMaze grid cell on medium:

```text
steps_per_cell ~= 19.8
H=32 -> h=16, which is too small to traverse one grid cell
```

## Matched Abundant-Label Comparison

Both runs used:

```text
env=pointmaze-medium-navigate-v0
budgets=(64,128)
batch_size=256
eval_pairs=512
steps=1000
hidden_dims=(256,256)
layer_norm=False
```

### Supervised Only

Output: `exp/bmm_grid_geodesic_value_supervised_64_128_1k.json`

| H | AUC | gap | pos_mean | neg_mean | ensemble-min AUC | ensemble-min gap |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 0.9636 | 0.5017 | 0.8683 | 0.3666 | 0.9641 | 0.5123 |
| 128 | 0.9753 | 0.5479 | 0.6490 | 0.1011 | 0.9760 | 0.5427 |

### Supervised + Transitive

Output: `exp/bmm_grid_geodesic_value_trans_medium_1k.json`

```text
lambda_trans=0.025
```

| H | AUC | gap | pos_mean | neg_mean | ensemble-min AUC | ensemble-min gap |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 0.9647 | 0.4908 | 0.8860 | 0.3952 | 0.9644 | 0.4999 |
| 128 | 0.9792 | 0.5725 | 0.6845 | 0.1121 | 0.9792 | 0.5676 |

Interpretation: no clear degradation. H=64 is essentially tied with a slightly
smaller score gap; H=128 improves both AUC and gap. Monotonicity stayed at
`0.0000`.

## Sparse-Label Comparison

Both sparse runs used the same setup as above except:

```text
batch_size=64
```

This reduces direct supervised pairs per budget by 4x.

### Sparse Supervised Only

Output: `exp/bmm_grid_geodesic_value_supervised_64_128_sparse_1k.json`

| H | AUC | gap | pos_mean | neg_mean | ensemble-min AUC | ensemble-min gap |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 0.9619 | 0.3991 | 0.8946 | 0.4955 | 0.9617 | 0.4018 |
| 128 | 0.9754 | 0.5007 | 0.6086 | 0.1079 | 0.9756 | 0.4947 |

### Sparse Supervised + Transitive

Output: `exp/bmm_grid_geodesic_value_trans_64_128_sparse_1k.json`

```text
lambda_trans=0.025
```

| H | AUC | gap | pos_mean | neg_mean | ensemble-min AUC | ensemble-min gap |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 0.9581 | 0.4310 | 0.8738 | 0.4428 | 0.9574 | 0.4377 |
| 128 | 0.9663 | 0.4679 | 0.5606 | 0.0926 | 0.9663 | 0.4615 |

Interpretation: sparse transitive does not yet show a clear label-efficiency
win. H=64 has a better score gap but slightly lower AUC; H=128 is lower on both
AUC and gap. Monotonicity stayed at `0.0000`, and both sparse runs still passed
the threshold gate.

## Takeaways

- Action-conditioned Q passed cleanly in the previous milestone.
- State-only geodesic transitive consistency is now implemented and does not
  break the value diagnostic at `lambda_trans=0.025`.
- The first sparse-label test is mixed, not a positive BMM result yet.
- Before implementing Q/V transitive consistency, it is worth deciding whether
  to:
  - sweep `lambda_trans` on state-only V first (`0.01`, `0.025`, `0.05`);
  - improve witness sampling coverage and report witness distance histograms;
  - use budgets aligned to calibrated cell distances rather than raw dyadic-ish
    step budgets;
  - or proceed to Q/V transitive now that the no-degradation check passed.

## Verification

Commands run:

```bash
python -m py_compile scripts/train_bmm_geodesic_value.py
conda run -n bmm-trl python scripts/train_bmm_geodesic_value.py --env_name=pointmaze-medium-navigate-v0 --reachability_label_type=grid_geodesic --budgets="(64, 128)" --batch_size=32 --eval_pairs=32 --steps=1 --eval_interval=1 --lambda_trans=0.025 --agent.value_hidden_dims="(64, 64)" --agent.actor_hidden_dims="(64, 64)" --agent.layer_norm=False
```

The four 1k diagnostic runs above were run with GPU escalation.
