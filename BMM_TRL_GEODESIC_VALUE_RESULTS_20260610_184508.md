# BMM-TRL Geodesic Value Results - 20260610_184508

## Summary

Following `BMM_TRL_NEXT_STEPS_GEODESIC_AND_BMM.md`, I added a multi-batch
state-only geodesic value diagnostic and ran it on PointMaze medium.

The run passed heldout thresholds for clean layout/grid geodesic labels on:

```text
H = 32, 64, 96, 128
```

This is stronger than the earlier fixed-batch graph-label smoke because the
critic trained on fresh supervised geodesic pairs every update and evaluated on
a heldout validation pair set.

## New Implementation

Added:

```text
scripts/train_bmm_geodesic_value.py
```

Extended:

```text
utils/pointmaze_grid.py
scripts/test_pointmaze_grid_bfs.py
utils/datasets.py
```

The new training script supports:

```text
reachability_label_type=grid_geodesic
reachability_label_type=graph
```

For this milestone, the important path is `grid_geodesic`, using:

```text
V_H(s, g) = 1[d_grid(xy_s, xy_g) <= H]
```

The run keeps the diagnostic clean:

```text
diagnostic_critic_mode=state
value_only=True
lambda_sup=1.0
lambda_trans=0.0
lambda_rank=0.0
num_rank_pairs=0
lambda_mono=0.0
lambda_pos=0.0
lambda_budget_neg=0.0
lambda_hard_neg=0.0
lambda_rand_hinge=0.0
```

## Command

```bash
conda run -n bmm-trl python scripts/train_bmm_geodesic_value.py \
    --env_name=pointmaze-medium-navigate-v0 \
    --reachability_label_type=grid_geodesic \
    --budgets="(32, 64, 96, 128)" \
    --batch_size=256 \
    --eval_pairs=512 \
    --steps=1000 \
    --eval_interval=250 \
    --agent.value_hidden_dims="(256, 256)" \
    --agent.actor_hidden_dims="(256, 256)" \
    --agent.layer_norm=False \
    --output_json=exp/bmm_grid_geodesic_value_medium_1k.json
```

## Target Context

The script confirmed the PointMaze medium grid context:

| Item | Value |
|---|---:|
| label type | `grid_geodesic` |
| maze type | `medium` |
| maze unit | `4.0` |
| median step xy | `0.202063` |
| steps per cell | `19.7958` |
| free cells | `26` |
| max grid distance | `11` cells / `217.75` steps |
| mean finite grid distance | `4.89` cells / `96.81` steps |

## Heldout Results

Final heldout result at step `1000`:

```text
Final heldout threshold pass: True
```

| H | AUC | Gap | Pos Mean | Neg Mean | Ensemble-Min AUC | Ensemble-Min Gap | Distance Oracle AUC | Euclidean AUC |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 0.9344 | 0.4309 | 0.6823 | 0.2514 | 0.9328 | 0.4266 | 1.0000 | 0.9478 |
| 64 | 0.9749 | 0.6086 | 0.7751 | 0.1665 | 0.9743 | 0.6064 | 1.0000 | 0.9657 |
| 96 | 0.9588 | 0.5916 | 0.7798 | 0.1882 | 0.9591 | 0.5903 | 1.0000 | 0.9196 |
| 128 | 0.9751 | 0.5476 | 0.6100 | 0.0625 | 0.9755 | 0.5402 | 1.0000 | 0.8809 |

Heldout monotonicity violation at final eval:

```text
0.0000
```

Each budget row had balanced heldout coverage:

```text
pos_count = 256
neg_count = 256
```

## Interpretation

The state-only BMM critic can learn the clean grid-geodesic reachability target
on PointMaze medium when budgets are below the calibrated maze diameter.

This supports the current diagnosis:

```text
The previous H=256/512 failure came from the logged-offset target, not from a
basic inability of the BMM critic to learn PointMaze reachability.
```

The next meaningful milestone is not more logged-offset tuning. It is the
action-conditioned Bellman next-state geodesic target:

```text
Q_H(s_t, a_t, g) = 1[d_grid(s_{t+1}, g) <= H - 1]
```

## Verification

Passed:

```bash
python -m py_compile utils/pointmaze_grid.py utils/datasets.py \
    scripts/train_bmm_geodesic_value.py scripts/test_pointmaze_grid_bfs.py

conda run -n bmm-trl python scripts/test_pointmaze_grid_bfs.py
conda run -n bmm-trl python scripts/test_bmm_pointmaze_graph.py
```

Also ran a 1-step smoke of `scripts/train_bmm_geodesic_value.py` before the
1000-step diagnostic.

## Recommended Next Steps

1. Add action-conditioned geodesic Q training:

```text
Q_H(s_t, a_t, g) = 1[d_grid(s_{t+1}, g) <= H - 1]
```

2. Keep losses clean for the first Q diagnostic:

```text
value_only=True
diagnostic_critic_mode=action
lambda_sup=1.0
all auxiliary losses = 0.0
```

3. After action-conditioned Q passes, add max-min consistency on geodesic-valid
witnesses with small `lambda_trans`.

4. Only after the geodesic supervised and transitive diagnostics pass should
policy-facing evaluation be restarted.
