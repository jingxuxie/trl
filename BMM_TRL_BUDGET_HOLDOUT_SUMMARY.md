# BMM-TRL Budget-Holdout Summary

Date: 2026-06-11

## Motivation

The earlier uniform sparse-label diagnostics were useful debugging tools, but they were not the cleanest test of the BMM hypothesis. They mostly asked whether a critic could survive broad label scarcity. The budget-holdout diagnostic is more direct: withhold the parent budget labels and test whether shorter-budget Q/V transitive knowledge improves the heldout longer-budget critic.

The key question is:

```text
Do shorter reachable branches help a longer heldout parent budget?
```

## Design

The default diagnostic setting is now frozen unless a specific failure appears:

```text
qv_trans_loss_type = bce_lower_bound
lambda_qv_trans = 0.01
num_trans_witnesses = 4
trans_witness_mode = slack_balanced
trans_pairs_per_update = 256
```

The main comparisons are:

```text
A: supervised short budgets only, no parent labels, no transitive loss
B: A + learned Q/V transitive lower-bound loss on the parent budget
C: supervised short budgets + a few parent labels, no transitive loss
D: C + learned Q/V transitive lower-bound loss
F: A + V-next distillation control
```

The V-next control is important because it checks whether the improvement is just from distilling a one-step value target rather than from the BMM-style max-min branch construction.

## Grid-Cell H8 Aggregate

Three seeds, parent budget H8 held out from direct labels.

| comparison | delta AUC | delta gap | delta BCE | delta ECE | delta Q-V abs | conclusion |
|---|---:|---:|---:|---:|---:|---|
| B-A | +0.0150 | +0.0751 | -0.3609 | -0.0575 | -0.0208 | learned Q/V transitive helps without parent labels |
| D-C | +0.0116 | +0.0507 | -0.1503 | -0.0758 | +0.0011 | learned Q/V transitive also helps with sparse parent labels |
| F-A | +0.0002 | +0.0017 | -0.0063 | -0.0010 | -0.0005 | V-next distillation is near-zero |
| G-B | +0.0014 | +0.0139 | -0.0560 | -0.0165 | -0.0036 | oracle branch, seed 0 only |

Ensemble-min deltas are consistent with the mean-score result: B-A improves H8 ensemble-min AUC by +0.0156 and gap by +0.0758.

## Env-Step H160 Aggregate

Three seeds, short budgets H40/H80 supervised, parent H160 held out from direct labels.

| comparison | delta AUC | delta gap | delta BCE | delta ECE | delta Q-V abs | conclusion |
|---|---:|---:|---:|---:|---:|---|
| B-A | +0.0243 | +0.0647 | -0.5815 | -0.0380 | -0.0102 | learned Q/V transitive helps heldout H160 |
| D-C | +0.0041 | +0.0498 | -0.1815 | -0.0547 | -0.0293 | sparse-parent setting improves mainly in gap/BCE/ECE |
| F-A | +0.0035 | +0.0030 | -0.0570 | -0.0016 | +0.0026 | V-next control is much smaller than B-A |

Ensemble-min deltas are again positive for B-A: H160 ensemble-min AUC improves by +0.0204 and gap by +0.0620.

## Interpolation Baseline Status

The planned non-BMM interpolation baseline is:

```text
score_Hparent_interp = max(score_Hshort1, score_Hshort2)
```

For the existing completed runs, this cannot be recovered from the saved JSON reports because those reports contain aggregate metrics, not raw per-example predictions. The new action-ranking diagnostic implements the same interpolation baseline for saved checkpoints, so future minimal A/B/F reruns can compare parent-budget scoring against short-budget monotone interpolation without additional training.

## Conclusion

The replicated budget-holdout result supports the core critic-level BMM claim:

```text
BMM-specific transitive bootstrapping improves heldout parent-budget reachability.
```

The effect appears in both grid-cell and env-step budget units, replicates across three seeds, and is much larger than the V-next distillation control. The next decision point is not another broad parameter sweep. It is whether the improved critic also improves offline action ranking.
