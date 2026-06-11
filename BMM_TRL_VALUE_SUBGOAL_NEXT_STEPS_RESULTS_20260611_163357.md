# BMM-TRL value-subgoal next-step results - 20260611_163357

## Summary

Following `BMM_TRL_NEXT_STEPS_AFTER_VALUE_SUBGOAL_CONTROLLER.md`, I focused on
the high-level BMM subgoal path and avoided more flat Q/QV action extraction.

The new evidence supports continuing only as:

```text
BMM high-level reachability/subgoal planning + separate low-level controller
```

It does not justify broad policy benchmarks yet.

## Implementation

Added selector comparison to:

```text
scripts/eval_bmm_value_subgoal_policy_smoke.py
```

The policy smoke now supports:

```text
--selectors=random,geometric_midpoint,BMM_V,oracle_midpoint
```

Added:

```text
scripts/eval_bmm_graph_value_subgoal.py
scripts/test_bmm_value_subgoal_policy_smoke.py
```

The graph diagnostic evaluates:

```text
score(w) = min(V_left(s, w), V_right(w, g))
```

on the dataset-position graph using the existing graph value checkpoint.

## Milestone 1: selector comparison under same-cell NN controller

Command:

```bash
conda run -n bmm-trl python scripts/eval_bmm_value_subgoal_policy_smoke.py \
    --env_name=pointmaze-medium-navigate-v0 \
    --geodesic_budget_unit=env_steps \
    --budgets=40,80,160 \
    --left_budget=80 \
    --right_budget=80 \
    --controller_hops=0 \
    --num_subgoal_candidates=64 \
    --selectors=random,geometric_midpoint,BMM_V,oracle_midpoint \
    --task_ids=1,2,3 \
    --episodes_per_task=1 \
    --max_steps=100 \
    --value_restore_path=exp/bmm_grid_value_qv_teacher_40_80_160 \
    --value_restore_epoch=1000
```

Output artifact:

```text
exp/bmm_qv_budget_holdout_20260611_021352/value_subgoal_policy_selectors_tasks123_hops0.md
```

Result:

| selector | success | final_d | improve | mean_step_goal | subgoal_valid | subgoal_reduce | goal_reduce |
|---|---:|---:|---:|---:|---:|---:|---:|
| random | 0.0000 | 118.7747 | 52.7888 | 0.5279 | 0.1333 | 0.0800 | 0.0600 |
| geometric_midpoint | 0.0000 | 112.1761 | 59.3874 | 0.5939 | 0.3833 | 0.2367 | 0.2267 |
| BMM_V | 0.0000 | 105.5775 | 65.9860 | 0.6599 | 0.6633 | 0.2267 | 0.2267 |
| oracle_midpoint | 0.0000 | 118.7747 | 52.7888 | 0.5279 | 0.4267 | 0.3867 | 0.2567 |

Interpretation:

```text
BMM/V beats random and geometric on final distance and total improvement.
No selector solved the tasks in this tiny smoke.
```

This passes the narrow Milestone 1 continue condition.

## Milestone 2: cheap low-level controller variants

I compared `controller_hops=0`, `1`, and `2` using the same tasks and selectors.

For BMM/V:

| controller_hops | final_d | improve | mean_step_goal | subgoal_valid | goal_reduce |
|---:|---:|---:|---:|---:|---:|
| 0 | 105.5775 | 65.9860 | 0.6599 | 0.6633 | 0.2267 |
| 1 | 151.7677 | 19.7958 | 0.1980 | 0.4267 | 0.1633 |
| 2 | 145.1691 | 26.3944 | 0.2639 | 0.3267 | 0.0367 |

Interpretation:

```text
The strict same-cell NN controller is the best cheap controller tested.
Neighbor-cell action borrowing hurts rollout progress.
```

This means the next low-level controller should be a real local/goal-conditioned
controller, not a looser nearest-neighbor action source.

## Milestone 4: dataset-support graph subgoal diagnostic

Command:

```bash
conda run -n bmm-trl python scripts/eval_bmm_graph_value_subgoal.py \
    --env_name=pointmaze-medium-navigate-v0 \
    --graph_path=exp/bmm_pointmaze_graph.npz \
    --budgets=40,80,120,500 \
    --budget=160 \
    --left_budget=80 \
    --right_budget=80 \
    --num_queries=512 \
    --num_candidates=64 \
    --value_restore_path=exp/bmm_graph_value_teacher_40_80_120_500 \
    --value_restore_epoch=500
```

Output artifact:

```text
exp/bmm_qv_budget_holdout_20260611_021352/graph_value_subgoal_h160_512.md
```

Result:

| scorer | state_valid | oracle_valid_exists | path_stretch | midpoint_err | source_d | right_d |
|---|---:|---:|---:|---:|---:|---:|
| random | 0.7422 | 1.0000 | 29.0078 | 57.9219 | 51.3867 | 62.3398 |
| euclidean_midpoint | 0.9902 | 1.0000 | 6.8711 | 68.5508 | 43.0156 | 48.5742 |
| oracle_graph_midpoint | 0.6387 | 1.0000 | 65.6875 | 15.2422 | 70.0078 | 80.3984 |
| BMM_V_graph | 0.9902 | 1.0000 | 4.7734 | 70.6406 | 44.4570 | 45.0352 |

Interpretation:

```text
BMM/V graph subgoals beat random clearly.
BMM/V ties Euclidean midpoint on validity and improves graph path stretch.
This is the first positive dataset-support graph subgoal result.
```

This supports the general offline story more than the previous grid-only
diagnostics, though the graph candidate sampler still includes broad dataset
support and is not yet a deployable learned proposal.

## Verification

Passed:

```bash
conda run -n bmm-trl python -m py_compile \
    scripts/eval_bmm_graph_value_subgoal.py \
    scripts/eval_bmm_value_subgoal_policy_smoke.py \
    scripts/test_bmm_value_subgoal_policy_smoke.py

conda run -n bmm-trl python scripts/test_bmm_value_subgoal_policy_smoke.py
```

The unit test prints the known non-escalated JAX CUDA discovery warning, then
passes. The PointMaze diagnostics were run with escalation/GPU access.

## Decision

Continue, but only in the reframed hierarchical direction:

```text
High level: BMM budgeted value chooses subgoals.
Low level: separate controller reaches selected subgoals.
```

Do not return to:

```text
flat Q action extraction;
QV joint action-subgoal extraction as the main path;
large policy benchmarks;
broad sweeps.
```

## Recommended next step

The next useful task is not more BMM critic tuning. It is to test a better
low-level controller while keeping the high-level selector fixed:

```text
random/geometric/BMM subgoal selector + same goal-conditioned low-level controller
```

Fast options:

```text
1. train a tiny goal-conditioned BC controller toward sampled subgoals;
2. use an existing repo actor toward selected subgoals if an appropriate checkpoint exists;
3. add a rollout-only comparison against direct-to-goal same-cell NN and oracle subgoal.
```

Continue only if BMM-selected subgoals improve progress over random/geometric
under the same low-level controller.
