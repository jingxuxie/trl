# BMM-TRL value-subgoal controller results

Date: 2026-06-11

This follows `BMM_TRL_NEXT_STEPS_AFTER_CANDIDATE_ACTION.md`.

## Decision

Recommendation: continue as a high-level BMM subgoal-planning project, but do not resume flat Q or Q/V action extraction.

The value-only subgoal diagnostic is positive:

```text
BMM/V subgoals are much better than random.
A simple nearest-neighbor low-level controller can make progress toward selected subgoals.
A tiny policy smoke makes goal-distance progress, but does not solve the tasks.
```

This is enough to keep the project alive in the high-level planning direction. It is not enough to claim policy performance.

## Code changes

- Added `scripts/eval_bmm_value_subgoal_controller.py`.
  - Scores candidate subgoals with `min(V_h(s,w), V_{H-h}(w,g))`.
  - Reports subgoal quality versus oracle and random baselines.
  - Adds a nearest-neighbor train-dataset controller diagnostic.
- Added `scripts/eval_bmm_value_subgoal_policy_smoke.py`.
  - Runs a tiny environment smoke using V-selected subgoals and same-cell nearest-neighbor logged actions.
  - Reports distance progress, not benchmark success.
- Added `scripts/test_bmm_value_subgoal_controller.py`.

## Verification

Commands run:

```bash
conda run -n bmm-trl python -m py_compile \
  scripts/eval_bmm_value_subgoal_controller.py \
  scripts/eval_bmm_value_subgoal_policy_smoke.py \
  scripts/test_bmm_value_subgoal_controller.py

conda run -n bmm-trl python scripts/test_bmm_value_subgoal_controller.py
conda run -n bmm-trl python scripts/test_bmm_subgoal_selection.py
```

All passed. Non-escalated JAX imports still print the known sandbox CUDA warning, but the commands exited successfully.

## Value-only subgoal diagnostic

All runs used:

```text
env: pointmaze-medium-navigate-v0
H=160, split 80/80
value checkpoint: exp/bmm_grid_value_qv_teacher_40_80_160:1000
candidate subgoals/query: 64
```

### 128-query run

Artifact:

- `exp/bmm_qv_budget_holdout_20260611_021352/value_subgoal_controller_h160_128.json`

Key rows:

| scorer | state valid | source stretch | midpoint err | nn query improve | nn query reduce |
|---|---:|---:|---:|---:|---:|
| random | 0.0156 | 48.2522 | 108.0570 | 53.6651 | 0.9844 |
| BMM/V value | 0.1875 | 1.5465 | 23.7883 | 59.3873 | 1.0000 |
| oracle midpoint | 0.3594 | 0.0000 | 13.2688 | 59.3873 | 1.0000 |

Interpretation:

- BMM/V is much better than random on state-valid fraction, path stretch, and midpoint error.
- Same diagnostic shows the local controller can reduce distance to the selected subgoal.

### 512-query run, controller hops 2

Artifact:

- `exp/bmm_qv_budget_holdout_20260611_021352/value_subgoal_controller_h160_512.json`

Key rows:

| scorer | state valid | source stretch | midpoint err | nn query improve | nn query reduce | nn source-to-query |
|---|---:|---:|---:|---:|---:|---:|
| random | 0.0059 | 52.8145 | 110.9444 | 51.6546 | 0.9512 | 37.0398 |
| BMM/V value | 0.1680 | 1.0826 | 24.6779 | 59.3873 | 1.0000 | 39.5916 |
| oracle midpoint | 0.3672 | 0.2320 | 13.3493 | 59.3873 | 1.0000 | 39.5916 |

Interpretation:

- The 512-query result confirms the 128-query signal.
- Controller hops 2 allows actions from nearby cells, so it is useful but not strict-local.

### 512-query run, controller hops 0

Artifact:

- `exp/bmm_qv_budget_holdout_20260611_021352/value_subgoal_controller_h160_512_hops0.json`

Key rows:

| scorer | state valid | source stretch | midpoint err | nn query improve | nn query reduce | nn source-to-query |
|---|---:|---:|---:|---:|---:|---:|
| random | 0.0059 | 52.8145 | 110.9444 | 18.8292 | 0.9512 | 0.0000 |
| BMM/V value | 0.1680 | 1.0826 | 24.6779 | 19.7958 | 1.0000 | 0.0000 |
| oracle midpoint | 0.3672 | 0.2320 | 13.3493 | 19.7958 | 1.0000 | 0.0000 |

Interpretation:

- Even same-cell logged actions reduce distance toward BMM/V-selected subgoals.
- This is the strongest low-level-controller diagnostic so far because the action source is local.

## Tiny policy smoke

The policy smoke uses:

```text
High level: select w by min(V_80(s,w), V_80(w,g)).
Low level: execute same-cell nearest-neighbor train-dataset action toward w.
Replan every step.
```

### One-task smoke

Artifact:

- `exp/bmm_qv_budget_holdout_20260611_021352/value_subgoal_policy_smoke_task1_hops0.json`

Result:

| metric | value |
|---|---:|
| success | 0.0000 |
| start goal distance | 197.9579 |
| final goal distance | 98.9789 |
| goal distance improvement | 98.9789 |
| mean step goal improvement | 0.9898 |
| subgoal valid fraction | 0.6100 |

### Three-task smoke

Artifact:

- `exp/bmm_qv_budget_holdout_20260611_021352/value_subgoal_policy_smoke_tasks123_hops0.json`

Result:

| metric | value |
|---|---:|
| success | 0.0000 |
| start goal distance | 171.5635 |
| final goal distance | 98.9789 |
| goal distance improvement | 72.5845 |
| mean step goal improvement | 0.7258 |
| mean step subgoal improvement | 0.8578 |
| subgoal reduce fraction | 0.2067 |
| goal reduce fraction | 0.2033 |
| subgoal valid fraction | 0.6667 |

Interpretation:

- The policy smoke does not solve the tasks.
- It does make consistent goal-distance progress using only value-selected subgoals and a simple same-cell nearest-neighbor controller.
- This is a weak-positive policy-facing signal for high-level planning, not a benchmark result.

## Final signal

Continue flat Q extraction: no.

Continue Q/V action-subgoal extraction: no, unless a new action-selection idea appears.

Continue high-level BMM subgoal planning: yes.

Run broad policy benchmarks now: no.

Recommended next steps:

1. Improve the low-level controller before more environment evaluation.
2. Compare value-selected subgoals against a goal-conditioned BC or existing actor toward the selected subgoal.
3. Add a non-oracle candidate-subgoal proposal mechanism if the project needs a deployable story.
4. Keep policy tests tiny until subgoal reach rate improves.
5. Preserve the current story as: BMM improves reachability/subgoal planning, not flat neural action extraction.
