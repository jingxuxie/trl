# BMM-TRL Action-Ranking Follow-Up Results

Date: 2026-06-11

## What Changed

Following `BMM_TRL_NEXT_STEPS_AFTER_ACTION_RANKING.md`, I kept this milestone evaluation-only and reused the saved A/B/F critics from:

```text
exp/bmm_qv_budget_holdout_20260611_021352
```

Added:

```text
scripts/eval_bmm_action_ranking.py
  -- query cache save/load
  -- oracle distance baseline
  -- source-distance baseline
  -- random baseline
  -- V-next teacher baseline
  -- V-teacher budget-scan action scoring

scripts/eval_bmm_subgoal_selection.py
  -- compact H160 -> H80/H80 subgoal-selection diagnostic
  -- oracle midpoint baselines
  -- V/V teacher baseline
  -- A/B/F Q/V subgoal scoring
```

## Cached Action-Ranking Sanity Check

Run:

```text
query cache: exp/bmm_qv_budget_holdout_20260611_021352/action_ranking_h160_queries.npz
budget: H160
queries: 512
candidates/query: 8
positive fraction: 0.4082
```

Baselines:

| score | pair_acc | AUC | gap | selected_d | selected_improve | selected_success |
|---|---:|---:|---:|---:|---:|---:|
| oracle_distance | 1.0000 | 1.0000 | 20.1881 | 157.2837 | 0.6875 | 1.0000 |
| source_distance | 0.5000 | 0.8472 | 13.7455 | 169.5014 | 0.1035 | 0.4277 |
| random | 0.4973 | 0.5036 | 0.0036 | 169.9267 | 0.0957 | 0.4062 |
| V-next teacher | 0.8359 | 0.7504 | 0.1755 | 163.4312 | 0.3828 | 0.7109 |

Interpretation:

```text
The candidate set is meaningful.
The frozen V teacher ranks actions much better than A/B/F Q critics.
The problem is not just the sampler; Q action-conditioned training/extraction is weak.
```

## A/B/F Flat Action Ranking

Parent H160 score:

| critic | pair_acc | AUC | gap | selected_d | selected_improve | selected_success |
|---|---:|---:|---:|---:|---:|---:|
| A parent | 0.6974 | 0.6004 | 0.0020 | 167.8775 | 0.1641 | 0.5078 |
| B parent | 0.6955 | 0.5848 | 0.0063 | 167.9935 | 0.1621 | 0.5000 |
| F parent | 0.6988 | 0.6009 | 0.0022 | 167.8775 | 0.1641 | 0.5078 |

Budget scan did not fix this. With `tau=0.5`, every query selected H160. With `tau=0.1`, only 51/512 queries selected H80 and the Q ranking metrics got worse.

Conclusion:

```text
Do not start flat policy smoke from the current Q critics.
```

## Subgoal-Selection Diagnostic

Run:

```text
budget: H160
split: H80 / H80
queries: 512
candidate subgoals/query: 64
first action: logged first candidate action from the cached query
```

Selected subgoal metrics:

| scorer | state_valid | action_valid | source_stretch | next_stretch | midpoint_err | action_mid_err |
|---|---:|---:|---:|---:|---:|---:|
| oracle_state_midpoint | 0.3672 | 0.0098 | 0.2320 | 0.7733 | 13.3493 | 14.2780 |
| oracle_action_midpoint | 0.3340 | 0.0059 | 0.2320 | 0.1547 | 14.6096 | 12.0699 |
| random | 0.0059 | 0.0000 | 52.8145 | 53.2012 | 110.9444 | 110.5231 |
| V/V teacher | 0.1680 | 0.0059 | 1.0826 | 1.0826 | 24.6779 | 24.2038 |
| A Q/V | 0.0215 | 0.0000 | 1.8559 | 2.3198 | 45.4557 | 45.6168 |
| B Q/V | 0.0391 | 0.0020 | 0.9279 | 1.2372 | 41.4243 | 41.7342 |
| F Q/V | 0.0234 | 0.0000 | 1.8559 | 2.3198 | 45.2269 | 45.3894 |

Interpretation:

```text
B is not good in absolute terms, but it is better than A/F on the compact Q/V subgoal diagnostic.
```

B selects more valid subgoals and has lower path stretch and midpoint error than A/F. This is the first policy-facing diagnostic after action-ranking where B has a measurable advantage.

## Decision

The evidence now points away from flat one-step action scoring and toward high-level BMM/subgoal use:

```text
critic holdout: B positive
flat action ranking: B negative
V-next teacher action ranking: strong
subgoal selection: B modestly positive versus A/F
```

Recommended next step:

1. Improve the subgoal diagnostic before any policy benchmark.
2. Add a version that selects `(w, a)` jointly from candidate actions and candidate subgoals, not only the first logged action.
3. If B still beats A/F after that, run one small high-level/subgoal policy smoke.
4. Do not run a flat TRL-style policy smoke yet.

## Artifacts

```text
exp/bmm_qv_budget_holdout_20260611_021352/action_ranking_h160_queries.npz
exp/bmm_qv_budget_holdout_20260611_021352/action_ranking_abf_h160_vnext_cache.json
exp/bmm_qv_budget_holdout_20260611_021352/action_ranking_abf_h160_vnext_cache_tau0p1.json
exp/bmm_qv_budget_holdout_20260611_021352/subgoal_selection_h160_abf.json
exp/bmm_qv_budget_holdout_20260611_021352/subgoal_selection_h160_abf.md
```
