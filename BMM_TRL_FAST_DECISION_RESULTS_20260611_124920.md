# BMM-TRL fast decision results

Date: 2026-06-11

This follows `BMM_TRL_FAST_DECISION_PLAN_20260611.md`. The goal was to get a fast signal for whether to continue, pivot, or stop, without another broad sweep.

## Decision

Recommendation: pivot, do not stop.

The evidence is not strong enough to continue toward flat one-step Q policy extraction. It is strong enough to keep the project alive as a reachability / subgoal-planning method.

Short version:

- Flat action ranking remains weak for learned Q, even after fixing the repeated-source counterfactual issue.
- Offline-support graph labels are learnable, and BMM Q/V gives a small heldout parent-budget AUC improvement over supervised-only and V-next distillation.
- Subgoal selection is weak absolutely, but BMM Q/V is modestly better than A/F in the existing diagnostic.

Next best milestone: improve and validate a joint `(a, w)` high-level subgoal/action diagnostic. Do not run policy evaluation yet.

## Code changes

- Added cached all-pairs graph distances in `utils/pointmaze_graph.py`.
- Routed graph distance matrices through:
  - `scripts/train_bmm_geodesic_value.py`
  - `scripts/train_bmm_geodesic_q.py`
- Added graph Q/V witness sampling for `reachability_label_type=graph`.
- Exposed graph-label budget holdout through `scripts/run_bmm_qv_budget_holdout.py`.
- Extended `scripts/eval_bmm_action_ranking.py` with:
  - `candidate_observations`
  - `candidate_next_cells`
  - query-cache backfill for older caches
  - candidate diagnostics
  - `q_repeated_source_*` versus `q_candidate_own_state_*` scores
- Extended synthetic tests for the new graph/action-ranking paths.

## Test 1: action-ranking evaluator validity

Command:

```bash
conda run -n bmm-trl python scripts/eval_bmm_action_ranking.py \
  --geodesic_budget_unit=env_steps \
  --budgets=40,80,160 \
  --budget=160 \
  --interp_budgets=40,80 \
  --num_queries=512 \
  --candidate_count=8 \
  --query_cache_path=exp/bmm_qv_budget_holdout_20260611_021352/action_ranking_h160_queries.npz \
  --load_query_cache \
  --value_restore_path=exp/bmm_grid_value_qv_teacher_40_80_160 \
  --value_restore_epoch=1000 \
  --critics \
    A=exp/bmm_qv_budget_holdout_20260611_021352/seed0_A_no_parent_no_trans_qv0_vnext0_parent0:1000 \
    B=exp/bmm_qv_budget_holdout_20260611_021352/seed0_B_no_parent_qv_trans_qv0p01_vnext0_parent0:1000 \
    F=exp/bmm_qv_budget_holdout_20260611_021352/seed0_F_no_parent_vnext_distill_qv0_vnext0p01_parent0:1000
```

Artifacts:

- `exp/bmm_qv_budget_holdout_20260611_021352/action_ranking_abf_h160_own_state.json`
- `exp/bmm_qv_budget_holdout_20260611_021352/action_ranking_abf_h160_own_state.md`

Candidate diagnostics:

| metric | value |
|---|---:|
| next-distance spread mean | 21.1104 |
| source-position spread mean | 2.6947 |
| unique next-cell count mean | 2.0762 |
| logged distance mean | 169.5014 |
| oracle best distance mean | 157.2837 |
| random distance mean | 169.9702 |

Baselines:

| scorer | AUC | pair acc | selected distance | selected success |
|---|---:|---:|---:|---:|
| oracle distance | 1.0000 | 1.0000 | 157.2837 | 1.0000 |
| V-next teacher | 0.7503 | 0.8354 | 163.4699 | 0.7090 |
| random | 0.5036 | 0.4973 | 169.9267 | 0.4062 |

Learned critics:

| critic | score mode | AUC | pair acc | selected distance | selected success |
|---|---|---:|---:|---:|---:|
| A | repeated source | 0.6003 | 0.6974 | 167.9162 | 0.5059 |
| A | candidate own state | 0.6307 | 0.7814 | 164.2045 | 0.6738 |
| B | repeated source | 0.5848 | 0.6955 | 167.8775 | 0.5059 |
| B | candidate own state | 0.6170 | 0.7890 | 163.8565 | 0.6895 |
| F | repeated source | 0.6009 | 0.6992 | 167.8775 | 0.5078 |
| F | candidate own state | 0.6321 | 0.7827 | 164.1272 | 0.6777 |

Interpretation:

- The old repeated-source counterfactual diagnostic was noisy: using each candidate action's own logged source state improves all learned critics.
- The candidate set is informative: oracle is perfect and V-next teacher is meaningfully above random.
- B does not beat A/F on AUC. B has slightly better selected distance/success in own-state mode, but the advantage is too small to justify flat Q policy extraction.

Flat action-ranking verdict: negative for flat Q extraction.

## Test 2: offline-support graph budget holdout

Graph context:

- `exp/bmm_pointmaze_graph.npz`
- 1,698 nodes, 5,548 edges
- max graph distance: 146 env-step units
- mean finite graph distance: 60.99 env-step units
- budgets: `40,80,120`
- supervised budgets: `40,80`
- heldout parent budget: `120`

Value teacher:

```bash
conda run -n bmm-trl python scripts/train_bmm_geodesic_value.py \
  --env_name=pointmaze-medium-navigate-v0 \
  --reachability_label_type=graph \
  --graph_path=exp/bmm_pointmaze_graph.npz \
  --budgets="(40, 80, 120)" \
  --batch_size=256 \
  --sup_pairs_per_budget=256 \
  --eval_pairs=512 \
  --steps=500 \
  --eval_interval=250 \
  --output_json=exp/bmm_graph_value_teacher_40_80_120_500.json \
  --save_dir=exp/bmm_graph_value_teacher_40_80_120_500 \
  --save_interval=500
```

Value teacher final eval:

| H | AUC | gap | ensemble-min AUC | ensemble-min gap |
|---:|---:|---:|---:|---:|
| 40 | 0.9749 | 0.4677 | 0.9714 | 0.5294 |
| 80 | 0.9802 | 0.5771 | 0.9736 | 0.6124 |
| 120 | 0.9967 | 0.6798 | 0.9956 | 0.6555 |

A/B/F graph Q run:

```bash
conda run -n bmm-trl python scripts/run_bmm_qv_budget_holdout.py \
  --run_dir=exp/bmm_graph_qv_budget_holdout_20260611_fast_decision \
  --env_name=pointmaze-medium-navigate-v0 \
  --reachability_label_type=graph \
  --graph_path=exp/bmm_pointmaze_graph.npz \
  --geodesic_budget_unit=env_steps \
  --budgets=40,80,120 \
  --eval_budgets=40,80,120 \
  --supervised_budgets=40,80 \
  --trans_budgets=120 \
  --variants=A,B,F \
  --seeds=0 \
  --sup_pairs_per_budget=256 \
  --batch_size=256 \
  --trans_pairs_per_update=256 \
  --eval_pairs=512 \
  --steps=500 \
  --eval_interval=250 \
  --num_trans_witnesses=4 \
  --trans_witness_mode=slack_balanced \
  --qv_lambda=0.01 \
  --vnext_lambda=0.01 \
  --qv_trans_loss_type=bce_lower_bound \
  --vnext_distill_loss_type=bce_lower_bound \
  --value_restore_path=exp/bmm_graph_value_teacher_40_80_120_500 \
  --value_restore_epoch=500
```

Heldout graph H120 result:

| variant | AUC | gap | ensemble-min AUC | ensemble-min gap | Q-V abs diff | Q-V rank corr |
|---|---:|---:|---:|---:|---:|---:|
| A supervised-only | 0.9783 | 0.3459 | 0.9799 | 0.3390 | 0.3306 | 0.7802 |
| B Q/V transitive | 0.9849 | 0.3411 | 0.9845 | 0.3312 | 0.3277 | 0.8084 |
| F V-next distill | 0.9780 | 0.3491 | 0.9797 | 0.3424 | 0.3265 | 0.7802 |

B Q/V sampler health:

- acceptance rate: 1.0000
- attempts/sample: 1.0000
- witnesses per parent: 4
- effective unique witnesses: 4.0000
- replacement used: 0.0000
- parent distance mean: 76.0312
- left distance mean: 44.0488
- right distance mean: 45.0293
- parent/left/right oracle labels: 1.0000

Interpretation:

- Graph labels are not a blocker; the graph value/Q diagnostics are learnable.
- A supervised-only extrapolates to H120 surprisingly well.
- B beats A and F on heldout H120 AUC, but only by about 0.006-0.007 AUC and not by score gap.
- F is near A, which supports that the small B gain is more likely transitive-specific than generic V-teacher distillation.

Graph-label verdict: mildly positive, but not enough by itself to justify policy runs.

## Test 3: subgoal selection

Existing artifact:

- `exp/bmm_qv_budget_holdout_20260611_021352/subgoal_selection_h160_abf.md`

Key H160 split 80/80 rows:

| scorer | state valid | action valid | source stretch | next stretch | midpoint err | action midpoint err |
|---|---:|---:|---:|---:|---:|---:|
| random | 0.0059 | 0.0000 | 52.8145 | 53.2012 | 110.9444 | 110.5231 |
| V/V teacher | 0.1680 | 0.0059 | 1.0826 | 1.0826 | 24.6779 | 24.2038 |
| A Q/V | 0.0215 | 0.0000 | 1.8559 | 2.3198 | 45.4557 | 45.6168 |
| B Q/V | 0.0391 | 0.0020 | 0.9279 | 1.2372 | 41.4243 | 41.7342 |
| F Q/V | 0.0234 | 0.0000 | 1.8559 | 2.3198 | 45.2269 | 45.3894 |

Interpretation:

- B is better than A/F on state-valid fraction, action-valid fraction, path stretch, and midpoint error.
- Absolute subgoal quality is still poor compared with V/V teacher and oracle midpoints.
- This is a better direction than flat Q ranking, but the diagnostic needs a joint `(a, w)` version before policy work.

Subgoal verdict: mildly positive for pivoting toward high-level planning.

## Verification

Lightweight checks run:

```bash
conda run -n bmm-trl python scripts/test_bmm_action_ranking.py
conda run -n bmm-trl python scripts/test_bmm_pointmaze_graph.py
conda run -n bmm-trl python scripts/test_bmm_qv_budget_holdout.py
conda run -n bmm-trl python scripts/test_bmm_subgoal_selection.py
```

All passed. Non-escalated JAX-importing tests still print the known CUDA plugin warning in the sandbox, but exit successfully.

## Final signal

Continue current flat-Q direction: no.

Stop the project: no.

Pivot: yes.

Recommended next steps:

1. Implement a joint `(a, w)` diagnostic that scores candidate action/subgoal pairs with `min(Q_h(s,a,w), V_{H-h}(w,g))`.
2. Compare A/B/F on selected action quality, selected subgoal validity, path stretch, and midpoint error.
3. Only run policy evaluation if B clearly beats A/F on this joint high-level diagnostic.
4. Avoid more broad sweeps until that diagnostic is positive.

