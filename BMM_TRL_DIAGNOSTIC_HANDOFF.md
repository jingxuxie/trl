# BMM-TRL Diagnostic Handoff

Date: 2026-06-10

## Workspace

- Workspace root: `TRL`
- TRL repo: `TRL/trl`
- OGBench repo: `TRL/ogbench`
- Conda env: `bmm-trl` with Python 3.10
- Main project spec: `TRL/bmm_trl_codex_handoff_v2.md`

The implementation work is in the TRL repo. OGBench has not been modified for this prototype.

## What Has Been Implemented

### Project setup

- Created root-level agent/project notes:
  - `TRL/AGENT_RULES.md`
  - `TRL/PROJECT_GIT.md`
  - `TRL/REPOS.md`
- Root project files are tracked with:

```bash
git --git-dir=.project-git --work-tree=. status --short
```

- Nested source repos remain separate:
  - Use normal `git` inside `TRL/trl`
  - Use normal `git` inside `TRL/ogbench`

### Committed TRL changes

Three BMM-related commits exist in `TRL/trl`:

```text
ddcda93 Add BMM hard negatives and report gate
512f4bb Add BMM reachability diagnostics
f757f53 Add BMM-TRL prototype
```

Commit `f757f53` added the first BMM-TRL prototype:

- `agents/bmm_trl.py`
  - New `BMMTRLAgent`
  - Critic sees `concat(goal, log2(H) / log2(max_budget))`
  - Actor input remains unaugmented: `pi(a | s, g)`
  - Vector/state goals only; dict/pixel goals fail explicitly
  - `oracle_distill=True` fails explicitly
  - Max-min one-witness transitive target:

```text
y_trans = min(R_h(s, a, w), R_{H-h}(w, a_w, g))
```

- `agents/__init__.py`
  - Registers `bmm_trl`

- `utils/datasets.py`
  - Adds BMM-only `GCDataset` fields for budgeted positives, midpoint witnesses, weak budget negatives, random-goal negatives, and monotonicity pairs
  - Keeps original TRL path untouched

- Synthetic checks:
  - `scripts/test_bmm_tabular.py`
  - `scripts/test_bmm_dataset_shapes.py`
  - `scripts/test_bmm_agent_shapes.py`

Commit `512f4bb` added reachability diagnostics:

- `scripts/bmm_reachability_utils.py`
- `scripts/eval_bmm_reachability.py`
- `scripts/test_bmm_reachability_eval.py`
- README instructions for running diagnostic evaluation

### Follow-up diagnostic changes

Commit `ddcda93` adds:

- Same-trajectory hard negatives where `offset > budget`
- `lambda_hard_neg`, `hard_neg_min_factor`, and `hard_neg_max_factor`
- Hard-negative metrics in `agent.update`
- A JSON report gate for reachability diagnostics
- Tests for hard-negative sampling and report gating

## Verification Status

Run from `TRL/trl`.

Passing checks:

```bash
conda run -n bmm-trl python scripts/test_bmm_tabular.py
conda run -n bmm-trl python scripts/test_bmm_dataset_shapes.py
conda run -n bmm-trl python scripts/test_bmm_agent_shapes.py
conda run -n bmm-trl python scripts/test_bmm_reachability_eval.py
conda run -n bmm-trl python scripts/test_bmm_hard_neg_shapes.py
conda run -n bmm-trl python scripts/test_bmm_reachability_gate.py
```

Notes:

- The CPU agent/reachability checks print a JAX CUDA plugin warning in the sandbox:

```text
operation cuInit(0) failed: Unknown CUDA error 100
```

- The tests still exit successfully. This is expected when Codex runs without GPU escalation and should not be treated as proof that CUDA is broken.
- GPU-dependent training/evaluation commands should be run with escalation.

## Current Diagnostic Problem

The basic implementation and shape checks pass, but trained BMM-TRL critics are not yet reliable budgeted reachability classifiers on PointMaze.

The failure is not a tensor-shape or missing-key failure. It is a calibration/separation failure:

- Low budgets often separate positives from negatives well.
- Large budgets, especially `H=256` and `H=512`, assign very high scores to negative pairs whose same-trajectory offset is greater than the budget.
- The report gate fails on low AUC, tiny positive-negative gaps, high negative means, and sometimes monotonicity violations.

Example gate command:

```bash
conda run -n bmm-trl python scripts/check_bmm_reachability_report.py \
  --report_path=exp/mrl/Debug/sd000_20260610_134556/bmm_reachability_20000.json
```

Representative failures from saved 20k-step runs:

```text
sd000_20260610_130441, before hard negatives:
  H=256: auc=0.6739, pos_mean=0.9538, neg_mean=0.9394
  H=512: auc=0.6032, pos_mean=0.9821, neg_mean=0.9805
  mean_monotonicity_violation=0.1830

sd000_20260610_134016, hard negatives, lambda_hard_neg=0.5:
  H=256: auc=0.6817, pos_mean=0.8115, neg_mean=0.7659
  H=512: auc=0.6143, pos_mean=0.9355, neg_mean=0.9298
  mean_monotonicity_violation=0.1976

sd000_20260610_134556, hard negatives, lambda_hard_neg=1.0 and lower lambda_trans:
  H=256: auc=0.6791, pos_mean=0.7621, neg_mean=0.7080
  H=512: auc=0.6005, pos_mean=0.9374, neg_mean=0.9316
  mean_monotonicity_violation=0.1778
```

The hard-negative change improves some mid-budget score scale, but it does not solve large-budget collapse. For `H=512`, unreachable pairs still get scores around `0.93`.

## Strong Symptom From Offset Buckets

The fixed-offset bucket diagnostics make the failure easier to see.

For `sd000_20260610_134556` at 20k steps:

```text
H=128:
  offset=128, label=1, mean_score=0.3252
  offset=256, label=0, mean_score=0.3019
  offset=512, label=0, mean_score=0.3073

H=256:
  offset=256, label=1, mean_score=0.7133
  offset=512, label=0, mean_score=0.7013

H=512:
  offset=128, label=1, mean_score=0.9354
  offset=256, label=1, mean_score=0.9327
  offset=512, label=1, mean_score=0.9306
```

This says the critic has learned some ordering at lower/mid budgets, but high budget values saturate near one and provide little useful separation.

## Current Hypotheses

These are hypotheses, not confirmed causes:

1. The loss is still too positive-biased at large budgets.
   - `loss_pos` anchors all sampled `(s, g, H)` positives to one.
   - For large `H`, many sampled offsets are valid positives.
   - The weak budget negatives and hard negatives may be too sparse or too weak at high `H`.

2. The transitive target can propagate high scores.
   - `y_trans = min(first_r, second_r)` is bootstrapped from the target critic.
   - If either/both target branches become overconfident for large budgets, transitive BCE can reinforce large-H optimism.
   - Lowering `lambda_trans` helped score scale in some places, but did not fix `H=512`.

3. The current hard-negative sampler may not put enough pressure on the exact failure distribution.
   - It samples same-trajectory goals beyond budget, but the valid fraction depends on remaining trajectory length.
   - The large-H evaluation negatives are rare because many random validation pairs are positive at high budgets.
   - We may need per-budget balanced hard negatives or a diagnostic-training sampler that explicitly targets `offset in (H, 4H]`.

4. The budget feature may be too easy to use as a monotone bias term.
   - The appended scalar budget feature may let the critic learn "larger H means reachable" without enough goal/action discrimination.
   - Monotonicity is desirable, but not if it swamps the pair-specific reachability signal.

5. The actor is probably not the main issue yet.
   - The failure appears in direct critic reachability diagnostics.
   - Policy performance tuning should stay deferred until critic diagnostics pass.

## Suggested Next Debugging Steps

1. Freeze policy concerns and focus on critic diagnostics.
   - Continue using `scripts/eval_bmm_reachability.py` and `scripts/check_bmm_reachability_report.py`.
   - Gate any change before running policy evaluation.

2. Add training metrics that expose hard-negative coverage.
   - Log `hard_neg_valid_frac`, `hard_neg_budget_mean`, `hard_neg_offset_mean`, and `hard_neg_offset_over_budget_mean` already exist in the WIP agent.
   - Also inspect these metrics from actual training logs.

3. Make negative training more balanced by budget.
   - Try explicit per-budget negatives for each sampled state:
     - choose `H`
     - choose `offset > H` where available
     - train BCE target zero
   - Consider sampling negatives from the exact diagnostic buckets: `1.25H`, `2H`, and `4H`.

4. Separate positive and transitive effects.
   - Run short jobs with:
     - `lambda_trans=0`
     - stronger hard negatives
     - lower `lambda_pos`
   - Check whether high-H negative scores drop before reintroducing transitive bootstrapping.

5. Consider making random-goal negatives proper BCE negatives.
   - The current random-goal loss is a hinge above `rho`.
   - For early diagnostics, direct BCE-to-zero may provide a clearer signal than a soft hinge.

6. Inspect action conditioning.
   - The diagnostic pairs use dataset action at the source state.
   - A same-trajectory goal with large offset may be reachable eventually but not necessarily via the current one-step action.
   - Confirm whether the intended critic semantics are "reachable within H after taking action a now" or "state-goal reachability within H"; the current implementation is action-conditioned like TRL.

## Reproduction Commands

Run short training with GPU escalation from `TRL/trl`:

```bash
WANDB_MODE=offline conda run -n bmm-trl python main.py \
  --env_name=pointmaze-medium-navigate-v0 \
  --agent=agents/bmm_trl.py \
  --offline_steps=20000 \
  --log_interval=100 \
  --eval_interval=0 \
  --video_episodes=0 \
  --save_interval=10000 \
  --agent.batch_size=512 \
  --agent.max_budget=512 \
  --agent.budgets="(1, 2, 4, 8, 16, 32, 64, 128, 256, 512)" \
  --agent.actor_hidden_dims="(256, 256)" \
  --agent.value_hidden_dims="(256, 256)"
```

Evaluate reachability:

```bash
conda run -n bmm-trl python scripts/eval_bmm_reachability.py \
  --env_name=pointmaze-medium-navigate-v0 \
  --agent=agents/bmm_trl.py \
  --agent.batch_size=512 \
  --agent.max_budget=512 \
  --agent.budgets="(1, 2, 4, 8, 16, 32, 64, 128, 256, 512)" \
  --agent.actor_hidden_dims="(256, 256)" \
  --agent.value_hidden_dims="(256, 256)" \
  --restore_path="exp/mrl/Debug/<run_dir>" \
  --restore_epoch=20000 \
  --output_json="exp/mrl/Debug/<run_dir>/bmm_reachability_20000.json"
```

Gate the report:

```bash
conda run -n bmm-trl python scripts/check_bmm_reachability_report.py \
  --report_path="exp/mrl/Debug/<run_dir>/bmm_reachability_20000.json"
```

## Files Most Relevant For Review

- `TRL/trl/agents/bmm_trl.py`
- `TRL/trl/utils/datasets.py`
- `TRL/trl/scripts/bmm_reachability_utils.py`
- `TRL/trl/scripts/eval_bmm_reachability.py`
- `TRL/trl/scripts/check_bmm_reachability_report.py`
- `TRL/trl/scripts/test_bmm_hard_neg_shapes.py`
- `TRL/trl/scripts/test_bmm_reachability_gate.py`

## Bottom Line

The prototype is structurally wired and passes shape/smoke tests. The blocking issue is that the learned budget-conditioned critic still overestimates reachability for large budgets. The next fix should target the critic loss and negative sampling distribution, not policy evaluation or OGBench integration.
