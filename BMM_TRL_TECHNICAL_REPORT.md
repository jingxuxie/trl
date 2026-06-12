# BMM-TRL Technical Report: Budgeted Max-Min Transitive RL for Offline Goal-Conditioned RL

Date: 2026-06-11

Repository context: this report summarizes the BMM-TRL prototype implemented in this repository and the diagnostics recorded in the `BMM_TRL_*.md` result files.

## Abstract

This project investigated **Budgeted Max-Min Transitive RL (BMM-TRL)**, a proposed variant of Transitive RL designed to reduce long-horizon error compounding. The motivating observation is that TRL's product backup is additive in distance/log-value space, so approximation bias can still compound additively along both branches of a recursive decomposition. BMM-TRL instead learns a **budgeted reachability predicate** and composes values with a **max-min backup**:

```text
R_H(s,g) = 1[d(s,g) <= H]
R_H(s,g) <- max_w min(R_h(s,w), R_{H-h}(w,g))
```

Because `max` and `min` are non-expansive in sup norm, the ideal balanced recursion satisfies an error recurrence of the form

```text
E(H) <= epsilon_H + max(E(h), E(H-h)),
```

which gives `O(epsilon log H)` accumulation under uniform residuals and dyadic/balanced budgets. This is the central mathematical motivation.

The empirical work found a clear split:

- **Positive:** BMM-style budgeted reachability works as a critic/subgoal diagnostic. It learns clean geodesic and support-graph reachability labels, and Q/V max-min transitive consistency improves heldout long-budget classification in budget-holdout settings.
- **Negative:** The current prototype did not produce a robust policy improvement path. Flat Q action ranking, Q/V joint action-subgoal extraction, and lightweight hierarchical controller experiments were weak or inconclusive.

The current recommended conclusion is: **pause active policy-facing experimentation**, but preserve the project as a critic/reachability/subgoal-planning result or as a possible future hierarchical RL project with a stronger low-level controller.

---

## 1. Motivation

### 1.1 The original TRL composition

TRL learns a discounted temporal-distance value of the form

```text
V*(s,g) = gamma^{d*(s,g)}
```

and uses a transitive product backup:

```text
V(s,g) <- max_w V(s,w) V(w,g).
```

In distance space this is a min-plus composition:

```text
d(s,g) <- min_w d(s,w) + d(w,g).
```

This can reduce the dependency depth of a long-horizon problem, but the numeric error in distance/log-value space still composes additively. If both branches have worst-case error `E(H/2)` and the parent regression has residual `epsilon`, then the worst-case distance-space recurrence is

```text
E(H) <= 2 E(H/2) + epsilon,
```

which is linear in horizon under uniform per-level residuals.

### 1.2 Desired property

The goal of this project was to construct a value object whose recursive composition has a max-error recurrence:

```text
E(H) <= epsilon + max(E(h), E(H-h)).
```

A natural object with this property is **budgeted reachability**:

```text
R_H(s,g) = 1[d*(s,g) <= H].
```

The corresponding transitive operator is

```text
T_H(R)(s,g) = max_w min(R_h(s,w), R_{H-h}(w,g)).
```

The `max-min` operator is the core of BMM-TRL.

---

## 2. Deterministic theory

We first state the clean deterministic result. Let `M` be a deterministic controllable graph or deterministic MDP with shortest-path distance `d*(s,g)`.

Define the state reachability predicate:

```text
V_H*(s,g) = 1[d*(s,g) <= H].
```

For an action-conditioned critic with deterministic transition `s' = f(s,a)`, define

```text
Q_H*(s,a,g) = 1[d*(f(s,a),g) <= H - 1].
```

### Proposition 1: Exact deterministic max-min identity

For any split `h in {0,...,H}` and allowing endpoint witnesses, the state reachability predicate satisfies

```text
V_H*(s,g) = max_w min(V_h*(s,w), V_{H-h}*(w,g)).
```

The action-conditioned version satisfies

```text
Q_H*(s,a,g) = max_w min(Q_h*(s,a,w), V_{H-h}*(w,g)).
```

#### Proof sketch

If `V_H*(s,g)=1`, then there is a path from `s` to `g` of length `L <= H`. Choose a witness `w` along the path after `min(h,L)` steps. Then

```text
d*(s,w) <= h,
d*(w,g) <= H-h,
```

so both branch predicates are one and the max-min value is one.

Conversely, if the max-min value is one, then there exists a witness `w` such that

```text
d*(s,w) <= h,
d*(w,g) <= H-h.
```

Concatenating the two paths gives a path from `s` to `g` of length at most `H`, so `V_H*(s,g)=1`.

The Q identity follows by applying the same argument after taking action `a` and moving to `f(s,a)`.

### Proposition 2: Non-expansive error propagation

Let estimates `V_hat_h` and `V_hat_{H-h}` satisfy

```text
||V_hat_h - V_h*||_infty <= E_h,
||V_hat_{H-h} - V_{H-h}*||_infty <= E_{H-h}.
```

Define

```text
T_hat_H(s,g) = max_w min(V_hat_h(s,w), V_hat_{H-h}(w,g)),
T*_H(s,g)   = max_w min(V_h*(s,w), V_{H-h}*(w,g)).
```

Then

```text
||T_hat_H - T*_H||_infty <= max(E_h, E_{H-h}).
```

#### Proof sketch

For scalars, `min` is 1-Lipschitz under the max norm:

```text
|min(a,b) - min(a',b')| <= max(|a-a'|, |b-b'|).
```

The `max_w` operation is also 1-Lipschitz in sup norm:

```text
|max_w z_w - max_w z'_w| <= sup_w |z_w - z'_w|.
```

Combining these two inequalities gives the bound.

If the function class projection/regression step adds residual `epsilon_H`, then

```text
E_H <= epsilon_H + max(E_h, E_{H-h}).
```

For balanced dyadic splits `H=2^k` and uniform residual `epsilon`, this yields

```text
E_H <= k epsilon = O(epsilon log H).
```

This is the main theoretical motivation for BMM-TRL.

### Contrast with distance-space composition

For additive distance composition,

```text
d_H(s,g) = min_w d_h(s,w) + d_{H-h}(w,g),
```

branch errors add:

```text
E_H <= epsilon_H + E_h + E_{H-h}.
```

For equal splits, this recurrence is linear in horizon under uniform residuals. This is the gap BMM-TRL was designed to address.

---

## 3. Stochastic environments

The stochastic case is more subtle. Exact finite-horizon success probability does **not** obey a max-min identity in general.

Let

```text
P_H*(s,g) = sup_pi Pr_pi[tau_g <= H | s_0=s].
```

If a policy first tries to reach `w` within `h` steps with probability `p`, and then tries to reach `g` from `w` within `H-h` steps with probability `q`, the two-stage success probability is approximately lower-bounded by `p q`, not by `min(p,q)`. Therefore an exact stochastic-probability Bellman equation is not the deterministic max-min operator.

BMM-TRL can still be extended to stochastic environments in two principled ways.

### 3.1 Support reachability

Define a support graph from the stochastic MDP:

```text
s -> s' is an edge if there exists a supported action a with P(s'|s,a) > 0.
```

Then define support reachability:

```text
S_H(s,g) = 1[there exists a positive-probability supported path from s to g of length <= H].
```

This is exactly deterministic graph reachability on the support graph. Therefore the deterministic max-min identity and non-expansive error result apply directly.

This is the most relevant stochastic interpretation for offline RL when only dataset support is observable.

### 3.2 Reliability-threshold reachability

For success probability targets, a conservative thresholded extension is possible. Let `rho = exp(-b)` be a required reliability threshold and define

```text
Z_{H,b}(s,g) = 1[P_H*(s,g) >= exp(-b)].
```

A two-stage sufficient condition is:

```text
P_h(s,w) >= exp(-b1),
P_{H-h}(w,g) >= exp(-b2)
=> P_H(s,g) >= exp(-(b1+b2))
```

under the usual Markov restart/two-stage policy interpretation. Thus a sound conservative recurrence is

```text
Z_{H,b}(s,g) >= max_w min(Z_{h,b/2}(s,w), Z_{H-h,b/2}(w,g)).
```

This preserves the max-min non-expansive structure, but it is only a sufficient lower-bound target and requires an additional reliability-budget dimension. This was not implemented in the current prototype.

### 3.3 What the current project actually implements

The current implementation should be viewed as deterministic/support-reachability BMM, not full stochastic success-probability BMM. For general offline RL, the correct target is not true environment reachability and not same-trajectory offset, but **offline support graph reachability**:

```text
R_H^D(s,g) = 1[d_D(phi(s), phi(g)) <= H],
```

where `d_D` is shortest-path distance in a graph built from offline trajectories. The experiments started with grid/geodesic targets as oracle diagnostics and later tested dataset-support graph targets.

---

## 4. Algorithm: BMM-TRL

### 4.1 Critic parameterization

The prototype keeps the TRL-style action-conditioned critic:

```text
R_theta(s,a,g,H) = sigmoid(f_theta(s,a,g,H)).
```

The first implementation appends a normalized budget feature to the goal input:

```text
budget_feature = log2(H) / log2(max_budget).
```

Later diagnostics also used budget sets such as env-step budgets `(40,80,160)` and grid-cell budgets `(2,4,8)`.

### 4.2 Supervised labels

The final clean labels used in successful diagnostics were:

State value labels:

```text
V_H(s,g) = 1[d(s,g) <= H].
```

Action-conditioned labels:

```text
Q_H(s_t,a_t,g) = 1[d(s_{t+1},g) <= H - 1].
```

The distance `d` was instantiated as either:

1. layout/grid geodesic distance, for oracle diagnostics;
2. dataset-position graph distance, for offline-support diagnostics.

### 4.3 Q/V transitive target

The policy-relevant transitive target uses an action-conditioned first branch and a state-value second branch:

```text
y_QV = max_w min(Q_h(s,a,w), V_{H-h}(w,g)).
```

A frozen state-value teacher was used for the second branch in the main Q/V diagnostics.

Because sampled witnesses give a lower-bound target rather than an equality target, the most sensible loss mode was a lower-bound BCE or hinge. The default used in later experiments was:

```text
qv_trans_loss_type = bce_lower_bound
lambda_qv_trans = 0.01
num_trans_witnesses = 4
trans_witness_mode = slack_balanced
```

### 4.4 Budget-holdout protocol

The most important diagnostic protocol was budget holdout:

```text
Train short budgets: H_short1, H_short2
Hold out parent budget: H_parent
Use Q/V transitive only for H_parent
Evaluate heldout parent classification
```

This directly tests whether shorter-budget reachability knowledge can compose into a longer-budget parent.

---

## 5. Experimental summary

All experiments were run on `pointmaze-medium-navigate-v0` unless otherwise noted. The current repository contains detailed logs in the `BMM_TRL_*.md` files.

### 5.1 Logged-offset labels failed

The initial idea used same-trajectory logged offset as the label:

```text
label = 1[offset <= H].
```

This failed at high budgets. Diagnostics showed:

- the same JAX critic path learns a deterministic chain through `H=512`;
- fixed-batch PointMaze overfit works through high budgets;
- heldout logged-offset labels fail at `H=256/512`;
- kNN on logged-offset labels is near chance: about `0.54` at `H=256` and `0.52` at `H=512`.

Conclusion: logged offset is behavior time, not clean reachability.

### 5.2 Grid/geodesic reachability labels worked

PointMaze medium has 2D observations with `xy` dimensions `(0,1)`. The layout/grid BFS target showed:

```text
median one-step xy displacement = 0.202063
maze unit = 4.0
steps per cell = 19.7958
max grid distance = 11 cells / 217.75 steps
```

Thus `H=256/512` are above the calibrated medium-maze diameter and should be one-class under true geodesic reachability.

The state-only geodesic value critic trained on fresh supervised batches and passed heldout thresholds:

| H | AUC | Gap | Ensemble-min AUC | Ensemble-min Gap |
|---:|---:|---:|---:|---:|
| 32 | 0.9344 | 0.4309 | 0.9328 | 0.4266 |
| 64 | 0.9749 | 0.6086 | 0.9743 | 0.6064 |
| 96 | 0.9588 | 0.5916 | 0.9591 | 0.5903 |
| 128 | 0.9751 | 0.5476 | 0.9755 | 0.5402 |

Monotonicity violation was `0.0000`.

The action-conditioned geodesic Q critic also passed clean supervised diagnostics with target

```text
Q_H(s_t,a_t,g) = 1[d_grid(s_{t+1},g) <= H-1].
```

### 5.3 Q/V transitive was stable but not enough in abundant-label settings

A frozen state-value teacher trained on budgets `(40,80,160)` had:

| Row | H=40 AUC/gap | H=80 AUC/gap | H=160 AUC/gap |
|---|---:|---:|---:|
| V supervised teacher | 0.9531 / 0.4625 | 0.9749 / 0.4995 | 0.9854 / 0.7048 |

Q supervised and Q+Q/V transitive both passed:

| Row | H=40 AUC/gap | H=80 AUC/gap | H=160 AUC/gap |
|---|---:|---:|---:|
| Q supervised | 0.9563 / 0.5359 | 0.9786 / 0.5138 | 0.9875 / 0.6421 |
| Q + Q/V transitive | 0.9565 / 0.5063 | 0.9763 / 0.4674 | 0.9857 / 0.6712 |

Q/V transitive reduced Q-V-next absolute probability difference from `0.2036` to `0.1345`, but did not produce an abundant-label classification win.

### 5.4 Budget-holdout showed the strongest BMM-specific positive result

The strongest result came from holding out the parent budget.

#### Grid-cell budget holdout

Setup:

```text
supervised budgets = (2,4)
heldout parent = 8
variants = A/B/C/D/F
```

Three-seed aggregate:

| Comparison | Seeds | Delta H8 AUC | Delta H8 Gap | Delta H8 BCE | Delta H8 ECE | Delta Q-V Abs | Interpretation |
|---|---:|---:|---:|---:|---:|---:|---|
| B-A | 0,1,2 | +0.0150 | +0.0751 | -0.3609 | -0.0575 | -0.0208 | no-parent BMM effect |
| D-C | 0,1,2 | +0.0116 | +0.0507 | -0.1503 | -0.0758 | +0.0011 | few-parent BMM effect |
| F-A | 0,1,2 | +0.0002 | +0.0017 | -0.0063 | -0.0010 | -0.0005 | V-next distill control |

#### Env-step budget holdout

Setup:

```text
supervised budgets = (40,80)
heldout parent = 160
variants = A/B/C/D/F
```

Three-seed aggregate:

| Comparison | Seeds | Delta H160 AUC | Delta H160 Gap | Delta H160 BCE | Delta H160 ECE | Delta Q-V Abs | Interpretation |
|---|---:|---:|---:|---:|---:|---:|---|
| B-A | 0,1,2 | +0.0243 | +0.0647 | -0.5815 | -0.0380 | -0.0102 | no-parent BMM effect |
| D-C | 0,1,2 | +0.0041 | +0.0498 | -0.1815 | -0.0547 | -0.0293 | few-parent BMM effect |
| F-A | 0,1,2 | +0.0035 | +0.0030 | -0.0570 | -0.0016 | +0.0026 | V-next distill control |

Conclusion: shorter-budget Q/V knowledge helps heldout longer-budget parent classification. The V-next distillation control is much smaller, suggesting the improvement is BMM-specific.

### 5.5 Dataset-support graph labels were learnable and mildly positive

A conservative dataset-position graph was built from observed transitions:

```text
nodes = 1698
edges = 5548
connected components = 1
graph diameter = 73 hops / 146 calibrated steps
```

Graph labels were clean and learnable for budgets below graph diameter. Example fixed-batch heldout graph-label results:

| H | Eval AUC | Eval Gap | Eval Min AUC | Eval Min Gap |
|---:|---:|---:|---:|---:|
| 32 | 0.9474 | 0.5239 | 0.9445 | 0.5277 |
| 64 | 0.9413 | 0.5047 | 0.9394 | 0.5066 |
| 96 | 0.9728 | 0.6839 | 0.9734 | 0.6815 |
| 128 | 0.9921 | 0.8095 | 0.9915 | 0.7899 |

Graph-label budget holdout was mildly positive:

| Variant | H120 AUC | H120 Gap | Q-V Abs Diff | Q-V Rank Corr |
|---|---:|---:|---:|---:|
| A supervised-only | 0.9783 | 0.3459 | 0.3306 | 0.7802 |
| B Q/V transitive | 0.9849 | 0.3411 | 0.3277 | 0.8084 |
| F V-next distill | 0.9780 | 0.3491 | 0.3265 | 0.7802 |

This weakens the concern that BMM only works with grid-oracle labels, though the graph experiment remains a diagnostic rather than a full general offline RL result.

### 5.6 Flat action ranking did not work

An offline action-ranking diagnostic compared A/B/F critics on candidate actions. The initial same-cell candidate diagnostic was noisy; an improved version added oracle and V-next baselines.

On the more credible own-state action-ranking mode:

| Critic | AUC | Pair Acc | Selected Distance | Selected Success |
|---|---:|---:|---:|---:|
| A | 0.6307 | 0.7814 | 164.2045 | 0.6738 |
| B | 0.6170 | 0.7890 | 163.8565 | 0.6895 |
| F | 0.6321 | 0.7827 | 164.1272 | 0.6777 |

BMM did not robustly beat A/F on action AUC. Flat action extraction was therefore not supported.

### 5.7 Joint Q/V action-subgoal extraction was also mixed

Candidate coverage was improved using `neighbor_cell`, `directional`, and `oracle_diverse` modes. Coverage was no longer the blocker:

| Mode | Oracle Any Action-Valid | Unique Next Cells | Next-Distance Spread |
|---|---:|---:|---:|
| same_cell_cached | 0.0547 | 2.0781 | 21.0330 |
| neighbor_cell | 1.0000 | 5.1641 | 75.0075 |
| directional | 0.9922 | 5.3984 | 73.3063 |
| oracle_diverse | 1.0000 | 7.5938 | 202.5975 |

Even with useful candidates, BMM Q/V did not robustly win the full joint objective. It often improved action-valid and action-midpoint metrics, but not state-valid, path-stretch, or midpoint quality. This weakened the case for neural Q/V policy extraction.

### 5.8 Value-only subgoal selection was positive, but controller bottleneck remained

The value-only high-level score

```text
score(w) = min(V_h(s,w), V_{H-h}(w,g))
```

worked better than Q/V action extraction.

Selector comparison with same-cell NN controller:

| Selector | Success | Final Distance | Improve | Mean Step Goal | Subgoal Valid |
|---|---:|---:|---:|---:|---:|
| random | 0.0000 | 118.7747 | 52.7888 | 0.5279 | 0.1333 |
| geometric_midpoint | 0.0000 | 112.1761 | 59.3874 | 0.5939 | 0.3833 |
| BMM_V | 0.0000 | 105.5775 | 65.9860 | 0.6599 | 0.6633 |
| oracle_midpoint | 0.0000 | 118.7747 | 52.7888 | 0.5279 | 0.4267 |

The graph-support subgoal diagnostic was also positive:

| Scorer | State Valid | Path Stretch | Midpoint Err |
|---|---:|---:|---:|
| random | 0.7422 | 29.0078 | 57.9219 |
| euclidean_midpoint | 0.9902 | 6.8711 | 68.5508 |
| oracle_graph_midpoint | 0.6387 | 65.6875 | 15.2422 |
| BMM_V_graph | 0.9902 | 4.7734 | 70.6406 |

However, the hierarchical policy pivot did not clear the final go/no-go criterion. With a stronger 5k-step, larger BC controller:

| Selector | Success | Final Distance | Improve | Subgoal Valid | Subgoal Reduce | Goal Reduce |
|---|---:|---:|---:|---:|---:|---:|
| random | 0.0000 | 134.6113 | 39.5916 | 0.1020 | 0.0400 | 0.0320 |
| geometric_midpoint | 0.0000 | 106.8972 | 67.3057 | 0.2820 | 0.0360 | 0.0360 |
| BMM_V | 0.0000 | 106.8972 | 67.3057 | 0.6100 | 0.0340 | 0.0340 |
| oracle_midpoint | 0.0000 | 122.7339 | 51.4690 | 0.1960 | 0.2800 | 0.1720 |

BMM/V tied geometric midpoint on final distance and improvement, so the hierarchical pivot did not pass the continue gate.

---

## 6. Overall conclusions

### 6.1 What worked

1. The mathematical BMM operator has the desired non-expansive error property for deterministic/support reachability.
2. The implementation can learn clean budgeted reachability classifiers.
3. Logged-offset high-budget targets were correctly diagnosed as flawed behavior-time labels.
4. Geodesic and support-graph targets are learnable.
5. Q/V transitive budget-holdout improves heldout long-budget classification across seeds.
6. The improvement is not explained by simple V-next distillation.
7. Value-based BMM subgoal selection gives useful intermediate goals in diagnostics.

### 6.2 What did not work

1. The current prototype did not show robust flat action-ranking gains.
2. Q/V joint action-subgoal selection did not become robust even after improving candidate coverage.
3. Lightweight hierarchical control did not make BMM/V outperform a simple geometric midpoint baseline.
4. No policy smoke solved the tasks.
5. The method is not ready for broad OGBench policy benchmarks.

### 6.3 Current project status

The evidence supports a **critic/reachability/subgoal diagnostic contribution**, not an end-to-end policy algorithm.

A fair statement is:

```text
BMM-TRL provides a max-min budgeted reachability objective with logarithmic-depth error propagation in deterministic/support settings. In PointMaze diagnostics, Q/V transitive consistency improves heldout parent-budget reachability. However, converting this critic-level improvement into robust policy improvement remains unresolved.
```

---

## 7. Recommended decision

### Recommended action

Pause active policy-facing experimentation.

Do not continue with:

```text
flat Q extraction;
Q/V action extraction;
more sparse-Q tables;
more loss/witness tuning;
PointMaze-large policy runs;
broad OGBench benchmarks.
```

### Viable paths

There are three possible next directions:

#### Option A: Write up as a diagnostic/methods note

This is the recommended low-compute path. Focus on:

```text
BMM theory;
logged-offset target failure;
geodesic/support reachability targets;
budget-holdout Q/V gains;
limitations of policy extraction.
```

#### Option B: Formal hierarchical RL pivot

Only pursue if the project is explicitly reframed around high-level planning plus a strong low-level controller. This would require:

```text
strong goal-conditioned low-level policy;
subgoal proposal over support states;
replanning;
comparison to geometric/graph/HIQL-style baselines.
```

This is effectively a new hierarchical RL project.

#### Option C: Pause and move on

This is reasonable if the goal is a high-impact offline RL policy result. The current policy-facing signal is not strong enough to justify more compute without a new low-level controller effort.

---

## 8. Suggested advisor-facing summary

The concise advisor message is:

```text
We proposed BMM-TRL to replace additive/product transitive value composition with a budgeted max-min reachability objective. The theory gives O(log H) sup-norm error accumulation for deterministic/support reachability. Experimentally, the critic/reachability side works: logged-offset targets fail, geodesic/support targets are learnable, and Q/V transitive improves heldout long-budget classification across seeds. However, flat action extraction and lightweight hierarchical control did not produce robust policy gains. The current recommendation is to pause policy-facing experimentation and either write this as a diagnostic/methods note or formally pivot to a hierarchical RL project with a stronger low-level controller.
```

---

## 9. Files and artifacts referenced

Key result files:

```text
BMM_TRL_STATUS_20260610_181801.md
BMM_TRL_GEODESIC_VALUE_RESULTS_20260610_184508.md
BMM_TRL_QV_TRANSITIVE_RESULTS_20260610_213730.md
BMM_TRL_BUDGET_HOLDOUT_REPLICATION_RESULTS_20260611_014354.md
BMM_TRL_FAST_DECISION_RESULTS_20260611_124920.md
BMM_TRL_CANDIDATE_ACTION_RESULTS_20260611_135918.md
BMM_TRL_VALUE_SUBGOAL_NEXT_STEPS_RESULTS_20260611_163357.md
BMM_TRL_HIERARCHICAL_PIVOT_QUICK_TRY_20260611_173408.md
```

Key implementation files:

```text
agents/bmm_trl.py
utils/pointmaze_grid.py
utils/pointmaze_graph.py
scripts/train_bmm_geodesic_value.py
scripts/train_bmm_geodesic_q.py
scripts/run_bmm_qv_budget_holdout.py
scripts/eval_bmm_action_ranking.py
scripts/eval_bmm_joint_action_subgoal.py
scripts/eval_bmm_value_subgoal_policy_smoke.py
scripts/eval_bmm_graph_value_subgoal.py
scripts/eval_bmm_subgoal_bc_controller.py
```
