# BMM-TRL next steps after graph/grid geodesic diagnostics

Date: 2026-06-11

This plan follows `BMM_TRL_STATUS_20260610_181801.md`.

## Executive conclusion

You made real progress. The project is not stuck at an implementation bug anymore; it has moved to the more important question of **what reachability target BMM should learn**.

The current evidence says:

```text
BMM implementation: mostly healthy.
Logged-offset high-budget label: not a clean PointMaze reachability target.
Graph/grid geodesic labels: clean and learnable on meaningful PointMaze-medium budgets.
```

The key discovery is that `H=256` and `H=512` are not valid binary reachability tests for PointMaze medium under calibrated geodesic distance. The medium maze diameter is about `218` calibrated steps under layout/grid BFS, so those horizons should be all-positive. The earlier high-budget negatives were behavior-policy detours, not true geodesic unreachability.

Therefore:

```text
Do not try to make PointMaze-medium geodesic labels separate positives and negatives at H=256/512.
Use budgets below the calibrated diameter, or move to a larger maze.
```

## What target should be used next?

### Primary diagnostic target: layout/grid geodesic reachability

Use the environment layout/grid BFS as the clean PointMaze diagnostic target:

```text
V_H(s, g) = 1[d_grid(xy_s, xy_g) <= H]
```

where `d_grid` is converted to calibrated environment-step units.

This is the best target for algorithm verification because it is:

- deterministic from `(s, g, H)`;
- independent of behavior-policy detours;
- topology-aware;
- monotone in `H`;
- compatible with max-min BMM composition;
- backed by a non-neural oracle.

### Secondary diagnostic target: dataset-position graph reachability

Use the dataset-position graph as an **offline-support reachability** target:

```text
V_H^graph(s, g) = 1[d_graph(bin_s, bin_g) <= H]
```

This target is useful because it follows observed transitions and avoids layout hardcoding. But it is not the same as true environment geodesic reachability. In current results it has a smaller calibrated diameter than layout BFS, so treat it as a support-graph proxy, not the final environment target.

### Action-conditioned target: Bellman next-state label

After state-only geodesic works, define action-conditioned labels as:

```text
Q_H(s_t, a_t, g) = 1[d(s_{t+1}, g) <= H - 1]
```

Do not label `Q(s,a,g,H)` directly from logged future offset. The action-conditioned object should mean: after taking this action and arriving at `s_next`, is the goal reachable within the remaining budget?

### Logged-offset target: auxiliary only

Keep logged same-trajectory offset as:

```text
behavior-time diagnostic
positive-anchor source
trajectory witness source
```

but no longer use:

```text
offset > H
```

as a hard high-budget negative in PointMaze.

## Answer to the four proposed options

### 1. Train BMM on graph/grid geodesic labels for budgets below the maze diameter?

Yes. This is the immediate next step.

For PointMaze medium, use budgets such as:

```text
32, 64, 96, 128
```

Optionally include `160` as a stress test, but it has few negative cell pairs under grid BFS. Avoid `224`, `256`, and `512` for binary separation on PointMaze medium because they are one-class or near-one-class.

For a split-closed BMM experiment, prefer:

```text
16, 32, 64, 128
```

because half-splits stay inside the budget set. Use `32,64,96,128` for supervised diagnostics if it gives better class balance, but use split-closed budgets for max-min recurrence tests.

### 2. Move to a larger maze where H=256/512 still has negatives?

Yes, but only after medium geodesic supervised and BMM-consistency diagnostics pass.

Before training on a larger maze, run the same layout/grid BFS diameter script and confirm:

```text
calibrated diameter > 512
```

or at least:

```text
negative cell pairs exist at H=256 and H=512.
```

If the calibrated diameter is still below 512, either use smaller budgets or change the budget unit. Do not force `H=512` to be a binary test if the task diameter says it is all-positive.

### 3. Redefine action-conditioned Q labels as Q_H(s,a,g)=1[d(s_next,g)<=H-1]?

Yes. This is the right action-conditioned diagnostic.

Use the logged transition only for the first step:

```text
s_t, a_t, s_{t+1}
```

Then use the state-only geodesic oracle for the remaining horizon:

```text
label_Q = 1[d_geo(s_{t+1}, g) <= H - 1]
```

This creates a Bellman-consistent `Q` target and avoids asking the action-conditioned critic to predict behavior-time trajectory labels.

### 4. Use logged-offset only as positive/behavior-time auxiliary signal, not hard negatives?

Yes.

Safe use:

```text
if offset <= H: positive anchor
```

Unsafe high-budget use:

```text
if offset > H: hard negative
```

The latter is exactly what failed: it creates false negatives whenever the behavior path is longer than a shorter feasible path.

## Code review notes

### 1. Graph construction is conservative and useful

`utils/pointmaze_graph.py` builds nodes from occupied xy bins and edges only from consecutive observed transitions. This is good because it avoids kNN shortcuts through walls.

Current graph metadata uses:

```text
env_steps_per_graph_edge = bin_size / median_step_xy
```

With the default `bin_size_factor=2.0`, one graph edge is calibrated as about two environment steps. This is a reasonable support-graph proxy, but it is not the same as layout BFS.

Recommendation:

- keep graph labels as `dataset_support_graph`;
- add a separate `grid_geodesic` label type;
- report both diameters and class coverage in every run.

### 2. The graph currently uses train + validation observations/edges

`build_dataset_position_graph(train_dataset, val_dataset, ...)` uses both train and validation data to create graph nodes and transition edges.

This is fine if the graph is an **oracle diagnostic** for support topology. But for a strict train/val generalization experiment, add a flag:

```text
graph_scope = train_only | train_val_oracle
```

Recommended use:

- `train_val_oracle` for target-definition diagnostics;
- `train_only` for offline-support generalization claims.

### 3. Graph-label sampler is good enough for diagnostics, but not yet integrated into main training

`sample_graph_budget_pairs()` can sample balanced graph positives/negatives for a budget. The fixed-batch graph overfit script proves the labels are learnable, but the next step needs a **multi-batch graph-label training path**.

Add a dataset config:

```python
reachability_label_type = "logged_offset"  # "logged_offset", "graph", "grid_geodesic"
graph_path = "exp/bmm_pointmaze_graph.npz"
geodesic_distance_source = "grid"          # "grid" or "graph"
```

Then make `GCDataset.add_bmm_supervised_fields` use graph/grid labels when requested.

### 4. Medium-maze H=256/512 should be skipped, not failed

The status file shows layout/grid BFS has max distance about `217.75` calibrated steps. Therefore `H=256` and `H=512` have zero negative free-cell pairs.

Update gates so that one-class geodesic rows are:

```text
SKIP: budget above geodesic diameter
```

not failures.

### 5. Keep diagnostic losses truly clean

For all geodesic supervised runs, keep:

```text
value_only=True
lambda_sup=1.0
lambda_trans=0.0
lambda_rank=0.0
num_rank_pairs=0
lambda_mono=0.0 initially
lambda_pos=0.0
lambda_budget_neg=0.0
lambda_hard_neg=0.0
lambda_rand_hinge=0.0
```

After the classifier passes, add max-min consistency. Do not add ranking or monotonicity until the clean supervised target is stable.

## Immediate next milestone: multi-batch state-only geodesic training

The current graph fixed-batch smoke is strong, but the next milestone should train on fresh graph/grid supervised pairs over many updates and evaluate on heldout graph/grid pairs.

### Script to add

```text
scripts/train_bmm_geodesic_value.py
```

or integrate this into `main.py` through `GCDataset`.

### Config

```text
diagnostic_critic_mode=state
value_only=True
reachability_label_type=grid_geodesic  # primary
fallback_label_type=graph              # secondary
budgets=(16,32,64,128)
max_budget=128
num_sup_pairs=8 or 16
lambda_sup=1.0
all other auxiliary losses=0.0
```

### Evaluation

Report by budget:

```text
AUC
gap
pos_mean
neg_mean
pos_count
neg_count
monotonicity violation
oracle distance AUC
Euclidean AUC
```

### Passing threshold

For PointMaze medium grid/geodesic labels:

```text
H=16:  AUC >= 0.90, gap >= 0.20
H=32:  AUC >= 0.90, gap >= 0.20
H=64:  AUC >= 0.90, gap >= 0.20
H=128: AUC >= 0.85, gap >= 0.15
```

If `H=16` has too few positives or too much discretization noise, start with:

```text
32,64,96,128
```

for supervised diagnostics, then return to split-closed budgets for BMM.

## Second milestone: action-conditioned geodesic Q

After state-only geodesic value passes, train:

```text
Q_H(s_t, a_t, g) = 1[d_geo(s_{t+1}, g) <= H - 1]
```

### Data fields

Add supervised fields:

```text
value_sup_observations = s_t
value_sup_actions      = a_t
value_sup_goals        = g
value_sup_budgets      = H
value_sup_labels       = 1[d_geo(s_{t+1}, g) <= H - 1]
```

For state-only mode, observations are `s_t` or `s_{t+1}` depending on the target. For action-conditioned mode, use `s_t,a_t` and compute label from `s_next`.

### Expected result

Action-conditioned metrics should be close to state-only metrics for large budgets. If they are much worse, debug transition indexing and the `H-1` conversion.

## Third milestone: max-min transitive consistency on geodesic labels

Only after the supervised geodesic classifier passes:

```text
lambda_sup=1.0
lambda_trans=0.025 or 0.05
lambda_mono=0.0
lambda_rank=0.0
```

### Witness selection

For a target pair `(s,g,H)` and split `h`, choose witnesses `w` satisfying:

```text
d_geo(s,w) <= h
d_geo(w,g) <= H-h
```

For PointMaze medium and split-closed budgets, use:

```text
H=32, h=16
H=64, h=32
H=128, h=64
```

Sample multiple valid witnesses when possible. If no valid witness exists, skip transitive loss for that pair.

### Target

```text
y_trans = max_w min(R_h(s,w), R_{H-h}(w,g))
```

Use target network and preferably ensemble-min lower-confidence branch values once the basic version works.

### Success criterion

The max-min version should not degrade supervised geodesic AUC/gap, and should improve budget consistency or reduce sample complexity versus supervised-only.

## Fourth milestone: larger maze for H=256/512

Once PointMaze medium works for meaningful budgets, move to:

```text
pointmaze-large-navigate-v0
pointmaze-large-stitch-v0
```

Before training, run:

```bash
python scripts/inspect_pointmaze_grid_bfs.py \
  --env_name=pointmaze-large-navigate-v0 \
  --budgets=64,128,256,512
```

Proceed only if the report shows both positive and negative cell pairs at `H=256` and preferably `H=512`.

If not, choose budgets based on the actual calibrated diameter. The budget list should be a property of the environment geometry, not copied from the original dyadic list blindly.

## Fifth milestone: policy-facing BMM

Policy evaluation should wait until:

1. state-only geodesic supervised labels pass;
2. action-conditioned geodesic Q labels pass;
3. max-min transitive consistency does not harm diagnostics.

Then extract policy with a budget scan:

```text
H_hat(s,g) = smallest H with R_H(s,g) >= tau
```

Use a tighter action budget rather than always using max budget:

```text
H_action = max(min_budget, H_hat(s,g))
```

For action selection:

```text
a = argmax_a Q_H_action(s,a,g)
```

For FRS, use candidate actions and score them with the budget-conditioned critic.

## What to stop doing

Until the geodesic pipeline passes, stop spending time on:

- high-budget logged-offset gates;
- hard negatives from `offset > H`;
- policy success rate;
- actor tuning;
- ranking loss;
- monotonicity penalties;
- `H=256/512` on PointMaze medium.

These are not the right bottleneck anymore.

## What this means for the research story

The research story is stronger now:

```text
BMM-TRL is not merely a different loss on TRL.
It requires a reachability target whose algebra matches max-min composition.
Logged behavior time is not that target at long horizons.
Maze/geodesic or support-graph reachability is the right diagnostic target.
```

This is real progress because it separates three things that were previously conflated:

1. implementation correctness;
2. target identifiability;
3. policy usefulness.

You have mostly cleared (1), found a failure mode in (2), and now have a clean path to test (2) before returning to (3).

## Concrete next run order

1. Add `reachability_label_type={logged_offset,graph,grid_geodesic}` to dataset/eval code.
2. Train state-only multi-batch grid/geodesic value on PointMaze medium budgets `(16,32,64,128)` or `(32,64,96,128)`.
3. Evaluate heldout grid/geodesic labels and skip one-class budgets above diameter.
4. Train action-conditioned Bellman next-state `Q_H` labels.
5. Add max-min transitive consistency with geodesic-valid witnesses.
6. Run the same pipeline on PointMaze large after checking its calibrated diameter.
7. Only then run policy comparisons.
