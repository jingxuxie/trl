# BMM-TRL next steps after final paper controls

Date: 2026-06-12

This plan follows `BMM_TRL_FINAL_PAPER_CONTROL_RESULTS_20260612_020718.md`.

## Executive decision

The final paper-control results are coherent and supportive of the **focused** BMM-TRL paper.

The right decision is:

```text
Stop running experiments for this version.
Proceed with the focused arXiv-style report.
Do not run more policy/controller experiments.
Do not run broad multi-environment sweeps.
```

The paper should be framed as:

```text
BMM-TRL is a non-expansive transitive reachability objective.
It is theoretically motivated by O(log H) error accumulation under deterministic/support reachability.
It improves heldout long-budget reachability in budget-holdout diagnostics.
It is empirically competitive with, and mildly better than, a product-style transitive control in the current diagnostics.
Policy extraction remains open.
```

This is a complete paper if presented as a **preliminary technical report / arXiv manuscript**, not as a broad empirical offline RL benchmark paper.

## What the final controls establish

### 1. Product is a strong control, but does not dominate BMM

The final grid-cell H8 three-seed comparison is:

```text
B-A: +0.0150 AUC, +0.0751 gap, -0.3609 BCE, -0.0575 ECE
P-A: +0.0139 AUC, +0.0702 gap, -0.3344 BCE, -0.0518 ECE
B-P: +0.0011 AUC, +0.0050 gap, -0.0265 BCE, -0.0057 ECE
F-A: +0.0002 AUC, +0.0017 gap, -0.0063 BCE, -0.0010 ECE
```

The final env-step H160 seed-0 comparison is:

```text
B-A: +0.0400 AUC, +0.0640 gap, -0.5976 BCE, -0.0398 ECE
P-A: +0.0345 AUC, +0.0517 gap, -0.5166 BCE, -0.0316 ECE
B-P: +0.0055 AUC, +0.0122 gap, -0.0810 BCE, -0.0083 ECE
F-A: +0.0035 AUC, +0.0067 gap, -0.0774 BCE, -0.0038 ECE
```

Interpretation:

```text
Product captures much of the transitive bootstrapping gain.
BMM is consistently better than product in the tested controls.
The margin is small.
```

Therefore the safe claim is:

```text
Max-min has cleaner non-expansive theory and is empirically competitive with, and mildly better than, product in these diagnostics.
```

Do **not** claim decisive empirical dominance over product.

### 2. The tabular scaling figure gives the strongest theory-to-experiment bridge

The tabular result shows the intended separation:

```text
H=1024, epsilon=0.02
BMM balanced sup error:       0.2200
additive/product sup error:  20.4800
```

This should be the main figure supporting the theoretical motivation.

### 3. The support-graph result is useful but mixed

The support-graph H120 three-seed result is:

```text
B-A: +0.0043 AUC, -0.0194 gap, +0.0224 BCE, +0.0099 ECE
```

This should be presented as:

```text
support-graph labels are feasible and learnable;
BMM gives a small AUC gain but not a robust all-metric win;
support-graph construction remains future work.
```

Do not make support-graph H120 the main empirical result.

## Do we need another environment?

### For this arXiv-style report: no

For the current scope, another environment is **not required**.

The paper is not claiming broad OGBench performance. It is claiming:

```text
1. a new reachability objective and error-algebra result;
2. target-diagnosis lessons;
3. budget-holdout reachability bootstrapping diagnostics;
4. limitations of policy extraction.
```

For that scope, the current evidence is enough:

```text
tabular chain/grid scaling;
PointMaze-medium geodesic diagnostics;
PointMaze-medium support-graph diagnostics;
product controls;
policy-extraction limitation diagnostics.
```

### For a stronger conference-style empirical paper: yes, probably

If the goal is a stronger empirical paper, one more environment would help. But this would also expand the project substantially and may restart the experimentation cycle.

A second environment should be added only if it is cheap and diagnostic-only:

```text
PointMaze-large value/reachability budget-holdout only;
no policy;
no controller;
no broad sweeps.
```

Recommended optional second-environment test:

```text
env = pointmaze-large-navigate-v0
run geometry inspection first
choose budgets below calibrated diameter
run A/B/P/F budget-holdout for one seed
```

Only expand to more seeds if BMM clearly beats A and product does not dominate.

However, this is optional. It is not necessary for the focused arXiv report.

## Do we have enough experiments now?

Yes, for a focused preliminary paper.

The current complete story is:

```text
1. Theory: max-min reachability is non-expansive.
2. Toy scaling: BMM shows log-depth error growth in controlled tabular diagnostics.
3. Target diagnosis: logged-offset labels fail; geodesic/support targets are cleaner.
4. Supervised learning: clean V/Q reachability labels are learnable.
5. Budget holdout: BMM improves heldout long-budget reachability.
6. Product control: product is strong, but BMM is mildly better and theoretically cleaner.
7. Support graph: feasible but mixed; future work.
8. Policy extraction: not robust; limitation.
```

That is enough for an arXiv report.

## Final paper claim to use

Use this claim:

```text
BMM-TRL changes the transitive value-learning algebra from additive distance composition to non-expansive reachability composition. In deterministic/support settings this yields an O(log H) error-propagation structure. Empirically, BMM improves heldout long-budget reachability in PointMaze budget-holdout diagnostics and is mildly better than a product-style transitive control, but policy extraction remains unresolved.
```

Avoid this claim:

```text
BMM-TRL solves long-horizon offline RL or outperforms TRL on OGBench.
```

## What to do next

## Milestone 1: freeze experiments

Do not run more experiments unless a reviewer/advisor specifically asks.

Freeze the current artifact set:

```text
exp/bmm_paper_tables_final.md
exp/bmm_paper_tables_final.json
assets/bmm_tabular_error_scaling.png
BMM_TRL_FINAL_PAPER_CONTROL_RESULTS_20260612_020718.md
```

## Milestone 2: update the arXiv report with final controls

Update `BMM_TRL_ARXIV_REPORT.md` to include:

```text
1. final H8 B/P/F three-seed product table;
2. env-step H160 product seed-0 table;
3. support-graph H120 mixed result;
4. explicit statement that product is a strong control;
5. explicit statement that BMM is only mildly better empirically;
6. policy extraction limitations.
```

## Milestone 3: improve academic presentation

Turn the report into a more paper-like document.

Add:

```text
1. short theorem boxes;
2. algorithm box for BMM Q/V target;
3. clean figure captions;
4. table captions with interpretation;
5. limitations section before conclusion;
6. future-work section.
```

Suggested figures/tables:

```text
Figure 1: additive vs max-min error scaling
Table 1: target diagnosis / label quality
Table 2: budget-holdout main results
Table 3: product-vs-max-min control
Table 4: policy-extraction limitations
```

## Milestone 4: optional second environment only if advisor insists

If advisor or arXiv framing requires more than one environment, run exactly one extra diagnostic:

```text
PointMaze-large budget-holdout, one seed, A/B/P/F only.
```

Do not run policy.

Decision for optional environment:

```text
If BMM >= product and both beat A: include as small additional evidence.
If product dominates or results are mixed: omit from main paper or put in appendix.
```

## Milestone 5: finalize paper narrative

Recommended title:

```text
Budgeted Max-Min Transitive Reachability for Offline Goal-Conditioned Reinforcement Learning
```

Recommended abstract structure:

```text
1. TRL/product composition is additive in distance error.
2. BMM learns budgeted reachability and uses max-min composition.
3. The operator is non-expansive and gives O(log H) error propagation under ideal assumptions.
4. Experiments validate reachability bootstrapping and target diagnosis.
5. Policy extraction remains open.
```

## What not to do

Do not run:

```text
more policy/controller experiments;
more large-maze policy runs;
more support-graph tuning;
more product-control sweeps;
more loss/witness sweeps;
more sparse-Q experiments.
```

These are unlikely to change the paper's core conclusion and could delay writing.

## Final go/no-go

### Go for arXiv report

Yes.

The paper is complete enough as a preliminary technical report.

### Go for broad benchmark paper

No.

The empirical scope is too narrow and policy extraction did not work.

### Need another environment

Not for arXiv.

Optional for a stronger empirical venue, but only as a one-seed diagnostic if cheap.

## Immediate task list

1. Update `BMM_TRL_ARXIV_REPORT.md` with final product-control numbers.
2. Add the tabular error-scaling figure to the report.
3. Add a paragraph explaining why product is a strong empirical control.
4. Add a paragraph explaining why support-graph results are mixed.
5. Add a clear limitations section:
   ```text
   no robust policy improvement;
   support-graph target construction remains open;
   stochastic extension not implemented;
   product control is close empirically.
   ```
6. Freeze experiments.
7. Convert Markdown to LaTeX only after the narrative is finalized.

## Bottom line

You have enough experiments for a focused arXiv report.

The final paper should be honest and precise:

```text
BMM has the better error algebra and improves heldout reachability diagnostics.
Product is a strong control and often close.
Policy extraction remains unsolved.
```

That is a coherent research contribution.
