#!/usr/bin/env python
"""Gate BMM-TRL reachability diagnostics for budget-collapse failures."""

import argparse
import json
import math
import sys


def is_finite(value):
    return value is not None and math.isfinite(float(value))


def check_report(
    report,
    max_budget=None,
    min_auc=0.75,
    min_gap=0.10,
    max_neg_mean=0.80,
    max_mono=0.15,
):
    """Return lists of failures and informational skips for a diagnostic report."""
    failures = []
    skipped = []
    if max_budget is None:
        budgets = report.get("budgets", [])
        max_budget = max(budgets) if budgets else float("inf")
    rows = report.get("balanced_budget_rows") or report.get("budget_rows", [])
    using_balanced = bool(report.get("balanced_budget_rows"))

    for key in ("mean_monotonicity_violation", "min_monotonicity_violation"):
        if key not in report:
            continue
        value = report[key]
        if not is_finite(value):
            failures.append(f"{key} is non-finite: {value}")
        elif float(value) > max_mono:
            failures.append(f"{key}={float(value):.4f} exceeds {max_mono:.4f}")

    for row in rows:
        budget = int(row["budget"])
        if budget > max_budget:
            skipped.append(f"H={budget}: above max_budget={max_budget}")
            continue

        budget_min_auc, budget_min_gap = thresholds_for_budget(
            budget, min_auc, min_gap, using_balanced
        )
        for metric_name in ("mean", "ensemble_min"):
            metrics = row.get(metric_name, {})
            pos_count = int(metrics.get("pos_count", 0))
            neg_count = int(metrics.get("neg_count", 0))
            if pos_count == 0 or neg_count == 0:
                skipped.append(
                    f"H={budget} {metric_name}: one-class labels "
                    f"(pos={pos_count}, neg={neg_count})"
                )
                continue

            auc = metrics.get("auc")
            pos_mean = metrics.get("pos_mean")
            neg_mean = metrics.get("neg_mean")
            for name, value in (
                ("auc", auc),
                ("pos_mean", pos_mean),
                ("neg_mean", neg_mean),
            ):
                if not is_finite(value):
                    failures.append(
                        f"H={budget} {metric_name}: {name} is non-finite: {value}"
                    )

            if not all(is_finite(value) for value in (auc, pos_mean, neg_mean)):
                continue

            gap = float(pos_mean) - float(neg_mean)
            if float(auc) < budget_min_auc:
                failures.append(
                    f"H={budget} {metric_name}: auc={float(auc):.4f} "
                    f"below {budget_min_auc:.4f}"
                )
            if gap < budget_min_gap:
                failures.append(
                    f"H={budget} {metric_name}: pos-neg gap={gap:.4f} "
                    f"below {budget_min_gap:.4f}"
                )
            if float(neg_mean) > max_neg_mean:
                failures.append(
                    f"H={budget} {metric_name}: neg_mean={float(neg_mean):.4f} "
                    f"exceeds {max_neg_mean:.4f}"
                )

    return failures, skipped


def thresholds_for_budget(budget, min_auc, min_gap, using_balanced):
    if not using_balanced:
        return min_auc, min_gap
    if budget <= 128:
        return 0.90, 0.20
    if budget == 256:
        return 0.85, 0.15
    if budget >= 512:
        return 0.80, 0.10
    return min_auc, min_gap


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report_path", required=True)
    parser.add_argument("--max_budget", type=int)
    parser.add_argument("--min_auc", type=float, default=0.75)
    parser.add_argument("--min_gap", type=float, default=0.10)
    parser.add_argument("--max_neg_mean", type=float, default=0.80)
    parser.add_argument("--max_mono", type=float, default=0.05)
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.report_path) as f:
        report = json.load(f)

    failures, skipped = check_report(
        report,
        max_budget=args.max_budget,
        min_auc=args.min_auc,
        min_gap=args.min_gap,
        max_neg_mean=args.max_neg_mean,
        max_mono=args.max_mono,
    )
    for item in skipped:
        print(f"SKIP {item}")
    if failures:
        for item in failures:
            print(f"FAIL {item}")
        return 1

    print("PASS BMM reachability report gate")
    return 0


if __name__ == "__main__":
    sys.exit(main())
