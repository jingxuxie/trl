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

    for key in ("mean_monotonicity_violation", "min_monotonicity_violation"):
        if key not in report:
            continue
        value = report[key]
        if not is_finite(value):
            failures.append(f"{key} is non-finite: {value}")
        elif float(value) > max_mono:
            failures.append(f"{key}={float(value):.4f} exceeds {max_mono:.4f}")

    for row in report.get("budget_rows", []):
        budget = int(row["budget"])
        if budget > max_budget:
            skipped.append(f"H={budget}: above max_budget={max_budget}")
            continue

        metrics = row["mean"]
        pos_count = int(metrics.get("pos_count", 0))
        neg_count = int(metrics.get("neg_count", 0))
        if pos_count == 0 or neg_count == 0:
            skipped.append(
                f"H={budget}: one-class labels (pos={pos_count}, neg={neg_count})"
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
                failures.append(f"H={budget}: {name} is non-finite: {value}")

        if not all(is_finite(value) for value in (auc, pos_mean, neg_mean)):
            continue

        gap = float(pos_mean) - float(neg_mean)
        if float(auc) < min_auc:
            failures.append(f"H={budget}: auc={float(auc):.4f} below {min_auc:.4f}")
        if gap < min_gap:
            failures.append(f"H={budget}: pos-neg gap={gap:.4f} below {min_gap:.4f}")
        if float(neg_mean) > max_neg_mean:
            failures.append(
                f"H={budget}: neg_mean={float(neg_mean):.4f} exceeds {max_neg_mean:.4f}"
            )

    return failures, skipped


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report_path", required=True)
    parser.add_argument("--max_budget", type=int)
    parser.add_argument("--min_auc", type=float, default=0.75)
    parser.add_argument("--min_gap", type=float, default=0.10)
    parser.add_argument("--max_neg_mean", type=float, default=0.80)
    parser.add_argument("--max_mono", type=float, default=0.15)
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
