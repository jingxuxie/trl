#!/usr/bin/env python
"""Unit checks for the BMM reachability diagnostic gate."""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


SCRIPT = REPO_ROOT / "scripts" / "check_bmm_reachability_report.py"


def row(
    budget,
    auc,
    pos_mean,
    neg_mean,
    pos_count=16,
    neg_count=16,
    min_auc=None,
    min_pos_mean=None,
    min_neg_mean=None,
):
    min_auc = auc if min_auc is None else min_auc
    min_pos_mean = pos_mean if min_pos_mean is None else min_pos_mean
    min_neg_mean = neg_mean if min_neg_mean is None else min_neg_mean
    return {
        "budget": budget,
        "mean": {
            "bce": 0.2,
            "accuracy": 0.8,
            "pos_mean": pos_mean,
            "neg_mean": neg_mean,
            "auc": auc,
            "pos_count": pos_count,
            "neg_count": neg_count,
        },
        "ensemble_min": {
            "bce": 0.2,
            "accuracy": 0.8,
            "pos_mean": min_pos_mean,
            "neg_mean": min_neg_mean,
            "auc": min_auc,
            "pos_count": pos_count,
            "neg_count": neg_count,
        },
    }


def write_report(path, budget_rows, mono=0.05, balanced_rows=None):
    with open(path, "w") as f:
        json.dump(
            {
                "budgets": [row["budget"] for row in budget_rows],
                "random_pair_count": 64,
                "balanced_pair_count": 64,
                "mean_monotonicity_violation": mono,
                "min_monotonicity_violation": mono,
                "budget_rows": budget_rows,
                "balanced_budget_rows": balanced_rows or [],
                "bucket_rows": [],
            },
            f,
        )


def run_gate(path):
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--report_path", str(path)],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        passing = tmpdir / "passing.json"
        write_report(
            passing,
            [
                row(64, auc=0.86, pos_mean=0.82, neg_mean=0.32),
                row(128, auc=0.78, pos_mean=0.76, neg_mean=0.60),
                row(
                    512,
                    auc=float("nan"),
                    pos_mean=0.98,
                    neg_mean=float("nan"),
                    neg_count=0,
                ),
            ],
        )
        result = run_gate(passing)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "SKIP H=512" in result.stdout

        collapsed = tmpdir / "collapsed.json"
        write_report(
            collapsed,
            [
                row(64, auc=0.86, pos_mean=0.82, neg_mean=0.32),
                row(128, auc=0.60, pos_mean=0.91, neg_mean=0.89),
            ],
        )
        result = run_gate(collapsed)
        assert result.returncode != 0, result.stdout
        assert "auc" in result.stdout
        assert "pos-neg gap" in result.stdout
        assert "neg_mean" in result.stdout

        bad_mono = tmpdir / "bad_mono.json"
        write_report(
            bad_mono,
            [row(64, auc=0.86, pos_mean=0.82, neg_mean=0.32)],
            mono=0.16,
        )
        result = run_gate(bad_mono)
        assert result.returncode != 0, result.stdout
        assert "monotonicity" in result.stdout

        balanced_passing = tmpdir / "balanced_passing.json"
        balanced_rows = [
            row(128, auc=0.93, pos_mean=0.80, neg_mean=0.40),
            row(256, auc=0.87, pos_mean=0.78, neg_mean=0.58),
            row(512, auc=0.82, pos_mean=0.76, neg_mean=0.64),
            row(
                1024,
                auc=float("nan"),
                pos_mean=0.98,
                neg_mean=float("nan"),
                neg_count=0,
            ),
        ]
        write_report(
            balanced_passing,
            [row(128, auc=0.1, pos_mean=0.5, neg_mean=0.5)],
            balanced_rows=balanced_rows,
        )
        result = run_gate(balanced_passing)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "SKIP H=1024" in result.stdout

        bad_ensemble_min = tmpdir / "bad_ensemble_min.json"
        write_report(
            bad_ensemble_min,
            [row(128, auc=0.95, pos_mean=0.85, neg_mean=0.35)],
            balanced_rows=[
                row(
                    128,
                    auc=0.95,
                    pos_mean=0.85,
                    neg_mean=0.35,
                    min_auc=0.70,
                    min_pos_mean=0.62,
                    min_neg_mean=0.50,
                )
            ],
        )
        result = run_gate(bad_ensemble_min)
        assert result.returncode != 0, result.stdout
        assert "ensemble_min" in result.stdout

    print("BMM reachability gate checks passed.")


if __name__ == "__main__":
    main()
