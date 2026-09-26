#!/usr/bin/env python3
"""Summarize subject-level strict-LOSO evidence against frozen PatchTST."""

from __future__ import annotations

import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "storage/results_mesa_v672_final_loso"
PUBLIC_ROOT = ROOT / "storage/results_mesa_strict_loso"
VARIANT = "mesanet_fixed_dual_coordinate"
COMPARATOR = "PatchTST"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def summary_paths(root: Path) -> list[Path]:
    return sorted(root.glob("*/*/_summary/v507_mesa_complete_benchmark_summary.csv"))


def key_from_path(path: Path) -> tuple[str, str]:
    return path.parts[-4], path.parts[-3]


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - position) + ordered[hi] * (position - lo)


def bootstrap_interval(values: list[float], seed: int) -> tuple[float, float]:
    rng = random.Random(seed)
    draws = [statistics.mean(rng.choice(values) for _ in values) for _ in range(20000)]
    return percentile(draws, 0.025), percentile(draws, 0.975)


def main() -> int:
    candidate: dict[tuple[str, str], dict[str, str]] = {}
    for path in summary_paths(RESULT_ROOT):
        rows = [row for row in read_csv(path) if row.get("variant") == VARIANT]
        if len(rows) != 1 or not rows[0].get("MSE") or not rows[0].get("MAE"):
            raise SystemExit(f"missing final MESANet row in {path}")
        candidate[key_from_path(path)] = rows[0]

    public: dict[tuple[str, str], dict[str, str]] = {}
    for path in summary_paths(PUBLIC_ROOT):
        rows = [
            row
            for row in read_csv(path)
            if row.get("family") == "sota_public"
            and row.get("model") == COMPARATOR
            and str(row.get("model_id", "")).endswith("_r5401")
            and row.get("MSE")
            and row.get("MAE")
        ]
        if len(rows) != 1:
            raise SystemExit(f"expected one frozen {COMPARATOR} row in {path}")
        public[key_from_path(path)] = rows[0]

    missing = sorted(set(public) - set(candidate))
    extra = sorted(set(candidate) - set(public))
    if missing or extra:
        raise SystemExit(f"strict-LOSO coverage mismatch: missing={missing}, extra={extra}")

    detail: list[dict[str, object]] = []
    for dataset, subject in sorted(candidate):
        mesa, base = candidate[(dataset, subject)], public[(dataset, subject)]
        mesa_mse, mesa_mae = float(mesa["MSE"]), float(mesa["MAE"])
        base_mse, base_mae = float(base["MSE"]), float(base["MAE"])
        detail.append(
            {
                "dataset": dataset,
                "test_subject": subject,
                "comparator": COMPARATOR,
                "MESANet_MSE": mesa_mse,
                "public_MSE": base_mse,
                "log_MSE_ratio": math.log(mesa_mse / base_mse),
                "MESANet_MAE": mesa_mae,
                "public_MAE": base_mae,
                "log_MAE_ratio": math.log(mesa_mae / base_mae),
                "MSE_win": mesa_mse < base_mse,
                "MAE_win": mesa_mae < base_mae,
            }
        )

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in detail:
        grouped[str(row["dataset"])].append(row)
        grouped["ALL"].append(row)

    aggregate: list[dict[str, object]] = []
    for index, (dataset, rows) in enumerate(sorted(grouped.items())):
        mse_log = [float(row["log_MSE_ratio"]) for row in rows]
        mae_log = [float(row["log_MAE_ratio"]) for row in rows]
        mse_lo, mse_hi = bootstrap_interval(mse_log, 20270717 + index)
        mae_lo, mae_hi = bootstrap_interval(mae_log, 20270817 + index)
        aggregate.append(
            {
                "dataset": dataset,
                "subjects": len(rows),
                "comparator": COMPARATOR,
                "geometric_MSE_gain_percent": 100.0 * (1.0 - math.exp(statistics.mean(mse_log))),
                "MSE_log_ratio_CI_low": mse_lo,
                "MSE_log_ratio_CI_high": mse_hi,
                "geometric_MAE_gain_percent": 100.0 * (1.0 - math.exp(statistics.mean(mae_log))),
                "MAE_log_ratio_CI_low": mae_lo,
                "MAE_log_ratio_CI_high": mae_hi,
                "MSE_wins": sum(bool(row["MSE_win"]) for row in rows),
                "MAE_wins": sum(bool(row["MAE_win"]) for row in rows),
                "dual_wins": sum(bool(row["MSE_win"]) and bool(row["MAE_win"]) for row in rows),
            }
        )

    output = RESULT_ROOT / "_summary"
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("subject_detail.csv", detail), ("subject_aggregate.csv", aggregate)):
        with (output / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (output / "analysis_scope.json").write_text(
        json.dumps(
            {
                "statistical_unit": "held-out released subject",
                "optimization_runs_per_subject": 1,
                "comparator": COMPARATOR,
                "comparator_selection": "predeclared from the validation-only main benchmark",
                "paired_optimization_run": 5401,
                "protocol": "eight-epoch MSE-only strict-LOSO stress test",
                "claim_scope": "supplementary subject-transfer evidence, not the main repeated benchmark",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(output / "subject_aggregate.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
