#!/usr/bin/env python3
"""Summarize frozen equal-HPO test runs and MESANet paired gains."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="storage/results_mesa_equal_hpo")
    args = parser.parse_args()
    root = Path(args.root)
    rows = [
        row for row in read_csv(root / "_summary/equal_hpo_test_runs.csv")
        if row.get("MSE") and row.get("MAE")
    ]
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["model"])].append(row)

    summary: list[dict[str, object]] = []
    for (dataset, model), group in sorted(grouped.items()):
        mse_mean, mse_std = mean_std([float(row["MSE"]) for row in group])
        mae_mean, mae_std = mean_std([float(row["MAE"]) for row in group])
        summary.append({
            "dataset": dataset,
            "model": model,
            "runs": len(group),
            "MSE_mean": mse_mean,
            "MSE_std": mse_std,
            "MAE_mean": mae_mean,
            "MAE_std": mae_std,
            "selected_trial": group[0]["selected_trial"],
        })
    write_csv(root / "_summary/equal_hpo_test_summary.csv", summary)

    paired: list[dict[str, object]] = []
    datasets = sorted({row["dataset"] for row in rows})
    for dataset in datasets:
        ds = [row for row in rows if row["dataset"] == dataset]
        mesa = {int(row["run"]): row for row in ds if row["model"] == "MESANet"}
        public_models = sorted({row["model"] for row in ds if row["model"] != "MESANet"})
        public_means = {
            model: statistics.mean(float(row["MSE"]) for row in ds if row["model"] == model)
            for model in public_models
        }
        if not mesa or not public_means:
            continue
        frontier = min(public_means, key=public_means.get)
        public = {int(row["run"]): row for row in ds if row["model"] == frontier}
        common = sorted(set(mesa) & set(public))
        mse_gains = [100.0 * (float(public[s]["MSE"]) - float(mesa[s]["MSE"])) / float(public[s]["MSE"]) for s in common]
        mae_gains = [100.0 * (float(public[s]["MAE"]) - float(mesa[s]["MAE"])) / float(public[s]["MAE"]) for s in common]
        paired.append({
            "dataset": dataset,
            "public_frontier_by_MSE": frontier,
            "paired_runs": len(common),
            "MSE_gain_pct_mean": statistics.mean(mse_gains) if mse_gains else math.nan,
            "MSE_gain_pct_std": statistics.stdev(mse_gains) if len(mse_gains) > 1 else 0.0,
            "MAE_gain_pct_mean": statistics.mean(mae_gains) if mae_gains else math.nan,
            "MAE_gain_pct_std": statistics.stdev(mae_gains) if len(mae_gains) > 1 else 0.0,
            "dual_wins": sum(mse > 0 and mae > 0 for mse, mae in zip(mse_gains, mae_gains)),
        })
    write_csv(root / "_summary/equal_hpo_paired_gains.csv", paired)
    print(f"complete_rows={len(rows)} model_summaries={len(summary)} paired_datasets={len(paired)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
