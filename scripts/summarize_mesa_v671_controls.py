#!/usr/bin/env python3
"""Summarize paired parameter-matched controls for the frozen MESANet."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORE = ("HumanActivity", "MHEALTH", "OPPORTUNITY", "PAMAP2")
CONTROLS = ("shared_transfer_matched", "random_basis")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="storage/results_mesa_v671_final_revision/_summary/final_revision_runs.csv",
    )
    parser.add_argument(
        "--output",
        default="storage/results_mesa_v671_final_revision/_summary",
    )
    args = parser.parse_args()

    with (ROOT / args.input).open(newline="", encoding="utf-8-sig") as handle:
        raw = list(csv.DictReader(handle))
    rows = [
        {
            **row,
            "run": int(row["run"]),
            "MSE": float(row["MSE"]),
            "MAE": float(row["MAE"]),
        }
        for row in raw
        if row.get("MSE") and row.get("MAE")
    ]
    groups: dict[tuple[str, str], dict[int, dict[str, object]]] = defaultdict(dict)
    for row in rows:
        groups[(str(row["dataset"]), str(row["variant"]))][int(row["run"])] = row

    detail: list[dict[str, object]] = []
    aggregate: list[dict[str, object]] = []
    decisions: dict[str, str] = {}
    for control in CONTROLS:
        mse_logs: list[float] = []
        mae_logs: list[float] = []
        mse_dataset_wins = 0
        mae_dataset_wins = 0
        dual_dataset_wins = 0
        for dataset in CORE:
            full = groups[(dataset, "fixed_rho_half")]
            alternative = groups[(dataset, control)]
            expected = {5401, 5402, 5403}
            if set(full) & expected != expected or set(alternative) & expected != expected:
                raise SystemExit(f"incomplete paired control: {dataset}/{control}")
            mse = [math.log(float(alternative[r]["MSE"]) / float(full[r]["MSE"])) for r in sorted(expected)]
            mae = [math.log(float(alternative[r]["MAE"]) / float(full[r]["MAE"])) for r in sorted(expected)]
            mse_mean, mae_mean = statistics.mean(mse), statistics.mean(mae)
            mse_win, mae_win = mse_mean > 0, mae_mean > 0
            mse_dataset_wins += int(mse_win)
            mae_dataset_wins += int(mae_win)
            dual_dataset_wins += int(mse_win and mae_win)
            mse_logs.extend(mse)
            mae_logs.extend(mae)
            detail.append(
                {
                    "control": control,
                    "dataset": dataset,
                    "runs": 3,
                    "control_degradation_MSE_percent": 100.0 * (math.exp(mse_mean) - 1.0),
                    "control_degradation_MAE_percent": 100.0 * (math.exp(mae_mean) - 1.0),
                    "full_MSE_win": mse_win,
                    "full_MAE_win": mae_win,
                }
            )
        mse_mean, mae_mean = statistics.mean(mse_logs), statistics.mean(mae_logs)
        promote = (
            mse_mean > 0
            and mae_mean > 0
            and mse_dataset_wins >= 3
            and mae_dataset_wins >= 3
        )
        decisions[control] = "SUPPORTED" if promote else "NOT_SUPPORTED"
        aggregate.append(
            {
                "control": control,
                "paired_cells": len(mse_logs),
                "control_geometric_MSE_degradation_percent": 100.0 * (math.exp(mse_mean) - 1.0),
                "control_geometric_MAE_degradation_percent": 100.0 * (math.exp(mae_mean) - 1.0),
                "full_MSE_dataset_wins": mse_dataset_wins,
                "full_MAE_dataset_wins": mae_dataset_wins,
                "full_dual_dataset_wins": dual_dataset_wins,
                "decision": decisions[control],
            }
        )

    output = (ROOT / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    for name, payload in (("control_detail.csv", detail), ("control_aggregate.csv", aggregate)):
        with (output / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(payload[0]))
            writer.writeheader()
            writer.writerows(payload)
    (output / "control_decision.json").write_text(
        json.dumps(
            {
                "registered_rule": (
                    "positive paired geometric MSE and MAE gains plus at least "
                    "3/4 dataset wins for each metric"
                ),
                "decisions": decisions,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(decisions, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
