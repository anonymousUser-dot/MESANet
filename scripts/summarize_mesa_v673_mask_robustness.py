#!/usr/bin/env python3
"""Summarize the validation-frozen alternate-mask stress test."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "storage/results_mesa_v673_mask_robustness/_summary/mask_robustness_runs.csv"
OUTPUT = ROOT / "storage/results_mesa_v673_mask_robustness/_summary/mask_robustness_comparison.csv"


def main() -> int:
    with INPUT.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault((row["protocol"], row["dataset"]), []).append(row)

    output: list[dict[str, object]] = []
    for (protocol, dataset), group in sorted(groups.items()):
        mesa = next(row for row in group if row["model"] == "MESANet")
        public = next(row for row in group if row["model"] != "MESANet")
        if not mesa["MSE"] or not public["MSE"]:
            raise SystemExit(f"incomplete cell: {protocol}/{dataset}")
        mesa_mse, mesa_mae = float(mesa["MSE"]), float(mesa["MAE"])
        public_mse, public_mae = float(public["MSE"]), float(public["MAE"])
        output.append(
            {
                "protocol": protocol,
                "dataset": dataset,
                "frozen_public_model": public["model"],
                "public_MSE": public_mse,
                "MESANet_MSE": mesa_mse,
                "MSE_gain_percent": 100.0 * (public_mse - mesa_mse) / public_mse,
                "public_MAE": public_mae,
                "MESANet_MAE": mesa_mae,
                "MAE_gain_percent": 100.0 * (public_mae - mesa_mae) / public_mae,
            }
        )
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
