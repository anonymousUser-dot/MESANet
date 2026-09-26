#!/usr/bin/env python3
"""Validation-only sensitivity of final MESANet method parameters."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_mesa_equal_hpo import Trial, command, model_id, read_metric, run_jobs


DATASETS = ("HumanActivity", "MHEALTH", "OPPORTUNITY", "PAMAP2")
RHO_VALUES = (0.0, 0.25, 0.5, 0.75, 1.0)
SCALE_VALUES = (1, 2, 3, 4, 6)


def tag(value: float | int) -> str:
    return str(value).replace(".", "p")


def append_override(cmd: list[str], key: str, value: str) -> None:
    if key in cmd:
        cmd[cmd.index(key) + 1] = value
    else:
        cmd.extend([key, value])


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--search_seed", type=int, default=5399)
    parser.add_argument("--search_epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--parallel", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--mae_weight", type=float, default=0.5)
    parser.add_argument("--wearable_mask_protocol", default="real_missing_only_if_available")
    parser.add_argument("--wearable_split_protocol", default="subject_heldout")
    # Short defaults keep Windows checkpoint paths below MAX_PATH.
    parser.add_argument("--prefix", default="v675")
    parser.add_argument("--root", default="storage/v675")
    parser.add_argument("--poll_seconds", type=int, default=5)
    parser.add_argument("--skip_done", type=int, default=1)
    parser.add_argument(
        "--mesa_ablation",
        default="mesa_final",
    )
    args = parser.parse_args()
    args.confirm_epochs = args.search_epochs

    result_root = (ROOT / args.root).resolve()
    summary_root = result_root / "_summary"
    summary_root.mkdir(parents=True, exist_ok=True)
    jobs: list[tuple[list[str], Path]] = []
    specs: list[dict[str, object]] = []
    manifest: dict[str, object] = {
        "arguments": vars(args),
        "purpose": "final-model method-parameter sensitivity on validation only",
        "rho_values": list(RHO_VALUES),
        "scale_values": list(SCALE_VALUES),
        "commands": [],
    }

    for dataset in DATASETS:
        for rho in RHO_VALUES:
            trial = Trial(f"rho_{tag(rho)}")
            cmd, mid = command(args, dataset, "MESANet", trial, args.search_seed, "search", result_root)
            append_override(cmd, "--mesa_dual_rho", f"{rho:g}")
            append_override(cmd, "--cag_num_scales", "4")
            specs.append({"dataset": dataset, "sweep": "rho", "value": rho, "trial": trial, "mid": mid})
            manifest["commands"].append({
                "dataset": dataset, "sweep": "rho", "value": rho,
                "model_id": mid, "sha256": hashlib.sha256("\0".join(cmd).encode()).hexdigest(),
                "command": cmd,
            })
            if not args.skip_done or read_metric(result_root, "MESANet", mid, "val") is None:
                jobs.append((cmd, result_root / "_logs" / f"{dataset}_rho_{tag(rho)}.log"))

        for scales in SCALE_VALUES:
            trial = Trial(f"scales_{scales}")
            cmd, mid = command(args, dataset, "MESANet", trial, args.search_seed, "search", result_root)
            append_override(cmd, "--mesa_dual_rho", "0.5")
            append_override(cmd, "--cag_num_scales", str(scales))
            specs.append({"dataset": dataset, "sweep": "scale_count", "value": scales, "trial": trial, "mid": mid})
            manifest["commands"].append({
                "dataset": dataset, "sweep": "scale_count", "value": scales,
                "model_id": mid, "sha256": hashlib.sha256("\0".join(cmd).encode()).hexdigest(),
                "command": cmd,
            })
            if not args.skip_done or read_metric(result_root, "MESANet", mid, "val") is None:
                jobs.append((cmd, result_root / "_logs" / f"{dataset}_scales_{scales}.log"))

    codes = run_jobs(jobs, args.parallel, args.poll_seconds)
    rows: list[dict[str, object]] = []
    for spec in specs:
        metric = read_metric(result_root, "MESANet", str(spec["mid"]), "val")
        rows.append({
            "dataset": spec["dataset"],
            "model": "MESANet",
            "sweep": spec["sweep"],
            "value": spec["value"],
            "validation_MSE": "" if metric is None else metric.get("MSE", ""),
            "validation_MAE": "" if metric is None else metric.get("MAE", ""),
            "default": (spec["sweep"] == "rho" and spec["value"] == 0.5)
            or (spec["sweep"] == "scale_count" and spec["value"] == 4),
        })

    write_csv(summary_root / "method_sensitivity.csv", rows)
    (summary_root / "command_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    missing = sum(row["validation_MSE"] == "" or row["validation_MAE"] == "" for row in rows)
    print(f"jobs={len(jobs)} rows={len(rows)} missing={missing}", flush=True)
    return 1 if any(code != 0 for code in codes) or missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
