#!/usr/bin/env python3
"""Run the frozen final MESANet on the registered strict-LOSO folds.

This is a subject-level stress protocol, not an additional optimization-seed
benchmark.  It uses the same eight-epoch MSE-only protocol as the retained
public strict-LOSO matrix so those public rows can be reused without retraining.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import subprocess
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_release_v507_mesa_complete_benchmark.py"
VARIANT = "mesanet_fixed_dual_coordinate"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dependencies.WearableActivity.WearableActivity import wearable_subject_id


def csv_values(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def subject_number(subject: str) -> int:
    digits = "".join(ch for ch in subject if ch.isdigit())
    return int(digits) if digits else 0


def available_subjects(dataset: str) -> list[str]:
    path = ROOT / "storage/datasets" / dataset / "processed/data_trnorm_subject_groupasync.pt"
    records = torch.load(path, map_location="cpu", weights_only=False)
    subjects = {wearable_subject_id(record[0]) for record in records}
    return sorted(subjects, key=subject_number)


def run_fold(args: argparse.Namespace, dataset: str, subject: str) -> None:
    output = Path(args.root) / dataset / subject
    command = [
        sys.executable,
        str(RUNNER),
        "--datasets", dataset,
        "--tasks", "ablation",
        "--ablation_variants", VARIANT,
        "--run_seeds", str(args.run_id),
        "--epochs", str(args.epochs),
        "--patience", str(args.patience),
        "--parallel", "1",
        "--num_workers", str(args.num_workers),
        "--loss", "MSE",
        "--mesa_mae_weight", "0",
        "--wearable_mask_protocol", "group_async_current",
        "--wearable_split_protocol", "subject_heldout",
        "--wearable_test_subject", subject,
        "--prefix", "mesa_v672_final_loso",
        "--root", output.as_posix(),
        "--cleanup_checkpoints", "1",
        "--continue_on_error", "0",
        "--skip_done", "1",
    ]
    print("RUN", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="MHEALTH,USCHAD")
    parser.add_argument("--run_id", type=int, default=5401)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--fold_parallel", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--root", default="storage/results_mesa_v672_final_loso")
    args = parser.parse_args()

    folds = [
        (dataset, subject)
        for dataset in csv_values(args.datasets)
        for subject in available_subjects(dataset)
    ]
    with ThreadPoolExecutor(max_workers=max(1, args.fold_parallel)) as pool:
        futures = [pool.submit(run_fold, args, dataset, subject) for dataset, subject in folds]
        for future in futures:
            future.result()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
