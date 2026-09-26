#!/usr/bin/env python3
"""Repeat only the metric-wise public frontier under frozen equal-HPO trials."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_mesa_equal_hpo import Trial, command, read_metric, run_jobs


FRONTIER_PAIRS = (
    ("HumanActivity", "KAFNet"),
    ("MHEALTH", "PatchTST"),
    ("OPPORTUNITY", "tPatchGNN"),
    ("OPPORTUNITY", "PatchTST"),
    ("PAMAP2", "GraFITi"),
    ("USCHAD", "PatchTST"),
    ("REALDISP", "GraFITi"),
    ("REALDISP", "PatchTST"),
)

DATASET_CODES = {
    "HumanActivity": "ha", "MHEALTH": "mh", "OPPORTUNITY": "op",
    "PAMAP2": "pa", "USCHAD": "uc", "REALDISP": "rd",
}
MODEL_CODES = {
    "KAFNet": "kf", "PatchTST": "pt", "tPatchGNN": "tg", "GraFITi": "gf",
}


def csv_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def load_frozen() -> dict[str, dict[str, dict[str, object]]]:
    paths = (
        "storage/results_mesa_equal_hpo/_summary/equal_hpo_frozen_configs.json",
        "storage/results_mesa_v664_uschad_public_equal_hpo/_summary/equal_hpo_frozen_configs.json",
        "storage/results_mesa_v664_realdisp_public_equal_hpo/_summary/equal_hpo_frozen_configs.json",
    )
    frozen: dict[str, dict[str, dict[str, object]]] = {}
    for raw in paths:
        frozen.update(json.loads((ROOT / raw).read_text(encoding="utf-8-sig")))
    return frozen


def load_existing_core() -> dict[tuple[str, str, int], dict[str, str]]:
    path = ROOT / "storage/results_mesa_equal_hpo/_summary/equal_hpo_test_runs.csv"
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    return {
        (row["dataset"], row["model"], int(row["run"])): row
        for row in rows
        if row.get("MSE") and row.get("MAE")
    }


def make_trial(payload: dict[str, object]) -> Trial:
    dropout = payload.get("dropout")
    return Trial(
        str(payload["name"]),
        float(payload.get("lr_mult", 1.0)),
        float(payload.get("width_mult", 1.0)),
        None if dropout is None else float(dropout),
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="5402,5403,5404,5405")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--parallel", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--mae_weight", type=float, default=0.5)
    parser.add_argument("--root", default="D:/p")
    parser.add_argument("--prefix", default="p8")
    parser.add_argument("--poll_seconds", type=int, default=5)
    parser.add_argument("--skip_done", type=int, default=1)
    parser.add_argument("--summary_only", type=int, default=0)
    parser.add_argument(
        "--reuse_core_runs",
        default="5402,5403",
        help="Reuse exact frozen equal-HPO core rows instead of recomputing them.",
    )
    parser.add_argument(
        "--exclude_pairs",
        default="",
        help="Comma-separated DATASET:MODEL pairs that require a larger GPU.",
    )
    args = parser.parse_args()

    seeds = [int(value) for value in csv_list(args.seeds)]
    frozen = load_frozen()
    existing_core = load_existing_core()
    reuse_core_runs = {int(value) for value in csv_list(args.reuse_core_runs)}
    excluded = {
        tuple(value.split(":", 1)) for value in csv_list(args.exclude_pairs)
    }
    result_root = Path(args.root).resolve()
    jobs: list[tuple[list[str], Path]] = []
    records: list[tuple[str, str, int, Trial, str]] = []
    manifest: list[dict[str, object]] = []

    for seed in seeds:
        for dataset, model in FRONTIER_PAIRS:
            if (dataset, model) in excluded:
                continue
            trial = make_trial(frozen[dataset][model])
            run_args = Namespace(
                gpu_id=args.gpu_id,
                search_epochs=args.epochs,
                confirm_epochs=args.epochs,
                patience=args.patience,
                mae_weight=args.mae_weight,
                num_workers=args.num_workers,
                wearable_mask_protocol="real_missing_only_if_available",
                wearable_split_protocol="subject_heldout",
                mesa_ablation="",
                prefix=args.prefix,
            )
            cmd, model_id = command(
                run_args, dataset, model, trial, seed, "confirm", result_root
            )
            model_id = (
                f"{args.prefix}_{MODEL_CODES[model]}_{DATASET_CODES[dataset]}_r{seed}"
            )
            cmd[cmd.index("--model_id") + 1] = model_id
            log = result_root / "_logs" / f"{dataset}_{model}_r{seed}.log"
            metric = read_metric(result_root, model, model_id, "test")
            reusable = (dataset, model, seed) in existing_core and seed in reuse_core_runs
            if not args.summary_only and not (args.skip_done and metric is not None) and not reusable:
                jobs.append((cmd, log))
            records.append((dataset, model, seed, trial, model_id))
            manifest.append({
                "dataset": dataset, "model": model, "run": seed,
                "selected_trial": trial.name, "command": cmd,
            })

    codes = [] if args.summary_only else run_jobs(jobs, args.parallel, args.poll_seconds)
    rows: list[dict[str, object]] = []
    for dataset, model, seed, trial, model_id in records:
        metric = read_metric(result_root, model, model_id, "test")
        source = "current_root"
        if metric is None and (dataset, model, seed) in existing_core and seed in reuse_core_runs:
            prior = existing_core[(dataset, model, seed)]
            metric = {"MSE": prior["MSE"], "MAE": prior["MAE"]}
            source = "frozen_equal_hpo_core"
        rows.append({
            "dataset": dataset, "model": model, "run": seed,
            "selected_trial": trial.name,
            "MSE": "" if metric is None else metric.get("MSE", ""),
            "MAE": "" if metric is None else metric.get("MAE", ""),
            "model_id": model_id, "source": source,
        })

    summary = result_root / "_summary"
    write_csv(summary / "frontier_repeated_runs.csv", rows)
    (summary / "command_manifest.json").write_text(
        json.dumps({"arguments": vars(args), "jobs": manifest}, indent=2),
        encoding="utf-8",
    )
    missing = sum(not row["MSE"] or not row["MAE"] for row in rows)
    print(f"jobs={len(jobs)} rows={len(rows)} missing={missing}", flush=True)
    return 1 if any(code != 0 for code in codes) or missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
