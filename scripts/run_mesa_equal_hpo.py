#!/usr/bin/env python3
"""Validation-only equal-budget HPO for MESANet and public IMTS backbones.

Every model-dataset pair receives the same registered six-trial budget. Search
runs evaluate the validation split only. The lowest validation MSE (MAE breaks
ties) is frozen before the selected configuration is evaluated on test runs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_release_v437_magnet_quick import DATASETS, csv_list
from scripts.run_release_v507_mesa_complete_benchmark import replace_many
from scripts.run_icdm2026_local_complete import baseline_extra


SUPPORTED_MODELS = ("APN", "KAFNet", "GraFITi", "tPatchGNN", "PatchTST", "MESANet")


@dataclass(frozen=True)
class Trial:
    name: str
    lr_mult: float = 1.0
    width_mult: float = 1.0
    dropout: float | None = None


TRIALS = (
    Trial("lr_half", lr_mult=0.5),
    Trial("default"),
    Trial("lr_double", lr_mult=2.0),
    Trial("width_half", width_mult=0.5),
    Trial("width_3half", width_mult=1.5),
    Trial("dropout_0p1", dropout=0.1),
)


def rounded_width(value: int, multiplier: float, heads: int) -> int:
    unit = max(int(heads), 1)
    target = max(unit, int(round(value * multiplier)))
    return max(unit, int(round(target / unit)) * unit)


def model_id(prefix: str, phase: str, dataset: str, model: str, tag: str, seed: int) -> str:
    safe_model = model.lower().replace("-", "")
    return f"{prefix}_{phase}_{dataset}_{safe_model}_{tag}_r{seed}"


def last_arg(cmd: list[str], key: str, default: str) -> str:
    value = default
    for index, part in enumerate(cmd[:-1]):
        if part == key:
            value = cmd[index + 1]
    return value


def command(
    args: argparse.Namespace,
    dataset: str,
    model: str,
    trial: Trial,
    seed: int,
    phase: str,
    result_root: Path,
) -> tuple[list[str], str]:
    spec = DATASETS[dataset]
    mid = model_id(args.prefix, phase, dataset, model, trial.name, seed)
    cmd = [
        sys.executable,
        "main.py",
        "--gpu_id", str(args.gpu_id),
        "--use_gpu", "1",
        "--use_multi_gpu", "0",
        "--is_training", "1",
        "--model_id", mid,
        "--model_name", model,
        "--dataset_root_path", spec.root,
        "--dataset_name", dataset,
        "--features", "M",
        "--seq_len", str(spec.seq_len),
        "--pred_len", str(spec.pred_len),
        "--enc_in", str(spec.enc_in),
        "--dec_in", str(spec.enc_in),
        "--c_out", str(spec.enc_in),
        "--train_epochs", str(args.search_epochs if phase == "search" else args.confirm_epochs),
        "--patience", str(args.patience),
        "--val_interval", "1",
        "--itr", "1",
        "--seed_base", str(seed),
        "--batch_size", str(spec.batch_size),
        "--learning_rate", str(spec.lr),
        "--d_model", str(spec.d_model),
        "--d_ff", str(spec.d_ff),
        "--dropout", str(spec.dropout),
        "--patch_len", str(spec.patch_len),
        "--n_heads", str(spec.n_heads),
        "--loss", "MSEMAE",
        "--mae_weight", str(args.mae_weight),
        "--num_workers", str(args.num_workers),
        "--pin_memory", "1" if args.num_workers else "0",
        "--persistent_workers", "1" if args.num_workers else "0",
        "--prefetch_factor", "2" if args.num_workers else "1",
        "--non_blocking_transfer", "1",
        "--tf32", "1",
        "--cudnn_benchmark", "1",
        "--amp_dtype", "bf16",
        "--disable_tqdm", "1",
        "--wearable_mask_protocol", args.wearable_mask_protocol,
        "--wearable_split_protocol", args.wearable_split_protocol,
        "--test_split", "val" if phase == "search" else "test",
        "--save_arrays", "0",
        "--checkpoints", str(result_root),
    ]
    if model != "MESANet":
        try:
            cmd.extend(baseline_extra(model, dataset))
        except KeyError:
            # Some regular-grid baselines use the common configuration only.
            pass
    elif args.mesa_ablation:
        cmd.extend(["--ablation_name", args.mesa_ablation])
    # Model-specific public defaults are loaded first. The registered HPO
    # perturbation and common objective are then applied last, so no baseline
    # helper can silently override the equal-budget search dimensions.
    base_heads = int(last_arg(cmd, "--n_heads", str(spec.n_heads)))
    base_d_model = int(last_arg(cmd, "--d_model", str(spec.d_model)))
    base_d_ff = int(last_arg(cmd, "--d_ff", str(spec.d_ff)))
    base_lr = float(last_arg(cmd, "--learning_rate", str(spec.lr)))
    base_dropout = float(last_arg(cmd, "--dropout", str(spec.dropout)))
    d_model = rounded_width(base_d_model, trial.width_mult, base_heads)
    d_ff = max(d_model, int(round(base_d_ff * trial.width_mult)))
    dropout = base_dropout if trial.dropout is None else trial.dropout
    cmd = replace_many(
        cmd,
        {
            "--learning_rate": str(base_lr * trial.lr_mult),
            "--d_model": str(d_model),
            "--d_ff": str(d_ff),
            "--dropout": str(dropout),
            "--loss": "MSEMAE",
            "--mae_weight": str(args.mae_weight),
        },
    )
    if model == "tPatchGNN" and dataset in {
        "MHEALTH", "PAMAP2", "OPPORTUNITY", "USCHAD", "REALDISP", "WESAD", "PPGDALIA"
    }:
        cmd = replace_many(cmd, {"--batch_size": "4"})
    if model == "PatchTST" and dataset in {"USCHAD", "REALDISP"}:
        cmd = replace_many(cmd, {"--batch_size": "4"})
    if model == "GraFITi" and dataset in {"OPPORTUNITY", "PAMAP2", "REALDISP", "WESAD", "PPGDALIA"}:
        cmd = replace_many(cmd, {"--batch_size": "4"})
    return cmd, mid


def metric_path(result_root: Path, model: str, mid: str, split: str) -> Path | None:
    candidates = [
        path
        for path in result_root.rglob("metric.json")
        if mid in path.parts and model in path.parts
    ]
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def read_metric(result_root: Path, model: str, mid: str, split: str) -> dict[str, float] | None:
    path = metric_path(result_root, model, mid, split)
    if path is None:
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {key: float(raw[key]) for key in ("MSE", "MAE") if key in raw}


def run_jobs(jobs: list[tuple[list[str], Path]], parallel: int, poll_seconds: int) -> list[int]:
    queue = list(jobs)
    running: list[tuple[subprocess.Popen, object, Path]] = []
    codes: list[int] = []
    while queue or running:
        while queue and len(running) < max(1, parallel):
            cmd, log_path = queue.pop(0)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handle = log_path.open("w", encoding="utf-8")
            handle.write("COMMAND " + subprocess.list2cmdline(cmd) + "\n")
            handle.flush()
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
            running.append((proc, handle, log_path))
            print(f"started pid={proc.pid} log={log_path}", flush=True)
        time.sleep(max(1, poll_seconds))
        alive: list[tuple[subprocess.Popen, object, Path]] = []
        for proc, handle, log_path in running:
            code = proc.poll()
            if code is None:
                alive.append((proc, handle, log_path))
                continue
            handle.close()
            codes.append(int(code))
            print(f"finished pid={proc.pid} code={code} log={log_path}", flush=True)
        running = alive
    return codes


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def select_trials(
    args: argparse.Namespace,
    result_root: Path,
    datasets: list[str],
    models: list[str],
) -> tuple[list[dict[str, object]], dict[tuple[str, str], Trial]]:
    rows: list[dict[str, object]] = []
    selected: dict[tuple[str, str], Trial] = {}
    for dataset in datasets:
        for model in models:
            candidates = []
            for trial in TRIALS:
                mid = model_id(args.prefix, "search", dataset, model, trial.name, args.search_seed)
                metric = read_metric(result_root, model, mid, "val")
                row: dict[str, object] = {
                    "dataset": dataset,
                    "model": model,
                    "trial": trial.name,
                    "learning_rate_multiplier": trial.lr_mult,
                    "width_multiplier": trial.width_mult,
                    "dropout": "dataset_default" if trial.dropout is None else trial.dropout,
                    "validation_MSE": "" if metric is None else metric.get("MSE", ""),
                    "validation_MAE": "" if metric is None else metric.get("MAE", ""),
                    "selected": False,
                }
                rows.append(row)
                if metric and math.isfinite(metric.get("MSE", math.nan)) and math.isfinite(metric.get("MAE", math.nan)):
                    candidates.append((metric["MSE"], metric["MAE"], trial.name, trial, row))
            if not candidates:
                raise RuntimeError(f"no valid validation result for {dataset}/{model}")
            _, _, _, winner, winner_row = min(candidates)
            winner_row["selected"] = True
            selected[(dataset, model)] = winner
    return rows, selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="HumanActivity,MHEALTH,OPPORTUNITY,PAMAP2")
    parser.add_argument("--models", default=",".join(SUPPORTED_MODELS))
    parser.add_argument("--search_seed", type=int, default=5399)
    parser.add_argument("--confirm_seeds", default="5401,5402,5403")
    parser.add_argument("--search_epochs", type=int, default=20)
    parser.add_argument("--confirm_epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--parallel", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--mae_weight", type=float, default=0.5)
    parser.add_argument("--mesa_ablation", default="")
    parser.add_argument("--wearable_mask_protocol", default="real_missing_only_if_available")
    parser.add_argument("--wearable_split_protocol", default="subject_heldout")
    parser.add_argument("--prefix", default="mesa_equal_hpo")
    parser.add_argument("--root", default="storage/results_mesa_equal_hpo")
    parser.add_argument("--phase", choices=("search", "confirm", "all", "summary"), default="all")
    parser.add_argument("--poll_seconds", type=int, default=5)
    parser.add_argument("--skip_done", type=int, default=1)
    args = parser.parse_args()

    datasets = csv_list(args.datasets)
    models = csv_list(args.models)
    unknown_models = sorted(set(models) - set(SUPPORTED_MODELS))
    unknown_datasets = sorted(set(datasets) - set(DATASETS))
    if unknown_models or unknown_datasets:
        raise ValueError(f"unsupported models={unknown_models} datasets={unknown_datasets}")

    result_root = (ROOT / args.root).resolve()
    summary_root = result_root / "_summary"
    summary_root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {"arguments": vars(args), "trials": [asdict(trial) for trial in TRIALS], "commands": []}

    if args.phase in {"search", "all"}:
        jobs = []
        for dataset in datasets:
            for model in models:
                for trial in TRIALS:
                    cmd, mid = command(args, dataset, model, trial, args.search_seed, "search", result_root)
                    manifest["commands"].append({
                        "phase": "search", "dataset": dataset, "model": model, "trial": trial.name,
                        "sha256": hashlib.sha256("\0".join(cmd).encode()).hexdigest(), "command": cmd,
                    })
                    if not args.skip_done or read_metric(result_root, model, mid, "val") is None:
                        jobs.append((cmd, result_root / "_logs" / f"search_{dataset}_{model}_{trial.name}.log"))
        codes = run_jobs(jobs, args.parallel, args.poll_seconds)
        if any(code != 0 for code in codes):
            print("warning: one or more search jobs failed; selection requires one valid result per pair", flush=True)

    search_rows, selected = select_trials(args, result_root, datasets, models)
    write_csv(summary_root / "equal_hpo_validation_trials.csv", search_rows)
    frozen = {
        dataset: {model: asdict(selected[(dataset, model)]) for model in models}
        for dataset in datasets
    }
    (summary_root / "equal_hpo_frozen_configs.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")

    if args.phase in {"confirm", "all"}:
        jobs = []
        for seed in (int(value) for value in csv_list(args.confirm_seeds)):
            for dataset in datasets:
                for model in models:
                    trial = selected[(dataset, model)]
                    cmd, mid = command(args, dataset, model, trial, seed, "confirm", result_root)
                    manifest["commands"].append({
                        "phase": "confirm", "dataset": dataset, "model": model, "trial": trial.name,
                        "seed": seed, "sha256": hashlib.sha256("\0".join(cmd).encode()).hexdigest(), "command": cmd,
                    })
                    if not args.skip_done or read_metric(result_root, model, mid, "test") is None:
                        jobs.append((cmd, result_root / "_logs" / f"confirm_{dataset}_{model}_r{seed}.log"))
        codes = run_jobs(jobs, args.parallel, args.poll_seconds)
        if any(code != 0 for code in codes):
            print("warning: one or more confirmation jobs failed", flush=True)

    confirm_rows: list[dict[str, object]] = []
    for seed in (int(value) for value in csv_list(args.confirm_seeds)):
        for dataset in datasets:
            for model in models:
                trial = selected[(dataset, model)]
                mid = model_id(args.prefix, "confirm", dataset, model, trial.name, seed)
                metric = read_metric(result_root, model, mid, "test")
                confirm_rows.append({
                    "dataset": dataset,
                    "model": model,
                    "run": seed,
                    "selected_trial": trial.name,
                    "MSE": "" if metric is None else metric.get("MSE", ""),
                    "MAE": "" if metric is None else metric.get("MAE", ""),
                    "model_id": mid,
                })
    write_csv(summary_root / "equal_hpo_test_runs.csv", confirm_rows)
    (summary_root / "command_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    missing = sum(not row["MSE"] or not row["MAE"] for row in confirm_rows)
    print(f"search_rows={len(search_rows)} confirmation_rows={len(confirm_rows)} missing_confirmation={missing}")
    return 0 if missing == 0 or args.phase in {"search", "summary"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
