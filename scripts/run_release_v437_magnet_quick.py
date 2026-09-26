#!/usr/bin/env python
"""v437 quick validation for MAGNet.

MAGNet is a mechanism-adaptive graph backbone: mechanism state controls
continuous-time patch scale, variable graph edges, and decoder state before the
forecast is made. The run compares MAGNet to the strongest recent
mechanism-coordinate baselines under the same single-seed protocol.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DatasetSpec:
    root: str
    seq_len: int
    pred_len: int
    enc_in: int
    d_model: int
    lr: float
    batch_size: int
    dropout: float
    patch_len: int
    n_heads: int
    d_ff: int


DATASETS = {
    "HumanActivity": DatasetSpec("storage/datasets/HumanActivity", 3000, 300, 12, 32, 1e-3, 8, 0.0, 60, 4, 128),
    "MHEALTH": DatasetSpec("storage/datasets/MHEALTH", 250, 50, 23, 48, 1e-3, 32, 0.05, 10, 4, 192),
    "OPPORTUNITY": DatasetSpec("storage/datasets/OPPORTUNITY", 300, 60, 64, 64, 1e-3, 16, 0.05, 12, 4, 256),
    "PAMAP2": DatasetSpec("storage/datasets/PAMAP2", 500, 100, 37, 64, 1e-3, 16, 0.05, 20, 4, 256),
    "USCHAD": DatasetSpec("storage/datasets/USCHAD", 300, 60, 6, 32, 1e-3, 32, 0.05, 15, 2, 128),
    "REALDISP": DatasetSpec("storage/datasets/REALDISP", 500, 100, 117, 96, 1e-3, 8, 0.05, 25, 4, 384),
    # Wrist BVP defines a 64 Hz union clock; masks retain native 64/32/4/4 Hz
    # BVP/ACC/EDA/TEMP arrivals. Windows cover 30 s history and 5 s horizon.
    "WESAD": DatasetSpec("storage/datasets/WESAD", 1920, 320, 6, 32, 1e-3, 8, 0.05, 64, 4, 128),
    "PPGDALIA": DatasetSpec("storage/datasets/PPGDALIA", 1920, 320, 6, 32, 1e-3, 8, 0.05, 64, 4, 128),
}


VARIANTS = {
    "mrps": ("AMPGNet", "", "fixed mechanism-state patch coordinate"),
    "ra_fixed_prior": ("RACPGNet", "fixed_prior_no_prob", "coordinate route initialized toward MRPS"),
    "ra_ct_prior": ("RACPGNet", "ct_prior_no_prob", "coordinate route initialized toward CTPG"),
    "magnet": ("MAGNet", "no_prob", "mechanism-adaptive scale, edge, and decoder state"),
    "magnet_scale_only": ("MAGNet", "no_state_edge_no_mech_token_no_decoder_state_no_prob", "mechanism-adaptive patch scale only"),
    "magnet_no_scale": ("MAGNet", "no_state_scale_no_prob", "no mechanism control over patch scale"),
    "magnet_no_edge": ("MAGNet", "no_state_edge_no_prob", "no mechanism control over variable graph edges"),
    "magnet_no_token": ("MAGNet", "no_mech_token_no_prob", "no mechanism token in patch representation"),
}


PUBLIC_BEST = {
    "HumanActivity": {"model": "KAFNet", "MSE": 0.0516, "MAE": 0.1283},
    "MHEALTH": {"model": "tPatchGNN", "MSE": 0.6941, "MAE": 0.4377},
    "OPPORTUNITY": {"model": "tPatchGNN", "MSE": 0.7150, "MAE": 0.5679},
    "PAMAP2": {"model": "tPatchGNN", "MSE": 0.763208, "MAE": 0.476635},
}


def csv_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def latest_metric(root: Path, dataset: str, model: str, mid: str) -> dict[str, float] | None:
    candidates = [root / dataset / model / mid]
    candidates.extend(sorted(root.glob(f"*/{dataset}/{model}/{mid}")))
    paths = []
    for base in candidates:
        paths.extend(base.glob("*/*/iter*/eval_*/metric.json"))
    paths = sorted(set(paths))
    if not paths:
        return None
    raw = json.loads(paths[-1].read_text(encoding="utf-8"))
    return {k: float(v) for k, v in raw.items() if isinstance(v, (int, float))}


def command_for(args: argparse.Namespace, root: Path, dataset: str, variant: str) -> tuple[list[str], str, str]:
    spec = DATASETS[dataset]
    model, ablation, _ = VARIANTS[variant]
    mid = f"{args.prefix}_{dataset}_{variant}_e{args.epochs}_s{args.seed}"
    cmd = [
        sys.executable, "main.py",
        "--gpu_id", str(args.gpu_id), "--use_gpu", "1", "--use_multi_gpu", "0", "--is_training", "1",
        "--model_id", mid, "--model_name", model,
        "--dataset_root_path", spec.root, "--dataset_name", dataset, "--features", "M",
        "--seq_len", str(spec.seq_len), "--pred_len", str(spec.pred_len),
        "--enc_in", str(spec.enc_in), "--dec_in", str(spec.enc_in), "--c_out", str(spec.enc_in),
        "--train_epochs", str(args.epochs), "--patience", str(args.patience), "--val_interval", "1",
        "--itr", "1", "--seed_base", str(args.seed), "--batch_size", str(spec.batch_size),
        "--learning_rate", str(spec.lr), "--d_model", str(spec.d_model), "--d_ff", str(spec.d_ff),
        "--dropout", str(spec.dropout), "--patch_len", str(spec.patch_len), "--n_heads", str(spec.n_heads),
        "--loss", args.loss, "--num_workers", str(args.num_workers),
        "--pin_memory", "1" if args.num_workers > 0 else "0",
        "--persistent_workers", "1" if args.num_workers > 0 else "0",
        "--prefetch_factor", "2" if args.num_workers > 0 else "1",
        "--non_blocking_transfer", "1", "--tf32", "1", "--cudnn_benchmark", "1",
        "--amp_dtype", "bf16", "--disable_tqdm", "1", "--checkpoints", str(root),
        "--ablation_name", ablation,
    ]
    return cmd, model, mid


def run_parallel(jobs: list[tuple[list[str], Path]], parallel: int, poll_seconds: int) -> list[int]:
    queue = list(jobs)
    running: list[tuple[subprocess.Popen, object, Path]] = []
    codes: list[int] = []
    while queue or running:
        while queue and len(running) < max(1, parallel):
            cmd, log = queue.pop(0)
            print("$", " ".join(shlex.quote(str(p)) for p in cmd), flush=True)
            log.parent.mkdir(parents=True, exist_ok=True)
            fh = log.open("w", encoding="utf-8", errors="replace")
            env = os.environ.copy()
            env.setdefault("PYTHONUNBUFFERED", "1")
            env.setdefault("TQDM_DISABLE", "1")
            proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=fh, stderr=subprocess.STDOUT, text=True)
            running.append((proc, fh, log))
        keep = []
        for proc, fh, log in running:
            code = proc.poll()
            if code is None:
                keep.append((proc, fh, log))
            else:
                fh.close()
                print(f"exit={code} log={log}", flush=True)
                codes.append(int(code))
        running = keep
        if running:
            time.sleep(poll_seconds)
    return codes


def summarize(root: Path, records: list[dict[str, str]]) -> Path:
    rows: list[dict[str, object]] = []
    metrics_by_dataset: dict[str, dict[str, dict[str, float]]] = {}
    for rec in records:
        metric = latest_metric(root, rec["dataset"], rec["model"], rec["model_id"])
        row: dict[str, object] = dict(rec)
        row["role"] = VARIANTS[rec["variant"]][2]
        if metric:
            row.update(metric)
            metrics_by_dataset.setdefault(rec["dataset"], {})[rec["variant"]] = metric
            best = PUBLIC_BEST.get(rec["dataset"], {})
            if best:
                row["public_best"] = best["model"]
                row["public_MSE_gain_pct"] = (best["MSE"] - metric["MSE"]) / best["MSE"] * 100.0
                row["public_MAE_gain_pct"] = (best["MAE"] - metric["MAE"]) / best["MAE"] * 100.0
        rows.append(row)
    for row in rows:
        dataset = str(row["dataset"])
        mrps = metrics_by_dataset.get(dataset, {}).get("mrps")
        ra = metrics_by_dataset.get(dataset, {}).get("ra_fixed_prior") or metrics_by_dataset.get(dataset, {}).get("ra_ct_prior")
        if mrps and "MSE" in row:
            row["vs_mrps_MSE_gain_pct"] = (mrps["MSE"] - float(row["MSE"])) / mrps["MSE"] * 100.0
            row["vs_mrps_MAE_gain_pct"] = (mrps["MAE"] - float(row["MAE"])) / mrps["MAE"] * 100.0
        if ra and "MSE" in row:
            row["vs_ra_MSE_gain_pct"] = (ra["MSE"] - float(row["MSE"])) / ra["MSE"] * 100.0
            row["vs_ra_MAE_gain_pct"] = (ra["MAE"] - float(row["MAE"])) / ra["MAE"] * 100.0
    out_dir = root / "_summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "v437_magnet_quick_summary.csv"
    fields = [
        "dataset", "variant", "model", "model_id", "role", "MSE", "MAE", "RMSE",
        "public_best", "public_MSE_gain_pct", "public_MAE_gain_pct",
        "vs_mrps_MSE_gain_pct", "vs_mrps_MAE_gain_pct", "vs_ra_MSE_gain_pct", "vs_ra_MAE_gain_pct",
    ]
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(out.read_text(encoding="utf-8"), flush=True)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="MHEALTH,OPPORTUNITY")
    parser.add_argument("--variants", default="mrps,ra_fixed_prior,ra_ct_prior,magnet,magnet_scale_only,magnet_no_scale,magnet_no_edge,magnet_no_token")
    parser.add_argument("--seed", type=int, default=5301)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--parallel", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--loss", default="MSEMAE")
    parser.add_argument("--prefix", default="v437")
    parser.add_argument("--root", default="storage/results_release_v437_magnet_quick")
    parser.add_argument("--poll_seconds", type=int, default=10)
    parser.add_argument("--skip_done", type=int, default=1)
    parser.add_argument("--summary_only", type=int, default=0)
    args = parser.parse_args()
    root = (REPO / args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, str]] = []
    jobs: list[tuple[list[str], Path]] = []
    for dataset in csv_list(args.datasets):
        if dataset not in DATASETS:
            raise ValueError(f"unknown dataset: {dataset}")
        for variant in csv_list(args.variants):
            if variant not in VARIANTS:
                raise ValueError(f"unknown variant: {variant}")
            cmd, model, mid = command_for(args, root, dataset, variant)
            records.append({"dataset": dataset, "variant": variant, "model": model, "model_id": mid})
            if args.summary_only or (args.skip_done and latest_metric(root, dataset, model, mid) is not None):
                print(f"skip done: {mid}", flush=True)
            else:
                jobs.append((cmd, root / "_logs" / f"{dataset}_{variant}.log"))
    codes = [] if args.summary_only else run_parallel(jobs, args.parallel, args.poll_seconds)
    summarize(root, records)
    return 1 if any(code != 0 for code in codes) else 0


if __name__ == "__main__":
    raise SystemExit(main())
