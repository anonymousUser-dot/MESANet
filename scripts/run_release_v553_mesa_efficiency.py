#!/usr/bin/env python
"""MESANet efficiency benchmark and Figure 4 generator.

The benchmark follows the common style used by recent IMTS papers: one dataset,
one batch size, one GPU, and four metrics where lower is better:
peak GPU memory, parameter count, training-step time, and inference-step time.

The script is intentionally single-run and does not add multi-seed experiments.
It reuses the repository's existing profiling hooks in ``Exp_Main.test``.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.run_release_v507_mesa_complete_benchmark import (  # noqa: E402
    public_command,
    mesa_command,
    replace_arg,
)


KINDS = ("gpu_memory", "flop", "train_time", "inference_time")


@dataclass(frozen=True)
class EffRecord:
    dataset: str
    model: str
    kind: str
    model_id: str
    log_name: str


def csv_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def add_or_replace(cmd: list[str], key: str, value: str) -> list[str]:
    # Public helpers may append a second copy of an argument. argparse keeps
    # the last copy, so replacing only the first one can silently break the
    # common profiling protocol. Remove every copy before appending one value.
    out: list[str] = []
    index = 0
    while index < len(cmd):
        if cmd[index] == key:
            index += 2
            continue
        out.append(cmd[index])
        index += 1
    out.extend([key, value])
    return out


def efficiency_command(args: argparse.Namespace, root: Path, dataset: str, model: str, kind: str) -> tuple[list[str], str]:
    if model == "MESANet":
        cmd, mid = mesa_command(args, root, dataset, args.mesa_variant, "efficiency")
        if args.mesa_width_multiplier != 1.0:
            for key in ("--d_model", "--d_ff"):
                if key in cmd:
                    index = cmd.index(key) + 1
                    cmd[index] = str(max(1, int(round(int(cmd[index]) * args.mesa_width_multiplier))))
    else:
        cmd, mid = public_command(args, root, dataset, model)

    mid = f"{args.prefix}_{dataset}_{model.lower()}_{kind}_b{args.batch_size}_r{args.run_seed}"
    cmd = replace_arg(cmd, "--model_id", mid)
    cmd = add_or_replace(cmd, "--is_training", "0")
    cmd = add_or_replace(cmd, "--batch_size", str(args.batch_size))
    cmd = add_or_replace(cmd, "--train_epochs", "1")
    cmd = add_or_replace(cmd, "--patience", "1")
    cmd = add_or_replace(cmd, "--num_workers", str(args.num_workers))
    cmd = add_or_replace(cmd, "--pin_memory", "1" if args.num_workers > 0 else "0")
    cmd = add_or_replace(cmd, "--persistent_workers", "1" if args.num_workers > 0 else "0")
    cmd = add_or_replace(cmd, "--prefetch_factor", "2" if args.num_workers > 0 else "1")
    cmd = add_or_replace(cmd, "--disable_tqdm", "1")

    if kind == "gpu_memory":
        cmd.extend(["--test_gpu_memory", "1"])
    elif kind == "flop":
        cmd.extend(["--test_flop", "1"])
    elif kind == "train_time":
        cmd.extend(["--test_train_time", "1"])
    elif kind == "inference_time":
        cmd.extend(["--test_inference_time", "1"])
    else:
        raise ValueError(f"unknown efficiency kind: {kind}")
    return cmd, mid


def run_jobs(jobs: list[tuple[list[str], Path]], parallel: int, poll_seconds: int) -> list[int]:
    queue = list(jobs)
    running: list[tuple[subprocess.Popen, object, Path]] = []
    codes: list[int] = []
    while queue or running:
        while queue and len(running) < max(1, parallel):
            cmd, log_path = queue.pop(0)
            print("$", " ".join(shlex.quote(str(p)) for p in cmd), flush=True)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handle = log_path.open("w", encoding="utf-8", errors="replace")
            env = os.environ.copy()
            env.setdefault("PYTHONUNBUFFERED", "1")
            env.setdefault("TQDM_DISABLE", "1")
            proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=handle, stderr=subprocess.STDOUT, text=True)
            running.append((proc, handle, log_path))
        keep: list[tuple[subprocess.Popen, object, Path]] = []
        for proc, handle, log_path in running:
            code = proc.poll()
            if code is None:
                keep.append((proc, handle, log_path))
            else:
                handle.close()
                print(f"exit={code} log={log_path}", flush=True)
                codes.append(int(code))
        running = keep
        if running:
            time.sleep(poll_seconds)
    return codes


def parse_log(log_path: Path) -> dict[str, float | str]:
    if not log_path.exists():
        return {}
    text = log_path.read_text(encoding="utf-8", errors="replace")
    out: dict[str, float | str] = {}
    patterns = {
        "peak_gpu_gb": r"Peak GPU memory usage: ([0-9.]+) GB",
        "params_m": r"Number of parameters \(M\).*?([0-9.]+)M",
        "total_params": r"Total parameters\s+([0-9]+)",
        "trainable_params": r"Trainable parameters\s+([0-9]+)",
        "frozen_params": r"Frozen parameters\s+([0-9]+)",
        "train_step_ms": r"Average training step time .*?: ([0-9.]+) ms",
        "inference_step_ms": r"Average inference step time .*?: ([0-9.]+) ms",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            out[key] = float(match.group(1))
    macs = re.search(r"Computational complexity \(MACs\).*?([0-9.]+)\s*([GMK])Mac", text)
    if macs:
        value = float(macs.group(1))
        unit = macs.group(2)
        out["macs_g"] = value if unit == "G" else value / (1000.0 if unit == "M" else 1_000_000.0)
    if "Traceback" in text or "RuntimeError" in text or "ERROR" in text:
        out["log_status"] = "check"
    elif out:
        out["log_status"] = "ok"
    return out


def summarize(root: Path, records: list[EffRecord], summary_name: str) -> Path:
    by_model: dict[str, dict[str, object]] = {}
    for rec in records:
        row = by_model.setdefault(rec.model, {"dataset": rec.dataset, "model": rec.model})
        parsed = parse_log(root / "_logs" / rec.log_name)
        row.update(parsed)
        if parsed.get("log_status") == "check":
            row["log_status"] = "check"
        elif parsed:
            row.setdefault("log_status", "ok")
    out = root / "_summary" / summary_name
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset",
        "model",
        "peak_gpu_gb",
        "params_m",
        "total_params",
        "trainable_params",
        "frozen_params",
        "train_step_ms",
        "inference_step_ms",
        "macs_g",
        "log_status",
    ]
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(by_model.values())
    print(out, flush=True)
    print(out.read_text(encoding="utf-8"), flush=True)
    return out


def plot(summary_csv: Path, out_prefix: Path, dataset: str, batch_size: int) -> None:
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.read_csv(summary_csv)
    metrics = [
        ("peak_gpu_gb", "(a) Peak GPU memory (GB)"),
        ("params_m", "(b) Parameters (M)"),
        ("train_step_ms", "(c) Training time / step (ms)"),
        ("inference_step_ms", "(d) Inference time / step (ms)"),
    ]
    colors = {
        "APN": "#4C78A8",
        "KAFNet": "#72B7B2",
        "tPatchGNN": "#F58518",
        "GraFITi": "#B279A2",
        "PatchTST": "#E45756",
        "MESANet": "#54A24B",
    }
    fig, axes = plt.subplots(1, 4, figsize=(10.8, 2.45), constrained_layout=True)
    for ax, (key, title) in zip(axes, metrics):
        plot_df = df.copy()
        x = list(range(len(plot_df)))
        values = plot_df[key].fillna(0.0)
        bars = ax.bar(x, values, color=[colors.get(m, "#8C8C8C") for m in plot_df["model"]], width=0.72)
        ymax = max(float(values.max()), 1e-6)
        for bar, missing, value in zip(bars, plot_df[key].isna(), values):
            if missing:
                bar.set_facecolor("white")
                bar.set_edgecolor("#555555")
                bar.set_hatch("//")
                ax.text(bar.get_x() + bar.get_width() / 2, 0.04 * ymax, "OOM", ha="center", va="bottom", fontsize=6, rotation=90)
            else:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    float(value),
                    f"{float(value):.2g}",
                    ha="center",
                    va="bottom",
                    fontsize=5.5,
                    rotation=90,
                )
        if key in {"train_step_ms", "inference_step_ms"}:
            ax.set_yscale("log")
        ax.margins(y=0.18)
        ax.set_title(title, fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(plot_df["model"], rotation=35, ha="right", fontsize=7)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(axis="y", linewidth=0.4, alpha=0.35)
        ax.set_axisbelow(True)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    column_fig, column_axes = plt.subplots(2, 2, figsize=(3.35, 5.55), constrained_layout=True)
    for ax, (key, title) in zip(column_axes.flat, metrics):
        plot_df = df.copy()
        x = list(range(len(plot_df)))
        values = plot_df[key].fillna(0.0)
        bars = ax.bar(
            x,
            values,
            color=[colors.get(m, "#8C8C8C") for m in plot_df["model"]],
            width=0.72,
        )
        ymax = max(float(values.max()), 1e-6)
        for bar, missing, value in zip(bars, plot_df[key].isna(), values):
            if missing:
                bar.set_facecolor("white")
                bar.set_edgecolor("#555555")
                bar.set_hatch("//")
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    0.04 * ymax,
                    "OOM",
                    ha="center",
                    va="bottom",
                    fontsize=5.5,
                    rotation=90,
                )
            else:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    float(value),
                    f"{float(value):.2g}",
                    ha="center",
                    va="bottom",
                    fontsize=5,
                    rotation=90,
                )
        if key in {"train_step_ms", "inference_step_ms"}:
            ax.set_yscale("log")
        ax.margins(y=0.22)
        ax.set_title(title, fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(plot_df["model"], rotation=38, ha="right", fontsize=5.5)
        ax.tick_params(axis="y", labelsize=6)
        ax.grid(axis="y", linewidth=0.4, alpha=0.35)
        ax.set_axisbelow(True)
    column_prefix = out_prefix.with_name(out_prefix.name + "_column")
    column_fig.savefig(column_prefix.with_suffix(".pdf"), bbox_inches="tight")
    column_fig.savefig(column_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(column_fig)

    wide_grid_fig, wide_grid_axes = plt.subplots(
        2, 2, figsize=(10.8, 8.4), constrained_layout=True
    )
    for ax, (key, title) in zip(wide_grid_axes.flat, metrics):
        plot_df = df.copy()
        x = list(range(len(plot_df)))
        values = plot_df[key].fillna(0.0)
        bars = ax.bar(
            x,
            values,
            color=[colors.get(m, "#8C8C8C") for m in plot_df["model"]],
            width=0.68,
        )
        ymax = max(float(values.max()), 1e-6)
        for bar, missing, value in zip(bars, plot_df[key].isna(), values):
            if missing:
                bar.set_facecolor("white")
                bar.set_edgecolor("#555555")
                bar.set_hatch("//")
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    0.04 * ymax,
                    "OOM",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=90,
                )
            else:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    float(value),
                    f"{float(value):.2g}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=90,
                )
        if key in {"train_step_ms", "inference_step_ms"}:
            ax.set_yscale("log")
        ax.margins(y=0.22)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(plot_df["model"], rotation=25, ha="right", fontsize=9)
        ax.tick_params(axis="y", labelsize=9)
        ax.grid(axis="y", linewidth=0.5, alpha=0.35)
        ax.set_axisbelow(True)
    wide_grid_prefix = out_prefix.with_name(out_prefix.name + "_wide_grid")
    wide_grid_fig.savefig(wide_grid_prefix.with_suffix(".pdf"), bbox_inches="tight")
    wide_grid_fig.savefig(
        wide_grid_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight"
    )
    plt.close(wide_grid_fig)
    print(out_prefix.with_suffix(".pdf"), flush=True)
    print(out_prefix.with_suffix(".png"), flush=True)
    print(column_prefix.with_suffix(".pdf"), flush=True)
    print(column_prefix.with_suffix(".png"), flush=True)
    print(wide_grid_prefix.with_suffix(".pdf"), flush=True)
    print(wide_grid_prefix.with_suffix(".png"), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="MHEALTH")
    parser.add_argument("--models", default="APN,KAFNet,tPatchGNN,GraFITi,MESANet")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--run_seed", type=int, default=5301)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--mesa_mae_weight", type=float, default=0.5)
    parser.add_argument("--mesa_model_name", default="MESANet")
    parser.add_argument(
        "--mesa_variant",
        default="mesa",
        help="Registered MESANet variant to profile.",
    )
    parser.add_argument(
        "--mesa_width_multiplier",
        type=float,
        default=1.0,
        help="Width multiplier for the selected MESANet configuration.",
    )
    parser.add_argument("--loss", default="MSEMAE")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--save_arrays", action="store_true")
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--poll_seconds", type=int, default=10)
    parser.add_argument("--prefix", default="v553")
    parser.add_argument("--root", default="storage/results_release_v553_mesa_efficiency")
    parser.add_argument("--summary_name", default="mesa_efficiency_summary.csv")
    parser.add_argument("--figure_name", default="mesa_fig4_efficiency_audit")
    parser.add_argument(
        "--figure_dir",
        default="paper/mesanet_paper/figures/final",
        help="Package-relative output directory for generated paper figures.",
    )
    parser.add_argument("--summary_only", action="store_true")
    parser.add_argument("--no_plot", action="store_true")
    parser.add_argument("--wearable_mask_protocol", default="group_async_current")
    parser.add_argument("--wearable_split_protocol", default="subject_heldout")
    parser.add_argument("--wearable_test_subject", default="")
    parser.add_argument("--wearable_subject_fold", type=int, default=0)
    args = parser.parse_args()

    root = (REPO / args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    records: list[EffRecord] = []
    jobs: list[tuple[list[str], Path]] = []
    for model in csv_list(args.models):
        for kind in KINDS:
            cmd, mid = efficiency_command(args, root, args.dataset, model, kind)
            log_name = f"eff_{args.dataset}_{model}_{kind}_b{args.batch_size}.log"
            records.append(EffRecord(args.dataset, model, kind, mid, log_name))
            log_path = root / "_logs" / log_name
            if not args.summary_only and not parse_log(log_path):
                jobs.append((cmd, log_path))

    codes = [] if args.summary_only else run_jobs(jobs, args.parallel, args.poll_seconds)
    summary = summarize(root, records, args.summary_name)
    if not args.no_plot:
        figure_dir = (REPO / args.figure_dir).resolve()
        if not figure_dir.exists():
            candidates = [
                path / "figures" / "final"
                for path in (REPO / "paper").iterdir()
                if (path / "figures" / "final").is_dir()
            ]
            if len(candidates) != 1:
                raise FileNotFoundError(
                    f"Cannot resolve figure directory {figure_dir}; pass --figure_dir explicitly."
                )
            figure_dir = candidates[0]
        plot(summary, figure_dir / args.figure_name, args.dataset, args.batch_size)
    return 1 if any(code != 0 for code in codes) else 0


if __name__ == "__main__":
    raise SystemExit(main())
