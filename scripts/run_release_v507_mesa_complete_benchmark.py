#!/usr/bin/env python
"""v507 fixed-MESA complete benchmark runner.

This script is the reset experiment for the conference backbone direction.  It uses
one fixed end-to-end MESANet model across all datasets and compares it against
the reproduced public IMTS baselines under the same protocol.  No dataset-level
admission-head selection is used for the main model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.run_release_v437_magnet_quick import DATASETS, csv_list
from scripts.run_icdm2026_local_complete import baseline_extra


PUBLIC_MODELS = [
    "APN",
    "KAFNet",
    "GraFITi",
    "tPatchGNN",
    "DLinear",
    "CausalInterpDLinear",
    "PatchTST",
    "mTAN",
    "CRU",
    "SeFT",
    "Raindrop",
]

MESA_ABLATIONS: dict[str, tuple[str, str]] = {
    "mesa": ("", "fixed affine-equivariant horizon-scale simplex backbone"),
    "dct_dual_soft_orthogonal_no_causal": (
        "aa_no_q_graph_no_prob_waveform_token_parallel_dct_dual_coordinate_"
        "dct_dual_soft_orthogonal_dct_dual_no_causal",
        "affine-anchored dual-coordinate backbone with learned soft DCT separation",
    ),
    "fixed_patch": ("fixed_only_no_prob", "fixed patch only"),
    "random_admit": ("random_admit_no_prob", "random admission control"),
    "constant_admit": ("mean_admit_no_prob", "constant mean admission control"),
    "fixed_patch_stateblind": ("fixed_only_no_q_graph_no_prob", "fixed patch only with state-blind graph transfer"),
    "random_admit_stateblind": ("random_admit_no_q_graph_no_prob", "random admission with state-blind graph transfer"),
    "constant_admit_stateblind": ("mean_admit_no_q_graph_no_prob", "constant admission with state-blind graph transfer"),
    "fixed_patch_nograph": ("fixed_only_no_graph_no_stconv_no_prob", "fixed patch only under graph-free core"),
    "random_admit_nograph": ("random_admit_no_graph_no_stconv_no_prob", "random admission under graph-free core"),
    "constant_admit_nograph": ("mean_admit_no_graph_no_stconv_no_prob", "constant admission under graph-free core"),
    "no_obs_state_nograph": ("no_obs_state_no_graph_no_stconv_no_prob", "remove observation-support cues under graph-free core"),
    "shuffle_state_nograph": ("shuffle_state_no_graph_no_stconv_no_prob", "shuffle observation-support cues under graph-free core"),
    "value_only_admit_nograph": ("value_only_admit_no_graph_no_stconv_no_prob", "learned admission without observation-support cues under graph-free core"),
    "value_only_scale_nograph": ("value_only_state_no_graph_no_stconv_no_prob", "scale-window selection without observation-support cues under graph-free core"),
    "no_state_scale_nograph": ("no_state_scale_no_graph_no_stconv_no_prob", "remove state-conditioned scale-window bias under graph-free core"),
    "no_state_scale_admit_nograph": (
        "no_state_scale_value_only_admit_no_graph_no_stconv_no_prob",
        "remove observation-support cues from both scale-window selection and admission under graph-free core",
    ),
    "fixed_patch_wide_nograph": (
        "fixed_only_no_graph_no_stconv_no_prob",
        "fixed patch only under graph-free core with doubled hidden capacity",
    ),
    "ms_concat_param_nograph": (
        "ms_concat_param_no_graph_no_stconv_no_prob",
        "same-parameter ungated multiscale concatenation under graph-free core",
    ),
    "scale_only_wide_nograph": (
        "scale_only_no_graph_no_stconv_no_prob",
        "always-open multiresolution scale under graph-free core with doubled hidden capacity",
    ),
    "value_only_admit_wide_nograph": (
        "value_only_admit_no_graph_no_stconv_no_prob",
        "learned admission without observation-support cues under graph-free core with doubled hidden capacity",
    ),
    "value_only_state": ("value_only_state_no_prob", "remove mechanism-state scale"),
    "scale_only_state": ("scale_only_no_prob", "scale-only mechanism state"),
    "no_patch_graph": ("no_graph_no_stconv_no_prob", "remove temporal patch graph"),
    "no_q_graph": ("no_q_graph_no_prob", "remove mechanism-state cues from graph transfer"),
    "graph_state_transfer": ("no_prob", "allow graph-transfer gate to read mechanism-state cues"),
    "no_obs_state": ("no_obs_state_no_prob", "remove observation-support cues from scale and admission"),
    "shuffle_state": ("shuffle_state_no_prob", "shuffle observation-support cues across samples"),
    "post_graph_admit": ("post_graph_admit_no_prob", "admit multiscale coordinate after graph transfer"),
    "no_obs_state_stateblind": ("no_obs_state_no_q_graph_no_prob", "remove observation-support cues under support-blind graph transfer"),
    "shuffle_state_stateblind": ("shuffle_state_no_q_graph_no_prob", "shuffle observation-support cues under support-blind graph transfer"),
    "post_graph_admit_stateblind": ("post_graph_admit_no_q_graph_no_prob", "post-graph admission under state-blind graph transfer"),
    "ms_concat_param": ("ms_concat_param_no_q_graph_no_prob", "same-parameter ungated multiscale concatenation"),
    "always_open_gate": ("always_open_gate_no_q_graph_no_prob", "always-open admission gate"),
    "global_trainable_gate": ("global_trainable_gate_no_q_graph_no_prob", "global trainable admission gate without patch-specific mechanism input"),
    "affine_anchor": (
        "aa_no_q_graph_no_prob",
        "masked affine-equivariant value anchor with the released scale-admission path",
    ),
    "affine_anchor_random_admission": (
        "aa_random_admit_no_q_graph_no_prob",
        "affine anchor with random patch admission and matched scale capacity",
    ),
    "affine_anchor_constant_admission": (
        "aa_mean_admit_no_q_graph_no_prob",
        "affine anchor with constant half-open patch admission",
    ),
    "affine_anchor_multiscale_concat": (
        "aa_ms_concat_param_no_q_graph_no_prob",
        "affine anchor with parameter-matched ungated multiscale concatenation",
    ),
    "affine_anchor_fixed_wide": (
        "aa_fixed_only_no_q_graph_no_prob",
        "affine-anchor fixed coordinate with matched feed-forward capacity",
    ),
    "affine_anchor_always_open": (
        "aa_always_open_gate_no_q_graph_no_prob",
        "affine anchor with an always-open multiresolution gate",
    ),
    "affine_anchor_global_admission": (
        "aa_global_trainable_gate_no_q_graph_no_prob",
        "affine anchor with one global trainable multiresolution gate",
    ),
    "affine_anchor_post_temporal_admit": (
        "aa_post_temporal_admit_no_q_graph_no_prob",
        "affine-anchor admission after temporal attention and before variable transfer",
    ),
    "affine_anchor_post_graph_admit": (
        "aa_post_graph_admit_no_q_graph_no_prob",
        "affine-anchor admission after temporal and variable dependency aggregation",
    ),
    "affine_anchor_decoder_level_admit": (
        "aa_decoder_level_admit_no_q_graph_no_prob",
        "affine-anchor admission at the shared direct-decoder input",
    ),
    "affine_anchor_fixed_support_param_matched": (
        "aa_fixed_support_param_matched_no_q_graph_no_prob",
        "parameter-identical control whose scale bank receives the fixed support statistic and center",
    ),
    "no_anchor_full": (
        "no_q_graph_no_prob",
        "dataset-normalized full multiresolution backbone without a sample-variable affine anchor",
    ),
    "standard_revin_full": (
        "standard_revin_no_q_graph_no_prob",
        "full multiresolution backbone with unmasked time-axis RevIN",
    ),
    "masked_revin_full": (
        "masked_revin_no_q_graph_no_prob",
        "full multiresolution backbone with learnable mask-aware RevIN",
    ),
    "masked_anchor_full": (
        "aa_no_q_graph_no_prob",
        "full multiresolution backbone with masked affine anchoring",
    ),
    "masked_anchor_subject_balanced": (
        "aa_no_q_graph_no_prob_subject_balanced",
        "affine-anchor backbone with subject-balanced forecasting risk",
    ),
    "masked_anchor_subject_groupdro": (
        "aa_no_q_graph_no_prob_subject_groupdro",
        "affine-anchor backbone with balanced soft worst-subject risk",
    ),
    "masked_anchor_subject_context": (
        "aa_no_q_graph_no_prob_subject_context",
        "affine-anchor backbone with history-conditioned context adaptation",
    ),
    "masked_anchor_subject_dg_context": (
        "aa_no_q_graph_no_prob_subject_groupdro_subject_context",
        "affine-anchor backbone with subject-robust risk and history-conditioned adaptation",
    ),
    "masked_anchor_pose_canonical": (
        "aa_no_q_graph_no_prob_pose_canonical",
        "affine-anchor backbone in a history-derived wearable sensor frame",
    ),
    "masked_anchor_pose_reliable": (
        "aa_no_q_graph_no_prob_pose_reliable",
        "affine-anchor backbone with reliability-gated wearable-frame canonicalization",
    ),
    "masked_anchor_ordered_patch_only": (
        "aa_no_q_graph_no_prob_ordered_patch_only",
        "affine-anchor backbone with an ordered patch trajectory decoder",
    ),
    "masked_anchor_ordered_patch_fusion": (
        "aa_no_q_graph_no_prob_ordered_patch_fusion",
        "affine-anchor backbone with closed pooled/ordered horizon fusion",
    ),
    "masked_anchor_horizon_patch_query": (
        "aa_no_q_graph_no_prob_horizon_patch_query",
        "affine-anchor backbone with future-query retrieval over ordered patch memory",
    ),
    "masked_anchor_waveform_token": (
        "aa_no_q_graph_no_prob_waveform_token",
        "affine-anchor backbone with a closed within-patch waveform coordinate",
    ),
    "masked_anchor_waveform_ordered_patch_fusion": (
        "aa_no_q_graph_no_prob_waveform_token_ordered_patch_fusion",
        "affine-anchor backbone retaining both within-patch waveform and ordered patch trajectory",
    ),
    "mesanet_fixed_dual_coordinate": (
        "aa_no_q_graph_no_prob_waveform_token_parallel_dct_dual_coordinate_"
        "dct_dual_rho_half_dct_dual_no_causal",
        "frozen affine-anchored dual-coordinate MESANet with fixed half-projection separation",
    ),
    "masked_anchor_lowrank_trajectory_only": (
        "aa_no_q_graph_no_prob_lowrank_trajectory_only",
        "affine-anchor backbone decoded from a shared low-rank ordered trajectory",
    ),
    "masked_anchor_lowrank_trajectory_fusion": (
        "aa_no_q_graph_no_prob_lowrank_trajectory_fusion",
        "affine-anchor backbone with closed pooled/low-rank trajectory fusion",
    ),
    "masked_anchor_spectral_trajectory": (
        "aa_no_q_graph_no_prob_spectral_trajectory",
        "affine-anchor backbone with reliability-adaptive spectral trajectory retrieval",
    ),
    "masked_anchor_spectral_global": (
        "aa_no_q_graph_no_prob_spectral_trajectory_spectral_global",
        "spectral trajectory backbone with a global per-frequency gate",
    ),
    "masked_anchor_spectral_support_only": (
        "aa_no_q_graph_no_prob_spectral_trajectory_spectral_support_only",
        "spectral trajectory backbone gated only by observation-support state",
    ),
    "no_anchor_fixed": (
        "fixed_only_no_q_graph_no_prob",
        "fixed-coordinate backbone without a sample-variable affine anchor",
    ),
    "standard_revin_fixed": (
        "standard_revin_fixed_only_no_q_graph_no_prob",
        "fixed-coordinate backbone with unmasked time-axis RevIN",
    ),
    "masked_revin_fixed": (
        "masked_revin_fixed_only_no_q_graph_no_prob",
        "fixed-coordinate backbone with learnable mask-aware RevIN",
    ),
    "masked_anchor_fixed": (
        "aa_fixed_only_no_q_graph_no_prob",
        "fixed-coordinate backbone with masked affine anchoring",
    ),
    "masked_anchor_scale": (
        "aa_scale_only_no_q_graph_no_prob",
        "always-open multiresolution coordinate with masked affine anchoring",
    ),
    "masked_anchor_no_restore": (
        "aa_no_anchor_restore_no_q_graph_no_prob",
        "masked normalization without restoring the sample-variable affine coordinate",
    ),
    "horizon_simplex": (
        "hs_no_q_graph_no_prob",
        "anchor-inclusive future-query-conditioned scale simplex",
    ),
    "affine_horizon_simplex": (
        "aa_hs_no_q_graph_no_prob",
        "affine-equivariant anchor plus future-query-conditioned scale simplex",
    ),
    "affine_horizon_simplex_no_price": (
        "aa_hs_no_reliability_price_no_q_graph_no_prob",
        "full backbone without the support-density reliability price",
    ),
}


@dataclass(frozen=True)
class JobRecord:
    family: str
    dataset: str
    model: str
    variant: str
    model_id: str
    role: str
    metric_model: str | None = None


def latest_metric(root: Path, dataset: str, model: str, mid: str) -> dict[str, float] | None:
    bases = [root / dataset / model / mid]
    bases.extend(sorted(root.glob(f"*/{dataset}/{model}/{mid}")))
    paths = []
    for base in bases:
        paths.extend(base.glob("*/*/iter*/eval_*/metric.json"))
    paths = sorted(set(paths), key=lambda p: p.stat().st_mtime)
    if not paths:
        return None
    raw = json.loads(paths[-1].read_text(encoding="utf-8"))
    return {k: float(v) for k, v in raw.items() if isinstance(v, (int, float))}


def replace_arg(cmd: list[str], key: str, value: str) -> list[str]:
    out: list[str] = []
    inserted = False
    i = 0
    while i < len(cmd):
        if cmd[i] == key and i + 1 < len(cmd):
            if not inserted:
                out.extend([key, value])
                inserted = True
            i += 2
            continue
        out.append(cmd[i])
        i += 1
    if not inserted:
        out.extend([key, value])
    return out


def replace_many(cmd: list[str], pairs: dict[str, str]) -> list[str]:
    out = list(cmd)
    for key, value in pairs.items():
        out = replace_arg(out, key, value)
    return out


def common_command(
    args: argparse.Namespace,
    root: Path,
    dataset: str,
    model: str,
    model_id: str,
    *,
    mae_weight: float,
    ablation: str = "",
) -> list[str]:
    spec = DATASETS[dataset]
    cmd = [
        sys.executable,
        "main.py",
        "--gpu_id",
        str(args.gpu_id),
        "--use_gpu",
        "1",
        "--use_multi_gpu",
        "0",
        "--is_training",
        "1",
        "--model_id",
        model_id,
        "--model_name",
        model,
        "--dataset_root_path",
        spec.root,
        "--dataset_name",
        dataset,
        "--features",
        "M",
        "--seq_len",
        str(spec.seq_len),
        "--pred_len",
        str(spec.pred_len),
        "--enc_in",
        str(spec.enc_in),
        "--dec_in",
        str(spec.enc_in),
        "--c_out",
        str(spec.enc_in),
        "--train_epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--val_interval",
        "1",
        "--itr",
        "1",
        "--seed_base",
        str(args.run_seed),
        "--batch_size",
        str(spec.batch_size),
        "--learning_rate",
        str(spec.lr),
        "--d_model",
        str(spec.d_model),
        "--d_ff",
        str(spec.d_ff),
        "--dropout",
        str(spec.dropout),
        "--patch_len",
        str(spec.patch_len),
        "--n_heads",
        str(spec.n_heads),
        "--loss",
        args.loss,
        "--mae_weight",
        str(mae_weight),
        "--num_workers",
        str(args.num_workers),
        "--pin_memory",
        "1" if args.num_workers > 0 else "0",
        "--persistent_workers",
        "1" if args.num_workers > 0 else "0",
        "--prefetch_factor",
        "2" if args.num_workers > 0 else "1",
        "--non_blocking_transfer",
        "1",
        "--tf32",
        "1",
        "--cudnn_benchmark",
        "1",
        "--amp_dtype",
        "bf16",
        "--disable_tqdm",
        "1",
        "--save_arrays",
        "1" if args.save_arrays else "0",
        "--wearable_mask_protocol",
        args.wearable_mask_protocol,
        "--wearable_split_protocol",
        args.wearable_split_protocol,
        "--wearable_subject_fold",
        str(args.wearable_subject_fold),
        "--wearable_test_subject",
        args.wearable_test_subject,
        "--checkpoints",
        str(root),
    ]
    if ablation:
        cmd.extend(["--ablation_name", ablation])
    return cmd


def public_command(args: argparse.Namespace, root: Path, dataset: str, model: str) -> tuple[list[str], str]:
    subject_tag = "".join(ch for ch in args.wearable_test_subject if ch.isalnum()).lower()
    fold_tag = f"_loso_{subject_tag}" if subject_tag else (f"_f{args.wearable_subject_fold}" if args.wearable_subject_fold else "")
    mid = f"{args.prefix}_{dataset}_{model.lower()}_public{fold_tag}_e{args.epochs}_r{args.run_seed}"
    cmd = common_command(args, root, dataset, model, mid, mae_weight=0.0)
    try:
        extra = baseline_extra(model, dataset)
    except KeyError:
        extra = []
    cmd.extend(extra)
    if model == "Raindrop" and dataset in {"MHEALTH", "OPPORTUNITY", "PAMAP2", "USCHAD"}:
        # In this implementation ``--d_model`` is the per-variable observation
        # width; Raindrop internally expands to enc_in * d_model.  High-channel
        # wearable datasets otherwise OOM on a 32GB GPU before producing a
        # comparable baseline row.
        cmd = replace_many(cmd, {"--d_model": "1", "--n_heads": "1", "--batch_size": "4"})
    if model == "Raindrop" and dataset == "REALDISP":
        cmd = replace_many(cmd, {"--d_model": "1", "--n_heads": "1", "--batch_size": "1"})
    if model == "tPatchGNN" and dataset in {"MHEALTH", "PAMAP2", "OPPORTUNITY"}:
        cmd = replace_many(cmd, {"--batch_size": "4"})
    if model == "tPatchGNN" and dataset == "REALDISP":
        cmd = replace_many(cmd, {"--batch_size": "1"})
    if model == "GraFITi" and dataset in {"OPPORTUNITY", "PAMAP2", "REALDISP"}:
        cmd = replace_many(cmd, {"--batch_size": "4"})
    return cmd, mid


def mesa_command(
    args: argparse.Namespace,
    root: Path,
    dataset: str,
    variant: str,
    family: str,
) -> tuple[list[str], str]:
    ablation, _ = MESA_ABLATIONS[variant]
    mw_tag = str(args.mesa_mae_weight).replace(".", "p")
    subject_tag = "".join(ch for ch in args.wearable_test_subject if ch.isalnum()).lower()
    fold_tag = f"_loso_{subject_tag}" if subject_tag else (f"_f{args.wearable_subject_fold}" if args.wearable_subject_fold else "")
    mid = f"{args.prefix}_{dataset}_{family}_{variant}{fold_tag}_mw{mw_tag}_e{args.epochs}_r{args.run_seed}"
    cmd = common_command(
        args,
        root,
        dataset,
        args.mesa_model_name,
        mid,
        mae_weight=args.mesa_mae_weight,
        ablation=ablation,
    )
    if "wide" in variant:
        spec = DATASETS[dataset]
        cmd = replace_arg(cmd, "--d_ff", str(int(spec.d_ff) * 2))
    return cmd, mid


def cleanup_finished_checkpoints(root: Path) -> None:
    """Drop bulky training weights after a metric exists; keep logs and arrays."""
    for metric_path in root.glob("*/*/*/*/iter*/eval_*/metric.json"):
        iter_dir = metric_path.parents[1]
        ckpt = iter_dir / "pytorch_model.bin"
        if ckpt.exists():
            try:
                ckpt.unlink()
                print(f"removed_checkpoint={ckpt}", flush=True)
            except OSError as exc:
                print(f"checkpoint_cleanup_failed={ckpt} error={exc}", flush=True)


def run_parallel(
    jobs: list[tuple[list[str], Path]],
    root: Path,
    parallel: int,
    poll_seconds: int,
    cleanup_checkpoints: bool,
) -> list[int]:
    queue = list(jobs)
    running: list[tuple[subprocess.Popen, object, Path]] = []
    codes: list[int] = []
    while queue or running:
        while queue and len(running) < max(1, parallel):
            cmd, log = queue.pop(0)
            print("$", " ".join(shlex.quote(str(p)) for p in cmd), flush=True)
            log.parent.mkdir(parents=True, exist_ok=True)
            handle = log.open("w", encoding="utf-8", errors="replace")
            env = os.environ.copy()
            env.setdefault("PYTHONUNBUFFERED", "1")
            env.setdefault("TQDM_DISABLE", "1")
            proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=handle, stderr=subprocess.STDOUT, text=True)
            running.append((proc, handle, log))
        keep: list[tuple[subprocess.Popen, object, Path]] = []
        for proc, handle, log in running:
            code = proc.poll()
            if code is None:
                keep.append((proc, handle, log))
            else:
                handle.close()
                print(f"exit={code} log={log}", flush=True)
                codes.append(int(code))
                if cleanup_checkpoints:
                    cleanup_finished_checkpoints(root)
        running = keep
        if running:
            time.sleep(poll_seconds)
    return codes


def summarize(root: Path, records: list[JobRecord]) -> Path:
    rows: list[dict[str, object]] = []
    by_dataset: dict[str, list[dict[str, object]]] = {}
    for rec in records:
        metric = latest_metric(root, rec.dataset, rec.metric_model or rec.model, rec.model_id)
        row: dict[str, object] = {
            "family": rec.family,
            "dataset": rec.dataset,
            "model": rec.model,
            "variant": rec.variant,
            "role": rec.role,
            "model_id": rec.model_id,
        }
        if metric:
            row.update(metric)
            by_dataset.setdefault(rec.dataset, []).append(row)
        rows.append(row)

    for dataset, ds_rows in by_dataset.items():
        public_rows = [r for r in ds_rows if r["family"] == "sota_public" and "MSE" in r]
        if public_rows:
            best_public = min(public_rows, key=lambda r: float(r["MSE"]))
            for row in ds_rows:
                row["public_best_model"] = best_public["model"]
                row["public_best_MSE"] = best_public["MSE"]
                row["public_best_MAE"] = best_public["MAE"]
                row["gain_vs_public_best_MSE_pct"] = (
                    (float(best_public["MSE"]) - float(row["MSE"])) / float(best_public["MSE"]) * 100.0
                )
                row["gain_vs_public_best_MAE_pct"] = (
                    (float(best_public["MAE"]) - float(row["MAE"])) / float(best_public["MAE"]) * 100.0
                )
        main_rows = [r for r in ds_rows if r["family"] == "mesa_main" and "MSE" in r]
        if main_rows:
            main = main_rows[0]
            for row in ds_rows:
                row["gain_vs_mesa_main_MSE_pct"] = (
                    (float(main["MSE"]) - float(row["MSE"])) / float(main["MSE"]) * 100.0
                )
                row["gain_vs_mesa_main_MAE_pct"] = (
                    (float(main["MAE"]) - float(row["MAE"])) / float(main["MAE"]) * 100.0
                )

    out = root / "_summary" / "v507_mesa_complete_benchmark_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "family",
        "dataset",
        "model",
        "variant",
        "role",
        "model_id",
        "MSE",
        "MAE",
        "RMSE",
        "public_best_model",
        "public_best_MSE",
        "public_best_MAE",
        "gain_vs_public_best_MSE_pct",
        "gain_vs_public_best_MAE_pct",
        "gain_vs_mesa_main_MSE_pct",
        "gain_vs_mesa_main_MAE_pct",
    ]
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(out.read_text(encoding="utf-8"), flush=True)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="HumanActivity,MHEALTH,OPPORTUNITY,PAMAP2,USCHAD,REALDISP")
    parser.add_argument("--tasks", default="sota,mesa,ablation")
    parser.add_argument("--public_models", default="APN,KAFNet,GraFITi,tPatchGNN,mTAN,CRU,SeFT,Raindrop")
    parser.add_argument(
        "--ablation_variants",
        default="mesa,fixed_patch_nograph,random_admit_nograph,constant_admit_nograph,no_obs_state_nograph,shuffle_state_nograph,graph_state_transfer",
    )
    parser.add_argument("--run_seed", type=int, default=5301)
    parser.add_argument("--run_seeds", default="", help="optional comma-separated seeds; overrides --run_seed")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--parallel", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--loss", default="MSEMAE")
    parser.add_argument("--mesa_mae_weight", type=float, default=0.5)
    parser.add_argument(
        "--mesa_model_name",
        default="MESANet",
        help="import name for MESANet runs; use MESANetLegacy to reproduce pre-v622 tables",
    )
    parser.add_argument("--prefix", default="v507")
    parser.add_argument("--root", default="storage/results_release_v507_mesa_complete_benchmark")
    parser.add_argument("--save_arrays", type=int, default=0)
    parser.add_argument("--poll_seconds", type=int, default=10)
    parser.add_argument("--skip_done", type=int, default=1)
    parser.add_argument("--summary_only", type=int, default=0)
    parser.add_argument("--continue_on_error", type=int, default=1)
    parser.add_argument("--cleanup_checkpoints", type=int, default=0)
    parser.add_argument(
        "--wearable_mask_protocol",
        default="group_async_current",
        choices=[
            "group_async_current",
            "random_mcar_matched_density",
            "block_dropout_matched_density",
            "real_missing_only_if_available",
        ],
    )
    parser.add_argument(
        "--wearable_split_protocol",
        default="record_chronological",
        choices=["record_chronological", "subject_heldout"],
    )
    parser.add_argument("--wearable_subject_fold", type=int, default=0)
    parser.add_argument("--wearable_test_subject", default="")
    args = parser.parse_args()

    root = (REPO / args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    datasets = csv_list(args.datasets)
    tasks = set(csv_list(args.tasks))

    jobs: list[tuple[list[str], Path]] = []
    records: list[JobRecord] = []

    seed_values = [args.run_seed]
    if str(args.run_seeds).strip():
        seed_values = [int(s) for s in csv_list(args.run_seeds)]

    for run_seed in seed_values:
        args.run_seed = int(run_seed)
        seed_tag = f"_r{args.run_seed}"
        if "sota" in tasks:
            for dataset in datasets:
                for model in csv_list(args.public_models):
                    if model not in PUBLIC_MODELS:
                        raise ValueError(f"unsupported public model: {model}")
                    cmd, mid = public_command(args, root, dataset, model)
                    records.append(JobRecord("sota_public", dataset, model, model.lower(), mid, "public backbone"))
                    if not args.summary_only and not (args.skip_done and latest_metric(root, dataset, model, mid)):
                        jobs.append((cmd, root / "_logs" / f"sota_{dataset}_{model}{seed_tag}.log"))

        if "mesa" in tasks:
            for dataset in datasets:
                cmd, mid = mesa_command(args, root, dataset, "mesa", "main")
                records.append(
                    JobRecord(
                        "mesa_main",
                        dataset,
                        "MESANet",
                        "mesa",
                        mid,
                        "fixed affine-equivariant horizon-scale simplex backbone",
                        args.mesa_model_name,
                    )
                )
                if not args.summary_only and not (
                    args.skip_done and latest_metric(root, dataset, args.mesa_model_name, mid)
                ):
                    jobs.append((cmd, root / "_logs" / f"mesa_{dataset}{seed_tag}.log"))

        if "ablation" in tasks:
            for dataset in datasets:
                for variant in csv_list(args.ablation_variants):
                    if variant not in MESA_ABLATIONS:
                        raise ValueError(f"unknown MESANet ablation: {variant}")
                    cmd, mid = mesa_command(args, root, dataset, variant, "ablation")
                    records.append(
                        JobRecord(
                            "ablation",
                            dataset,
                            "MESANet",
                            variant,
                            mid,
                            MESA_ABLATIONS[variant][1],
                            args.mesa_model_name,
                        )
                    )
                    if not args.summary_only and not (
                        args.skip_done and latest_metric(root, dataset, args.mesa_model_name, mid)
                    ):
                        jobs.append((cmd, root / "_logs" / f"ablation_{dataset}_{variant}{seed_tag}.log"))

    manifest = {
        "runner": str(Path(__file__).relative_to(REPO)),
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=False
        ).stdout.strip(),
        "arguments": vars(args),
        "jobs": [
            {
                "command": [str(part) for part in cmd],
                "command_sha256": hashlib.sha256(
                    "\0".join(str(part) for part in cmd).encode("utf-8")
                ).hexdigest(),
                "log": str(log.relative_to(root)),
            }
            for cmd, log in jobs
        ],
    }
    manifest_path = root / "_summary" / "command_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    codes = [] if args.summary_only else run_parallel(
        jobs,
        root,
        args.parallel,
        args.poll_seconds,
        bool(args.cleanup_checkpoints),
    )
    summarize(root, records)
    missing = [
        rec.model_id
        for rec in records
        if latest_metric(root, rec.dataset, rec.metric_model or rec.model, rec.model_id) is None
    ]
    if missing:
        print(f"missing_metrics={len(missing)}", flush=True)
        for model_id in missing:
            print(f"missing_metric_model_id={model_id}", flush=True)
    if any(code != 0 for code in codes) or missing:
        return 0 if args.continue_on_error else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
