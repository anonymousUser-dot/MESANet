#!/usr/bin/env python3
"""Run frozen-config repeated evidence for the final dual-coordinate MESANet.

The script never performs test-set model selection. It reads the validation-
selected trial for each dataset, changes only the registered model variant and
run identifier, and writes one row per dataset/variant/run.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_mesa_equal_hpo import Trial, command, read_metric, run_jobs

FINAL_MODE = (
    "aa_no_q_graph_no_prob_waveform_token_parallel_dct_dual_coordinate_"
    "dct_dual_soft_orthogonal_dct_dual_no_causal"
)


@dataclass(frozen=True)
class Variant:
    mode: str
    interpretation: str


VARIANTS = {
    "full_learned": Variant(FINAL_MODE, "final learned soft-separation backbone"),
    "fixed_rho_half": Variant(
        "aa_no_q_graph_no_prob_waveform_token_parallel_dct_dual_coordinate_"
        "dct_dual_rho_half_dct_dual_no_causal",
        "fixed half-projection separation",
    ),
    "low_subspace_only_param_matched": Variant(
        "aa_no_q_graph_no_prob_waveform_token_parallel_dct_dual_coordinate_"
        "dct_dual_low_subspace_only_dct_dual_no_causal",
        "parameter-matched single low-subspace prediction coordinate",
    ),
    "no_waveform_coordinate": Variant(
        "aa_no_q_graph_no_prob_waveform_token_parallel_dct_dual_coordinate_"
        "dct_dual_no_high_dct_dual_no_causal",
        "remove the ordered waveform prediction coordinate",
    ),
    "no_multiresolution_coordinate": Variant(
        "aa_no_q_graph_no_prob_waveform_token_parallel_dct_dual_coordinate_"
        "dct_dual_fixed_level_dct_dual_soft_orthogonal_dct_dual_no_causal",
        "replace admitted multiresolution support with the fixed patch coordinate",
    ),
    "rho_zero": Variant(
        "aa_no_q_graph_no_prob_waveform_token_parallel_dct_dual_coordinate_"
        "dct_dual_rho_zero_dct_dual_no_causal",
        "no low-subspace removal from the waveform forecast",
    ),
    "rho_one": Variant(
        "aa_no_q_graph_no_prob_waveform_token_parallel_dct_dual_coordinate_"
        "dct_dual_rho_one_dct_dual_no_causal",
        "complete low-subspace removal from the waveform forecast",
    ),
    "no_affine_anchor": Variant(
        "no_q_graph_no_prob_waveform_token_parallel_dct_dual_coordinate_"
        "dct_dual_soft_orthogonal_dct_dual_no_causal",
        "remove masked affine normalization and coordinate restoration",
    ),
}

DATASET_CODES = {
    "HumanActivity": "ha",
    "MHEALTH": "mh",
    "OPPORTUNITY": "op",
    "PAMAP2": "pa",
    "USCHAD": "uc",
    "REALDISP": "rd",
}


def csv_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_frozen(paths: list[str]) -> dict[str, dict[str, dict[str, object]]]:
    frozen: dict[str, dict[str, dict[str, object]]] = {}
    for raw in paths:
        payload = json.loads((ROOT / raw).read_text(encoding="utf-8-sig"))
        frozen.update(payload)
    return frozen


def load_prior_rho_controls() -> dict[tuple[str, str, int], dict[str, str]]:
    path = ROOT / "storage/results_mesa_v662_rho_control/_summary/backbone_upgrade_quick.csv"
    mapping = {
        "dct_dual_rho_zero_no_causal": "rho_zero",
        "dct_dual_rho_half_no_causal": "fixed_rho_half",
        "dct_dual_rho_one_no_causal": "rho_one",
    }
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    return {
        (row["dataset"], mapping[row["candidate"]], 5401): row
        for row in rows
        if row.get("candidate") in mapping and row.get("MSE") and row.get("MAE")
    }


def trial_from(payload: dict[str, object]) -> Trial:
    dropout = payload.get("dropout")
    return Trial(
        name=str(payload["name"]),
        lr_mult=float(payload.get("lr_mult", 1.0)),
        width_mult=float(payload.get("width_mult", 1.0)),
        dropout=None if dropout is None else float(dropout),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        default="HumanActivity,MHEALTH,OPPORTUNITY,PAMAP2,USCHAD,REALDISP",
    )
    parser.add_argument("--variants", default="full_learned")
    parser.add_argument("--seeds", default="5402,5403,5404,5405")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--parallel", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--mae_weight", type=float, default=0.5)
    parser.add_argument("--prefix", default="mesa_v667_final")
    parser.add_argument("--root", default="storage/results_mesa_v667_final_evidence")
    parser.add_argument(
        "--frozen_configs",
        default=(
            "storage/results_mesa_v661_soft_no_causal_equal_hpo/_summary/"
            "equal_hpo_frozen_configs.json,"
            "storage/results_mesa_v663_boundary_equal_hpo/_summary/"
            "equal_hpo_frozen_configs.json"
        ),
    )
    parser.add_argument("--wearable_mask_protocol", default="real_missing_only_if_available")
    parser.add_argument("--wearable_split_protocol", default="subject_heldout")
    parser.add_argument("--poll_seconds", type=int, default=5)
    parser.add_argument("--skip_done", type=int, default=1)
    parser.add_argument("--summary_only", type=int, default=0)
    parser.add_argument("--reuse_prior_rho_run", type=int, default=1)
    args = parser.parse_args()

    datasets = csv_list(args.datasets)
    variant_names = csv_list(args.variants)
    seeds = [int(seed) for seed in csv_list(args.seeds)]
    unknown = sorted(set(variant_names) - set(VARIANTS))
    if unknown:
        raise ValueError(f"unknown variants: {unknown}")

    frozen = load_frozen(csv_list(args.frozen_configs))
    prior_rho = load_prior_rho_controls() if args.reuse_prior_rho_run else {}
    result_root = (ROOT / args.root).resolve()
    jobs: list[tuple[list[str], Path]] = []
    records: list[tuple[str, str, int, Trial, str]] = []
    manifest: list[dict[str, object]] = []

    for variant_name in variant_names:
        variant = VARIANTS[variant_name]
        for seed in seeds:
            for dataset in datasets:
                trial = trial_from(frozen[dataset]["MESANet"])
                run_args = Namespace(
                    gpu_id=args.gpu_id,
                    search_epochs=args.epochs,
                    confirm_epochs=args.epochs,
                    patience=args.patience,
                    mae_weight=args.mae_weight,
                    num_workers=args.num_workers,
                    wearable_mask_protocol=args.wearable_mask_protocol,
                    wearable_split_protocol=args.wearable_split_protocol,
                    mesa_ablation=variant.mode,
                    prefix=f"{args.prefix}_{variant_name}",
                )
                cmd, model_id = command(
                    run_args, dataset, "MESANet", trial, seed, "confirm", result_root
                )
                short_variant = "".join(part[0] for part in variant_name.split("_"))[:5]
                model_id = f"{args.prefix}_{short_variant}_{DATASET_CODES[dataset]}_r{seed}"
                cmd[cmd.index("--model_id") + 1] = model_id
                records.append((dataset, variant_name, seed, trial, model_id))
                log_path = result_root / "_logs" / f"{variant_name}_{dataset}_r{seed}.log"
                metric = read_metric(result_root, "MESANet", model_id, "test")
                reusable = (dataset, variant_name, seed) in prior_rho
                if not args.summary_only and not (args.skip_done and metric is not None) and not reusable:
                    jobs.append((cmd, log_path))
                manifest.append(
                    {
                        "dataset": dataset,
                        "variant": variant_name,
                        "run": seed,
                        "selected_trial": trial.name,
                        "mode": variant.mode,
                        "interpretation": variant.interpretation,
                        "command": cmd,
                    }
                )

    codes = [] if args.summary_only else run_jobs(jobs, args.parallel, args.poll_seconds)
    rows: list[dict[str, object]] = []
    for dataset, variant_name, seed, trial, model_id in records:
        metric = read_metric(result_root, "MESANet", model_id, "test")
        source = "current_root"
        if metric is None and (dataset, variant_name, seed) in prior_rho:
            prior = prior_rho[(dataset, variant_name, seed)]
            metric = {"MSE": prior["MSE"], "MAE": prior["MAE"]}
            source = "v662_frozen_rho_control"
        rows.append(
            {
                "dataset": dataset,
                "variant": variant_name,
                "run": seed,
                "selected_trial": trial.name,
                "MSE": "" if metric is None else metric.get("MSE", ""),
                "MAE": "" if metric is None else metric.get("MAE", ""),
                "model_id": model_id,
                "source": source,
            }
        )

    summary = result_root / "_summary"
    write_csv(summary / "final_evidence_runs.csv", rows)
    (summary / "command_manifest.json").write_text(
        json.dumps({"arguments": vars(args), "jobs": manifest}, indent=2),
        encoding="utf-8",
    )
    missing = sum(not row["MSE"] or not row["MAE"] for row in rows)
    print(f"jobs={len(jobs)} rows={len(rows)} missing={missing}", flush=True)
    return 1 if any(code != 0 for code in codes) or missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
