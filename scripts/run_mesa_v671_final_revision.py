#!/usr/bin/env python3
"""Run the frozen-rho MESANet revision and parameter-matched controls.

The final architecture fixes the DCT separation coefficient to rho=0.5.  The
two registered controls preserve the complete training protocol and parameter
count while changing only the claimed dual-coordinate structure:

* shared_transfer_matched shares variable transfer between coordinates and
  replaces the removed parameters with coordinate-local channel mixers;
* random_basis replaces the DCT basis with a deterministic random orthobasis.

Existing protocol-compatible fixed-rho runs are reused verbatim.  No result is
selected or filtered using test error.
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
    "dct_dual_rho_half_dct_dual_no_causal"
)


@dataclass(frozen=True)
class Variant:
    mode: str
    interpretation: str


VARIANTS = {
    "fixed_rho_half": Variant(
        FINAL_MODE,
        "frozen MESANet with fixed half-projection DCT separation",
    ),
    "shared_transfer_matched": Variant(
        FINAL_MODE + "_dct_dual_shared_transfer",
        "parameter-matched shared coordinate-transfer control",
    ),
    "random_basis": Variant(
        FINAL_MODE + "_dct_dual_random_basis",
        "parameter-matched deterministic random-orthobasis control",
    ),
    "no_multiresolution_fixed": Variant(
        "aa_no_q_graph_no_prob_waveform_token_parallel_dct_dual_coordinate_"
        "dct_dual_fixed_level_dct_dual_rho_half_dct_dual_no_causal",
        "remove multiresolution support while retaining fixed half-projection",
    ),
    "no_affine_anchor_fixed": Variant(
        "no_q_graph_no_prob_waveform_token_parallel_dct_dual_coordinate_"
        "dct_dual_rho_half_dct_dual_no_causal",
        "remove affine anchoring while retaining fixed half-projection",
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


def load_existing_fixed() -> dict[tuple[str, str, int], dict[str, str]]:
    """Load only the protocol-compatible fixed-rho rows already completed."""
    path = (
        ROOT
        / "storage/results_mesa_v667_final_ablation/_summary/final_evidence_runs.csv"
    )
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    return {
        (row["dataset"], row["variant"], int(row["run"])): row
        for row in rows
        if row.get("variant") == "fixed_rho_half"
        and row.get("MSE")
        and row.get("MAE")
    }


def trial_from(payload: dict[str, object]) -> Trial:
    dropout = payload.get("dropout")
    return Trial(
        name=str(payload["name"]),
        lr_mult=float(payload.get("lr_mult", 1.0)),
        width_mult=float(payload.get("width_mult", 1.0)),
        dropout=None if dropout is None else float(dropout),
    )


def registered_pairs(variant: str) -> list[tuple[str, int]]:
    if variant == "fixed_rho_half":
        datasets = tuple(DATASET_CODES)
        seeds = (5401, 5402, 5403, 5404, 5405)
    else:
        datasets = ("HumanActivity", "MHEALTH", "OPPORTUNITY", "PAMAP2")
        seeds = (5401, 5402, 5403)
    return [(dataset, seed) for seed in seeds for dataset in datasets]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variants",
        default="fixed_rho_half,shared_transfer_matched,random_basis",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--parallel", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--mae_weight", type=float, default=0.5)
    parser.add_argument("--prefix", default="mesa_v671_final")
    parser.add_argument("--root", default="storage/results_mesa_v671_final_revision")
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
    parser.add_argument("--reuse_existing_fixed", type=int, default=1)
    parser.add_argument(
        "--diagnostic_run",
        type=int,
        default=5404,
        help="Fixed-rho run that also saves per-example decomposition arrays.",
    )
    args = parser.parse_args()

    variant_names = csv_list(args.variants)
    unknown = sorted(set(variant_names) - set(VARIANTS))
    if unknown:
        raise ValueError(f"unknown variants: {unknown}")

    frozen = load_frozen(csv_list(args.frozen_configs))
    existing = load_existing_fixed() if args.reuse_existing_fixed else {}
    result_root = (ROOT / args.root).resolve()
    jobs: list[tuple[list[str], Path]] = []
    records: list[tuple[str, str, int, Trial, str]] = []
    manifest: list[dict[str, object]] = []

    for variant_name in variant_names:
        variant = VARIANTS[variant_name]
        for dataset, seed in registered_pairs(variant_name):
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
            cmd, _ = command(
                run_args, dataset, "MESANet", trial, seed, "confirm", result_root
            )
            if variant_name == "fixed_rho_half" and seed == args.diagnostic_run:
                cmd[cmd.index("--save_arrays") + 1] = "1"
            short_variant = "".join(part[0] for part in variant_name.split("_"))[:5]
            model_id = f"{args.prefix}_{short_variant}_{DATASET_CODES[dataset]}_r{seed}"
            cmd[cmd.index("--model_id") + 1] = model_id
            records.append((dataset, variant_name, seed, trial, model_id))
            reusable = (dataset, variant_name, seed) in existing
            metric = read_metric(result_root, "MESANet", model_id, "test")
            if (
                not args.summary_only
                and not reusable
                and not (args.skip_done and metric is not None)
            ):
                log_path = result_root / "_logs" / f"{variant_name}_{dataset}_r{seed}.log"
                jobs.append((cmd, log_path))
            manifest.append(
                {
                    "dataset": dataset,
                    "variant": variant_name,
                    "run": seed,
                    "selected_trial": trial.name,
                    "mode": variant.mode,
                    "interpretation": variant.interpretation,
                    "reused_existing": reusable,
                    "command": cmd,
                }
            )

    codes = [] if args.summary_only else run_jobs(jobs, args.parallel, args.poll_seconds)
    rows: list[dict[str, object]] = []
    for dataset, variant_name, seed, trial, model_id in records:
        metric = read_metric(result_root, "MESANet", model_id, "test")
        source = "current_root"
        key = (dataset, variant_name, seed)
        if metric is None and key in existing:
            prior = existing[key]
            metric = {"MSE": prior["MSE"], "MAE": prior["MAE"]}
            source = "v667_protocol_compatible_fixed_rho"
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
    write_csv(summary / "final_revision_runs.csv", rows)
    (summary / "command_manifest.json").write_text(
        json.dumps({"arguments": vars(args), "jobs": manifest}, indent=2),
        encoding="utf-8",
    )
    missing = sum(not row["MSE"] or not row["MAE"] for row in rows)
    print(f"jobs={len(jobs)} rows={len(rows)} missing={missing}", flush=True)
    return 1 if any(code != 0 for code in codes) or missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
