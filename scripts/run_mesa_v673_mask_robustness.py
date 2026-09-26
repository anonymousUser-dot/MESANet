#!/usr/bin/env python3
"""Run a compact alternate-mask stress test for frozen MESANet.

Each dataset uses the single public comparator frozen by validation MSE in the
main benchmark.  The experiment changes only the observation protocol and uses
an eight-epoch stress budget.  It evaluates mechanism robustness, not a
train/test mechanism-mismatch claim.
"""

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


FINAL_MODE = (
    "aa_no_q_graph_no_prob_waveform_token_parallel_dct_dual_coordinate_"
    "dct_dual_rho_half_dct_dual_no_causal"
)
DATASETS = ("HumanActivity", "MHEALTH", "OPPORTUNITY", "PAMAP2")
PROTOCOLS = (
    "group_async_current",
    "random_mcar_matched_density",
    "block_dropout_matched_density",
)
COMPARATORS = {
    "HumanActivity": "KAFNet",
    "MHEALTH": "PatchTST",
    "OPPORTUNITY": "tPatchGNN",
    "PAMAP2": "GraFITi",
}
SHORT = {
    "group_async_current": "group",
    "random_mcar_matched_density": "mcar",
    "block_dropout_matched_density": "block",
}


def load_json(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8-sig"))


def trial(payload: dict[str, object]) -> Trial:
    value = payload.get("dropout")
    return Trial(
        name=str(payload["name"]),
        lr_mult=float(payload.get("lr_mult", 1.0)),
        width_mult=float(payload.get("width_mult", 1.0)),
        dropout=None if value is None else float(value),
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=int, default=5401)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--root", default="storage/results_mesa_v673_mask_robustness")
    parser.add_argument("--summary_only", action="store_true")
    args = parser.parse_args()

    mesa_frozen = load_json(
        "storage/results_mesa_v661_soft_no_causal_equal_hpo/_summary/"
        "equal_hpo_frozen_configs.json"
    )
    public_frozen = load_json(
        "storage/results_mesa_equal_hpo/_summary/equal_hpo_frozen_configs.json"
    )
    result_root = (ROOT / args.root).resolve()
    jobs: list[tuple[list[str], Path]] = []
    records: list[tuple[str, str, str, Trial, str]] = []
    commands: list[dict[str, object]] = []

    for protocol in PROTOCOLS:
        for dataset in DATASETS:
            for model in ("MESANet", COMPARATORS[dataset]):
                chosen = trial(
                    mesa_frozen[dataset]["MESANet"]
                    if model == "MESANet"
                    else public_frozen[dataset][model]
                )
                run_args = Namespace(
                    gpu_id=args.gpu_id,
                    search_epochs=args.epochs,
                    confirm_epochs=args.epochs,
                    patience=args.patience,
                    mae_weight=0.5,
                    num_workers=args.num_workers,
                    wearable_mask_protocol=protocol,
                    wearable_split_protocol="subject_heldout",
                    mesa_ablation=FINAL_MODE if model == "MESANet" else "",
                    prefix=f"mesa_v673_{SHORT[protocol]}",
                )
                cmd, model_id = command(
                    run_args, dataset, model, chosen, args.run, "confirm", result_root
                )
                records.append((protocol, dataset, model, chosen, model_id))
                metric = read_metric(result_root, model, model_id, "test")
                if not args.summary_only and metric is None:
                    log = result_root / "_logs" / f"{SHORT[protocol]}_{dataset}_{model}.log"
                    jobs.append((cmd, log))
                commands.append(
                    {
                        "protocol": protocol,
                        "dataset": dataset,
                        "model": model,
                        "trial": chosen.name,
                        "command": cmd,
                    }
                )

    codes = [] if args.summary_only else run_jobs(jobs, args.parallel, 5)
    rows: list[dict[str, object]] = []
    for protocol, dataset, model, chosen, model_id in records:
        metric = read_metric(result_root, model, model_id, "test")
        rows.append(
            {
                "protocol": protocol,
                "dataset": dataset,
                "model": model,
                "validation_frozen_trial": chosen.name,
                "MSE": "" if metric is None else metric.get("MSE", ""),
                "MAE": "" if metric is None else metric.get("MAE", ""),
                "model_id": model_id,
            }
        )

    summary = result_root / "_summary"
    write_csv(summary / "mask_robustness_runs.csv", rows)
    (summary / "command_manifest.json").write_text(
        json.dumps({"arguments": vars(args), "comparators": COMPARATORS, "jobs": commands}, indent=2),
        encoding="utf-8",
    )
    missing = sum(not row["MSE"] or not row["MAE"] for row in rows)
    print(f"jobs={len(jobs)} rows={len(rows)} missing={missing}")
    return 1 if any(code != 0 for code in codes) or missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
