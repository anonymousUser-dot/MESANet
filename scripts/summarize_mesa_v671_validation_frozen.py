#!/usr/bin/env python3
"""Build a validation-frozen repeated benchmark for the final MESANet.

One public comparator is selected per dataset using validation MSE (validation
MAE breaks exact ties).  The same frozen comparator is then used for both test
metrics and every repeated run.  Test errors never participate in comparator
selection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASETS = (
    "HumanActivity",
    "MHEALTH",
    "OPPORTUNITY",
    "PAMAP2",
    "USCHAD",
    "REALDISP",
)
PUBLIC_MODELS = {"APN", "KAFNet", "GraFITi", "tPatchGNN", "PatchTST"}
ABBREVIATIONS = {
    "HumanActivity": "HA",
    "MHEALTH": "MH",
    "OPPORTUNITY": "OPP",
    "PAMAP2": "PAM",
    "USCHAD": "USC",
    "REALDISP": "REAL",
}

VALIDATION_SOURCES = (
    "storage/results_mesa_equal_hpo/_summary/equal_hpo_validation_trials.csv",
    "storage/results_mesa_v664_uschad_public_equal_hpo/_summary/equal_hpo_validation_trials.csv",
    "storage/results_mesa_v664_realdisp_public_equal_hpo/_summary/equal_hpo_validation_trials.csv",
)
PUBLIC_RUN_SOURCES = (
    "storage/results_mesa_equal_hpo/_summary/equal_hpo_test_runs.csv",
    "storage/results_mesa_v664_uschad_public_equal_hpo/_summary/equal_hpo_test_runs.csv",
    "storage/results_mesa_v664_realdisp_public_equal_hpo/_summary/equal_hpo_test_runs.csv",
    "storage/results_mesa_v668_frontier_repeated/_summary/frontier_repeated_runs.csv",
)
MESA_RUN_SOURCES = (
    "storage/results_mesa_v667_final_ablation/_summary/final_evidence_runs.csv",
    "storage/results_mesa_v671_final_revision/_summary/final_revision_runs.csv",
)


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def choose_comparators() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    validation_rows: list[dict[str, str]] = []
    source_hashes: list[dict[str, object]] = []
    for relative in VALIDATION_SOURCES:
        path = ROOT / relative
        validation_rows.extend(read_rows(path))
        source_hashes.append({"path": relative, "sha256": sha256(path)})

    manifest: list[dict[str, object]] = []
    for dataset in DATASETS:
        candidates = [
            row
            for row in validation_rows
            if row.get("dataset") == dataset
            and row.get("model") in PUBLIC_MODELS
            and row.get("selected", "").lower() == "true"
            and row.get("validation_MSE")
            and row.get("validation_MAE")
        ]
        if not candidates:
            raise RuntimeError(f"no validation-selected public candidates for {dataset}")
        chosen = min(
            candidates,
            key=lambda row: (float(row["validation_MSE"]), float(row["validation_MAE"])),
        )
        manifest.append(
            {
                "dataset": dataset,
                "public_model": chosen["model"],
                "trial": chosen["trial"],
                "validation_MSE": float(chosen["validation_MSE"]),
                "validation_MAE": float(chosen["validation_MAE"]),
                "selection_rule": "minimum validation MSE; validation MAE tie-break",
            }
        )
    return manifest, source_hashes


def numerical(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for row in rows:
        if not row.get("MSE") or not row.get("MAE") or not row.get("run"):
            continue
        item: dict[str, object] = dict(row)
        item["run"] = int(row["run"])
        item["MSE"] = float(row["MSE"])
        item["MAE"] = float(row["MAE"])
        output.append(item)
    return output


def load_public(manifest: list[dict[str, object]]) -> list[dict[str, object]]:
    selected = {str(row["dataset"]): str(row["public_model"]) for row in manifest}
    rows: dict[tuple[str, str, int], dict[str, object]] = {}
    for relative in PUBLIC_RUN_SOURCES:
        for row in numerical(read_rows(ROOT / relative)):
            dataset = str(row.get("dataset"))
            model = str(row.get("model"))
            run = int(row["run"])
            if dataset in selected and model == selected[dataset] and 5401 <= run <= 5405:
                rows[(dataset, model, run)] = row
    return list(rows.values())


def load_mesa() -> list[dict[str, object]]:
    rows: dict[tuple[str, int], dict[str, object]] = {}
    for relative in MESA_RUN_SOURCES:
        for row in numerical(read_rows(ROOT / relative)):
            variant = str(row.get("variant", ""))
            run = int(row["run"])
            if variant == "fixed_rho_half" and 5401 <= run <= 5405:
                item = dict(row)
                item["model"] = "MESANet"
                rows[(str(row["dataset"]), run)] = item
    return list(rows.values())


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - position) + ordered[hi] * (position - lo)


def paired_interval(values: list[float], draws: int = 20000) -> tuple[float, float]:
    rng = random.Random(20270717)
    means = [statistics.mean(rng.choice(values) for _ in values) for _ in range(draws)]
    return percentile(means, 0.025), percentile(means, 0.975)


def summarize(
    manifest: list[dict[str, object]],
    public: list[dict[str, object]],
    mesa: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[str]]:
    selected = {str(row["dataset"]): str(row["public_model"]) for row in manifest}
    public_groups: dict[tuple[str, str], dict[int, dict[str, object]]] = defaultdict(dict)
    mesa_groups: dict[str, dict[int, dict[str, object]]] = defaultdict(dict)
    for row in public:
        public_groups[(str(row["dataset"]), str(row["model"]))][int(row["run"])] = row
    for row in mesa:
        mesa_groups[str(row["dataset"])][int(row["run"])] = row

    missing: list[str] = []
    output: list[dict[str, object]] = []
    expected = set(range(5401, 5406))
    for dataset in DATASETS:
        model = selected[dataset]
        public_by_run = public_groups[(dataset, model)]
        mesa_by_run = mesa_groups[dataset]
        if set(public_by_run) != expected:
            missing.append(f"public {dataset}/{model}: {sorted(expected - set(public_by_run))}")
        if set(mesa_by_run) != expected:
            missing.append(f"MESANet {dataset}: {sorted(expected - set(mesa_by_run))}")
        common = sorted(expected & set(public_by_run) & set(mesa_by_run))
        for metric in ("MSE", "MAE"):
            if not common:
                continue
            public_values = [float(public_by_run[run][metric]) for run in common]
            mesa_values = [float(mesa_by_run[run][metric]) for run in common]
            differences = [p - m for p, m in zip(public_values, mesa_values)]
            lo, hi = paired_interval(differences)
            public_mean = statistics.mean(public_values)
            output.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "public_model": model,
                    "runs": len(common),
                    "public_mean": public_mean,
                    "public_std": statistics.stdev(public_values) if len(common) > 1 else 0.0,
                    "mesanet_mean": statistics.mean(mesa_values),
                    "mesanet_std": statistics.stdev(mesa_values) if len(common) > 1 else 0.0,
                    "paired_public_minus_mesa": statistics.mean(differences),
                    "gain_percent": 100.0 * statistics.mean(differences) / public_mean,
                    "interval_low": lo,
                    "interval_high": hi,
                    "dagger": "yes" if lo > 0 else "no",
                    "mesanet_run_wins": sum(value > 0 for value in differences),
                }
            )
    return output, missing


def fmt(value: float) -> str:
    return f"{value:.4f}"


def write_tex(path: Path, rows: list[dict[str, object]]) -> None:
    by_key = {(str(row["dataset"]), str(row["metric"])): row for row in rows}
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\setlength{\tabcolsep}{2.2pt}",
        r"\caption{Validation-frozen five-run comparison. One public comparator is selected per dataset by validation MSE and used for both test metrics. Values are mean $\pm$ standard deviation; positive gains favor MESANet. $\dagger$ marks a run-level paired stability interval above zero.}",
        r"\label{tab:v671-validation-frozen}",
        r"\begin{tabular}{@{}llrrrrrr@{}}",
        r"\toprule",
        r"Dataset & Frozen public comparator & Public MSE & MESANet MSE & Gain & Public MAE & MESANet MAE & Gain \\",
        r"\midrule",
    ]
    for dataset in DATASETS:
        mse = by_key.get((dataset, "MSE"))
        mae = by_key.get((dataset, "MAE"))
        if mse is None or mae is None:
            continue
        mse_mark = r"$^\dagger$" if mse["dagger"] == "yes" else ""
        mae_mark = r"$^\dagger$" if mae["dagger"] == "yes" else ""
        lines.append(
            f"{ABBREVIATIONS[dataset]} & {mse['public_model']} & "
            f"{fmt(float(mse['public_mean']))}$\\pm${fmt(float(mse['public_std']))} & "
            f"{fmt(float(mse['mesanet_mean']))}$\\pm${fmt(float(mse['mesanet_std']))} & "
            f"{float(mse['gain_percent']):+.2f}\\%{mse_mark} & "
            f"{fmt(float(mae['public_mean']))}$\\pm${fmt(float(mae['public_std']))} & "
            f"{fmt(float(mae['mesanet_mean']))}$\\pm${fmt(float(mae['mesanet_std']))} & "
            f"{float(mae['gain_percent']):+.2f}\\%{mae_mark} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", default="storage/results_mesa_v671_validation_frozen/_summary"
    )
    parser.add_argument(
        "--generated", default="paper/mesanet_paper/generated"
    )
    parser.add_argument("--allow_incomplete", action="store_true")
    args = parser.parse_args()

    output = (ROOT / args.output).resolve()
    generated = (ROOT / args.generated).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest, source_hashes = choose_comparators()
    write_rows(output / "validation_frozen_comparators.csv", manifest)
    (output / "validation_frozen_comparators.json").write_text(
        json.dumps(
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "selection_stage": "validation only",
                "primary_selection_metric": "validation MSE",
                "tie_breaker": "validation MAE",
                "same_comparator_for_test_MSE_and_test_MAE": True,
                "source_files": source_hashes,
                "comparators": manifest,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    rows, missing = summarize(manifest, load_public(manifest), load_mesa())
    write_rows(output / "validation_frozen_five_run.csv", rows)
    (output / "missing_runs.json").write_text(json.dumps(missing, indent=2), encoding="utf-8")
    if not missing:
        write_tex(generated / "main_table_v671_validation_frozen.tex", rows)
    print(json.dumps({"comparators": manifest, "missing": missing}, indent=2))
    return 0 if not missing or args.allow_incomplete else 1


if __name__ == "__main__":
    raise SystemExit(main())
