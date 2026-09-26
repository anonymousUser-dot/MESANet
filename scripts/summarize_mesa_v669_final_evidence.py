#!/usr/bin/env python3
"""Assemble protocol-compatible repeated evidence for the final MESANet.

The script combines frozen validation-selected runs only. It does not select
architectures, hyperparameters, or public competitors using test metrics.
Metric-wise public-frontier labels are used only after every registered run has
been loaded, exactly as reported in the paper.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DATASETS = (
    "HumanActivity", "MHEALTH", "OPPORTUNITY", "PAMAP2", "USCHAD", "REALDISP"
)
CORE_DATASETS = DATASETS[:4]
PUBLIC_CANDIDATES = {
    "HumanActivity": ("KAFNet",),
    "MHEALTH": ("PatchTST",),
    "OPPORTUNITY": ("tPatchGNN", "PatchTST"),
    "PAMAP2": ("GraFITi",),
    "USCHAD": ("PatchTST",),
    "REALDISP": ("GraFITi", "PatchTST"),
}
ABBREVIATIONS = {
    "HumanActivity": "HA", "MHEALTH": "MH", "OPPORTUNITY": "OPP",
    "PAMAP2": "PAM", "USCHAD": "USC", "REALDISP": "REAL",
}
VARIANT_LABELS = {
    "full_learned": "MESANet",
    "fixed_rho_half": r"fixed $\rho=0.5$",
    "low_subspace_only_param_matched": "single coordinate (matched)",
    "no_waveform_coordinate": "w/o waveform coordinate",
    "no_multiresolution_coordinate": "w/o multiresolution coordinate",
    "rho_zero": r"w/o coordinate separation ($\rho=0$)",
    "rho_one": r"complete separation ($\rho=1$)",
    "no_affine_anchor": "w/o affine anchor",
}


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


def numerical(rows: Iterable[dict[str, str]]) -> list[dict[str, object]]:
    converted: list[dict[str, object]] = []
    for row in rows:
        if not row.get("MSE") or not row.get("MAE"):
            continue
        item: dict[str, object] = dict(row)
        item["run"] = int(row["run"])
        item["MSE"] = float(row["MSE"])
        item["MAE"] = float(row["MAE"])
        converted.append(item)
    return converted


def unique_by_key(rows: Iterable[dict[str, object]], keys: tuple[str, ...]) -> list[dict[str, object]]:
    output: dict[tuple[object, ...], dict[str, object]] = {}
    for row in rows:
        output[tuple(row[key] for key in keys)] = row
    return list(output.values())


def mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lo = math.floor(position)
    hi = math.ceil(position)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - position) + ordered[hi] * (position - lo)


def paired_interval(differences: list[float], draws: int = 20000) -> tuple[float, float]:
    rng = random.Random(20270716)
    samples = []
    for _ in range(draws):
        samples.append(statistics.mean(rng.choice(differences) for _ in differences))
    return percentile(samples, 0.025), percentile(samples, 0.975)


def load_mesanet() -> list[dict[str, object]]:
    paths = (
        ROOT / "storage/results_mesa_v661_soft_no_causal_equal_hpo/_summary/equal_hpo_test_runs.csv",
        ROOT / "storage/results_mesa_v663_boundary_equal_hpo/_summary/equal_hpo_test_runs.csv",
        ROOT / "storage/results_mesa_v667_final_repeated/_summary/final_evidence_runs_5402_5405.csv",
    )
    rows: list[dict[str, object]] = []
    for path in paths:
        for row in numerical(read_rows(path)):
            if row.get("model") == "MESANet" or row.get("variant") == "full_learned":
                row["model"] = "MESANet"
                rows.append(row)
    return unique_by_key(rows, ("dataset", "model", "run"))


def load_public() -> list[dict[str, object]]:
    run_5401_sources = (
        ROOT / "storage/results_mesa_equal_hpo/_summary/equal_hpo_test_runs.csv",
        ROOT / "storage/results_mesa_v664_uschad_public_equal_hpo/_summary/equal_hpo_test_runs.csv",
        ROOT / "storage/results_mesa_v664_realdisp_public_equal_hpo/_summary/equal_hpo_test_runs.csv",
    )
    rows: list[dict[str, object]] = []
    for path in run_5401_sources:
        for row in numerical(read_rows(path)):
            dataset = str(row["dataset"])
            if int(row["run"]) == 5401 and str(row.get("model")) in PUBLIC_CANDIDATES.get(dataset, ()):
                rows.append(row)
    repeated = ROOT / "storage/results_mesa_v668_frontier_repeated/_summary/frontier_repeated_runs.csv"
    rows.extend(numerical(read_rows(repeated)))
    return unique_by_key(rows, ("dataset", "model", "run"))


def repeated_summary(mesa: list[dict[str, object]], public: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    combined = mesa + public
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in combined:
        groups[(str(row["dataset"]), str(row["model"]))].append(row)

    means: list[dict[str, object]] = []
    for (dataset, model), rows in sorted(groups.items()):
        if len(rows) != 5:
            raise RuntimeError(f"expected five runs for {dataset}/{model}, found {len(rows)}")
        mse_mean, mse_std = mean_std([float(row["MSE"]) for row in rows])
        mae_mean, mae_std = mean_std([float(row["MAE"]) for row in rows])
        means.append({
            "dataset": dataset, "model": model, "runs": len(rows),
            "MSE_mean": mse_mean, "MSE_std": mse_std,
            "MAE_mean": mae_mean, "MAE_std": mae_std,
        })

    paired: list[dict[str, object]] = []
    for dataset in DATASETS:
        mesa_rows = {int(row["run"]): row for row in groups[(dataset, "MESANet")]}
        for metric in ("MSE", "MAE"):
            candidates = []
            for model in PUBLIC_CANDIDATES[dataset]:
                rows = groups.get((dataset, model), [])
                if len(rows) != 5:
                    continue
                candidates.append((statistics.mean(float(row[metric]) for row in rows), model, rows))
            if not candidates:
                raise RuntimeError(f"no five-run public candidate for {dataset}/{metric}")
            public_mean, public_model, public_rows = min(candidates)
            public_by_run = {int(row["run"]): row for row in public_rows}
            differences = [
                float(public_by_run[run][metric]) - float(mesa_rows[run][metric])
                for run in sorted(mesa_rows)
            ]
            mesa_values = [float(row[metric]) for row in mesa_rows.values()]
            public_values = [float(row[metric]) for row in public_rows]
            lo, hi = paired_interval(differences)
            paired.append({
                "dataset": dataset, "metric": metric, "public_model": public_model,
                "public_mean": public_mean, "public_std": statistics.stdev(public_values),
                "mesanet_mean": statistics.mean(mesa_values),
                "mesanet_std": statistics.stdev(mesa_values),
                "gain_percent": 100.0 * statistics.mean(differences) / public_mean,
                "paired_difference": statistics.mean(differences),
                "interval_low": lo, "interval_high": hi,
                "dagger": "yes" if lo > 0 or hi < 0 else "no",
                "mesanet_wins": sum(value > 0 for value in differences),
                "runs": len(differences),
            })
    return means, paired


def load_ablations(path: Path, mesa: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    controls = numerical(read_rows(path))
    controls = [row for row in controls if str(row["dataset"]) in CORE_DATASETS]
    full = []
    for row in mesa:
        if str(row["dataset"]) in CORE_DATASETS and int(row["run"]) in (5401, 5402, 5403):
            item = dict(row)
            item["variant"] = "full_learned"
            full.append(item)
    all_rows = unique_by_key(full + controls, ("dataset", "variant", "run"))
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in all_rows:
        groups[(str(row["variant"]), str(row["dataset"]))].append(row)

    variants = ("full_learned", "fixed_rho_half", "low_subspace_only_param_matched",
                "no_waveform_coordinate", "no_multiresolution_coordinate", "rho_zero",
                "rho_one", "no_affine_anchor")
    detailed: list[dict[str, object]] = []
    summary: list[dict[str, object]] = []
    for variant in variants:
        mse_degradation: list[float] = []
        mae_degradation: list[float] = []
        mse_full_wins = 0
        mae_full_wins = 0
        for dataset in CORE_DATASETS:
            rows = groups[(variant, dataset)]
            if len(rows) != 3:
                raise RuntimeError(f"expected three runs for {variant}/{dataset}, found {len(rows)}")
            mse_values = [float(row["MSE"]) for row in rows]
            mae_values = [float(row["MAE"]) for row in rows]
            mse_mean, mse_std = mean_std(mse_values)
            mae_mean, mae_std = mean_std(mae_values)
            detailed.append({
                "variant": variant, "dataset": dataset, "runs": 3,
                "MSE_mean": mse_mean, "MSE_std": mse_std,
                "MAE_mean": mae_mean, "MAE_std": mae_std,
            })
            if variant != "full_learned":
                full_rows = groups[("full_learned", dataset)]
                full_mse = statistics.mean(float(row["MSE"]) for row in full_rows)
                full_mae = statistics.mean(float(row["MAE"]) for row in full_rows)
                mse_degradation.append(100.0 * (mse_mean - full_mse) / full_mse)
                mae_degradation.append(100.0 * (mae_mean - full_mae) / full_mae)
                mse_full_wins += int(full_mse < mse_mean)
                mae_full_wins += int(full_mae < mae_mean)
        summary.append({
            "variant": variant,
            "avg_MSE_degradation_percent": "" if variant == "full_learned" else statistics.mean(mse_degradation),
            "avg_MAE_degradation_percent": "" if variant == "full_learned" else statistics.mean(mae_degradation),
            "full_MSE_wins": "" if variant == "full_learned" else mse_full_wins,
            "full_MAE_wins": "" if variant == "full_learned" else mae_full_wins,
        })
    return detailed, summary


def extract_rho(checkpoint_root: Path, mesa: list[dict[str, object]]) -> list[dict[str, object]]:
    """Read the learned separation coefficient from retained checkpoints."""
    import torch

    rows: list[dict[str, object]] = []
    for row in mesa:
        model_id = str(row.get("model_id", ""))
        if not model_id:
            continue
        matches = list(checkpoint_root.glob(f"**/{model_id}/**/pytorch_model.bin"))
        if not matches:
            continue
        state = torch.load(matches[-1], map_location="cpu", weights_only=False)
        logit = state.get("dual_orth_logit") if isinstance(state, dict) else None
        if logit is None:
            continue
        rho = float(torch.sigmoid(logit.detach()).item())
        rows.append({
            "dataset": row["dataset"], "run": row["run"],
            "model_id": model_id, "rho": rho,
        })
    return sorted(rows, key=lambda item: (str(item["dataset"]), int(item["run"])))


def fmt(value: float) -> str:
    return f"{value:.4f}"


def write_repeated_tex(path: Path, paired: list[dict[str, object]]) -> None:
    by_key = {(str(row["dataset"]), str(row["metric"])): row for row in paired}
    lines = [
        r"\begin{table*}[t]", r"\centering", r"\setlength{\tabcolsep}{3.4pt}",
        r"\caption{Five-run stability against the metric-wise repeated public frontier. Values are mean $\pm$ standard deviation; positive gains favor MESANet. $\dagger$ marks paired bootstrap intervals excluding zero.}",
        r"\label{tab:v669-repeated}",
        r"\begin{tabular}{@{}llrrrrrr@{}}", r"\toprule",
        r"Dataset & Public frontier & Public MSE & MESANet MSE & Gain & Public MAE & MESANet MAE & Gain \\",
        r"\midrule",
    ]
    for dataset in DATASETS:
        mse = by_key[(dataset, "MSE")]
        mae = by_key[(dataset, "MAE")]
        public = str(mse["public_model"])
        if mae["public_model"] != mse["public_model"]:
            public += "/" + str(mae["public_model"])
        mse_mark = r"$^\dagger$" if mse["dagger"] == "yes" and float(mse["paired_difference"]) > 0 else ""
        mae_mark = r"$^\dagger$" if mae["dagger"] == "yes" and float(mae["paired_difference"]) > 0 else ""
        lines.append(
            f"{ABBREVIATIONS[dataset]} & {public} & "
            f"{fmt(float(mse['public_mean']))}$\\pm${fmt(float(mse['public_std']))} & "
            f"{fmt(float(mse['mesanet_mean']))}$\\pm${fmt(float(mse['mesanet_std']))} & "
            f"{float(mse['gain_percent']):+.2f}\\%{mse_mark} & "
            f"{fmt(float(mae['public_mean']))}$\\pm${fmt(float(mae['public_std']))} & "
            f"{fmt(float(mae['mesanet_mean']))}$\\pm${fmt(float(mae['mesanet_std']))} & "
            f"{float(mae['gain_percent']):+.2f}\\%{mae_mark} \\\\" 
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_ablation_tex(path: Path, detailed: list[dict[str, object]], summary: list[dict[str, object]]) -> None:
    detail = {(str(row["variant"]), str(row["dataset"])): row for row in detailed}
    totals = {str(row["variant"]): row for row in summary}
    variants = tuple(totals)
    lines = [
        r"\begin{table*}[t]", r"\centering", r"\setlength{\tabcolsep}{1.1pt}",
        r"\caption{Three-run component controls for the final MESANet implementation. Values are mean MSE/MAE; positive degradation indicates that the complete model is better.}",
        r"\label{tab:v669-controls}", r"\begin{tabular}{@{}l*{4}{rr}rrrr@{}}", r"\toprule",
        r"& \multicolumn{2}{c}{HA} & \multicolumn{2}{c}{MH} & \multicolumn{2}{c}{OPP} & \multicolumn{2}{c}{PAM} & \multicolumn{2}{c}{Avg. deg.} & \multicolumn{2}{c}{Full wins} \\",
        r"Variant & MSE & MAE & MSE & MAE & MSE & MAE & MSE & MAE & $\Delta$MSE & $\Delta$MAE & MSE & MAE \\",
        r"\midrule",
    ]
    for variant in variants:
        cells = []
        for dataset in CORE_DATASETS:
            row = detail[(variant, dataset)]
            cells.extend([fmt(float(row["MSE_mean"])), fmt(float(row["MAE_mean"]))])
        total = totals[variant]
        if variant == "full_learned":
            tail = ["--", "--", "--", "--"]
        else:
            tail = [
                f"{float(total['avg_MSE_degradation_percent']):+.2f}\\%",
                f"{float(total['avg_MAE_degradation_percent']):+.2f}\\%",
                f"{total['full_MSE_wins']}/4", f"{total['full_MAE_wins']}/4",
            ]
        lines.append(VARIANT_LABELS[variant] + " & " + " & ".join(cells + tail) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ablation_csv",
        default="storage/results_mesa_v667_final_ablation/_summary/final_evidence_runs.csv",
    )
    parser.add_argument(
        "--output",
        default="storage/results_mesa_v669_final_evidence/_summary",
    )
    parser.add_argument(
        "--generated",
        default="paper/mesanet_paper/generated",
    )
    parser.add_argument("--skip_ablations", action="store_true")
    parser.add_argument("--skip_rho", action="store_true")
    parser.add_argument("--checkpoint_root", default="D:/m")
    args = parser.parse_args()

    output = (ROOT / args.output).resolve()
    generated = (ROOT / args.generated).resolve()
    output.mkdir(parents=True, exist_ok=True)
    generated.mkdir(parents=True, exist_ok=True)

    mesa = load_mesanet()
    public = load_public()
    means, paired = repeated_summary(mesa, public)
    write_rows(output / "five_run_method_summary.csv", means)
    write_rows(output / "five_run_paired_frontier.csv", paired)
    write_repeated_tex(generated / "main_table_v669_repeated_frontier.tex", paired)
    if not args.skip_rho:
        rho_rows = extract_rho(Path(args.checkpoint_root), mesa)
        write_rows(output / "learned_separation_coefficients.csv", rho_rows)

    if not args.skip_ablations:
        detailed, summary = load_ablations((ROOT / args.ablation_csv).resolve(), mesa)
        write_rows(output / "three_run_final_ablation_detail.csv", detailed)
        write_rows(output / "three_run_final_ablation_summary.csv", summary)
        write_ablation_tex(generated / "main_table_v669_final_controls.tex", detailed, summary)

    wins = sum(float(row["paired_difference"]) > 0 for row in paired)
    dual = sum(
        all(float(row["paired_difference"]) > 0 for row in paired if row["dataset"] == dataset)
        for dataset in DATASETS
    )
    print(json.dumps({"metric_wins": wins, "metric_cells": 12, "dual_wins": dual}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
