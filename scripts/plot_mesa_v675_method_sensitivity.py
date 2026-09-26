#!/usr/bin/env python3
"""Plot method-specific MESANet sensitivity for the conference main paper."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "storage/v675/_summary/method_sensitivity.csv"
OUT = ROOT / "paper/mesanet_paper/figures/final/mesa_fig4_v675_method_sensitivity"
TABLE = ROOT / "paper/mesanet_paper/generated/supp_table_v675_method_sensitivity.tex"
RHO_VALUES = (0.0, 0.25, 0.5, 0.75, 1.0)
SCALE_VALUES = (1, 2, 3, 4, 6)


def macro_ratio(frame: pd.DataFrame, metric: str, default_value: float) -> pd.Series:
    pivot = frame.pivot(index="value", columns="dataset", values=metric).astype(float)
    return pivot.div(pivot.loc[default_value], axis=1).mean(axis=1)


def write_table(frame: pd.DataFrame) -> None:
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\setlength{\tabcolsep}{5pt}",
        r"\caption{Validation MSE/MAE for the method-parameter sweeps. Lower is better.}",
        r"\label{tab:v675-method-sensitivity}",
        r"\begin{tabular}{@{}lccccc@{}}",
        r"\toprule",
        r"\multicolumn{6}{c}{DCT overlap $\rho$} \\",
        r"\midrule",
    ]
    for sweep, values in (
        ("rho", RHO_VALUES),
        ("scale_count", SCALE_VALUES),
    ):
        lines.append("Dataset & " + " & ".join(f"{value:g}" for value in values) + r" \\")
        lines.append(r"\midrule")
        part = frame[frame["sweep"] == sweep]
        for dataset in ("HumanActivity", "MHEALTH", "OPPORTUNITY", "PAMAP2"):
            dataset_rows = part[part["dataset"] == dataset].set_index("value")
            cells = [
                f"{dataset_rows.loc[float(value), 'validation_MSE']:.4f}/"
                f"{dataset_rows.loc[float(value), 'validation_MAE']:.4f}"
                for value in values
            ]
            lines.append(dataset + " & " + " & ".join(cells) + r" \\")
        if sweep == "rho":
            lines.extend([r"\midrule", r"\multicolumn{6}{c}{Causal support count $K$} \\", r"\midrule"])
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    TABLE.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    frame = pd.read_csv(CSV)
    if frame[["validation_MSE", "validation_MAE"]].isna().any().any():
        raise ValueError("method sensitivity contains missing metrics")

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 7,
        "lines.linewidth": 1.5,
        "lines.markersize": 4,
    })
    fig, axes = plt.subplots(1, 2, figsize=(3.35, 1.75), constrained_layout=True)
    panels = [
        ("rho", 0.5, r"Overlap $\rho$", "(a) Coordinate overlap"),
        ("scale_count", 4.0, "Support scales $K$", "(b) Support count"),
    ]
    colors = {"validation_MSE": "#0072B2", "validation_MAE": "#D55E00"}
    labels = {"validation_MSE": "MSE", "validation_MAE": "MAE"}
    markers = {"validation_MSE": "o", "validation_MAE": "s"}
    for panel_index, (ax, (sweep, default, xlabel, title)) in enumerate(zip(axes, panels)):
        part = frame[frame["sweep"] == sweep]
        for metric in ("validation_MSE", "validation_MAE"):
            values = macro_ratio(part, metric, default)
            ax.plot(values.index, values.values, color=colors[metric], marker=markers[metric], label=labels[metric])
        ax.axhline(1.0, color="#6B7280", linestyle=":", linewidth=0.9)
        ax.axvline(default, color="#9CA3AF", linestyle="--", linewidth=0.8)
        ax.set_title(title, loc="left", fontweight="semibold")
        ax.set_xlabel(xlabel)
        if panel_index == 0:
            ax.set_ylabel("Macro error / default")
        ax.grid(axis="y", color="#D1D5DB", linewidth=0.5, alpha=0.8)
        if panel_index == 0:
            ax.legend(frameon=False, ncol=2, loc="best", handlelength=2.0, columnspacing=0.8)
        if sweep == "scale_count":
            ax.set_xticks(sorted(part["value"].unique()))

    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), dpi=300, bbox_inches="tight")
    write_table(frame)
    print(OUT.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
