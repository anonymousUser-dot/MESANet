#!/usr/bin/env python3
"""Estimate observable proxies for the dual-coordinate risk decomposition.

The analysis uses one predeclared saved-array run per dataset.  Missing future
targets are handled by a mask-weighted least-squares projection onto the same
rank-four DCT subspace used by MESANet.  The reported quantities are empirical
diagnostics, not estimates of an unobservable population conditional mean.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATASETS = (
    "HumanActivity",
    "MHEALTH",
    "OPPORTUNITY",
    "PAMAP2",
    "USCHAD",
    "REALDISP",
)


def dct_basis(length: int, rank: int) -> np.ndarray:
    t = np.arange(length, dtype=np.float64)[:, None]
    k = np.arange(rank, dtype=np.float64)[None, :]
    basis = np.sqrt(2.0 / length) * np.cos(np.pi * (t + 0.5) * k / length)
    basis[:, 0] = 1.0 / math.sqrt(length)
    return basis


def locate_array_dir(root: Path, dataset: str, run: int) -> Path:
    candidates = [
        path.parent
        for path in root.rglob("output_dual_low.npy")
        if dataset in path.parts and f"r{run}" in "_".join(path.parts)
    ]
    if not candidates:
        raise FileNotFoundError(f"no saved dual-coordinate arrays for {dataset}/run {run}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def masked_target_low(target: np.ndarray, mask: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Project each sample-variable target using only released target entries."""
    n, horizon, variables = target.shape
    projected = np.zeros_like(target, dtype=np.float64)
    for sample in range(n):
        for variable in range(variables):
            observed = mask[sample, :, variable] > 0.5
            if not np.any(observed):
                continue
            design = basis[observed]
            values = target[sample, observed, variable]
            coefficient, *_ = np.linalg.lstsq(design, values, rcond=None)
            projected[sample, :, variable] = basis @ coefficient
    return projected


def masked_mse(error: np.ndarray, mask: np.ndarray) -> float:
    denom = float(mask.sum())
    return float((error.square() if hasattr(error, "square") else error ** 2).sum() / denom) if denom else math.nan


def load_gain_map(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    by_key: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        if not row.get("MSE"):
            continue
        key = (row["dataset"], row["variant"])
        by_key.setdefault(key, []).append(float(row["MSE"]))
    gains: dict[str, float] = {}
    for dataset in DATASETS:
        full = by_key.get((dataset, "fixed_rho_half"), [])
        single = by_key.get((dataset, "low_subspace_only_param_matched"), [])
        if full and single:
            full_mean = float(np.mean(full))
            single_mean = float(np.mean(single))
            gains[dataset] = 100.0 * (single_mean - full_mean) / single_mean
    return gains


def analyze_dataset(array_dir: Path, rank: int) -> dict[str, float | int | str]:
    target = np.load(array_dir / "input_y.npy").astype(np.float64)
    mask = np.load(array_dir / "input_y_mask.npy").astype(np.float64)
    low = np.load(array_dir / "output_dual_low.npy").astype(np.float64)
    high = np.load(array_dir / "output_dual_high.npy").astype(np.float64)
    raw = np.load(array_dir / "output_dual_high_raw.npy").astype(np.float64)
    horizon = target.shape[1]
    basis = dct_basis(horizon, min(rank, horizon))
    target_low = masked_target_low(target, mask, basis)
    target_high = target - target_low

    # P_B r is well defined for the complete model output, independent of the
    # future-availability mask used only for target scoring.
    raw_coeff = np.einsum("nht,hk->nkt", raw, basis)
    raw_low = np.einsum("nkt,hk->nht", raw_coeff, basis)
    raw_energy = float(np.mean(raw ** 2))
    overlap_energy = float(np.mean(raw_low ** 2))
    prediction = low + high
    observed_error = (prediction - target) * mask
    per_window_denom = mask.sum(axis=(1, 2)).clip(min=1.0)
    per_window_mse = (observed_error ** 2).sum(axis=(1, 2)) / per_window_denom

    return {
        "samples": int(target.shape[0]),
        "horizon": int(horizon),
        "variables": int(target.shape[2]),
        "target_observation_rate": float(mask.mean()),
        "epsilon_L_proxy": masked_mse((low - target_low) * mask, mask),
        "epsilon_W_proxy": masked_mse((high - target_high) * mask, mask),
        "Omega_raw_low_energy": overlap_energy,
        "Omega_raw_low_fraction": overlap_energy / max(raw_energy, 1e-12),
        "forecast_MSE": masked_mse(observed_error, mask),
        "forecast_MAE": float(np.abs(observed_error).sum() / max(mask.sum(), 1.0)),
        "window_MSE_p50": float(np.quantile(per_window_mse, 0.50)),
        "window_MSE_p95": float(np.quantile(per_window_mse, 0.95)),
        "window_MSE_p99": float(np.quantile(per_window_mse, 0.99)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="storage/results_mesa_v671_final_revision")
    parser.add_argument("--run", type=int, default=5404)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument(
        "--ablation_csv",
        default="storage/results_mesa_v667_final_ablation/_summary/final_evidence_runs.csv",
    )
    parser.add_argument(
        "--output",
        default="storage/results_mesa_v671_validation_frozen/_summary/dual_coordinate_diagnostic.csv",
    )
    args = parser.parse_args()

    root = (ROOT / args.root).resolve()
    gains = load_gain_map((ROOT / args.ablation_csv).resolve())
    rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        directory = locate_array_dir(root, dataset, args.run)
        row: dict[str, object] = {
            "dataset": dataset,
            "array_dir": str(directory.relative_to(ROOT)),
            **analyze_dataset(directory, args.rank),
            "gain_vs_single_coordinate_MSE_percent": gains.get(dataset, ""),
        }
        rows.append(row)

    output = (ROOT / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    core = [row for row in rows if row["gain_vs_single_coordinate_MSE_percent"] != ""]
    correlations: dict[str, float] = {}
    if len(core) >= 4:
        gain = np.asarray([float(row["gain_vs_single_coordinate_MSE_percent"]) for row in core])
        for key in ("epsilon_L_proxy", "epsilon_W_proxy", "Omega_raw_low_fraction"):
            values = np.asarray([float(row[key]) for row in core])
            correlations[key] = float(np.corrcoef(values, gain)[0, 1])
    (output.with_suffix(".json")).write_text(
        json.dumps({"rows": rows, "correlation_with_single_coordinate_gain": correlations}, indent=2),
        encoding="utf-8",
    )
    print(output)
    print(json.dumps(correlations, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
