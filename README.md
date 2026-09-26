# MESANet Reproducibility Package

This package reproduces the experiments and paper artifacts for:

> MESANet: Affine-Anchored Dual-Coordinate Forecasting for Wearable Irregular Time Series

All commands below are run from the package root. Historical result-directory version numbers are retained only to preserve provenance. They do not denote different submitted models unless the artifact map explicitly says so.

## 1. Package Contents

```text
main.py                         training/test entry point
models/                         MESANet and reproduced public baselines
data/                           wearable loaders, split and mask protocols
exp/, layers/, loss_fns/        training and model dependencies
utils/, configs/                CLI configuration and evaluation utilities
scripts/                        paper-facing runners, summarizers, and plotters
storage/results_*/_summary/     archived CSV/JSON evidence used by the paper
storage/datasets/README.md      dataset placement instructions
paper/mesanet_paper/           paper tables, figures, and LaTeX sources
```

The package includes result summaries and expanded command manifests, but not the 12.62 GB processed dataset cache or training checkpoints. Checkpoints were deleted after metric extraction; every retained result can be retrained with the commands below.

## 2. Environment

Reference environment:

- Python 3.10+
- PyTorch 2.6.0
- CUDA-capable GPU; the reported efficiency experiment used one RTX 4080
- 12 vCPU for the reported host configuration

Install dependencies:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Place the six converted datasets as described in `storage/datasets/README.md`.
The exact subject assignment and reviewed cache hashes are recorded there;
the corresponding frozen split manifest is retained in
`paper/mesanet_paper/generated/supp_table_split_manifest.tex`.

No activity label, subject identifier, future mask, or test target is provided as model input. Subject identifiers are used only to construct the split.

## 3. Frozen Submitted Model

The submitted model is `models/MESANet.py`. For historical experiment
compatibility, this public entry point subclasses the implementation container
in `models/CAGNet.py`; it does not load a pretrained or frozen CAGNet
forecaster. All MESANet modules are optimized jointly end to end. Its empty
`--ablation_name` selects the fixed forward path:

```text
aa_no_q_graph_no_prob_waveform_token_parallel_dct_dual_coordinate_
dct_dual_rho_half_dct_dual_no_causal
```

The deployed separation coefficient is fixed at `rho=0.5`. MSE is the primary endpoint and MAE is the secondary endpoint.

## 4. Core Experiment Commands

### C1. Shared validation search and descriptive benchmark

Produces the validation trials and run-5401 test rows used by Main Table 2 and Supplement Tables 3-4.

```bash
python scripts/run_mesa_equal_hpo.py \
  --datasets HumanActivity,MHEALTH,OPPORTUNITY,PAMAP2,USCHAD,REALDISP \
  --models APN,KAFNet,GraFITi,tPatchGNN,PatchTST,MESANet \
  --search_epochs 20 --confirm_epochs 20 --patience 5 \
  --confirm_seeds 5401 --mae_weight 0.5 \
  --wearable_split_protocol subject_heldout \
  --wearable_mask_protocol real_missing_only_if_available \
  --mesa_ablation aa_no_q_graph_no_prob_waveform_token_parallel_dct_dual_coordinate_dct_dual_rho_half_dct_dual_no_causal \
  --root storage/results_mesa_equal_hpo \
  --prefix mesa_equal_hpo
python scripts/summarize_mesa_equal_hpo.py \
  --root storage/results_mesa_equal_hpo
```

Archived reviewed sources are under:

- `storage/results_mesa_equal_hpo/_summary/`
- `storage/results_mesa_v661_soft_no_causal_equal_hpo/_summary/`
- `storage/results_mesa_v663_boundary_equal_hpo/_summary/`
- `storage/results_mesa_v664_uschad_public_equal_hpo/_summary/`
- `storage/results_mesa_v664_realdisp_public_equal_hpo/_summary/`

### C2. Validation-frozen five-run comparison

Produces Main Table 3 and Supplement Tables 5-7. One public comparator is frozen per dataset by validation MSE, with validation MAE used only as a tie breaker. The same comparator is used for both test metrics.

```bash
python scripts/run_mesa_v668_frontier_repeated.py \
  --seeds 5402,5403,5404,5405 --epochs 20 --patience 6 \
  --root storage/results_mesa_v668_frontier_repeated \
  --prefix mesa_v668_frontier
python scripts/run_mesa_v671_final_revision.py \
  --variants fixed_rho_half --parallel 4 --epochs 20 --patience 6 \
  --reuse_existing_fixed 0 \
  --root storage/results_mesa_v671_final_revision \
  --prefix mesa_v671_final
python scripts/summarize_mesa_v671_validation_frozen.py
```

Reviewed outputs are in `storage/results_mesa_v671_validation_frozen/_summary/`. The exact LaTeX table is `paper/mesanet_paper/generated/main_table_v671_validation_frozen.tex`.

### C3. Fixed-model component and equal-capacity controls

Produces Main Table 4 and Supplement Table 10.

```bash
# Dual-coordinate, single-coordinate, and waveform controls.
python scripts/run_mesa_v667_final_evidence.py \
  --datasets HumanActivity,MHEALTH,OPPORTUNITY,PAMAP2 \
  --variants fixed_rho_half,low_subspace_only_param_matched,no_waveform_coordinate \
  --seeds 5401,5402,5403 --epochs 20 --patience 6 \
  --parallel 4 --reuse_prior_rho_run 0 \
  --root storage/results_mesa_component_core_reproduced

# Shared-transfer and random-orthobasis parameter-matched controls.
python scripts/run_mesa_v671_final_revision.py \
  --variants fixed_rho_half,shared_transfer_matched,random_basis \
  --parallel 4 --epochs 20 --patience 6 --reuse_existing_fixed 0 \
  --root storage/results_mesa_capacity_controls_reproduced \
  --prefix mesa_capacity_controls_reproduced

# Exact fixed-rho multiresolution and affine-anchor removals.
python scripts/run_mesa_v671_final_revision.py \
  --variants fixed_rho_half,no_multiresolution_fixed,no_affine_anchor_fixed \
  --parallel 4 --epochs 20 --patience 6 --reuse_existing_fixed 0 \
  --root storage/results_mesa_fixed_components_reproduced \
  --prefix mesa_fixed_components_reproduced

python scripts/summarize_mesa_v671_controls.py \
  --input storage/results_mesa_capacity_controls_reproduced/_summary/final_revision_runs.csv \
  --output storage/results_mesa_capacity_controls_reproduced/_summary
```

Reviewed table inputs are:

- `storage/results_mesa_v667_final_ablation/_summary/final_evidence_runs.csv`
- `storage/results_mesa_v671_final_revision/_summary/final_revision_runs.csv`
- `storage/results_mesa_v674_fixed_component_ablation/_summary/final_revision_runs.csv`

The assembled reviewed LaTeX table is `paper/mesanet_paper/generated/main_table_v671_final_controls.tex`.

### C4. Strict leave-one-subject-out stress test

Produces the LOSO rows in Main Table 5 and Supplement Table 8.

```bash
python scripts/run_mesa_v672_final_loso.py \
  --datasets MHEALTH,USCHAD --run_id 5401 \
  --epochs 8 --patience 2 --fold_parallel 4
python scripts/summarize_mesa_v672_final_loso.py
```

The summarizer pairs each MESANet fold with the predeclared archived run-5401
PatchTST rows in `storage/results_mesa_strict_loso/`. The package includes those
comparator rows and the exact commands that produced the MESANet side. It does
not include a single wrapper that retrains both LOSO sides from an empty result
tree. Outputs are
`storage/results_mesa_v672_final_loso/_summary/subject_detail.csv` and
`subject_aggregate.csv`.

### C5. Observation-mask stress test

Produces the mask rows in Main Table 6 and Supplement Table A10. HumanActivity
is excluded because its loader does not implement the mask intervention.

```bash
python scripts/run_mesa_v673_mask_robustness.py \
  --run 5401 --epochs 8 --patience 2 --parallel 4
python scripts/summarize_mesa_v673_mask_robustness.py
```

Reviewed outputs are in `storage/results_mesa_v673_mask_robustness/_summary/`. This is a one-run stress test, not the repeated main benchmark.

### C6. Parameter sensitivity

Produces Main Figure 3 and Supplement Figure A1:

```bash
python scripts/run_mesa_v675_method_sensitivity.py \
  --root storage/v675 --search_seed 5399 \
  --search_epochs 20 --patience 6 --parallel 3
python scripts/plot_mesa_v675_method_sensitivity.py
```

This sweep varies the dual-coordinate overlap $\rho$ and causal support count
$K$ while keeping the final backbone and validation protocol fixed. Its input
CSV is `storage/v675/_summary/method_sensitivity.csv`.

### C7. Efficiency profile

Produces the main efficiency table and the expanded supplementary profile.

```bash
python scripts/run_release_v553_mesa_efficiency.py \
  --dataset MHEALTH \
  --models APN,KAFNet,tPatchGNN,GraFITi,PatchTST,MESANet \
  --batch_size 4 --run_seed 5401 --epochs 4 --patience 2 \
  --mesa_variant mesanet_fixed_dual_coordinate \
  --mesa_width_multiplier 1.5 \
  --wearable_mask_protocol real_missing_only_if_available \
  --root storage/results_mesa_v666_efficiency \
  --summary_name mesa_v666_efficiency_b4.csv \
  --figure_name mesa_fig4_v665_efficiency
```

Input CSV: `storage/results_mesa_v666_efficiency/_summary/mesa_v666_efficiency_b4.csv`.

## 5. Main-Paper Figure and Table Map

| Item | Paper content | Data/source | Reproduction command |
|---|---|---|---|
| Figure 1 | Adaptive support versus multiscale evidence motivation | Conceptual figure; no experimental data | Edit `paper/mesanet_paper/figures/final/mesa_fig1_apn_like_v540.svg`; export to PDF/PNG with Inkscape or draw.io |
| Figure 2 | MESANet framework | Architecture defined in `models/MESANet.py`; no measured data | Edit/export `paper/mesanet_paper/figures/final/mesa_fig2_v676.drawio` |
| Figure 3 | DCT-overlap and support-count sensitivity | `storage/v675/_summary/method_sensitivity.csv` | C6 |
| Table 1 | Dataset dimensions and forecasting setup | Dataset loaders, `generated/main_table_v665_datasets.tex`, and the frozen subject manifest | Inspect the retained table and `storage/datasets/README.md`; retraining commands C1-C5 use the same registry |
| Table 2 | Full shared-search benchmark matrix | Equal-HPO summaries listed under C1 | C1 |
| Table 3 | Validation-frozen five-run comparison | `results_mesa_v671_validation_frozen/_summary/` | C2 |
| Table 4 | Three-run fixed-model controls | v667, v671 and v674 run CSVs | C3 |
| Table 5 | Efficiency: parameters, MACs, memory and latency | `results_mesa_v666_efficiency/_summary/mesa_v666_efficiency_b4.csv` | C7 |
| Table 6 | LOSO and mask stress summary | v672 subject aggregate and the nine valid v673 mask cells | C4 and C5 |

## 6. Supplement Figure and Table Map

| Item | Content | Source/command |
|---|---|---|
| Table A1 | Dataset dimensions | Dataset registry and retained `generated/main_table_v665_datasets.tex` |
| Table A2 | Released-subject split manifest | `generated/supp_table_split_manifest.tex` |
| Table A3 | Shared optimization and MESANet settings | CLI defaults plus the frozen configuration JSONs produced by C1 |
| Table A4 | Validation-selected trial per method/dataset | `equal_hpo_frozen_configs.json` from C1 |
| Table A5 | Complete validation-search matrix | C1 |
| Table A6 | Five-run validation-frozen comparison | C2 |
| Table A7 | Frozen comparator manifest | `validation_frozen_comparators.csv/json` generated by C2 |
| Table A8 | Paired stability intervals | `validation_frozen_five_run.csv` generated by C2 |
| Table A9 | Strict LOSO | C4 |
| Table A10 | Mask-protocol stress | C5 |
| Table A11 | Fixed-model component controls | C3 |
| Table A12 | Observable dual-coordinate proxies | Retained `dual_coordinate_diagnostic.csv` produced by the analysis command after a saved-array run |
| Table A13 | Exact method-sensitivity values | C6 |
| Table A14 | Exact efficiency values | C7 |
| Figure A1 | DCT-overlap and support-count sensitivity | C6 |
| Figure A2 | Expanded efficiency profile | C7; values are also preserved in `generated/main_table_v665_efficiency.tex` |

For Supplement Table A12, the package retains the reviewed derived CSV in
`storage/results_mesa_v671_validation_frozen/_summary/dual_coordinate_diagnostic.csv`.
Per-example arrays were not retained, so the archived CSV reproduces the table
but does not permit an independent recomputation of its per-example projections.

## 7. Archived-Result Regeneration Versus Full Retraining

To regenerate plots and summaries from the included archived CSV/JSON files without training:

```bash
python scripts/summarize_mesa_v671_validation_frozen.py
python scripts/summarize_mesa_v671_controls.py
python scripts/summarize_mesa_v672_final_loso.py
python scripts/summarize_mesa_v673_mask_robustness.py
python scripts/plot_mesa_v675_method_sensitivity.py
python scripts/run_release_v553_mesa_efficiency.py \
  --dataset MHEALTH --models APN,KAFNet,tPatchGNN,GraFITi,PatchTST,MESANet \
  --batch_size 4 --summary_only \
  --root storage/results_mesa_v666_efficiency \
  --summary_name mesa_v666_efficiency_b4.csv \
  --figure_name mesa_fig4_v665_efficiency
```

Full retraining requires the processed datasets and a CUDA GPU. Runners skip completed metric rows by default; use fresh `--root` paths for a clean replication.

## 8. Paper Build

```bash
cd paper/mesanet_paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error supplement.tex
```

## 9. Integrity and Scope Notes

- Main Table 3 reports five optimization runs. These runs quantify optimization stability, not uncertainty over unseen subjects.
- Subject transfer is evaluated separately by the 24-fold strict-LOSO stress test.
- Main Table 6 mask results are an eight-epoch, one-run protocol stress test on
  MHEALTH, OPPORTUNITY, and PAMAP2.
- HumanActivity MSE and both USC-HAD metrics are retained as negative main-benchmark evidence.
- The shared-transfer control is nearly tied and is not claimed as a source of improvement.
- Server addresses, passwords, private keys, and machine-specific absolute paths are excluded from this package.
