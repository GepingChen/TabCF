# TabCF

Publication-oriented code and artifacts for **"TabCF: Distributional Control Function Estimation with Tabular Foundation Models"**.

This repository is a double-blind review package for the paper-facing TabCF experiments. It keeps the reproducibility code and compact artifacts while excluding scheduler scripts, local caches, large generated output trees, and machine-specific workspace assumptions.

The map below follows the current NeurIPS manuscript so reviewers can go directly from a paper section to the relevant code.

## Paper Section Map

| Paper location | What it covers | Repository folders and files |
| --- | --- | --- |
| Section 2, **TabCF framework** | Two-stage control-function estimator, predictive CDF construction, and plug-in interventional distributions. | `tabcf_core/` contains the shared DGP utilities, first-stage control construction, second-stage outcome models, foundation-model backends, and tests reused by the experiments. |
| Section 3, **Extending to multivariate responses** | Bivariate response extension using TabCF marginal estimators plus a Gaussian copula for the joint interventional law. | `multivar/core/` contains the DGPs, Gaussian-copula logic, and experiment runner; `multivar/pipeline/` aggregates and plots the official multivariate benchmark. |
| Section 4, **Synthetic Experiments** | Shared synthetic IV designs for scalar and bivariate evaluations. | `tabcf_core/dgp.py` and `tabcf_core/dgp_test_utils.py` hold reusable scalar DGP pieces; `interv_mean/`, `interv_qtl/`, and `multivar/` contain the section-specific benchmark pipelines. |
| Section 4.1, **Interventional means** | Mean-curve benchmark against linear IV, control-function, neural IV, and naive foundation-model baselines. | `interv_mean/` is the manifest-driven mean benchmark. Use `scripts/reproduce_mean.sh` to regenerate the shipped mean figure from `artifacts/aggregated_csv/interv_mean/mean_benchmark_results.csv`. |
| Section 4.2, **Interventional quantiles** | Quantile-curve benchmark against DIV and IVQR. | `interv_qtl/` contains the quantile pipeline, R baseline wrappers, and visualization code. Use `scripts/reproduce_quantile.sh` to regenerate the shipped quantile RMSE figure from `artifacts/aggregated_csv/interv_qtl/`. |
| Section 4.3, **Joint interventional distribution** | Bivariate response benchmark evaluated by sliced Wasserstein distance. | `multivar/` contains the joint-distribution implementation and aggregation code. Use `scripts/reproduce_multivar.sh` to regenerate the shipped Wasserstein figure from `artifacts/aggregated_csv/multivar/`. |
| Section 4.4, **Runtime analysis** | Runtime comparison for mean and quantile benchmarks. | Runtime helpers are `interv_mean/pipeline/benchmark_runtime.py` and `interv_qtl/benchmark_runtime.py`. Full timing runs are environment-dependent; see `docs/hpc.md` and `docs/advanced_recompute.md`. |
| Section 5, **Real Data Examples** | AJR, Fulton Fish, Card, and CigarettesSW empirical IV examples; Fulton Fish also has a quantile example. | `empirical/` contains real-data download/drop-in logic, TabCF runs, R baselines, and plotting. Use `scripts/reproduce_empirical.sh` to regenerate the shipped 2x2 empirical mean figure from `artifacts/aggregated_csv/empirical/`. Fulton quantile entrypoints are `empirical/run_empirical_quantile.py` and `empirical/run_empirical_quantile.sh`. |
| Appendices C-D | Benchmark-method notes and real-data application details. | Baseline implementations are under `interv_mean/pipeline/`, `interv_qtl/baselines/`, and `empirical/`; dataset notes are in `docs/data.md` and `empirical/README.md`. |

## Folder Summary

- `tabcf_core/`: shared scalar-IV TabCF core used across the paper-facing tasks.
- `interv_mean/`: Section 4.1 interventional-mean benchmark pipeline.
- `interv_qtl/`: Section 4.2 interventional-quantile benchmark code and shipped comparison-report artifacts.
- `multivar/`: Sections 3 and 4.3 bivariate/joint-distribution extension.
- `empirical/`: Section 5 real-data tasks:
  - AJR colonial origins
  - Fulton Fish Market
  - Card college proximity
  - CigarettesSW cigarette demand
- `artifacts/`: compact CSVs and paper-facing figures used by the lightweight reproduction scripts.
- `scripts/`: one-command refreshers for the main-text figures shipped with this review package.
- `docs/`: data, reproducibility, HPC, and full-recompute notes.

## Quick Start

Create a lightweight Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r environment/requirements-core.txt
```

If you want to run TabCF rather than only re-render shipped artifacts:

```bash
pip install -r environment/requirements-baselines.txt
```

## Data Policy

- `card`, `fulton`, and `cigarettes` can be downloaded by `empirical/run_empirical_mean.py`.
- `AJR` is **manual drop-in only**. Place the Stata file at `empirical/data/manual/ajr_colonial_origins.dta`, or pass `--data-path`.
- This review package does not vendor restricted datasets, access tokens, SLURM scripts, or local HPC paths.

## Reproduce The Paper-Facing Figures

The shipped artifacts live under `artifacts/`. The easiest entrypoint is:

```bash
bash scripts/reproduce_main_text.sh
```

Individual figure refreshers:

```bash
bash scripts/reproduce_mean.sh
bash scripts/reproduce_quantile.sh
bash scripts/reproduce_multivar.sh
bash scripts/reproduce_empirical.sh
```

These commands use the shipped CSV artifacts to regenerate publication-facing plots without re-running the heavy baseline pipelines from scratch.

Representative shipped artifact names:

- `artifacts/aggregated_csv/interv_mean/mean_benchmark_results.csv`
- `artifacts/aggregated_csv/multivar/multivar_wasserstein_curves.csv`
- `artifacts/aggregated_csv/interv_qtl/quantile_median_curve_comparison.csv`
- `artifacts/paper_figures/original/empirical/empirical_comparison_grid.pdf`

## Notes

- The repository keeps the original script-first layout on purpose.
- `tabcf_core` remains the shared core layer used by `interv_qtl`, `multivar`, and `empirical`.
- HPC and full recomputation notes are in [docs/hpc.md](docs/hpc.md) and [docs/advanced_recompute.md](docs/advanced_recompute.md).
