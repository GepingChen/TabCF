# TabCF

Publication-oriented code and artifacts for **"TabCF: Plug-and-play Distributional Causal Inference"**.

This repository is a double-blind review package for the paper-facing TabCF experiments. It keeps the reproducibility code and compact artifacts while excluding scheduler scripts, local caches, large generated output trees, and machine-specific workspace assumptions.

## Scope

- `tabcf_core/`: shared scalar-IV TabCF core used across paper-facing tasks.
- `interv_mean/`: Section 5.1 interventional-mean benchmark pipeline.
- `interv_qtl/`: interventional-quantile benchmark code and shipped comparison-report artifacts.
- `multivar/`: bivariate/joint-distribution extension used for the paper's multivariate response results.
- `empirical/`: four empirical tasks used in the paper:
  - AJR colonial origins
  - Fulton Fish Market
  - Card college proximity
  - CigarettesSW cigarette demand

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
