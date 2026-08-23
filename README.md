# TabCF

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](environment/requirements-core.txt)
[![arXiv](https://img.shields.io/badge/arXiv-2605.05993-b31b1b.svg)](https://arxiv.org/abs/2605.05993)

This repository contains the implementation code for the paper **"TabCF: Distributional Control Function Estimation with Tabular Foundation Models"** ([arXiv:2605.05993](https://arxiv.org/abs/2605.05993)).

The map below links each paper section to the relevant code.

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
- `scripts/`: one-command refreshers for the shipped main-text figures.
- `docs/`: data, reproducibility, HPC, and full-recompute notes.

## Quick Start

Requires **Python 3.9+** (tested with 3.9.21).

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

The DIV and IVQR baselines used in Section 4.2 and Section 5 are R scripts. If you want to
re-run those baselines rather than only using the shipped comparison CSVs, install R (tested
with R 4.4.3) and the packages listed in `environment/requirements-r.txt`:

```bash
grep -v '^#' environment/requirements-r.txt | Rscript -e 'install.packages(readLines("stdin"))'
```

### Running the tests

Each module ships lightweight unit tests under its own `tests/` directory. After installing
`environment/requirements-core.txt`, verify your setup with:

```bash
pytest tabcf_core interv_mean interv_qtl multivar empirical
```

## Data Policy

- `card`, `fulton`, and `cigarettes` can be downloaded by `empirical/run_empirical_mean.py`.
- `AJR` is **manual drop-in only**. Place the Stata file at `empirical/data/manual/ajr_colonial_origins.dta`, or pass `--data-path`.

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

- `tabcf_core` is the shared core layer used by `interv_qtl`, `multivar`, and `empirical`.
- HPC and full recomputation notes are in [docs/hpc.md](docs/hpc.md) and [docs/advanced_recompute.md](docs/advanced_recompute.md).

## Related Projects

- [DCFA](https://github.com/GepingChen/DCFA) — an agentic implementation of TabCF, currently under
  active development.

## Citation

If you use this repository, please cite the paper:

```bibtex
@article{chen2026tabcf,
  title   = {TabCF: Distributional Control Function Estimation with Tabular Foundation Models},
  author  = {Chen, Geping and Li, Chunlin and Yang, Tianzhong and Zhu, Zhengyuan and Zhou, Jing},
  journal = {arXiv preprint arXiv:2605.05993},
  year    = {2026}
}
```

## License

This project is released under the [MIT License](LICENSE).
