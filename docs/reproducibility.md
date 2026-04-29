# Reproducibility

This public release is intentionally **paper-facing**.

- It ships curated CSV artifacts and official plots under `artifacts/`.
- The default reproduction scripts regenerate the publication-facing figures from those shipped artifacts.
- Full baseline recomputation is still possible, but it is documented separately because it needs larger environments and, in some cases, manual data placement.

Recommended entrypoints:

```bash
bash scripts/reproduce_main_text.sh
bash scripts/reproduce_mean.sh
bash scripts/reproduce_quantile.sh
bash scripts/reproduce_multivar.sh
bash scripts/reproduce_empirical.sh
```

Expected reproduced outputs land under `artifacts/paper_figures/reproduced/` with publication-facing names such as:

- `mean_benchmark_figure.png`
- `multivar_wasserstein_figure.png`
- `quantile_rmse_figure.pdf`
- `empirical_comparison_grid.pdf`
