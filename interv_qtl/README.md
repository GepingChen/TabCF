# interv_qtl

Paper-facing interventional-quantile evaluation for TabCF.

This review package keeps the quantile pipeline code plus compact report CSVs under:

`artifacts/aggregated_csv/interv_qtl/`

The shipped report artifacts can be turned back into the public RMSE figure with:

```bash
bash scripts/reproduce_quantile.sh
```

Key shipped filenames:

- `artifacts/aggregated_csv/interv_qtl/quantile_median_curve_comparison.csv`
- `artifacts/aggregated_csv/interv_qtl/quantile_rmse_by_quantile_level.csv`
- `artifacts/paper_figures/original/interv_qtl/quantile_rmse_figure.pdf`

Heavyweight Stage 1 / Stage 2 output directories and scheduler scripts are intentionally not part of this review package.
