# Data Notes

## AJR

- Expected location: `empirical/data/manual/ajr_colonial_origins.dta`
- This repository does not redistribute the AJR Stata file.
- If you keep a local copy elsewhere, run:

```bash
python empirical/run_empirical_mean.py --dataset ajr --data-path /path/to/ajr_colonial_origins.dta
```

## Fulton Fish, Card, CigarettesSW

- These tasks have public upstream CSV sources.
- The real-data pipeline will download them automatically when `--data-path` is omitted.

## Shipped Artifacts

The repository includes paper-facing CSV artifacts under `artifacts/aggregated_csv/` so that plots can be regenerated without re-running the heavy estimators.

Examples:

- `artifacts/aggregated_csv/interv_mean/mean_benchmark_results.csv`
- `artifacts/aggregated_csv/multivar/multivar_wasserstein_curves.csv`
- `artifacts/aggregated_csv/interv_qtl/quantile_summary_metrics.csv`
- `artifacts/aggregated_csv/empirical/ajr/ajr_analysis_data.csv`
