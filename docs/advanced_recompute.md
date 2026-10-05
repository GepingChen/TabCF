# Advanced Recompute

This repository keeps the public, reviewable core of the TabCF paper. Full recomputation is heavier than the default artifact-based refresh path.

## Mean benchmark

Use the manifest-driven benchmark pipeline under `interv_mean/pipeline/`.

## Quantile benchmark

The full quantile run depends on `interv_qtl/run_interv_quantile.py`, the R baseline wrappers, and larger Stage 1 / Stage 2 outputs than the shipped report CSVs.

## Copula benchmark

Use `multivar/core/run_pipeline.py`, then aggregate with `multivar/pipeline/evaluate_wasserstein.py`.

## Real data

- `card`, `fulton`, and `cigarettes` can be redownloaded.
- `AJR` remains manual drop-in.

This repository does not ship scheduler scripts, local virtualenv/module commands, or machine-specific runtime settings, since those are specific to the original compute environment. Wrap the CLI entrypoints above in your own site-local scheduler templates as needed.
