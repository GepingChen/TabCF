# multivar

Bivariate response / joint-distribution extension of TabCF used in the paper's multivariate results.

The public release keeps:

- `core/`: Gaussian dependence model and marginal-bridge logic;
- `pipeline/`: aggregation and official visualization scripts;
- `tests/`: lightweight checks for the shipped code;
- shipped paper-facing aggregation CSVs under `artifacts/aggregated_csv/multivar/`, including `multivar_wasserstein_curves.csv` and `multivar_wasserstein_summary.csv`.

Refresh the official Wasserstein figure with:

```bash
bash scripts/reproduce_multivar.sh
```
