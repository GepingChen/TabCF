#!/usr/bin/env python3
"""Compare Stage 2 summary metrics across two V-integration implementations."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Dict, Iterable, List


REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_DIR = REPO_ROOT / "tabcf_core"
PIPELINE_DIR = REPO_ROOT / "interv_mean" / "pipeline"
for candidate in (CORE_DIR, PIPELINE_DIR, REPO_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from tabcf_core.foundation_backends import stage2_base_prefix
from interv_mean.pipeline import utils


DEFAULT_METRICS = (
    "metric_mse_do_pred_vs_clean",
    "metric_iae_mean",
    "metric_iae_max",
)


def _summary_path(stage2_dir: Path, *, code: str, train_size: int, seed: int, backend: str) -> Path:
    prefix = stage2_base_prefix(
        code,
        train_sample_size=int(train_size),
        seed=int(seed),
        backend_name=backend,
    )
    return stage2_dir / f"{prefix}_summary.csv"


def _read_summary_kv(summary_path: Path) -> Dict[str, str]:
    with summary_path.open(newline="") as handle:
        return {str(row["key"]): str(row["value"]) for row in csv.DictReader(handle)}


def _parse_metric(summary: Dict[str, str], metric: str) -> float:
    if metric not in summary:
        raise KeyError(f"{metric} not found in summary.")
    return float(summary[metric])


def build_comparison_rows(
    *,
    reference_stage2_dir: Path,
    candidate_stage2_dir: Path,
    codes: Iterable[str],
    train_sizes: Iterable[int],
    seeds: Iterable[int],
    backend: str,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for code in codes:
        for train_size in train_sizes:
            for seed in seeds:
                reference_summary_path = _summary_path(
                    reference_stage2_dir,
                    code=str(code),
                    train_size=int(train_size),
                    seed=int(seed),
                    backend=backend,
                )
                candidate_summary_path = _summary_path(
                    candidate_stage2_dir,
                    code=str(code),
                    train_size=int(train_size),
                    seed=int(seed),
                    backend=backend,
                )
                if not reference_summary_path.exists():
                    raise FileNotFoundError(f"Missing reference summary: {reference_summary_path}")
                if not candidate_summary_path.exists():
                    raise FileNotFoundError(f"Missing candidate summary: {candidate_summary_path}")

                reference_summary = _read_summary_kv(reference_summary_path)
                candidate_summary = _read_summary_kv(candidate_summary_path)
                row: Dict[str, object] = {
                    "code": str(code),
                    "scenario": utils.CODE_SCENARIO_MAP[str(code)]["scenario"],
                    "train_size": int(train_size),
                    "seed": int(seed),
                    "backend": backend,
                    "reference_stage2_summary": str(reference_summary_path),
                    "candidate_stage2_summary": str(candidate_summary_path),
                    "reference_mu_integrator": reference_summary.get("mu_integrator", "simpson"),
                    "candidate_mu_integrator": candidate_summary.get("mu_integrator", "simpson"),
                    "candidate_gauss_legendre_order": candidate_summary.get("gauss_legendre_order", ""),
                }
                for metric in DEFAULT_METRICS:
                    reference_value = _parse_metric(reference_summary, metric)
                    candidate_value = _parse_metric(candidate_summary, metric)
                    row[f"reference_{metric}"] = reference_value
                    row[f"candidate_{metric}"] = candidate_value
                    row[f"delta_{metric}"] = candidate_value - reference_value
                rows.append(row)

    rows.sort(key=lambda item: (str(item["code"]), int(item["train_size"]), int(item["seed"])))
    return rows


def aggregate_comparison_rows(rows: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[tuple[str, int, str], List[Dict[str, object]]] = {}
    for row in rows:
        key = (str(row["code"]), int(row["train_size"]), str(row["backend"]))
        grouped.setdefault(key, []).append(dict(row))

    aggregated_rows: List[Dict[str, object]] = []
    for (code, train_size, backend), group_rows in sorted(grouped.items()):
        aggregate_row: Dict[str, object] = {
            "code": code,
            "scenario": group_rows[0]["scenario"],
            "train_size": int(train_size),
            "backend": backend,
            "n_runs": len(group_rows),
            "reference_mu_integrator": group_rows[0]["reference_mu_integrator"],
            "candidate_mu_integrator": group_rows[0]["candidate_mu_integrator"],
            "candidate_gauss_legendre_order": group_rows[0]["candidate_gauss_legendre_order"],
        }
        for metric in DEFAULT_METRICS:
            reference_key = f"reference_{metric}"
            candidate_key = f"candidate_{metric}"
            delta_key = f"delta_{metric}"
            reference_mean = sum(float(row[reference_key]) for row in group_rows) / len(group_rows)
            candidate_mean = sum(float(row[candidate_key]) for row in group_rows) / len(group_rows)
            delta_mean = sum(float(row[delta_key]) for row in group_rows) / len(group_rows)
            aggregate_row[f"mean_{reference_key}"] = reference_mean
            aggregate_row[f"mean_{candidate_key}"] = candidate_mean
            aggregate_row[f"mean_{delta_key}"] = delta_mean
        aggregated_rows.append(aggregate_row)
    return aggregated_rows


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Stage 2 summary metrics across two integration methods.")
    parser.add_argument("--reference-stage2-dir", type=Path, required=True, help="Directory with reference Simpson summaries.")
    parser.add_argument("--candidate-stage2-dir", type=Path, required=True, help="Directory with candidate summaries.")
    parser.add_argument("--codes", nargs="+", required=True, help="DGP codes to compare.")
    parser.add_argument("--train-sizes", nargs="+", type=int, required=True, help="Training sizes to compare.")
    parser.add_argument("--seeds", nargs="+", type=int, required=True, help="Seeds to compare.")
    parser.add_argument("--backend", type=str, required=True, help="Backend suffix used in summary filenames.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for comparison CSVs. Defaults to <candidate-stage2-dir>/comparison.",
    )
    parser.add_argument(
        "--output-stem",
        type=str,
        default=None,
        help="Optional filename stem. Defaults to '<backend>_integrator_compare'.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (Path(args.candidate_stage2_dir) / "comparison")
    output_stem = args.output_stem or f"{args.backend}_integrator_compare"

    per_run_rows = build_comparison_rows(
        reference_stage2_dir=Path(args.reference_stage2_dir),
        candidate_stage2_dir=Path(args.candidate_stage2_dir),
        codes=[str(code).upper() for code in args.codes],
        train_sizes=[int(size) for size in args.train_sizes],
        seeds=[int(seed) for seed in args.seeds],
        backend=str(args.backend),
    )
    aggregated_rows = aggregate_comparison_rows(per_run_rows)

    per_run_path = output_dir / f"{output_stem}_per_run.csv"
    aggregated_path = output_dir / f"{output_stem}_aggregated.csv"
    _write_csv(per_run_path, per_run_rows)
    _write_csv(aggregated_path, aggregated_rows)

    print(f"Saved per-run comparison CSV to {per_run_path}")
    print(f"Saved aggregated comparison CSV to {aggregated_path}")


if __name__ == "__main__":
    main()
