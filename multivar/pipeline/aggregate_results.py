#!/usr/bin/env python3
"""Aggregate per-run multivariate simulation summaries."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multivar.core.run_pipeline import _summarise_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate multivariate run-level CSV results.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("multivar/io/results/multivar_batch_results.csv"),
        help="Run-level results CSV path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("multivar/io/aggregated/multivar_grouped_summary.csv"),
        help="Grouped output CSV path.",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input}")

    df = pd.read_csv(args.input)
    grouped = _summarise_results(df)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(args.output, index=False)
    print(f"Loaded {len(df)} run rows from {args.input}")
    print(f"Wrote {len(grouped)} grouped rows to {args.output}")


if __name__ == "__main__":
    main()

