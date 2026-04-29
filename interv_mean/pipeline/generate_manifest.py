#!/usr/bin/env python3
"""Generate sec5.1 manifest under the new v2 schema."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from interv_mean.pipeline import utils


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate sec5.1 v2 manifest.")
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=utils.DEFAULT_MANIFEST_GENERATED_DIR / "tabpfn_runs_interv_mean.json",
        help="Output path for the manifest JSON.",
    )
    parser.add_argument(
        "--stage2-dir",
        type=Path,
        default=utils.DEFAULT_STAGE2_DIR,
        help="Directory containing TabPFN Stage2 summaries.",
    )
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=utils.DEFAULT_TRAIN_DIR,
        help="Directory containing TabPFN training CSVs.",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=utils.DEFAULT_TEST_DIR,
        help="Directory containing TabPFN test CSVs.",
    )
    parser.add_argument(
        "--bridge-dir",
        type=Path,
        default=utils.DEFAULT_BRIDGE_DIR,
        help="Default bridge directory stored in manifest meta.",
    )
    parser.add_argument(
        "--allow-missing-stage2",
        action="store_true",
        help="Do not fail if a Stage2 summary is missing; record null.",
    )
    parser.add_argument(
        "--codes",
        nargs="+",
        default=None,
        help="Optional subset of codes to include.",
    )
    parser.add_argument(
        "--train-sizes",
        nargs="+",
        type=int,
        default=utils.TRAIN_SIZES,
        help="Training sizes to include.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=utils.SEEDS,
        help="Seeds to include.",
    )
    args = parser.parse_args()

    runs = utils.build_runs(
        train_sizes=args.train_sizes,
        seeds=args.seeds,
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        stage2_dir=args.stage2_dir,
        require_stage2=not args.allow_missing_stage2,
        codes=args.codes,
    )
    utils.save_manifest(
        runs,
        args.manifest_path,
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        stage2_dir=args.stage2_dir,
        default_bridge_dir=args.bridge_dir,
    )

    print(f"Manifest written to {args.manifest_path}")
    print(f"Runs enumerated: {len(runs)}")


if __name__ == "__main__":
    main()
