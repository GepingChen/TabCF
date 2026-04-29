#!/usr/bin/env python3
"""
Run a naive TabPFN baseline on bridged sec5.1 datasets.
This baseline ignores the IV/endogeneity setting and fits Y ~ X directly.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from interv_mean.pipeline import utils


def set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except Exception:
        pass
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except Exception:
        pass


def load_bridged_csvs(train_path: Path, test_path: Path):
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required to load bridged CSVs") from exc

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    for col in ("X", "Y", "Z"):
        if col not in train_df.columns:
            raise ValueError(f"{train_path} missing column {col}")
    for col in ("Xint", "mean_int"):
        if col not in test_df.columns:
            raise ValueError(f"{test_path} missing column {col}")
    return train_df, test_df


def _tabpfn_naive_predictor(train_df, seed: int):
    os.environ.setdefault("TABPFN_MODEL_CACHE_DIR", str(REPO_ROOT / "tabpfn_home_config" / "models"))
    os.environ.setdefault("TABPFN_MODEL_VERSION", os.environ.get("TABPFN_MODEL_VERSION", "v2.5"))
    os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")
    try:
        from importlib.metadata import version as pkg_version
        from packaging.version import Version

        skl_version = Version(pkg_version("scikit-learn"))
        if skl_version < Version("1.3"):
            raise RuntimeError(
                f"tabpfn_naive requires scikit-learn>=1.3 for TabPFN preprocessing; found {skl_version}."
            )
    except Exception:
        # If metadata lookup fails, defer to import errors below.
        pass
    try:
        from tabpfn.regressor import TabPFNRegressor
    except Exception:
        try:
            from tabpfn import TabPFNRegressor  # type: ignore[attr-defined]
        except Exception as exc:
            raise RuntimeError("TabPFNRegressor is required for the tabpfn_naive baseline.") from exc

    set_global_seed(seed)
    x = train_df["X"].to_numpy(dtype=np.float32).reshape(-1, 1)
    y = train_df["Y"].to_numpy(dtype=np.float32)

    model = TabPFNRegressor(random_state=seed, ignore_pretraining_limits=True)
    model.fit(x, y)

    def _predict(x_values: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x_values, dtype=np.float32).reshape(-1, 1)
        preds = model.predict(x_arr)
        return np.asarray(preds, dtype=float).reshape(-1)

    return _predict, "tabpfn_naive_y_on_x"


def compute_mse(pred: np.ndarray, target: np.ndarray) -> float:
    pred_arr = np.asarray(pred, dtype=float).reshape(-1)
    target_arr = np.asarray(target, dtype=float).reshape(-1)
    if pred_arr.shape != target_arr.shape:
        raise ValueError(f"Shape mismatch for MSE: pred {pred_arr.shape}, target {target_arr.shape}")
    return float(np.mean((pred_arr - target_arr) ** 2))


def save_metrics(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _should_skip_run(metrics_path: Path, overwrite: bool) -> bool:
    if not metrics_path.exists():
        return False
    if overwrite:
        return False
    try:
        json.loads(metrics_path.read_text())
    except Exception as exc:
        print(f"   Existing metrics at {metrics_path} unreadable ({exc}); recomputing.")
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run naive TabPFN baseline (Y~X) on bridged sec5.1 datasets."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=utils.DEFAULT_MANIFEST_PATH,
        help="Manifest produced by manifest_manager.py / generate_manifest.py.",
    )
    parser.add_argument(
        "--bridge-dir",
        type=Path,
        default=utils.DEFAULT_BRIDGE_DIR,
        help="Directory containing bridged DIV-style CSVs.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=utils.DEFAULT_RESULTS_DIR,
        help="Directory to store baseline metrics/predictions.",
    )
    parser.add_argument(
        "--codes",
        nargs="+",
        default=None,
        help="Optional subset of DGP codes.",
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=None,
        help="Optional subset of train sizes.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Optional subset of seeds.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute even if metrics already exist.",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Persist per-sample predictions alongside metrics.",
    )
    args = parser.parse_args()

    manifest = utils.load_manifest(args.manifest)
    runs: List[Dict[str, object]] = manifest.get("runs", [])
    if not runs:
        raise SystemExit("Manifest contained no runs.")

    def _run_filter(run: Dict[str, object]) -> bool:
        return (
            (not args.codes or run["code"] in args.codes)
            and (not args.sizes or int(run["train_size"]) in args.sizes)
            and (not args.seeds or int(run["seed"]) in args.seeds)
        )

    runs = [r for r in runs if _run_filter(r)]
    if not runs:
        raise SystemExit("No runs matched the provided filters.")

    for idx, run in enumerate(runs, start=1):
        code = run["code"]
        size = int(run["train_size"])
        seed = int(run["seed"])
        resolved_paths = utils.resolve_run_paths(run, manifest=manifest, bridge_dir=args.bridge_dir)
        bridge_train = resolved_paths["bridge_train"]
        bridge_test = resolved_paths["bridge_test"]
        assert bridge_train is not None and bridge_test is not None
        if not bridge_train.exists() or not bridge_test.exists():
            raise FileNotFoundError(f"Missing bridged CSVs for {code}, n={size}, seed={seed}")
        train_df, test_df = load_bridged_csvs(bridge_train, bridge_test)
        x_test = test_df["Xint"].to_numpy()
        target = test_df["mean_int"].to_numpy()

        model_name = "tabpfn_naive"
        metrics_path = args.results_dir / f"{model_name}_{code}_n{size}_seed{seed}.json"
        preds_path = args.results_dir / f"{model_name}_{code}_n{size}_seed{seed}_pred.csv"
        if _should_skip_run(metrics_path, args.overwrite):
            print(f"[{idx}/{len(runs)}] {model_name} {code} n={size} seed={seed} already done, skipping.")
            continue
        if metrics_path.exists():
            print(f"[{idx}/{len(runs)}] Overwriting {metrics_path}")

        print(f"[{idx}/{len(runs)}] Running {model_name} on {code}, n={size}, seed={seed}")
        model_start = time.time()
        predictor, variant = _tabpfn_naive_predictor(train_df, seed=seed)
        preds = predictor(x_test)

        mse = compute_mse(preds, target)

        payload: Dict[str, object] = {
            "model": model_name,
            "variant": variant,
            "code": code,
            "scenario": run["scenario"],
            "train_size": size,
            "seed": seed,
            "bridge_train": str(bridge_train),
            "bridge_test": str(bridge_test),
            "mse_vs_mean_int": mse,
            "n_train": len(train_df),
            "n_test": len(test_df),
        }

        if args.save_predictions:
            preds_path.parent.mkdir(parents=True, exist_ok=True)
            pred_rows = np.column_stack([x_test, target, preds])
            header = "Xint,mean_int,pred"
            np.savetxt(preds_path, pred_rows, delimiter=",", header=header, comments="")
            payload["pred_path"] = str(preds_path)

        save_metrics(metrics_path, payload)
        elapsed = time.time() - model_start
        print(
            f"   Done {model_name} {code} n={size} seed={seed} in {elapsed:.2f}s "
            f"(saved to {metrics_path.name}) at {time.strftime('%Y-%m-%d %H:%M:%S')}"
        )

    print("Naive TabPFN baseline runs completed.")


if __name__ == "__main__":
    main()
