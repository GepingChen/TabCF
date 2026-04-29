#!/usr/bin/env python3
"""
Run Python baselines (DeepIV, DeepGMM, HSIC-X, TabPFN control-function) on bridged sec5.1 datasets.
"""

from __future__ import annotations

import argparse
import os
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEEPGMM_ROOT = REPO_ROOT / "DIV-main" / "DeepGMM-master"
if DEEPGMM_ROOT.exists() and str(DEEPGMM_ROOT) not in sys.path:
    sys.path.insert(0, str(DEEPGMM_ROOT))

HSICX_ROOT = REPO_ROOT / "DIV-main" / "HSIC-X-master"
if HSICX_ROOT.exists() and str(HSICX_ROOT) not in sys.path:
    # Do not shadow DeepGMM's modules (both repos define `models/`); keep HSIC later.
    sys.path.append(str(HSICX_ROOT))

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


def _deepiv_predictor(train_df, seed: int, n_restarts: int = 10):
    try:
        from econml.iv.nnet import DeepIV
        import tensorflow as tf
        from tensorflow import keras
    except ImportError as exc:
        raise RuntimeError(
            "DeepIV dependencies missing. Install tensorflow/keras and econml to run DeepIV."
        ) from exc

    set_global_seed(seed)
    z = train_df["Z"].to_numpy().reshape(-1, 1).astype(np.float32)
    t = train_df["X"].to_numpy().reshape(-1, 1).astype(np.float32)
    y = train_df["Y"].to_numpy().astype(np.float32)
    x_context = np.zeros_like(t)

    treatment_model = keras.Sequential(
        [
            keras.layers.Dense(256, activation="relu", input_shape=(2,)),
            keras.layers.Dropout(0.17),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dropout(0.17),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dropout(0.17),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dropout(0.17),
        ]
    )
    response_model = keras.Sequential(
        [
            keras.layers.Dense(256, activation="relu", input_shape=(2,)),
            keras.layers.Dropout(0.17),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dropout(0.17),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dropout(0.17),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dropout(0.17),
            keras.layers.Dense(1),
        ]
    )
    keras_fit_options = {
        "epochs": 30,
        "validation_split": 0.2,
        "callbacks": [keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)],
    }

    # Train multiple restarts (as in DIV sec5.1 notebooks) and average predictions.
    models = []
    for restart in range(max(1, n_restarts)):
        set_global_seed(seed + restart)
        deepiv = DeepIV(
            n_components=10,
            m=lambda z_in, x_in: treatment_model(keras.layers.concatenate([z_in, x_in])),
            h=lambda t_in, x_in: response_model(keras.layers.concatenate([t_in, x_in])),
            n_samples=1,
            use_upper_bound_loss=False,
            n_gradient_samples=1,
            optimizer="adam",
            first_stage_options=keras_fit_options,
            second_stage_options=keras_fit_options,
        )
        deepiv.fit(Y=y, T=t, X=x_context, Z=z)
        models.append(deepiv)

    def _predict(x_values: np.ndarray) -> np.ndarray:
        x_vals = np.asarray(x_values, dtype=np.float32).reshape(-1, 1)
        ctx = np.zeros_like(x_vals)
        preds = []
        for mdl in models:
            preds.append(mdl.predict(x_vals, ctx))
        mean_preds = np.mean(np.stack(preds, axis=0), axis=0)
        return np.asarray(mean_preds, dtype=float).reshape(-1)

    return _predict, f"deepiv_restarts{len(models)}"


def _linear_iv_fallback(train_df) -> Tuple[np.ndarray, np.ndarray]:
    """Simple 2SLS fallback used when DeepGMM dependencies are unavailable."""
    z = train_df["Z"].to_numpy(dtype=float)
    x = train_df["X"].to_numpy(dtype=float)
    y = train_df["Y"].to_numpy(dtype=float)
    z_design = np.column_stack([np.ones_like(z), z])
    beta_stage1, *_ = np.linalg.lstsq(z_design, x, rcond=None)
    x_hat = z_design @ beta_stage1
    x_design = np.column_stack([np.ones_like(x_hat), x_hat])
    beta_stage2, *_ = np.linalg.lstsq(x_design, y, rcond=None)
    return beta_stage1, beta_stage2


def _linear_iv_predictor(train_df):
    beta_stage1, beta_stage2 = _linear_iv_fallback(train_df)

    def _predict(x_values: np.ndarray) -> np.ndarray:
        x_vals = np.asarray(x_values, dtype=float)
        x_design = np.column_stack([np.ones_like(x_vals), x_vals])
        return x_design @ beta_stage2

    return _predict, "linear_iv_fallback"


def _deepgmm_predictor(train_df, val_fraction: float, seed: int):
    """
    Run DeepGMM (ToyModelSelectionMethod) on the bridged data.
    Mirrors the sec5.1 notebook: split train/val, fit on doubles, then predict.
    """
    if not DEEPGMM_ROOT.exists():
        raise ImportError(f"DeepGMM source tree not found at {DEEPGMM_ROOT}")
    try:
        # pylint: disable=import-error
        from methods.toy_model_selection_method import ToyModelSelectionMethod
    except Exception as exc:
        raise ImportError(
            "Failed to import DeepGMM (methods.toy_model_selection_method). "
            "Ensure PYTHONPATH includes DIV-main/DeepGMM-master and dependencies are installed."
        ) from exc
    try:
        import torch
        from torch.utils.data import random_split, TensorDataset
    except Exception as exc:
        raise ImportError("PyTorch is required to run DeepGMM.") from exc

    set_global_seed(seed)
    require_cuda = os.environ.get("DEEPGMM_REQUIRE_CUDA", "1") == "1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
    print(
        "   [deepgmm] torch.cuda.is_available="
        f"{torch.cuda.is_available()}, device={device}, "
        f"CUDA_VISIBLE_DEVICES={visible}, torch.cuda.device_count={torch.cuda.device_count()}"
    )
    if torch.cuda.is_available():
        try:
            gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            print(f"   [deepgmm] GPU names: {gpu_names}")
        except Exception as exc:  # pragma: no cover - best-effort diagnostics
            print(f"   [deepgmm] Unable to query GPU names: {exc}")
    elif require_cuda:
        raise RuntimeError("DEEPGMM_REQUIRE_CUDA=1 but CUDA is not available inside the job.")

    # Follow notebook logic: make tensors, split 90/10 (val_fraction), use double precision.
    z_all = torch.tensor(train_df["Z"].to_numpy(dtype=np.float64)).reshape(-1, 1)
    x_all = torch.tensor(train_df["X"].to_numpy(dtype=np.float64)).reshape(-1, 1)
    y_all = torch.tensor(train_df["Y"].to_numpy(dtype=np.float64)).reshape(-1, 1)
    dataset = TensorDataset(x_all, z_all, y_all)
    n_total = len(dataset)
    val_n = max(1, int(val_fraction * n_total))
    train_n = max(1, n_total - val_n)
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(dataset, [train_n, val_n], generator=generator)

    def _split_tensors(subset):
        idx = subset.indices
        return (
            x_all[idx].to(device),
            z_all[idx].to(device),
            y_all[idx].to(device),
        )

    X_train, Z_train, Y_train = _split_tensors(train_subset)
    X_val, Z_val, Y_val = _split_tensors(val_subset)

    model = ToyModelSelectionMethod(enable_cuda=torch.cuda.is_available())
    model.fit(
        X_train.double(),
        Z_train.double(),
        Y_train.double(),
        X_val.double(),
        Z_val.double(),
        Y_val.double(),
        g_dev=None,
        verbose=False,
    )

    def _predict(x_values: np.ndarray) -> np.ndarray:
        x_tensor = torch.tensor(np.asarray(x_values, dtype=np.float64)).reshape(-1, 1)
        preds = (
            model.predict(x_tensor.to(device).double())
            .flatten()
            .detach()
            .cpu()
            .numpy()
        )
        return preds

    return _predict, "deepgmm"


def _hsic_predictor(train_df, test_df, instrument: str, seed: int, n_runs: int = 10):
    """
    HSIC-X predictor matching HSIC-X_interv_mean.ipynb: MSE warm start, median-bandwidth kernel on Z,
    4 restart heuristic, and 10 independent runs whose mean prediction is returned.
    """
    # Ensure HSIC-X modules shadow similarly named modules (e.g., DeepGMM's models).
    if HSICX_ROOT.exists():
        try:
            sys.path.remove(str(HSICX_ROOT))
        except ValueError:
            pass
        sys.path.insert(0, str(HSICX_ROOT))

    import shutil

    if shutil.which("R") is None:
        raise RuntimeError(
            "HSIC-X requires R (binary 'R' not found in PATH). Please load/install R and the dHSIC package."
        )
    try:
        import torch
        from helpers.trainer import train_HSIC_IV, train_mse
        from helpers.utils import med_sigma, to_torch
        from models.hsicx import NNHSICX
        from models.kernel import RBFKernel, CategoryKernel
    except Exception as exc:
        raise RuntimeError(f"HSIC-X dependencies unavailable: {exc}") from exc

    # Extract numpy arrays (float32) to mirror the notebook input.
    required_test_cols = [col for col in ("Xtest_grid", "mean_grid") if col not in test_df.columns]
    if required_test_cols:
        missing_cols = ", ".join(required_test_cols)
        raise ValueError(f"Test CSV missing required HSIC-X columns: {missing_cols}")

    x_train = train_df["X"].to_numpy(dtype=np.float32)
    y_train = train_df["Y"].to_numpy(dtype=np.float32)
    z_train = train_df["Z"].to_numpy(dtype=np.float32)
    x_test = test_df["Xint"].to_numpy(dtype=np.float32)
    x_grid = test_df["Xtest_grid"].to_numpy(dtype=np.float32)
    grid_target = test_df["mean_grid"].to_numpy(dtype=np.float32)

    instrument_lower = instrument.lower()
    is_binary_instrument = instrument_lower.startswith("zbin") or instrument_lower.startswith("binary")
    if not is_binary_instrument and np.unique(z_train).size <= 5:
        is_binary_instrument = True

    kernel_e = RBFKernel(sigma=1.0)
    config_hsic = {"batch_size": 256, "max_epoch": 700, "num_restart": 4, "lr": 1e-3}
    config_mse = {"batch_size": 256, "max_epoch": 300, "lr": 5e-4}

    def _single_run(run_idx: int):
        set_global_seed(seed + run_idx)

        mse_net = NNHSICX(input_dim=1, lr=config_mse["lr"], lmd=-99)
        mse_net = train_mse(mse_net, config_mse, x_train, y_train, z_train)

        if is_binary_instrument:
            kernel_z = CategoryKernel()
        else:
            sigma_z = float(med_sigma(z_train))
            kernel_z = RBFKernel(sigma=sigma_z)

        hsic_net = NNHSICX(
            input_dim=1,
            lr=config_hsic["lr"],
            kernel_e=kernel_e,
            kernel_z=kernel_z,
            lmd=0,
        )
        hsic_net.load_state_dict(mse_net)
        hsic_net = train_HSIC_IV(
            hsic_net,
            config_hsic,
            x_train,
            y_train,
            z_train,
            verbose=True,
        )
        hsic_net.eval()
        device = next(hsic_net.parameters()).device

        x_train_t = to_torch(x_train).to(device)
        y_train_t = to_torch(y_train).to(device)
        intercept_adjust = (y_train_t.mean() - hsic_net(x_train_t).mean()).detach()

        def _predict(arr: np.ndarray) -> np.ndarray:
            x_tensor = to_torch(arr).to(device)
            with torch.no_grad():
                preds = intercept_adjust + hsic_net(x_tensor)
            return preds.detach().cpu().numpy().reshape(-1)

        preds_test = _predict(x_test)
        preds_grid = _predict(x_grid)
        return preds_test, preds_grid

    per_run_preds: List[np.ndarray] = []
    per_run_grid_preds: List[np.ndarray] = []
    for run_idx in range(n_runs):
        preds_test, preds_grid = _single_run(run_idx)
        per_run_preds.append(preds_test)
        per_run_grid_preds.append(preds_grid)

    per_run_pred_mat = np.stack(per_run_preds, axis=1)  # (n_test, n_runs)
    mean_pred = np.mean(per_run_pred_mat, axis=1)

    per_run_grid_mat = np.stack(per_run_grid_preds, axis=1)  # (n_grid, n_runs)
    mean_grid_pred = np.mean(per_run_grid_mat, axis=1)

    variant = f"hsicx_notebook_runs{per_run_pred_mat.shape[1]}"
    return mean_pred, mean_grid_pred, per_run_pred_mat, per_run_grid_mat, grid_target, variant


def _tabpfn_cf_predictor(train_df, seed: int):
    """
    Two-stage control-function baseline using TabPFN regressors:
      1) X ~ Z with TabPFN -> residual e = X - X_hat
      2) Y ~ [X, e] with TabPFN, evaluate at e = 0 for interventional mean.
    """
    os.environ.setdefault("TABPFN_MODEL_CACHE_DIR", str(REPO_ROOT / "tabpfn_home_config" / "models"))
    os.environ.setdefault("TABPFN_MODEL_VERSION", os.environ.get("TABPFN_MODEL_VERSION", "v2.5"))
    os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")
    try:
        from importlib.metadata import version as pkg_version
        from packaging.version import Version

        skl_version = Version(pkg_version("scikit-learn"))
        if skl_version < Version("1.3"):
            raise RuntimeError(
                f"tabpfn_cf requires scikit-learn>=1.3 for TabPFN preprocessing; found {skl_version}."
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
            raise RuntimeError("TabPFNRegressor is required for the tabpfn_cf baseline.") from exc

    set_global_seed(seed)
    z = train_df["Z"].to_numpy(dtype=np.float32).reshape(-1, 1)
    x = train_df["X"].to_numpy(dtype=np.float32)
    y = train_df["Y"].to_numpy(dtype=np.float32)

    stage1 = TabPFNRegressor(random_state=seed, ignore_pretraining_limits=True)
    stage1.fit(z, x)
    x_hat = stage1.predict(z).reshape(-1).astype(np.float32)
    resid = x - x_hat

    stage2_features = np.column_stack([x, resid]).astype(np.float32)
    stage2 = TabPFNRegressor(random_state=seed + 1, ignore_pretraining_limits=True)
    stage2.fit(stage2_features, y)

    def _predict(x_values: np.ndarray, residual_value: float = 0.0) -> np.ndarray:
        x_arr = np.asarray(x_values, dtype=np.float32).reshape(-1)
        resid_feat = np.full_like(x_arr, fill_value=residual_value, dtype=np.float32)
        feats = np.column_stack([x_arr, resid_feat])
        preds = stage2.predict(feats)
        return np.asarray(preds, dtype=float).reshape(-1)

    diagnostics = {
        "stage1_rmse_x": float(np.sqrt(np.mean((x_hat - x) ** 2))),
        "stage1_mae_x": float(np.mean(np.abs(x_hat - x))),
    }
    return _predict, "tabpfn_cf_resid0", diagnostics


def compute_mse(pred: np.ndarray, target: np.ndarray) -> float:
    pred_arr = np.asarray(pred, dtype=float).reshape(-1)
    target_arr = np.asarray(target, dtype=float).reshape(-1)
    if pred_arr.shape != target_arr.shape:
        raise ValueError(f"Shape mismatch for MSE: pred {pred_arr.shape}, target {target_arr.shape}")
    return float(np.mean((pred_arr - target_arr) ** 2))


def save_metrics(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _should_skip_run(metrics_path: Path, model: str, overwrite: bool) -> bool:
    """Decide whether to skip a run based on existing metrics."""
    if not metrics_path.exists():
        return False
    if overwrite:
        return False
    try:
        existing = json.loads(metrics_path.read_text())
    except Exception as exc:  # corrupt or partial file → recompute
        print(f"   Existing metrics at {metrics_path} unreadable ({exc}); recomputing.")
        return False

    if model == "deepgmm" and existing.get("variant") == "linear_iv_fallback":
        # Previously fell back because deps (torch/DeepGMM) were missing; force rerun.
        print(f"   Existing DeepGMM metrics use linear_iv_fallback; recomputing.")
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Run sec5.1 TabPFN baselines (DeepIV, DeepGMM).")
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
        "--models",
        nargs="+",
        default=["deepiv", "deepgmm"],
        choices=["deepiv", "deepgmm", "hsic", "tabpfn_cf"],
        help="Which models to run.",
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

        for model in args.models:
            metrics_path = args.results_dir / f"{model}_{code}_n{size}_seed{seed}.json"
            preds_path = args.results_dir / f"{model}_{code}_n{size}_seed{seed}_pred.csv"
            if _should_skip_run(metrics_path, model, args.overwrite):
                print(f"[{idx}/{len(runs)}] {model} {code} n={size} seed={seed} already done, skipping.")
                continue
            if metrics_path.exists():
                print(f"[{idx}/{len(runs)}] Overwriting {metrics_path}")

            print(f"[{idx}/{len(runs)}] Running {model} on {code}, n={size}, seed={seed}")
            model_start = time.time()
            variant: str
            preds: np.ndarray
            grid_pred: Optional[np.ndarray] = None
            per_run_preds: Optional[np.ndarray] = None
            per_run_grid_preds: Optional[np.ndarray] = None
            grid_target: Optional[np.ndarray] = None
            diagnostics: Optional[Dict[str, object]] = None
            if model == "deepiv":
                try:
                    predictor, variant = _deepiv_predictor(train_df, seed=seed, n_restarts=10)
                except RuntimeError as exc:
                    print(f"   ⚠️ DeepIV unavailable ({exc}); falling back to linear IV.")
                    predictor, variant = _linear_iv_predictor(train_df)
                preds = predictor(x_test)
            elif model == "deepgmm":
                predictor, variant = _deepgmm_predictor(train_df, val_fraction=0.1, seed=seed)
                if variant == "linear_iv_fallback":
                    print(
                        "   ⚠️ DeepGMM dependencies (torch/DeepGMM) missing; using linear IV fallback."
                        " Install the required packages and rerun with --overwrite to refresh results."
                    )
                preds = predictor(x_test)
            elif model == "tabpfn_cf":
                predictor, variant, diagnostics = _tabpfn_cf_predictor(train_df, seed=seed)
                preds = predictor(x_test)
            elif model == "hsic":
                (
                    preds,
                    grid_pred,
                    per_run_preds,
                    per_run_grid_preds,
                    grid_target,
                    variant,
                ) = _hsic_predictor(
                    train_df,
                    test_df,
                    instrument=run["instrument"],
                    seed=seed,
                )
            else:
                raise ValueError(f"Unsupported model: {model}")

            mse = compute_mse(preds, target)

            payload: Dict[str, object] = {
                "model": model,
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
            if diagnostics:
                payload.update(diagnostics)
            if per_run_preds is not None:
                payload["n_hsic_runs"] = int(per_run_preds.shape[1])
                payload["per_run_mse_vs_mean_int"] = [
                    compute_mse(per_run_preds[:, run_idx], target) for run_idx in range(per_run_preds.shape[1])
                ]
            if grid_pred is not None and grid_target is not None:
                payload["mse_vs_mean_grid"] = compute_mse(grid_pred, grid_target)
                if per_run_grid_preds is not None:
                    payload["per_run_mse_vs_mean_grid"] = [
                        compute_mse(per_run_grid_preds[:, run_idx], grid_target)
                        for run_idx in range(per_run_grid_preds.shape[1])
                    ]
            if args.save_predictions:
                preds_path.parent.mkdir(parents=True, exist_ok=True)
                if model == "hsic":
                    import pandas as pd

                    pred_df = pd.DataFrame({"Xint": x_test, "mean_int": target, "pred_mean": preds})
                    if per_run_preds is not None:
                        for run_idx in range(per_run_preds.shape[1]):
                            pred_df[f"pred_run{run_idx + 1}"] = per_run_preds[:, run_idx]
                    pred_df.to_csv(preds_path, index=False)
                    payload["pred_path"] = str(preds_path)

                    if grid_pred is not None:
                        grid_path = preds_path.with_name(f"{preds_path.stem}_grid.csv")
                        grid_x = test_df["Xtest_grid"].to_numpy() if "Xtest_grid" in test_df.columns else None
                        grid_df = pd.DataFrame(
                            {
                                "Xtest_grid": grid_x if grid_x is not None else np.arange(len(grid_pred)),
                                "pred_mean": grid_pred,
                            }
                        )
                        if grid_target is not None:
                            grid_df.insert(1, "mean_grid", grid_target)
                        if per_run_grid_preds is not None:
                            for run_idx in range(per_run_grid_preds.shape[1]):
                                grid_df[f"pred_run{run_idx + 1}"] = per_run_grid_preds[:, run_idx]
                        grid_df.to_csv(grid_path, index=False)
                        payload["grid_pred_path"] = str(grid_path)
                else:
                    pred_rows = np.column_stack([x_test, target, preds])
                    header = "Xint,mean_int,pred"
                    np.savetxt(preds_path, pred_rows, delimiter=",", header=header, comments="")
                    payload["pred_path"] = str(preds_path)

            save_metrics(metrics_path, payload)
            elapsed = time.time() - model_start
            print(
                f"   ✅ {model} {code} n={size} seed={seed} done in {elapsed:.2f}s "
                f"(saved to {metrics_path.name}) at {time.strftime('%Y-%m-%d %H:%M:%S')}"
            )

    print("Baseline runs completed.")


if __name__ == "__main__":
    main()
