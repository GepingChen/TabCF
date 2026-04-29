#!/usr/bin/env python3
"""Thin TabCF adapter for multivariate marginal CDF estimation."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
from scipy.integrate import simpson


@dataclass
class _TabCFModules:
    foundation_backends: ModuleType
    stage1: ModuleType
    stage2: ModuleType


_MODULES_CACHE: _TabCFModules | None = None
_STAGE1_MODULE_CACHE: ModuleType | None = None


def _batch_core_dir() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    core_dir = repo_root / "tabcf_core"
    if not core_dir.exists():
        raise ImportError(f"TabCF core directory not found: {core_dir}")
    core_dir_str = str(core_dir)
    if core_dir_str not in sys.path:
        sys.path.insert(0, core_dir_str)
    return core_dir


def _load_stage1_module() -> ModuleType:
    """Lazy-load Stage-1 helpers shared by backend-aware TabCF and TabPFN-naive marginals."""
    global _STAGE1_MODULE_CACHE
    if _STAGE1_MODULE_CACHE is not None:
        return _STAGE1_MODULE_CACHE

    _ = _batch_core_dir()
    try:
        stage1 = importlib.import_module("stage1_control")
    except Exception as exc:  # pragma: no cover - import errors are environment-dependent
        raise ImportError(
            "Unable to import Stage-1 module from tabcf_core. "
            "Backend-aware marginal modes require a compatible core-model environment."
        ) from exc
    _STAGE1_MODULE_CACHE = stage1
    return stage1


def _load_foundation_backends_module() -> ModuleType:
    _ = _batch_core_dir()
    try:
        return importlib.import_module("foundation_backends")
    except Exception as exc:  # pragma: no cover - import errors are environment-dependent
        raise ImportError(
            "Unable to import foundation_backends from tabcf_core. "
            "Backend-aware marginal modes require the shared core backend helpers."
        ) from exc


def _load_tabcf_modules() -> _TabCFModules:
    """Lazy-load tabcf_core modules only when TabCF-compatible marginals are requested."""
    global _MODULES_CACHE
    if _MODULES_CACHE is not None:
        return _MODULES_CACHE

    _ = _batch_core_dir()
    foundation_backends = _load_foundation_backends_module()
    stage1 = _load_stage1_module()
    try:
        stage2 = importlib.import_module("stage2_outcome")
    except Exception as exc:  # pragma: no cover - import errors are environment-dependent
        raise ImportError(
            "Unable to import TabCF modules from tabcf_core. "
            "TabCF mode requires a TabPFN-ready environment."
        ) from exc

    _MODULES_CACHE = _TabCFModules(
        foundation_backends=foundation_backends,
        stage1=stage1,
        stage2=stage2,
    )
    return _MODULES_CACHE


@dataclass
class FittedMarginalEstimators:
    """Fitted TabCF marginal CDF estimators for Y1 and Y2."""

    model_y1: object
    model_y2: object
    v_grid: np.ndarray
    eval_batch_size: int
    cdf_from_full_output: object
    backend_name: str
    backend_package: str
    checkpoint_version: str
    model_path: str

    @property
    def metadata(self) -> dict[str, str]:
        return {
            "core_backend": self.backend_name,
            "backend_package": self.backend_package,
            "checkpoint_version": self.checkpoint_version,
            "model_path": self.model_path,
        }

    def _model_for_component(self, component: int) -> object:
        if component == 1:
            return self.model_y1
        if component == 2:
            return self.model_y2
        raise ValueError(f"component must be 1 or 2, got {component}")

    def cdf_pointwise(
        self,
        component: int,
        x: np.ndarray | float,
        y: np.ndarray | float,
    ) -> np.ndarray:
        """Evaluate F_hat_component(y | do(X=x)) pointwise with vectorized V-integration."""
        model = self._model_for_component(component)

        x_b, y_b = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        flat_x = x_b.reshape(-1)
        flat_y = y_b.reshape(-1)
        out = np.empty_like(flat_x, dtype=float)

        n_v = len(self.v_grid)
        batch_size = max(1, int(self.eval_batch_size))

        for start in range(0, len(flat_x), batch_size):
            end = min(start + batch_size, len(flat_x))
            x_chunk = flat_x[start:end]
            y_chunk = flat_y[start:end]

            x_rep = np.repeat(x_chunk, n_v)
            v_rep = np.tile(self.v_grid, len(x_chunk))
            y_rep = np.repeat(y_chunk, n_v)

            full_output = model.predict_full_distribution(x_rep, v_rep)
            cdf_vals = np.asarray(
                self.cdf_from_full_output(full_output, y_rep, squeeze_last=True),
                dtype=float,
            )
            if cdf_vals.size != len(x_rep):
                raise ValueError(
                    "Unexpected TabCF CDF output shape while evaluating marginals: "
                    f"expected {len(x_rep)}, got {cdf_vals.size}"
                )

            cdf_matrix = cdf_vals.reshape(len(x_chunk), n_v)
            integrated = simpson(cdf_matrix, x=self.v_grid, axis=1)
            out[start:end] = np.clip(integrated, 0.0, 1.0)

        return out.reshape(x_b.shape)


@dataclass
class FittedTabPFNNaiveMarginalEstimators:
    """Fitted TabPFN-naive marginal CDF estimators for Y1 and Y2 (ignoring IV)."""

    model_y1: object
    model_y2: object

    def _model_for_component(self, component: int) -> object:
        if component == 1:
            return self.model_y1
        if component == 2:
            return self.model_y2
        raise ValueError(f"component must be 1 or 2, got {component}")

    def cdf_pointwise(
        self,
        component: int,
        x: np.ndarray | float,
        y: np.ndarray | float,
    ) -> np.ndarray:
        """Evaluate F_hat_component(y | x) using TabPFN-naive Y~X marginals."""
        model = self._model_for_component(component)
        x_b, y_b = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        flat_x = x_b.reshape(-1)
        flat_y = y_b.reshape(-1)
        cdf_vals = np.asarray(model.predict(flat_x, flat_y), dtype=float)
        if cdf_vals.shape != flat_x.shape:
            raise ValueError(
                "Unexpected TabPFN-naive CDF output shape while evaluating marginals: "
                f"expected {flat_x.shape}, got {cdf_vals.shape}"
            )
        return np.clip(cdf_vals, 0.0, 1.0).reshape(x_b.shape)


def fit_marginal_estimators(
    df_obs: pd.DataFrame,
    *,
    v_points: int = 101,
    eval_batch_size: int = 256,
    backend_name: str = "tabpfn",
    model_path: str = "auto",
    random_state: int = 1,
) -> FittedMarginalEstimators:
    """Fit TabCF marginals for Y1 and Y2 using shared Stage-1 V-hat estimates."""
    required_cols = {"Z", "X", "Y1", "Y2"}
    missing = sorted(required_cols.difference(df_obs.columns))
    if missing:
        raise ValueError(f"df_obs is missing required columns for TabCF fitting: {missing}")

    modules = _load_tabcf_modules()
    backend_name = modules.foundation_backends.normalize_backend_name(backend_name)
    backend_info = modules.foundation_backends.backend_metadata(backend_name, model_path)
    CondCDFModel = modules.stage1.CondCDFModel
    ConditionalCDFEstimator = modules.stage2.ConditionalCDFEstimator
    create_v_integration_grid = modules.stage2.create_v_integration_grid
    cdf_from_full_output = modules.stage2.cdf_from_full_output

    z = df_obs["Z"].to_numpy(dtype=float)
    x = df_obs["X"].to_numpy(dtype=float)
    y1 = df_obs["Y1"].to_numpy(dtype=float)
    y2 = df_obs["Y2"].to_numpy(dtype=float)

    stage1_model = CondCDFModel(
        quantiles=(),
        backend_name=backend_name,
        model_path=model_path,
        random_state=random_state,
    )
    stage1_model.fit(z, x)
    v_hat = np.asarray(stage1_model.predict(z, x), dtype=float)
    v_hat = np.clip(v_hat, 0.0, 1.0)

    use_tabpfn = backend_name != modules.foundation_backends.TABICL_BACKEND
    model_y1 = ConditionalCDFEstimator(
        use_tabpfn=use_tabpfn,
        backend_name=backend_name,
        model_path=model_path,
        random_state=random_state,
    )
    _ = model_y1.fit_full(x, v_hat, y1)

    model_y2 = ConditionalCDFEstimator(
        use_tabpfn=use_tabpfn,
        backend_name=backend_name,
        model_path=model_path,
        random_state=random_state,
    )
    _ = model_y2.fit_full(x, v_hat, y2)

    v_grid = np.asarray(create_v_integration_grid(int(v_points)), dtype=float)

    return FittedMarginalEstimators(
        model_y1=model_y1,
        model_y2=model_y2,
        v_grid=v_grid,
        eval_batch_size=int(eval_batch_size),
        cdf_from_full_output=cdf_from_full_output,
        backend_name=str(backend_info["backend"]),
        backend_package=str(backend_info["backend_package"]),
        checkpoint_version=str(backend_info["checkpoint_version"]),
        model_path=str(backend_info["model_path"]),
    )


def fit_tabpfn_naive_marginal_estimators(
    df_obs: pd.DataFrame,
) -> FittedTabPFNNaiveMarginalEstimators:
    """Fit TabPFN-naive marginals for Y1 and Y2 with Y_j ~ X (ignoring IV)."""
    required_cols = {"X", "Y1", "Y2"}
    missing = sorted(required_cols.difference(df_obs.columns))
    if missing:
        raise ValueError(f"df_obs is missing required columns for tabpfn-naive fitting: {missing}")

    stage1 = _load_stage1_module()
    CondCDFModel = stage1.CondCDFModel

    x = df_obs["X"].to_numpy(dtype=float)
    y1 = df_obs["Y1"].to_numpy(dtype=float)
    y2 = df_obs["Y2"].to_numpy(dtype=float)

    model_y1 = CondCDFModel(quantiles=())
    model_y1.fit(x, y1)

    model_y2 = CondCDFModel(quantiles=())
    model_y2.fit(x, y2)

    return FittedTabPFNNaiveMarginalEstimators(model_y1=model_y1, model_y2=model_y2)
