#!/usr/bin/env python3
"""Thin DistributionIV adapter for multivariate marginal CDF estimation."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


DIV_EPSX_DIM = 50
DIV_EPSY_DIM = 50
DIV_EPSH_DIM = 50
DIV_NUM_LAYER = 3
DIV_NUM_EPOCHS = 1000
DIV_LR = 1e-3
DIV_PREDICT_NSAMPLE = 1000


@dataclass
class _DIVRuntime:
    ro: object
    base: object
    predict_fn: object
    div_fn: object
    default_converter: object
    numpy_converter: object


_DIV_RUNTIME_CACHE: _DIVRuntime | None = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _crash_message(prefix: str) -> str:
    return (
        f"{prefix} For multivar DIV mode, load R first "
        "(`source /etc/profile && module load r/4.4.3-py311-xspgsan`) "
        "and run with `venv_hsic` activated."
    )


def _load_div_runtime() -> _DIVRuntime:
    global _DIV_RUNTIME_CACHE
    if _DIV_RUNTIME_CACHE is not None:
        return _DIV_RUNTIME_CACHE

    if shutil.which("R") is None:
        raise ImportError(_crash_message("R executable not found in PATH."))

    try:
        import rpy2.robjects as ro
        from rpy2.robjects import numpy2ri
        from rpy2.robjects import default_converter
        from rpy2.robjects.packages import PackageNotInstalledError
        from rpy2.robjects.packages import importr
    except Exception as exc:  # pragma: no cover - environment-dependent
        raise ImportError(_crash_message(f"Unable to import rpy2: {exc}")) from exc

    repo_root = _repo_root()
    r_libs = repo_root / "R_libs"
    ro.r[".libPaths"](ro.StrVector([str(r_libs), *tuple(ro.r[".libPaths"]())]))

    try:
        utils = importr("utils")
        base = importr("base")
        distributioniv = importr("DistributionIV")
    except PackageNotInstalledError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            _crash_message(
                "R package `DistributionIV` is not installed or not visible in R_LIBS."
            )
        ) from exc
    except Exception as exc:  # pragma: no cover - environment-dependent
        raise ImportError(_crash_message(f"Unable to load R packages for DIV: {exc}")) from exc

    try:
        repo_field = str(utils.packageDescription("DistributionIV").rx2("Repository")[0]).strip()
    except Exception:  # pragma: no cover - package metadata can vary
        repo_field = ""
    if "cran" not in repo_field.lower():
        raise ImportError(
            _crash_message(
                "multivar DIV mode requires the CRAN DistributionIV install (div_2 semantics); "
                f"current Repository field is `{repo_field or '<unknown>'}`."
            )
        )

    _DIV_RUNTIME_CACHE = _DIVRuntime(
        ro=ro,
        base=base,
        predict_fn=ro.r["predict"],
        div_fn=distributioniv.div,
        default_converter=default_converter,
        numpy_converter=numpy2ri.converter,
    )
    return _DIV_RUNTIME_CACHE


def _stable_seed(base_seed: int, component: int, cache_key: str) -> int:
    digest = hashlib.sha1(f"{component}|{cache_key}".encode("utf-8")).hexdigest()
    return (int(base_seed) + int(digest[:8], 16)) % (2**31 - 1)


def _cache_key(x: np.ndarray) -> str:
    digest = hashlib.sha1(np.ascontiguousarray(x, dtype=np.float64).tobytes()).hexdigest()
    return f"{x.shape}:{digest}"


def _coerce_samples_array(runtime: _DIVRuntime, value: object, n_rows: int) -> np.ndarray:
    with runtime.ro.conversion.localconverter(
        runtime.default_converter + runtime.numpy_converter
    ):
        arr = runtime.ro.conversion.rpy2py(value)
    samples = np.asarray(arr, dtype=float)
    if samples.ndim == 3:
        if samples.shape[1] != 1:
            raise ValueError(
                "Unexpected DIV sample output shape; expected singleton outcome axis, "
                f"got {samples.shape}"
            )
        samples = samples[:, 0, :]
    if samples.ndim != 2:
        raise ValueError(f"Unexpected DIV sample output rank: {samples.shape}")
    if samples.shape[0] != n_rows and samples.shape[1] == n_rows:
        samples = samples.T
    if samples.shape[0] != n_rows:
        raise ValueError(
            "Unexpected DIV sample output shape; "
            f"expected first dim {n_rows}, got {samples.shape}"
        )
    return samples


def _to_r_vector(runtime: _DIVRuntime, values: np.ndarray) -> object:
    return runtime.ro.FloatVector(np.asarray(values, dtype=float).reshape(-1))


@dataclass
class FittedDIVMarginalEstimators:
    """Fitted DIV marginal CDF estimators for Y1 and Y2."""

    model_y1: object
    model_y2: object
    base_seed: int
    runtime: _DIVRuntime
    sample_cache: dict[int, dict[str, np.ndarray]] = field(
        default_factory=lambda: {1: {}, 2: {}}
    )

    def _model_for_component(self, component: int) -> object:
        if component == 1:
            return self.model_y1
        if component == 2:
            return self.model_y2
        raise ValueError(f"component must be 1 or 2, got {component}")

    def _sorted_samples_for_x(self, component: int, x_flat: np.ndarray) -> np.ndarray:
        cache_key = _cache_key(x_flat)
        cached = self.sample_cache[component].get(cache_key)
        if cached is not None:
            return cached

        seed = _stable_seed(self.base_seed, component, cache_key)
        self.runtime.ro.r["set.seed"](seed)
        samples_r = self.runtime.predict_fn(
            self._model_for_component(component),
            Xtest=_to_r_vector(self.runtime, x_flat),
            type="sample",
            nsample=DIV_PREDICT_NSAMPLE,
            drop=False,
        )
        samples = _coerce_samples_array(self.runtime, samples_r, n_rows=len(x_flat))
        sorted_samples = np.sort(samples, axis=1)
        self.sample_cache[component][cache_key] = sorted_samples
        return sorted_samples

    def cdf_pointwise(
        self,
        component: int,
        x: np.ndarray | float,
        y: np.ndarray | float,
    ) -> np.ndarray:
        """Evaluate F_hat_component(y | do(X=x)) via empirical CDF of DIV samples."""
        x_b, y_b = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        x_flat = x_b.reshape(-1)
        y_flat = y_b.reshape(-1)
        sorted_samples = self._sorted_samples_for_x(component, x_flat)
        counts = np.fromiter(
            (
                np.searchsorted(row, value, side="right")
                for row, value in zip(sorted_samples, y_flat, strict=False)
            ),
            dtype=float,
            count=len(y_flat),
        )
        cdf = np.clip(counts / float(sorted_samples.shape[1]), 0.0, 1.0)
        return cdf.reshape(x_b.shape)


def _fit_one_div_model(runtime: _DIVRuntime, z: np.ndarray, x: np.ndarray, y: np.ndarray) -> object:
    return runtime.div_fn(
        Z=_to_r_vector(runtime, z),
        X=_to_r_vector(runtime, x),
        Y=_to_r_vector(runtime, y),
        epsx_dim=DIV_EPSX_DIM,
        epsy_dim=DIV_EPSY_DIM,
        epsh_dim=DIV_EPSH_DIM,
        num_epochs=DIV_NUM_EPOCHS,
        num_layer=DIV_NUM_LAYER,
        lr=DIV_LR,
    )


def fit_div_marginal_estimators(df_obs: pd.DataFrame, *, seed: int) -> FittedDIVMarginalEstimators:
    """Fit DIV marginals for Y1 and Y2 using CRAN DistributionIV defaults from sec5.1."""
    required_cols = {"Z", "X", "Y1", "Y2"}
    missing = sorted(required_cols.difference(df_obs.columns))
    if missing:
        raise ValueError(f"df_obs is missing required columns for DIV fitting: {missing}")

    runtime = _load_div_runtime()
    z = df_obs["Z"].to_numpy(dtype=float)
    x = df_obs["X"].to_numpy(dtype=float)
    y1 = df_obs["Y1"].to_numpy(dtype=float)
    y2 = df_obs["Y2"].to_numpy(dtype=float)

    runtime.ro.r["set.seed"](int(seed))
    model_y1 = _fit_one_div_model(runtime, z, x, y1)
    runtime.ro.r["set.seed"](int(seed) + 1)
    model_y2 = _fit_one_div_model(runtime, z, x, y2)

    return FittedDIVMarginalEstimators(
        model_y1=model_y1,
        model_y2=model_y2,
        base_seed=int(seed),
        runtime=runtime,
    )
