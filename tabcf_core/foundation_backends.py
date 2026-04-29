"""
Unified backend helpers for core batch-simulation pipelines.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np

try:
    import torch
except ModuleNotFoundError as torch_import_error:  # pragma: no cover - exercised in non-torch environments
    class _MissingTorchProxy:
        def __getattr__(self, name: str) -> Any:
            raise ModuleNotFoundError(
                "PyTorch is required for distribution-backed TabCF helpers. "
                "Install 'torch' before calling foundation_backends features that need predictive distributions."
            ) from torch_import_error

    torch = _MissingTorchProxy()  # type: ignore[assignment]

from local_context_backends import (
    LocalKNNRegressorWrapper,
    ensure_local_context_config,
    local_context_suffix,
)


TABPFN_BACKEND = "tabpfn"
TABPFN_REAL_BACKEND = "tabpfn_real"
TABICL_BACKEND = "tabicl"
SUPPORTED_BACKENDS = (TABPFN_BACKEND, TABPFN_REAL_BACKEND, TABICL_BACKEND)

DEFAULT_TABPFN_V25_REGRESSOR_CHECKPOINT = "tabpfn-v2.5-regressor-v2.5_default.ckpt"
DEFAULT_TABPFN_REAL_REGRESSOR_CHECKPOINT = "tabpfn-v2.5-regressor-v2.5_real.ckpt"
DEFAULT_TABICL_REGRESSOR_CHECKPOINT = "tabicl-regressor-v2-20260212.ckpt"

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TABPFN_CACHE_DIR = REPO_ROOT / "tabpfn_home_config" / "models"
TABICL_SRC_DIR = REPO_ROOT / "tabicl" / "src"


def normalize_backend_name(backend_name: Optional[str], *, use_tabpfn: Optional[bool] = None) -> str:
    """Resolve backend names while preserving legacy bool-based callers."""
    candidate = (backend_name or "").strip().lower()
    if not candidate:
        if use_tabpfn is False:
            raise ValueError("Legacy use_tabpfn=False is no longer supported without an explicit backend_name.")
        candidate = TABPFN_BACKEND

    if candidate not in SUPPORTED_BACKENDS:
        raise ValueError(f"Unsupported backend '{backend_name}'. Expected one of {SUPPORTED_BACKENDS}.")
    return candidate


def backend_file_suffix(backend_name: str) -> str:
    backend = normalize_backend_name(backend_name)
    return "" if backend == TABPFN_BACKEND else f"_{backend}"


def normalize_softmax_temperature_tag(softmax_temperature: float | None) -> str:
    if softmax_temperature is None:
        return ""
    normalized = format(float(softmax_temperature), "g").replace("-", "m").replace(".", "p")
    return f"_st{normalized}"


def stage1_output_filename(
    subset: str,
    code: str,
    *,
    train_sample_size: Optional[int] = None,
    seed: Optional[int] = None,
    backend_name: str = TABPFN_BACKEND,
    softmax_temperature: float | None = None,
    timestamp: Optional[str] = None,
) -> str:
    sample_tag = f"_n{train_sample_size}" if train_sample_size is not None else ""
    seed_tag = f"_seed{seed}" if seed is not None else ""
    backend_tag = backend_file_suffix(backend_name)
    temperature_tag = normalize_softmax_temperature_tag(softmax_temperature)
    time_tag = f"_{timestamp}" if timestamp else ""
    return f"iv_stage1_{subset}_{code}{sample_tag}{seed_tag}{backend_tag}{temperature_tag}{time_tag}.csv"


def stage2_base_prefix(
    code: str,
    *,
    train_sample_size: Optional[int] = None,
    seed: Optional[int] = None,
    backend_name: str = TABPFN_BACKEND,
    local_context: object | None = None,
    resolved_local_k: Optional[int] = None,
) -> str:
    sample_tag = f"_n{train_sample_size}" if train_sample_size is not None else ""
    seed_tag = f"_seed{seed}" if seed is not None else ""
    backend_tag = backend_file_suffix(backend_name)
    local_tag = local_context_suffix(
        local_context,
        n_train_samples=train_sample_size,
        resolved_k=resolved_local_k,
    )
    return f"s2_{code}{sample_tag}{seed_tag}{backend_tag}{local_tag}"


def _tabpfn_cache_dir() -> Path:
    cache_dir = os.environ.get("TABPFN_MODEL_CACHE_DIR")
    if cache_dir:
        return Path(cache_dir)
    return DEFAULT_TABPFN_CACHE_DIR


def _ensure_tabicl_src_path() -> None:
    """Put the real tabicl package root ahead of the repo root namespace package."""
    if not TABICL_SRC_DIR.exists():
        return
    tabicl_src = str(TABICL_SRC_DIR)
    if tabicl_src in sys.path:
        sys.path.remove(tabicl_src)
    sys.path.insert(0, tabicl_src)


def default_checkpoint_for_backend(backend_name: str) -> str:
    backend = normalize_backend_name(backend_name)
    if backend == TABPFN_BACKEND:
        return DEFAULT_TABPFN_V25_REGRESSOR_CHECKPOINT
    if backend == TABPFN_REAL_BACKEND:
        return DEFAULT_TABPFN_REAL_REGRESSOR_CHECKPOINT
    return DEFAULT_TABICL_REGRESSOR_CHECKPOINT


def resolve_model_path(backend_name: str, model_path: str | os.PathLike[str] | None = "auto") -> str | None:
    backend = normalize_backend_name(backend_name)
    if model_path is None or str(model_path).strip() == "" or str(model_path).strip().lower() == "auto":
        if backend == TABPFN_BACKEND:
            return "auto"
        if backend == TABPFN_REAL_BACKEND:
            return str(_tabpfn_cache_dir() / DEFAULT_TABPFN_REAL_REGRESSOR_CHECKPOINT)
        return None
    return str(model_path)


def backend_checkpoint_version(backend_name: str, model_path: str | os.PathLike[str] | None = "auto") -> str:
    resolved = resolve_model_path(backend_name, model_path)
    if resolved in (None, "auto"):
        return default_checkpoint_for_backend(backend_name)
    return Path(str(resolved)).name


def backend_metadata(backend_name: str, model_path: str | os.PathLike[str] | None = "auto") -> Dict[str, str]:
    backend = normalize_backend_name(backend_name)
    resolved = resolve_model_path(backend, model_path)
    package_name = "tabicl" if backend == TABICL_BACKEND else "tabpfn"
    return {
        "backend": backend,
        "backend_package": package_name,
        "checkpoint_version": backend_checkpoint_version(backend, model_path),
        "model_path": "auto" if resolved in (None, "auto") else str(resolved),
    }


def import_tabpfn_regressor():
    try:
        from tabpfn.regressor import TabPFNRegressor
    except Exception:
        try:
            from tabpfn import TabPFNRegressor  # type: ignore[attr-defined]
        except Exception as exc:  # pragma: no cover - exercised in integration environments
            raise RuntimeError("TabPFNRegressor is required for the selected backend.") from exc
    return TabPFNRegressor


def import_tabicl_regressor():
    _ensure_tabicl_src_path()
    try:
        from tabicl import TabICLRegressor
    except Exception as exc:  # pragma: no cover - exercised in integration environments
        raise RuntimeError("TabICLRegressor is required for the selected backend.") from exc
    return TabICLRegressor


def import_tabicl_quantile_distribution():
    _ensure_tabicl_src_path()
    try:
        from tabicl.model.quantile_dist import QuantileDistribution
    except Exception as exc:  # pragma: no cover - exercised in integration environments
        raise RuntimeError("TabICL QuantileDistribution is required for the selected backend.") from exc
    return QuantileDistribution


def _make_base_regressor_backend(
    backend_name: str,
    *,
    random_state: int,
    model_path: str | os.PathLike[str] | None = "auto",
    device: Optional[str] = None,
    n_estimators: Optional[int] = None,
    ignore_pretraining_limits: bool = True,
    softmax_temperature: float | None = None,
) -> Any:
    backend = normalize_backend_name(backend_name)

    if backend in {TABPFN_BACKEND, TABPFN_REAL_BACKEND}:
        regressor_cls = import_tabpfn_regressor()
        kwargs: Dict[str, Any] = {
            "random_state": random_state,
            "ignore_pretraining_limits": ignore_pretraining_limits,
        }
        resolved_model_path = resolve_model_path(backend, model_path)
        if resolved_model_path != "auto":
            kwargs["model_path"] = resolved_model_path
        if device is not None:
            kwargs["device"] = device
        if n_estimators is not None:
            kwargs["n_estimators"] = n_estimators
        if softmax_temperature is not None:
            kwargs["softmax_temperature"] = float(softmax_temperature)
        return regressor_cls(**kwargs)

    if softmax_temperature is not None:
        raise ValueError(
            "softmax_temperature is only supported for TabPFN backends ('tabpfn' and 'tabpfn_real')."
        )

    regressor_cls = import_tabicl_regressor()
    kwargs = {
        "random_state": random_state,
        "allow_auto_download": True,
        "checkpoint_version": DEFAULT_TABICL_REGRESSOR_CHECKPOINT,
    }
    if model_path not in (None, "", "auto"):
        kwargs["model_path"] = str(model_path)
    if device is not None:
        kwargs["device"] = device
    if n_estimators is not None:
        kwargs["n_estimators"] = n_estimators
    return regressor_cls(**kwargs)


def make_regressor_backend(
    backend_name: str,
    *,
    random_state: int,
    model_path: str | os.PathLike[str] | None = "auto",
    device: Optional[str] = None,
    n_estimators: Optional[int] = None,
    ignore_pretraining_limits: bool = True,
    local_context: object | None = None,
    softmax_temperature: float | None = None,
) -> Any:
    backend = normalize_backend_name(backend_name)
    local_cfg = ensure_local_context_config(local_context)
    if local_cfg.strategy == "global":
        return _make_base_regressor_backend(
            backend,
            random_state=random_state,
            model_path=model_path,
            device=device,
            n_estimators=n_estimators,
            ignore_pretraining_limits=ignore_pretraining_limits,
            softmax_temperature=softmax_temperature,
        )

    return LocalKNNRegressorWrapper(
        backend_name=backend,
        local_context=local_cfg,
        base_estimator_factory=lambda: _make_base_regressor_backend(
            backend,
            random_state=random_state,
            model_path=model_path,
            device=device,
            n_estimators=n_estimators,
            ignore_pretraining_limits=ignore_pretraining_limits,
            softmax_temperature=softmax_temperature,
        ),
    )


def _broadcast_eval_values(values: np.ndarray, n_samples: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        return np.full((n_samples, 1), float(arr))
    if arr.ndim == 1:
        if arr.size == n_samples:
            return arr.reshape(n_samples, 1)
        return np.tile(arr.reshape(1, -1), (n_samples, 1))
    if arr.ndim == 2 and arr.shape[0] == n_samples:
        return arr
    raise ValueError(
        f"Incompatible evaluation shape {arr.shape}; expected scalar, (n,), or (n, m) with n={n_samples}."
    )


class DistributionAdapter:
    """Small adapter interface for backend-specific predictive distributions."""

    def cdf(self, values: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def pdf(self, values: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def icdf(self, alphas: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def mean(self) -> np.ndarray:
        raise NotImplementedError

    def variance(self) -> np.ndarray:
        raise NotImplementedError


@dataclass
class TabPFNDistributionAdapter(DistributionAdapter):
    criterion: Any
    logits: Any

    def __post_init__(self) -> None:
        if not torch.is_tensor(self.logits):
            self.logits = torch.as_tensor(self.logits)
        self.device = self.criterion.borders.device  # type: ignore[attr-defined]
        self.dtype = self.criterion.borders.dtype  # type: ignore[attr-defined]
        self.logits = self.logits.to(self.device)

    @property
    def n_samples(self) -> int:
        return int(self.logits.shape[0])

    def _to_tensor(self, values: np.ndarray) -> torch.Tensor:
        value_matrix = _broadcast_eval_values(values, self.n_samples)
        return torch.as_tensor(value_matrix, dtype=self.dtype, device=self.device)

    def cdf(self, values: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            out = self.criterion.cdf(self.logits, self._to_tensor(values))  # type: ignore[attr-defined]
        return out.detach().cpu().numpy()

    def pdf(self, values: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            value_tensor = self._to_tensor(values)
            delta = torch.full_like(value_tensor, 1e-3)
            cdf_hi = self.criterion.cdf(self.logits, value_tensor + delta)  # type: ignore[attr-defined]
            cdf_lo = self.criterion.cdf(self.logits, value_tensor - delta)  # type: ignore[attr-defined]
            out = torch.clamp((cdf_hi - cdf_lo) / (2.0 * 1e-3), min=0.0)
        return out.detach().cpu().numpy()

    def icdf(self, alphas: np.ndarray) -> np.ndarray:
        alpha_arr = np.asarray(alphas, dtype=float).reshape(-1)
        outputs = []
        with torch.no_grad():
            for alpha in alpha_arr:
                out = self.criterion.icdf(self.logits, float(alpha))  # type: ignore[attr-defined]
                outputs.append(out.detach().cpu().numpy())
        return np.column_stack(outputs)

    def mean(self) -> np.ndarray:
        with torch.no_grad():
            out = self.criterion.mean(self.logits)  # type: ignore[attr-defined]
        return out.detach().cpu().numpy()

    def variance(self) -> np.ndarray:
        with torch.no_grad():
            out = self.criterion.variance(self.logits)  # type: ignore[attr-defined]
        return out.detach().cpu().numpy()


@dataclass
class TabICLDistributionAdapter(DistributionAdapter):
    raw_quantiles: Any

    def __post_init__(self) -> None:
        quantile_tensor = torch.as_tensor(self.raw_quantiles, dtype=torch.float32)
        if quantile_tensor.ndim != 2:
            raise ValueError(
                f"TabICL raw quantiles must have shape (n_samples, n_quantiles); got {tuple(quantile_tensor.shape)}."
            )
        quantile_dist_cls = import_tabicl_quantile_distribution()
        self.raw_quantiles = quantile_tensor
        self.device = quantile_tensor.device
        self.dtype = quantile_tensor.dtype
        self.distribution = quantile_dist_cls(quantile_tensor)

    @property
    def n_samples(self) -> int:
        return int(self.raw_quantiles.shape[0])

    def _to_tensor(self, values: np.ndarray) -> torch.Tensor:
        value_matrix = _broadcast_eval_values(values, self.n_samples)
        return torch.as_tensor(value_matrix, dtype=self.dtype, device=self.device)

    def cdf(self, values: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            out = self.distribution.cdf(self._to_tensor(values))
        return out.detach().cpu().numpy()

    def pdf(self, values: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            out = self.distribution.pdf(self._to_tensor(values))
        return out.detach().cpu().numpy()

    def icdf(self, alphas: np.ndarray) -> np.ndarray:
        alpha_tensor = torch.as_tensor(np.asarray(alphas, dtype=float), dtype=self.dtype, device=self.device)
        if alpha_tensor.ndim == 0:
            alpha_tensor = alpha_tensor.reshape(1)
        with torch.no_grad():
            out = self.distribution.icdf(alpha_tensor)
        return out.detach().cpu().numpy()

    def mean(self) -> np.ndarray:
        with torch.no_grad():
            out = self.distribution.mean()
        return out.detach().cpu().numpy()

    def variance(self) -> np.ndarray:
        with torch.no_grad():
            out = self.distribution.variance()
        return out.detach().cpu().numpy()


def distribution_adapter_from_output(full_output: Mapping[str, object]) -> DistributionAdapter:
    maybe_dist = full_output.get("distribution")
    if maybe_dist is not None and all(hasattr(maybe_dist, method) for method in ("cdf", "pdf", "icdf", "mean", "variance")):
        return maybe_dist

    if "criterion" in full_output and "logits" in full_output:
        return TabPFNDistributionAdapter(
            criterion=full_output["criterion"],
            logits=full_output["logits"],
        )

    raise KeyError("Full output did not contain a supported predictive distribution payload.")


def cdf_from_distribution_output(
    full_output: Mapping[str, object],
    values: np.ndarray,
    *,
    squeeze_last: bool = True,
) -> np.ndarray:
    cdf_np = np.asarray(distribution_adapter_from_output(full_output).cdf(values), dtype=float)
    if squeeze_last and cdf_np.ndim == 2 and cdf_np.shape[1] == 1:
        cdf_np = cdf_np[:, 0]
    if squeeze_last and cdf_np.ndim > 1 and cdf_np.shape[0] == 1:
        cdf_np = cdf_np[0]
    return cdf_np


def pdf_from_distribution_output(
    full_output: Mapping[str, object],
    values: np.ndarray,
    *,
    squeeze_last: bool = True,
) -> np.ndarray:
    pdf_np = np.asarray(distribution_adapter_from_output(full_output).pdf(values), dtype=float)
    if squeeze_last and pdf_np.ndim == 2 and pdf_np.shape[1] == 1:
        pdf_np = pdf_np[:, 0]
    if squeeze_last and pdf_np.ndim > 1 and pdf_np.shape[0] == 1:
        pdf_np = pdf_np[0]
    return pdf_np


def predict_distribution(
    estimator: Any,
    features: np.ndarray,
    *,
    backend_name: str,
    quantiles: tuple[float, ...] = (),
) -> Dict[str, object]:
    backend = normalize_backend_name(backend_name)
    if getattr(estimator, "is_local_context_wrapper_", False):
        full_output = estimator.predict(
            features,
            output_type="full",
            quantiles=list(quantiles),
            alphas=list(quantiles),
        )
        if not isinstance(full_output, dict):
            raise RuntimeError("LocalKNNRegressorWrapper full output not returned as a mapping.")
        full_output = dict(full_output)
        full_output["backend_name"] = backend
        return full_output

    if backend in {TABPFN_BACKEND, TABPFN_REAL_BACKEND}:
        kwargs: Dict[str, Any] = {}
        if quantiles:
            kwargs["quantiles"] = list(quantiles)
        full_output = estimator.predict(features, output_type="full", **kwargs)
        if not isinstance(full_output, dict):
            raise RuntimeError("TabPFN full output not returned as a mapping.")
        full_output = dict(full_output)
        full_output["backend_name"] = backend
        return full_output

    raw_quantiles = estimator.predict(features, output_type="raw_quantiles")
    return {
        "backend_name": backend,
        "raw_quantiles": raw_quantiles,
        "distribution": TabICLDistributionAdapter(raw_quantiles=raw_quantiles),
    }


def predict_mean(
    estimator: Any,
    features: np.ndarray,
    *,
    backend_name: str,
) -> np.ndarray:
    """Predict the mean response explicitly from the backend predictive interface."""
    backend = normalize_backend_name(backend_name)
    del backend
    mean_values = estimator.predict(features, output_type="mean")
    return np.asarray(mean_values, dtype=float).reshape(-1)


def predict_quantiles(
    estimator: Any,
    features: np.ndarray,
    *,
    backend_name: str,
    quantiles: tuple[float, ...],
) -> np.ndarray:
    backend = normalize_backend_name(backend_name)
    if not quantiles:
        raise ValueError("At least one quantile level is required.")
    if backend in {TABPFN_BACKEND, TABPFN_REAL_BACKEND}:
        q_values = estimator.predict(features, output_type="quantiles", quantiles=list(quantiles))
        if isinstance(q_values, list):
            matrix = np.column_stack([np.asarray(arr, dtype=float).reshape(-1) for arr in q_values])
        else:
            matrix = np.asarray(q_values, dtype=float)
        return matrix
    return np.asarray(
        estimator.predict(features, output_type="quantiles", alphas=list(quantiles)),
        dtype=float,
    )
