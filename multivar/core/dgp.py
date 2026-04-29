#!/usr/bin/env python3
"""Data-generating processes and ground-truth interventional quantities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal, norm


DGP1_LINEAR = "DGP1_LINEAR"
DGP2_NONLINEAR = "DGP2_NONLINEAR"
DGP3_PRE_ADDITIVE = "DGP3_PRE_ADDITIVE"
DGP4_PIECEWISE = "DGP4_PIECEWISE"
DGP5_SOFTPLUS = "DGP5_SOFTPLUS"
SUPPORTED_DGPS = (DGP1_LINEAR, DGP2_NONLINEAR, DGP3_PRE_ADDITIVE, DGP4_PIECEWISE, DGP5_SOFTPLUS)
_PRE_ADDITIVE_DGPS = (DGP3_PRE_ADDITIVE, DGP4_PIECEWISE, DGP5_SOFTPLUS)

_DGP_ALIASES: Dict[str, str] = {
    "D1": DGP1_LINEAR,
    "DGP1": DGP1_LINEAR,
    DGP1_LINEAR: DGP1_LINEAR,
    "LINEAR": DGP1_LINEAR,
    "D2": DGP2_NONLINEAR,
    "DGP2": DGP2_NONLINEAR,
    DGP2_NONLINEAR: DGP2_NONLINEAR,
    "NONLINEAR": DGP2_NONLINEAR,
    "D3": DGP3_PRE_ADDITIVE,
    "DGP3": DGP3_PRE_ADDITIVE,
    DGP3_PRE_ADDITIVE: DGP3_PRE_ADDITIVE,
    "PRE_ADDITIVE": DGP3_PRE_ADDITIVE,
    "D4": DGP4_PIECEWISE,
    "DGP4": DGP4_PIECEWISE,
    DGP4_PIECEWISE: DGP4_PIECEWISE,
    "PIECEWISE": DGP4_PIECEWISE,
    "D5": DGP5_SOFTPLUS,
    "DGP5": DGP5_SOFTPLUS,
    DGP5_SOFTPLUS: DGP5_SOFTPLUS,
    "SOFTPLUS": DGP5_SOFTPLUS,
}

_DGP2_GH_ORDER = 31
_DGP2_GH_RAW_NODES, _DGP2_GH_RAW_WEIGHTS = np.polynomial.hermite.hermgauss(_DGP2_GH_ORDER)
_DGP2_GH_NODES = np.sqrt(2.0) * _DGP2_GH_RAW_NODES
_DGP2_GH_WEIGHTS = _DGP2_GH_RAW_WEIGHTS / np.sqrt(np.pi)

_DGP3_GH_ORDER_1D = 301
_DGP3_GH_RAW_NODES_1D, _DGP3_GH_RAW_WEIGHTS_1D = np.polynomial.hermite.hermgauss(_DGP3_GH_ORDER_1D)
_DGP3_GH_STD_NODES_1D = np.sqrt(2.0) * _DGP3_GH_RAW_NODES_1D
_DGP3_GH_WEIGHTS_1D = _DGP3_GH_RAW_WEIGHTS_1D / np.sqrt(np.pi)
_DGP3_GH_VAR2_NODES_1D = np.sqrt(2.0) * _DGP3_GH_STD_NODES_1D

_DGP3_GH_ORDER_2D = 31
_DGP3_GH_RAW_NODES_2D, _DGP3_GH_RAW_WEIGHTS_2D = np.polynomial.hermite.hermgauss(_DGP3_GH_ORDER_2D)
_DGP3_GH_STD_NODES_2D = np.sqrt(2.0) * _DGP3_GH_RAW_NODES_2D
_DGP3_GH_WEIGHTS_2D = _DGP3_GH_RAW_WEIGHTS_2D / np.sqrt(np.pi)
_DGP3_GH_WEIGHT_GRID_2D = np.outer(_DGP3_GH_WEIGHTS_2D, _DGP3_GH_WEIGHTS_2D)


@dataclass(frozen=True)
class DGPConfig:
    n: int
    seed: int
    dgp_code: str
    rho_eps: float

    def normalize(self) -> "DGPConfig":
        return DGPConfig(
            n=int(self.n),
            seed=int(self.seed),
            dgp_code=normalize_dgp_code(self.dgp_code),
            rho_eps=float(self.rho_eps),
        )


def normalize_dgp_code(code: str) -> str:
    key = str(code).strip().upper()
    if key not in _DGP_ALIASES:
        supported = ", ".join(SUPPORTED_DGPS)
        raise ValueError(f"Unsupported DGP code '{code}'. Supported: {supported}")
    return _DGP_ALIASES[key]


def validate_rho_eps(rho_eps: float) -> float:
    value = float(rho_eps)
    if not -0.999 < value < 0.999:
        raise ValueError(f"rho_eps must be in (-0.999, 0.999), got {value}")
    return value


def _draw_correlated_eps(n: int, rho_eps: float, rng: np.random.Generator) -> np.ndarray:
    rho = validate_rho_eps(rho_eps)
    cov = np.array([[1.0, rho], [rho, 1.0]], dtype=float)
    return rng.multivariate_normal(mean=np.zeros(2), cov=cov, size=n)


def g1(x: np.ndarray | float) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    return 2.0 * x_arr + 3.0 * np.sin(2.0 * x_arr)


def g2(x: np.ndarray | float) -> np.ndarray:
    return 0.5 * g1(x)


def _piecewise_g1(x: np.ndarray | float) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    return np.where(
        x_arr < 0.0,
        x_arr,
        np.where(x_arr <= 1.0, 2.0 * x_arr, 2.0 + 0.5 * (x_arr - 1.0)),
    )


def _softplus_g1(x: np.ndarray | float) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    return np.logaddexp(0.0, 2.0 * x_arr)


def _pre_additive_g1(dgp_code: str, x: np.ndarray | float) -> np.ndarray:
    if dgp_code == DGP3_PRE_ADDITIVE:
        return g1(x)
    if dgp_code == DGP4_PIECEWISE:
        return _piecewise_g1(x)
    if dgp_code == DGP5_SOFTPLUS:
        return _softplus_g1(x)
    raise ValueError(f"Expected a pre-additive DGP code, got {dgp_code}")


def _pre_additive_g2(dgp_code: str, x: np.ndarray | float) -> np.ndarray:
    return 0.5 * _pre_additive_g1(dgp_code, x)


def _is_pre_additive_code(dgp_code: str) -> bool:
    return dgp_code in _PRE_ADDITIVE_DGPS


def _require_scalar_x(x: np.ndarray | float | None) -> float:
    if x is None:
        raise ValueError("x is required for this DGP quantity")
    x_arr = np.asarray(x, dtype=float)
    if x_arr.size != 1:
        raise ValueError(f"Expected scalar x, got shape {x_arr.shape}")
    return float(x_arr.reshape(-1)[0])


def _dgp2_latent_grid(x: np.ndarray | float) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    return x_arr[..., np.newaxis] + _DGP2_GH_NODES


def _dgp2_signal_moments(x: np.ndarray | float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    latent = _dgp2_latent_grid(x)
    g1_vals = np.asarray(g1(latent), dtype=float)
    g2_vals = np.asarray(g2(latent), dtype=float)

    w = _DGP2_GH_WEIGHTS
    mean1 = np.sum(g1_vals * w, axis=-1)
    mean2 = np.sum(g2_vals * w, axis=-1)
    e11 = np.sum((g1_vals * g1_vals) * w, axis=-1)
    e22 = np.sum((g2_vals * g2_vals) * w, axis=-1)
    e12 = np.sum((g1_vals * g2_vals) * w, axis=-1)
    return mean1, mean2, e11, e22, e12


def _dgp2_outcome_moments_under_do(
    x: np.ndarray | float,
    rho_eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rho = validate_rho_eps(rho_eps)
    mean1, mean2, e11, e22, e12 = _dgp2_signal_moments(x)
    var1 = np.maximum(e11 - mean1 * mean1, 0.0) + 1.0
    var2 = np.maximum(e22 - mean2 * mean2, 0.0) + 1.0
    cov12 = (e12 - mean1 * mean2) + rho
    return mean1, mean2, var1, var2, cov12


def _dgp2_marginal_mean_var(component: int, x: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
    if component not in (1, 2):
        raise ValueError(f"component must be 1 or 2, got {component}")
    mean1, mean2, e11, e22, _ = _dgp2_signal_moments(x)
    if component == 1:
        mean = mean1
        var = np.maximum(e11 - mean1 * mean1, 0.0) + 1.0
    else:
        mean = mean2
        var = np.maximum(e22 - mean2 * mean2, 0.0) + 1.0
    return mean, var


def _dgp2_marginal_cdf(component: int, x: np.ndarray | float, y: np.ndarray | float) -> np.ndarray:
    if component not in (1, 2):
        raise ValueError(f"component must be 1 or 2, got {component}")
    x_arr, y_arr = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    latent = x_arr[..., np.newaxis] + _DGP2_GH_NODES
    signal_vals = g1(latent) if component == 1 else g2(latent)
    cdf_terms = norm.cdf(y_arr[..., np.newaxis] - np.asarray(signal_vals, dtype=float))
    cdf_vals = np.sum(cdf_terms * _DGP2_GH_WEIGHTS, axis=-1)
    return np.clip(np.asarray(cdf_vals, dtype=float), 0.0, 1.0)


def _dgp2_marginal_ppf(component: int, x: np.ndarray | float, u: np.ndarray | float) -> np.ndarray:
    return _marginal_ppf_via_numerical_cdf(
        component,
        x,
        u,
        marginal_mean_var_fn=_dgp2_marginal_mean_var,
        marginal_cdf_fn=_dgp2_marginal_cdf,
    )


def _marginal_ppf_via_numerical_cdf(
    component: int,
    x: np.ndarray | float,
    u: np.ndarray | float,
    *,
    marginal_mean_var_fn,
    marginal_cdf_fn,
) -> np.ndarray:
    u_arr = np.asarray(u, dtype=float)
    if np.any((u_arr <= 0.0) | (u_arr >= 1.0)):
        raise ValueError("u must be strictly inside (0,1)")

    x_arr, u_target = np.broadcast_arrays(np.asarray(x, dtype=float), u_arr)
    mean, var = marginal_mean_var_fn(component, x_arr)
    scale = np.maximum(np.sqrt(np.maximum(var, 1e-12)), 1.0)

    lo_span = 12.0 * scale
    hi_span = 12.0 * scale
    lo = mean - lo_span
    hi = mean + hi_span

    for _ in range(8):
        cdf_lo = marginal_cdf_fn(component, x_arr, lo)
        cdf_hi = marginal_cdf_fn(component, x_arr, hi)
        need_lo = cdf_lo > u_target
        need_hi = cdf_hi < u_target
        if not np.any(need_lo | need_hi):
            break
        lo_span = np.where(need_lo, lo_span * 2.0, lo_span)
        hi_span = np.where(need_hi, hi_span * 2.0, hi_span)
        lo = mean - lo_span
        hi = mean + hi_span

    for _ in range(60):
        mid = 0.5 * (lo + hi)
        cdf_mid = marginal_cdf_fn(component, x_arr, mid)
        move_right = cdf_mid < u_target
        lo = np.where(move_right, mid, lo)
        hi = np.where(move_right, hi, mid)
    return 0.5 * (lo + hi)


def _dgp2_joint_cdf_vectorized(
    rho_eps: float,
    x: float,
    y1: np.ndarray | float,
    y2: np.ndarray | float,
) -> np.ndarray:
    rho = validate_rho_eps(rho_eps)
    x_scalar = _require_scalar_x(x)
    y1_arr, y2_arr = np.broadcast_arrays(np.asarray(y1, dtype=float), np.asarray(y2, dtype=float))
    y1_flat = y1_arr.reshape(-1)
    y2_flat = y2_arr.reshape(-1)

    cov_eps = np.array([[1.0, rho], [rho, 1.0]], dtype=float)
    eps_dist = multivariate_normal(mean=np.zeros(2), cov=cov_eps)
    latent = x_scalar + _DGP2_GH_NODES
    mu1_nodes = np.asarray(g1(latent), dtype=float).reshape(-1)
    mu2_nodes = np.asarray(g2(latent), dtype=float).reshape(-1)

    cdf_flat = np.zeros_like(y1_flat, dtype=float)
    for w, mu1, mu2 in zip(_DGP2_GH_WEIGHTS, mu1_nodes, mu2_nodes):
        points = np.column_stack([y1_flat - mu1, y2_flat - mu2])
        cdf_flat += float(w) * np.asarray(eps_dist.cdf(points), dtype=float)
    cdf_flat = np.clip(cdf_flat, 0.0, 1.0)
    return cdf_flat.reshape(y1_arr.shape)


def _pre_additive_latent_grid(x: np.ndarray | float) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    return x_arr[..., np.newaxis] + _DGP3_GH_VAR2_NODES_1D


def _pre_additive_signal_moments(
    dgp_code: str,
    x: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    latent = _pre_additive_latent_grid(x)
    g1_vals = np.asarray(_pre_additive_g1(dgp_code, latent), dtype=float)
    g2_vals = np.asarray(_pre_additive_g2(dgp_code, latent), dtype=float)

    w = _DGP3_GH_WEIGHTS_1D
    mean1 = np.sum(g1_vals * w, axis=-1)
    mean2 = np.sum(g2_vals * w, axis=-1)
    e11 = np.sum((g1_vals * g1_vals) * w, axis=-1)
    e22 = np.sum((g2_vals * g2_vals) * w, axis=-1)
    return mean1, mean2, e11, e22


def _pre_additive_outcome_moments_under_do(
    dgp_code: str,
    x: np.ndarray | float,
    rho_eps: float,
) -> tuple[float, float, float, float, float]:
    rho = validate_rho_eps(rho_eps)
    x_scalar = _require_scalar_x(x)

    mean1_arr, mean2_arr, e11_arr, e22_arr = _pre_additive_signal_moments(dgp_code, x_scalar)
    mean1 = float(np.asarray(mean1_arr, dtype=float))
    mean2 = float(np.asarray(mean2_arr, dtype=float))
    e11 = float(np.asarray(e11_arr, dtype=float))
    e22 = float(np.asarray(e22_arr, dtype=float))

    cov_s = np.array([[2.0, 1.0 + rho], [1.0 + rho, 2.0]], dtype=float)
    chol = np.linalg.cholesky(cov_s)
    z1 = _DGP3_GH_STD_NODES_2D.reshape(-1, 1)
    z2 = _DGP3_GH_STD_NODES_2D.reshape(1, -1)
    s1 = x_scalar + chol[0, 0] * z1 + chol[0, 1] * z2
    s2 = x_scalar + chol[1, 0] * z1 + chol[1, 1] * z2
    e12 = float(
        np.sum(
            np.asarray(_pre_additive_g1(dgp_code, s1), dtype=float)
            * np.asarray(_pre_additive_g2(dgp_code, s2), dtype=float)
            * _DGP3_GH_WEIGHT_GRID_2D
        )
    )

    var1 = max(e11 - mean1 * mean1, 0.0)
    var2 = max(e22 - mean2 * mean2, 0.0)
    cov12 = e12 - mean1 * mean2
    return mean1, mean2, var1, var2, cov12


def _pre_additive_marginal_mean_var(
    dgp_code: str,
    component: int,
    x: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray]:
    if component not in (1, 2):
        raise ValueError(f"component must be 1 or 2, got {component}")
    mean1, mean2, e11, e22 = _pre_additive_signal_moments(dgp_code, x)
    if component == 1:
        mean = mean1
        var = np.maximum(e11 - mean1 * mean1, 0.0)
    else:
        mean = mean2
        var = np.maximum(e22 - mean2 * mean2, 0.0)
    return mean, var


def _pre_additive_marginal_cdf(
    dgp_code: str,
    component: int,
    x: np.ndarray | float,
    y: np.ndarray | float,
) -> np.ndarray:
    if component not in (1, 2):
        raise ValueError(f"component must be 1 or 2, got {component}")
    x_arr, y_arr = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    latent = _pre_additive_latent_grid(x_arr)
    signal_fn = _pre_additive_g1 if component == 1 else _pre_additive_g2
    signal_vals = np.asarray(signal_fn(dgp_code, latent), dtype=float)
    cdf_terms = (signal_vals <= y_arr[..., np.newaxis]).astype(float)
    cdf_vals = np.sum(cdf_terms * _DGP3_GH_WEIGHTS_1D, axis=-1)
    return np.clip(np.asarray(cdf_vals, dtype=float), 0.0, 1.0)


def _pre_additive_marginal_ppf(
    dgp_code: str,
    component: int,
    x: np.ndarray | float,
    u: np.ndarray | float,
) -> np.ndarray:
    return _marginal_ppf_via_numerical_cdf(
        component,
        x,
        u,
        marginal_mean_var_fn=lambda comp, x_val: _pre_additive_marginal_mean_var(dgp_code, comp, x_val),
        marginal_cdf_fn=lambda comp, x_val, y_val: _pre_additive_marginal_cdf(dgp_code, comp, x_val, y_val),
    )


def _pre_additive_joint_cdf_vectorized(
    dgp_code: str,
    rho_eps: float,
    x: float,
    y1: np.ndarray | float,
    y2: np.ndarray | float,
) -> np.ndarray:
    rho = validate_rho_eps(rho_eps)
    x_scalar = _require_scalar_x(x)
    y1_arr, y2_arr = np.broadcast_arrays(np.asarray(y1, dtype=float), np.asarray(y2, dtype=float))
    y1_flat = y1_arr.reshape(-1)
    y2_flat = y2_arr.reshape(-1)

    cov_s = np.array([[2.0, 1.0 + rho], [1.0 + rho, 2.0]], dtype=float)
    chol = np.linalg.cholesky(cov_s)
    z1 = _DGP3_GH_STD_NODES_2D.reshape(-1, 1)
    z2 = _DGP3_GH_STD_NODES_2D.reshape(1, -1)
    s1 = x_scalar + chol[0, 0] * z1 + chol[0, 1] * z2
    s2 = x_scalar + chol[1, 0] * z1 + chol[1, 1] * z2

    y1_nodes = np.asarray(_pre_additive_g1(dgp_code, s1), dtype=float).reshape(-1)
    y2_nodes = np.asarray(_pre_additive_g2(dgp_code, s2), dtype=float).reshape(-1)
    weights = _DGP3_GH_WEIGHT_GRID_2D.reshape(-1)

    cdf_terms = (y1_nodes[:, np.newaxis] <= y1_flat[np.newaxis, :]) & (
        y2_nodes[:, np.newaxis] <= y2_flat[np.newaxis, :]
    )
    cdf_flat = np.sum(cdf_terms * weights[:, np.newaxis], axis=0, dtype=float)
    cdf_flat = np.clip(cdf_flat, 0.0, 1.0)
    return cdf_flat.reshape(y1_arr.shape)


def sample_observational_data(cfg: DGPConfig) -> pd.DataFrame:
    cfg = cfg.normalize()
    rho_eps = validate_rho_eps(cfg.rho_eps)
    rng = np.random.default_rng(cfg.seed)

    z = rng.uniform(0.0, 3.0, size=cfg.n)
    h = rng.normal(0.0, 1.0, size=cfg.n)
    eps_x = rng.normal(0.0, 1.0, size=cfg.n)
    eps = _draw_correlated_eps(cfg.n, rho_eps, rng)

    x = z + h + eps_x
    eps1 = eps[:, 0]
    eps2 = eps[:, 1]

    if cfg.dgp_code == DGP1_LINEAR:
        y1 = x - 3.0 * h + eps1
        y2 = 0.5 * x - 1.0 * h + eps2
    elif cfg.dgp_code == DGP2_NONLINEAR:
        y1 = g1(x + h) + eps1
        y2 = g2(x + h) + eps2
    elif _is_pre_additive_code(cfg.dgp_code):
        y1 = _pre_additive_g1(cfg.dgp_code, x + h + eps1)
        y2 = _pre_additive_g2(cfg.dgp_code, x + h + eps2)
    else:
        raise ValueError(f"Unexpected dgp_code: {cfg.dgp_code}")

    return pd.DataFrame(
        {
            "Z": z,
            "H": h,
            "eps_X": eps_x,
            "X": x,
            "eps1": eps1,
            "eps2": eps2,
            "Y1": y1,
            "Y2": y2,
        }
    )


def sample_interventional_data(
    dgp_code: str,
    rho_eps: float,
    x: float,
    n: int,
    *,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Sample the true joint law of (Y1, Y2) under do(X=x)."""
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if seed is not None and rng is not None:
        raise ValueError("Provide at most one of seed or rng")

    code = normalize_dgp_code(dgp_code)
    x_scalar = _require_scalar_x(x)
    rho = validate_rho_eps(rho_eps)
    rng = rng or np.random.default_rng(seed)

    h = rng.normal(0.0, 1.0, size=int(n))
    eps = _draw_correlated_eps(int(n), rho, rng)
    eps1 = eps[:, 0]
    eps2 = eps[:, 1]
    x_arr = np.full(int(n), x_scalar, dtype=float)

    if code == DGP1_LINEAR:
        y1 = x_arr - 3.0 * h + eps1
        y2 = 0.5 * x_arr - 1.0 * h + eps2
    elif code == DGP2_NONLINEAR:
        y1 = g1(x_arr + h) + eps1
        y2 = g2(x_arr + h) + eps2
    elif _is_pre_additive_code(code):
        y1 = _pre_additive_g1(code, x_arr + h + eps1)
        y2 = _pre_additive_g2(code, x_arr + h + eps2)
    else:
        raise ValueError(f"Unexpected dgp_code: {code}")

    return pd.DataFrame(
        {
            "X": x_arr,
            "H": h,
            "eps1": eps1,
            "eps2": eps2,
            "Y1": y1,
            "Y2": y2,
        }
    )


def mean_under_do(dgp_code: str, x: np.ndarray | float) -> np.ndarray:
    code = normalize_dgp_code(dgp_code)
    x_arr = np.asarray(x, dtype=float)
    if code == DGP1_LINEAR:
        mu1 = x_arr
        mu2 = 0.5 * x_arr
    elif code == DGP2_NONLINEAR:
        mu1, mu2, _, _, _ = _dgp2_outcome_moments_under_do(x_arr, rho_eps=0.0)
    elif _is_pre_additive_code(code):
        mu1, mu2, _, _ = _pre_additive_signal_moments(code, x_arr)
    else:
        raise ValueError(f"Unexpected dgp_code: {code}")
    return np.stack([mu1, mu2], axis=-1)


def covariance_under_do(
    dgp_code: str,
    rho_eps: float,
    x: np.ndarray | float | None = None,
) -> np.ndarray:
    code = normalize_dgp_code(dgp_code)
    rho = validate_rho_eps(rho_eps)
    if code == DGP1_LINEAR:
        return np.array([[10.0, 3.0 + rho], [3.0 + rho, 2.0]], dtype=float)
    if code == DGP2_NONLINEAR:
        x_scalar = _require_scalar_x(x)
        _, _, var1, var2, cov12 = _dgp2_outcome_moments_under_do(x_scalar, rho)
        return np.array([[var1, cov12], [cov12, var2]], dtype=float)
    if _is_pre_additive_code(code):
        x_scalar = _require_scalar_x(x)
        _, _, var1, var2, cov12 = _pre_additive_outcome_moments_under_do(code, x_scalar, rho)
        return np.array([[var1, cov12], [cov12, var2]], dtype=float)
    raise ValueError(f"Unexpected dgp_code: {code}")


def true_marginal_mean(dgp_code: str, component: int, x: np.ndarray | float) -> np.ndarray:
    means = mean_under_do(dgp_code, x)
    if component not in (1, 2):
        raise ValueError(f"component must be 1 or 2, got {component}")
    return means[..., component - 1]


def true_marginal_std(
    dgp_code: str,
    rho_eps: float,
    component: int,
    x: np.ndarray | float | None = None,
) -> np.ndarray:
    code = normalize_dgp_code(dgp_code)
    if component not in (1, 2):
        raise ValueError(f"component must be 1 or 2, got {component}")
    if code == DGP1_LINEAR:
        cov = covariance_under_do(code, rho_eps, x=x)
        return np.sqrt(cov[component - 1, component - 1])
    if code == DGP2_NONLINEAR:
        if x is None:
            raise ValueError("x is required for DGP2 marginal standard deviation")
        _, var = _dgp2_marginal_mean_var(component, x)
        return np.sqrt(var)
    if _is_pre_additive_code(code):
        if x is None:
            raise ValueError(f"x is required for {code} marginal standard deviation")
        _, var = _pre_additive_marginal_mean_var(code, component, x)
        return np.sqrt(var)
    raise ValueError(f"Unexpected dgp_code: {code}")


def true_marginal_cdf(
    dgp_code: str,
    rho_eps: float,
    component: int,
    x: np.ndarray | float,
    y: np.ndarray | float,
) -> np.ndarray:
    code = normalize_dgp_code(dgp_code)
    if code == DGP1_LINEAR:
        mu = true_marginal_mean(code, component, x)
        sigma = true_marginal_std(code, rho_eps, component, x)
        y_arr = np.asarray(y, dtype=float)
        return norm.cdf((y_arr - mu) / sigma)
    if code == DGP2_NONLINEAR:
        return _dgp2_marginal_cdf(component, x, y)
    if _is_pre_additive_code(code):
        return _pre_additive_marginal_cdf(code, component, x, y)
    raise ValueError(f"Unexpected dgp_code: {code}")


def true_marginal_ppf(
    dgp_code: str,
    rho_eps: float,
    component: int,
    x: np.ndarray | float,
    u: np.ndarray | float,
) -> np.ndarray:
    code = normalize_dgp_code(dgp_code)
    if code == DGP1_LINEAR:
        mu = true_marginal_mean(code, component, x)
        sigma = true_marginal_std(code, rho_eps, component, x)
        u_arr = np.asarray(u, dtype=float)
        return mu + sigma * norm.ppf(u_arr)
    if code == DGP2_NONLINEAR:
        return _dgp2_marginal_ppf(component, x, u)
    if _is_pre_additive_code(code):
        return _pre_additive_marginal_ppf(code, component, x, u)
    raise ValueError(f"Unexpected dgp_code: {code}")


def true_joint_cdf(
    dgp_code: str,
    rho_eps: float,
    x: float,
    y1: float,
    y2: float,
) -> float:
    return float(true_joint_cdf_vectorized(dgp_code, rho_eps, x, y1, y2))


def true_joint_cdf_vectorized(
    dgp_code: str,
    rho_eps: float,
    x: float,
    y1: np.ndarray | float,
    y2: np.ndarray | float,
) -> np.ndarray:
    code = normalize_dgp_code(dgp_code)
    y1_arr, y2_arr = np.broadcast_arrays(np.asarray(y1, dtype=float), np.asarray(y2, dtype=float))

    if code == DGP1_LINEAR:
        points = np.column_stack([y1_arr.reshape(-1), y2_arr.reshape(-1)])
        mu = mean_under_do(code, x).reshape(-1)
        cov = covariance_under_do(code, rho_eps, x=x)
        cdf_vals = multivariate_normal(mean=mu, cov=cov).cdf(points)
        return np.asarray(cdf_vals, dtype=float).reshape(y1_arr.shape)
    if code == DGP2_NONLINEAR:
        return _dgp2_joint_cdf_vectorized(rho_eps, x, y1_arr, y2_arr)
    if _is_pre_additive_code(code):
        return _pre_additive_joint_cdf_vectorized(code, rho_eps, x, y1_arr, y2_arr)
    raise ValueError(f"Unexpected dgp_code: {code}")


def true_joint_tail_probability(
    dgp_code: str,
    rho_eps: float,
    x: float,
    a: float,
    b: float,
) -> float:
    f1 = float(true_marginal_cdf(dgp_code, rho_eps, 1, x, a))
    f2 = float(true_marginal_cdf(dgp_code, rho_eps, 2, x, b))
    f12 = true_joint_cdf(dgp_code, rho_eps, x, a, b)
    return float(1.0 - f1 - f2 + f12)


def true_corr_under_do(
    dgp_code: str,
    rho_eps: float,
    x: np.ndarray | float | None = None,
) -> float:
    cov = covariance_under_do(normalize_dgp_code(dgp_code), rho_eps, x=x)
    return float(cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]))


def true_tau_under_do(
    dgp_code: str,
    rho_eps: float,
    x: np.ndarray | float | None = None,
) -> float:
    corr = true_corr_under_do(dgp_code, rho_eps, x=x)
    return float((2.0 / np.pi) * np.arcsin(corr))


def parse_code_list(codes: Iterable[str]) -> list[str]:
    return [normalize_dgp_code(code) for code in codes]
