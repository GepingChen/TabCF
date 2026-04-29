"""
Shared V-integral helpers for Stage 2 mean and distribution recovery.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.integrate import simpson


DEFAULT_MU_INTEGRATOR = "gauss_legendre"
DEFAULT_GAUSS_LEGENDRE_ORDER = 18
DEFAULT_V_INTEGRATION_POINTS = 100
DEFAULT_TARGET_PRODUCTS = 20000
SUPPORTED_MU_INTEGRATORS = ("simpson", "gauss_legendre")
V_EPSILON = 1e-6


@dataclass(frozen=True)
class MuIntegratorConfig:
    method: str = DEFAULT_MU_INTEGRATOR
    n_v_points: int = DEFAULT_V_INTEGRATION_POINTS
    gauss_legendre_order: int = DEFAULT_GAUSS_LEGENDRE_ORDER
    max_points_per_batch: int | None = None

    def __post_init__(self) -> None:
        method = str(self.method).strip().lower()
        if method not in SUPPORTED_MU_INTEGRATORS:
            raise ValueError(f"Unsupported mu integrator '{self.method}'. Expected one of {SUPPORTED_MU_INTEGRATORS}.")
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "n_v_points", int(self.n_v_points))
        object.__setattr__(self, "gauss_legendre_order", int(self.gauss_legendre_order))
        if self.max_points_per_batch is not None:
            object.__setattr__(self, "max_points_per_batch", int(self.max_points_per_batch))


def create_simpson_v_grid(n_points: int = DEFAULT_V_INTEGRATION_POINTS) -> np.ndarray:
    n_points = int(n_points)
    if n_points <= 1:
        raise ValueError("Simpson integration requires at least 2 points.")
    if n_points % 2 == 0:
        n_points += 1
    return np.linspace(V_EPSILON, 1.0 - V_EPSILON, n_points, dtype=float)


def gauss_legendre_rule(order: int = DEFAULT_GAUSS_LEGENDRE_ORDER) -> tuple[np.ndarray, np.ndarray]:
    order = int(order)
    if order <= 0:
        raise ValueError("Gauss-Legendre order must be positive.")
    nodes, weights = np.polynomial.legendre.leggauss(order)
    mapped_nodes = 0.5 * (nodes + 1.0)
    mapped_weights = 0.5 * weights
    return mapped_nodes.astype(float), mapped_weights.astype(float)


def resolve_v_rule(integrator_cfg: MuIntegratorConfig) -> tuple[np.ndarray, np.ndarray | None]:
    if integrator_cfg.method == "simpson":
        return create_simpson_v_grid(integrator_cfg.n_v_points), None
    if integrator_cfg.method == "gauss_legendre":
        return gauss_legendre_rule(integrator_cfg.gauss_legendre_order)
    raise ValueError(f"Unsupported mu integrator: {integrator_cfg.method}")


def resolve_max_points_per_batch(
    n_x_points: int,
    n_v_points: int,
    *,
    max_points_per_batch: int | None = None,
    target_products: int = DEFAULT_TARGET_PRODUCTS,
) -> int:
    if max_points_per_batch is not None:
        return max(1, int(max_points_per_batch))
    if n_x_points <= 0:
        return 1
    if n_x_points * n_v_points <= int(target_products):
        return int(n_x_points)
    return max(1, int(target_products) // int(n_v_points))


def integrate_values(
    values: np.ndarray,
    v_grid: np.ndarray,
    *,
    integrator_cfg: MuIntegratorConfig,
    axis: int = -1,
    v_weights: np.ndarray | None = None,
) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    grid = np.asarray(v_grid, dtype=float).reshape(-1)

    if arr.shape[axis] != grid.size:
        raise ValueError(
            f"Grid length mismatch: values axis has {arr.shape[axis]} entries, "
            f"but v_grid has {grid.size}."
        )

    if integrator_cfg.method == "simpson":
        return np.asarray(simpson(arr, x=grid, axis=axis), dtype=float)

    if v_weights is None:
        raise ValueError("Gauss-Legendre integration requires explicit quadrature weights.")
    weights = np.asarray(v_weights, dtype=float).reshape(-1)
    if weights.size != grid.size:
        raise ValueError(
            f"Weight length mismatch: v_weights has {weights.size} entries, but v_grid has {grid.size}."
        )
    return np.tensordot(arr, weights, axes=([axis], [0]))


def integrate_mean_function_over_v(
    predict_mean_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    x_grid: np.ndarray,
    *,
    integrator_cfg: MuIntegratorConfig,
) -> np.ndarray:
    x_arr = np.asarray(x_grid, dtype=float).reshape(-1)
    if x_arr.size == 0:
        return np.empty(0, dtype=float)

    v_grid, v_weights = resolve_v_rule(integrator_cfg)
    n_v = len(v_grid)
    batch_size = resolve_max_points_per_batch(
        len(x_arr),
        n_v,
        max_points_per_batch=integrator_cfg.max_points_per_batch,
    )

    mu_c_grid = np.empty(len(x_arr), dtype=float)
    for start in range(0, len(x_arr), batch_size):
        end = min(start + batch_size, len(x_arr))
        x_chunk = np.asarray(x_arr[start:end], dtype=float)
        repeat_count = len(x_chunk)
        x_col = np.repeat(x_chunk, n_v).astype(float)
        v_col = np.tile(v_grid, repeat_count).astype(float)
        mean_vals = np.asarray(predict_mean_fn(x_col, v_col), dtype=float)
        if mean_vals.size != repeat_count * n_v:
            raise ValueError(
                f"Unexpected structural predictions shape {mean_vals.shape}; "
                f"expected {repeat_count * n_v} entries for the XxV grid."
            )
        mean_matrix = mean_vals.reshape(repeat_count, n_v)
        mu_c_grid[start:end] = integrate_values(
            mean_matrix,
            v_grid,
            integrator_cfg=integrator_cfg,
            axis=1,
            v_weights=v_weights,
        )
    return mu_c_grid
