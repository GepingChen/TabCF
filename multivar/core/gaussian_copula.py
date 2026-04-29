#!/usr/bin/env python3
"""Gaussian copula helpers for estimation and evaluation."""

from __future__ import annotations

import numpy as np
from scipy.stats import multivariate_normal, norm


def clip_rho(rho: float, margin: float = 1e-6) -> float:
    return float(np.clip(float(rho), -1.0 + margin, 1.0 - margin))


def clip_u(u: np.ndarray | float, eps: float = 1e-6) -> np.ndarray:
    return np.clip(np.asarray(u, dtype=float), eps, 1.0 - eps)


def fit_gaussian_copula_rho(u1: np.ndarray, u2: np.ndarray, eps: float = 1e-6) -> float:
    if u1.shape != u2.shape:
        raise ValueError("u1 and u2 must have matching shapes")
    z1 = norm.ppf(clip_u(u1, eps=eps))
    z2 = norm.ppf(clip_u(u2, eps=eps))
    std1 = np.std(z1)
    std2 = np.std(z2)
    if std1 < 1e-12 or std2 < 1e-12:
        return 0.0
    rho = float(np.corrcoef(z1, z2)[0, 1])
    if np.isnan(rho):
        return 0.0
    return clip_rho(rho)


def gaussian_copula_cdf(u1: float, u2: float, rho: float, eps: float = 1e-6) -> float:
    z1 = float(norm.ppf(clip_u(u1, eps=eps)))
    z2 = float(norm.ppf(clip_u(u2, eps=eps)))
    rho_c = clip_rho(rho)
    cov = np.array([[1.0, rho_c], [rho_c, 1.0]], dtype=float)
    return float(multivariate_normal(mean=np.zeros(2), cov=cov).cdf([z1, z2]))


def gaussian_copula_cdf_vectorized(
    u1: np.ndarray | float,
    u2: np.ndarray | float,
    rho: float,
    eps: float = 1e-6,
) -> np.ndarray:
    u1_arr, u2_arr = np.broadcast_arrays(np.asarray(u1, dtype=float), np.asarray(u2, dtype=float))
    z1 = norm.ppf(clip_u(u1_arr, eps=eps))
    z2 = norm.ppf(clip_u(u2_arr, eps=eps))
    rho_c = clip_rho(rho)
    cov = np.array([[1.0, rho_c], [rho_c, 1.0]], dtype=float)
    points = np.column_stack([z1.reshape(-1), z2.reshape(-1)])
    cdf_vals = multivariate_normal(mean=np.zeros(2), cov=cov).cdf(points)
    return np.asarray(cdf_vals, dtype=float).reshape(u1_arr.shape)


def gaussian_copula_logpdf(u1: np.ndarray, u2: np.ndarray, rho: float, eps: float = 1e-6) -> np.ndarray:
    z1 = norm.ppf(clip_u(u1, eps=eps))
    z2 = norm.ppf(clip_u(u2, eps=eps))
    rho_c = clip_rho(rho)
    one_minus_rho2 = 1.0 - rho_c * rho_c
    quad = (rho_c * rho_c * (z1 * z1 + z2 * z2) - 2.0 * rho_c * z1 * z2) / (2.0 * one_minus_rho2)
    return -0.5 * np.log(one_minus_rho2) - quad


def implied_kendall_tau(rho: float) -> float:
    rho_c = clip_rho(rho)
    return float((2.0 / np.pi) * np.arcsin(rho_c))


def implied_spearman_rho(rho: float) -> float:
    rho_c = clip_rho(rho)
    return float((6.0 / np.pi) * np.arcsin(rho_c / 2.0))


def sample_gaussian_copula(
    rho: float,
    n: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    rng = rng or np.random.default_rng()
    rho_c = clip_rho(rho)
    cov = np.array([[1.0, rho_c], [rho_c, 1.0]], dtype=float)
    z = rng.multivariate_normal(mean=np.zeros(2), cov=cov, size=n)
    return norm.cdf(z)


def upper_tail_probability_from_copula(f1_at_a: float, f2_at_b: float, rho: float) -> float:
    cdf_ab = gaussian_copula_cdf(f1_at_a, f2_at_b, rho)
    return float(1.0 - f1_at_a - f2_at_b + cdf_ab)
