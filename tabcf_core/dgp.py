"""
Data Generating Process (DGP) Module
====================================

This module contains all data generation functions for the IV estimation project.
It implements triangular DGPs with instrument Z, endogenous regressor X, and outcome Y.

Key Functions:
- training_data_generation(): Generate dynamic training data (2000 samples)
- testdata_generation(): Generate and cache fixed test data (10000 samples)
- simulate_dataset(): Core data generation function

DGP Specifications:
- First-stage: X = G(Z, η) with options A1 (additive), A2 (non-additive), A3 (latent H), and A4 (Z-scaled heteroskedastic)
- Second-stage: Y = H(X, ε) with options B1 (polynomial) and B2 (nonlinear)
- Endogeneity: corr(ε, Φ^{-1}(η)) = ρ creates correlation between X and Y errors
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict, replace
from typing import Dict, Tuple, Literal, Optional
from scipy.stats import norm
from pathlib import Path
import os


LATENT_H_WEIGHT = 0.6  # Shared latent-factor loading used across DGP variants


# =========================================================
# Configuration
# =========================================================
@dataclass
class DGPConfig:
    """Configuration for Data Generating Process"""
    n: int = 2000
    seed: int = 42
    rho: Optional[float] = None             # Legacy parameter (unused; kept for backwards compatibility)

    # First-stage family: X = G(Z, η)
    first_stage: Literal["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9"] = "A1"  # A1: additive location–scale; A2: monotone non-additive; A3: additive with latent H; A4: Z-scaled heteroskedastic; A5: Z-quadratic heteroskedastic; A6: Z-quadratic with sin-scaled heteroskedasticity; A7: Z-quadratic with exp-scaled heteroskedasticity; A8: additive latent H + Z-scaled heteroskedastic latent/noise mix; A9: reviewer-facing bounded-support A8* with nonlinear mean and linear scale
    alpha0: float = 0.0
    alpha1: float = 1.0
    gamma0: float = -0.2
    gamma1: float = 0.2
    a2_h: Literal["exp", "cubic"] = "exp"   # A2 transform

    # Second-stage family: Y = H(X, ε)
    second_stage: Literal["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10", "B11"] = "B1"
    beta1: float = 1.0
    beta2: float = 0.5
    delta0: float = -0.5                    # σ_Y(V) = exp(δ0 + δ1 V)
    delta1: float = 0.7
    
    # B2 bimodal parameters (B2 always uses bimodal mixture)
    b2_mixture_weight: float = 0.5              # w: mixture weight for bimodal
    b2_peak_separation: float = 2.0             # Δμ: separation between peaks
    b2_sigma1: float = 0.3                      # σ₁: first component std
    b2_sigma2: float = 0.3                      # σ₂: second component std
    b2_beta_offset: float = 1.0                 # offset for second peak

    # Legacy B6 parameters kept for backwards compatibility with old configs.
    b6_h_weight: float = 0.8
    b6_eps_slope: float = 0.5
    a3_b6_cross_coef: float = 0.3

    # B7 bounded, tail-robust parameters (designed for heavy-tailed X in A6)
    b7_x_scale: float = 20.0          # controls arctan compression of X
    b7_beta1: float = 2.0             # main slope on arctan(X / scale)
    b7_beta2: float = 1.0             # amplitude for sin(3 * arctan(.))
    b7_h_weight: float = 1.0          # confounding strength via tanh(H)
    b7_eps_scale: float = 0.5         # additive noise scale (bounded vs. B6's X*eps)

    # B10 periodic mean with nonadditive latent/noise response. The default
    # "affine" family is the first-pass search target; the softplus family is
    # used only by exploratory fallback runs.
    b10_theta0: float = -0.6
    b10_theta1: float = 0.4
    b10_omega: float = 0.8
    b10_phi: float = 0.0
    b10_response_type: Literal["affine", "softplus"] = "affine"
    b10_lambda: float = 1.0


# =========================================================
# DGP Utility Functions
# =========================================================
def set_seed(seed: int):
    """Set random seed for reproducibility"""
    np.random.seed(seed)


def mu_of_Z(z: np.ndarray, cfg: DGPConfig) -> np.ndarray:
    """Location parameter as function of Z"""
    return cfg.alpha0 + cfg.alpha1 * z


def sigma_of_Z(z: np.ndarray, cfg: DGPConfig) -> np.ndarray:
    """Scale parameter as function of Z"""
    return np.exp(cfg.gamma0 + cfg.gamma1 * z)


def sigmaY_of_X(x: np.ndarray, cfg: DGPConfig) -> np.ndarray:
    """Outcome noise scale σ_Y(X) for second-stage B1 specification."""
    x_arr = np.asarray(x, dtype=float)
    if cfg.second_stage != "B1":
        raise ValueError("sigmaY_of_X is only defined for second_stage='B1'.")
    return np.log1p(np.exp(cfg.delta0 + cfg.delta1 * x_arr))


def stable_softplus(values: np.ndarray | float) -> np.ndarray:
    """Numerically stable softplus for scalar or vector inputs."""
    arr = np.asarray(values, dtype=float)
    return np.log1p(np.exp(-np.abs(arr))) + np.maximum(arr, 0.0)


def b10_mu_of_x(x: np.ndarray | float) -> np.ndarray:
    """Shared periodic mean used by the exploratory B10 family."""
    x_arr = np.asarray(x, dtype=float)
    return 3.0 * np.sin(2.0 * x_arr) + np.cos(0.5 * x_arr) + 2.0 * x_arr


def b10_scale_of_x(x: np.ndarray | float, cfg: DGPConfig) -> np.ndarray:
    """Positive periodic scale used by the exploratory B10 family."""
    x_arr = np.asarray(x, dtype=float)
    periodic_arg = cfg.b10_theta0 + cfg.b10_theta1 * np.sin(cfg.b10_omega * x_arr + cfg.b10_phi)
    return 0.15 + stable_softplus(periodic_arg)


def b10_response(
    cfg: DGPConfig,
    x: np.ndarray | float,
    latent_h: np.ndarray | float,
    eps_y: np.ndarray | float,
) -> np.ndarray:
    """Evaluate the B10 outcome model for arrays of matching shape."""
    x_arr = np.asarray(x, dtype=float)
    h_arr = np.asarray(latent_h, dtype=float)
    eps_arr = np.asarray(eps_y, dtype=float)
    u_arr = h_arr + eps_arr
    mu_arr = b10_mu_of_x(x_arr)
    scale_arr = b10_scale_of_x(x_arr, cfg)
    if cfg.b10_response_type == "softplus":
        return mu_arr + cfg.b10_lambda * stable_softplus(scale_arr * u_arr)
    if cfg.b10_response_type == "affine":
        return mu_arr + scale_arr * u_arr
    raise ValueError(f"Unknown b10_response_type: {cfg.b10_response_type}")


def a2_h_transform(t: np.ndarray, cfg: DGPConfig) -> np.ndarray:
    """Non-linear transformation for A2 first-stage"""
    if cfg.a2_h == "exp":
        return np.exp(t)
    elif cfg.a2_h == "cubic":
        return t + (t ** 3) / 3.0
    else:
        raise ValueError("Unknown A2 transform")


def simulate_eps_eta(n: int, rho: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Draw (ε, η) with corr(ε, Φ^{-1}(η)) = ρ.
    This creates endogeneity between X and Y through correlated errors.
    
    Args:
        n: Number of samples
        rho: Correlation coefficient between ε and Φ^{-1}(η)
    
    Returns:
        eps: (n,) array of error terms for Y
        eta: (n,) array of uniform(0,1) quantiles for X
    """
    cov = np.array([[1.0, rho], [rho, 1.0]])
    L = np.linalg.cholesky(cov)
    Z0 = np.random.randn(2, n)
    E, T = L @ Z0
    eps = E
    eta = norm.cdf(T)  # in (0,1)
    return eps, eta


def simulate_bimodal_y(
    cfg: DGPConfig,
    X: np.ndarray,
    V_true: np.ndarray,
    eps: np.ndarray,
    latent_h: np.ndarray,
) -> np.ndarray:
    """
    Generate Y from bimodal mixture of normals for B2-bimodal.
    
    Y follows mixture: w·N(μ₁, σ₁) + (1-w)·N(μ₂, σ₂)
    where each component mean depends on (X, V, ε).
    
    Args:
        cfg: DGP configuration
        X: (n,) endogenous regressor
        V_true: (n,) control function values
        eps: (n,) error terms from MARGINAL N(0,1)
    
    Returns:
        Y: (n,) bimodal outcomes
    """
    n = len(X)
    h_arr = np.asarray(latent_h, dtype=float)
    if h_arr.shape != X.shape:
        raise ValueError("latent_h must have the same shape as X for B2.")
    
    # Draw mixture indicators (which component each obs comes from)
    mixture_indicators = np.random.binomial(1, cfg.b2_mixture_weight, size=n)
    
    # Component means (both depend on X, eps)
    # First peak: similar to original sin-based B2
    mu1 = np.sin(X) + 0.3 * X * h_arr
    
    # Second peak: offset version
    mu2 = np.sin(X + cfg.b2_beta_offset) + cfg.b2_peak_separation + 0.3 * X * h_arr
    
    # Component standard deviations (X-dependent heteroskedasticity)
    sigma1 = cfg.b2_sigma1 * (1.0 + 0.2 * np.abs(X))
    sigma2 = cfg.b2_sigma2 * (1.0 + 0.3 * np.abs(X))
    
    # Generate from each component
    y1 = mu1 + sigma1 * np.random.randn(n)
    y2 = mu2 + sigma2 * np.random.randn(n)
    
    # Mix according to indicators
    Y = mixture_indicators * y1 + (1 - mixture_indicators) * y2
    
    return Y


def simulate_first_stage(
    cfg: DGPConfig,
    Z: np.ndarray,
    latent_h: np.ndarray,
    eps_x: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate X = G(Z, H, ε_X) according to first-stage specification.

    All first-stage families now share the same latent-factor structure to create
    endogeneity via the common shock H.

    Args:
        cfg: DGP configuration
        Z: (n,) instrument array
        latent_h: (n,) latent factor shared with the second stage
        eps_x: (n,) idiosyncratic shock for X

    Returns:
        X: (n,) endogenous regressor
        V_true: (n,) oracle control function values (uniform on (0,1))
    """
    if cfg.first_stage not in {"A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9"}:
        raise ValueError("Unknown first_stage")

    if cfg.first_stage == "A4":
        X = Z + (Z + 1.0) * (eps_x + latent_h)
        V_true = norm.cdf((X - Z) / ((Z + 1.0) * np.sqrt(2.0)))
        return X, V_true
    if cfg.first_stage == "A5":
        # Sinusoidal heteroskedasticity with milder scale so Z's nonlinear signal is identifiable.
        scale = 5 * (2.0 + np.sin(3.0 * Z))  # ≈ [0.5, 1.5]
        base = (Z - 1.5)**2
        X = base + scale * (eps_x + latent_h)
        # Oracle control function for X|Z when shocks are N(0,1): (X - base - scale*H) / (scale*sqrt(2))
        V_true = norm.cdf((X - base - scale * latent_h) / (scale * np.sqrt(2.0)))
        return X, V_true
    if cfg.first_stage == "A6":
        # Stronger Z-dependent heteroskedasticity while remaining monotone in H and eps_x.
        base = (Z - 1.5) ** 2
        scale = 5.0 * (2.0 + np.sin(3.0 * Z))  # strictly positive on Z ∈ [0, 3]
        X = base + scale * (1 * eps_x + 0.2 *latent_h)
        V_true = norm.cdf((X - base - scale * latent_h) / (scale * np.sqrt(2.0)))
        return X, V_true
#    if cfg.first_stage == "A7":# 1
#        # Stronger Z-dependent heteroskedasticity while remaining monotone in H and eps_x.
#        base = (Z - 1.5) ** 2
#        scale = np.exp(0.3 * (Z - 1.5))  # strictly positive on Z ∈ [0, 3]
#        X = base + scale * (eps_x + latent_h)
#        V_true = norm.cdf((X - base - scale * latent_h) / (scale * np.sqrt(2.0)))
#        return X, V_true
#    if cfg.first_stage == "A7":# 2
#        # Stronger Z-dependent heteroskedasticity while remaining monotone in H and eps_x.
#        base = (Z - 1.5) ** 2
#        scale = np.exp(0.4 * (Z - 1.5))  # strictly positive on Z ∈ [0, 3]
#        X = base + scale * (eps_x + latent_h)
#        V_true = norm.cdf((X - base - scale * latent_h) / (scale * np.sqrt(2.0)))
#        return X, V_true    
#    if cfg.first_stage == "A7":# 3
#        # Stronger Z-dependent heteroskedasticity while remaining monotone in H and eps_x.
#        base = (Z - 1.5) ** 2 + (Z - 1.5)
#        scale = np.exp(0.4 * (Z - 1.5))  # strictly positive on Z ∈ [0, 3]
#        X = base + scale * (eps_x + latent_h)
#        V_true = norm.cdf((X - base - scale * latent_h) / (scale * np.sqrt(2.0)))
#        return X, V_true   
    if cfg.first_stage == "A7":# 4
        # Stronger Z-dependent heteroskedasticity while remaining monotone in H and eps_x.
        base = (Z - 1.5) ** 2 + 0.6 * (Z - 1.5)
        scale = np.exp(0.4 * (Z - 1.5))  # strictly positive on Z ∈ [0, 3]
        X = base + scale * (eps_x + latent_h)
        V_true = norm.cdf((X - base - scale * latent_h) / (scale * np.sqrt(2.0)))
        return X, V_true
    if cfg.first_stage == "A8":
        # A8 mixes additive latent confounding with A4-style heteroskedastic shocks.
        X = Z + latent_h + (Z + 1.0) * (eps_x + latent_h)
        sd = np.sqrt((Z + 1.0) ** 2 + (Z + 2.0) ** 2)
        V_true = norm.cdf((X - Z) / sd)
        return X, V_true
    if cfg.first_stage == "A9":
        # A9 is the bounded-support A8* variant:
        #   nonlinear mean in Z with a mild, strictly positive scale,
        #   but a fixed latent/noise direction H + eps_x so one-dimensional ranks
        #   remain well aligned with the control-function assumptions.
        base = 2.0 * Z + 0.25 * (Z ** 2)
        scale = 1.0 + 0.15 * Z
        X = base + scale * (latent_h + eps_x)
        V_true = norm.cdf((X - base) / (scale * np.sqrt(2.0)))
        return X, V_true
    # A3 baseline: additive first stage with latent confounding.
    X = Z + latent_h + eps_x
    #if cfg.first_stage == "A3" and cfg.second_stage == "B6":
    #    X = X + cfg.a3_b6_cross_coef * Z * latent_h
    V_true = norm.cdf((X - Z) / np.sqrt(2.0))
    return X, V_true


def simulate_second_stage(
    cfg: DGPConfig,
    X: np.ndarray,
    V_true: np.ndarray,
    eps: np.ndarray,
    *,
    latent_h: np.ndarray | None = None,
) -> np.ndarray:
    """
    Generate Y = H(X, ε) according to second-stage specification.
    
    Args:
        cfg: DGP configuration
        X: (n,) array of endogenous regressors
        V_true: (n,) array of true control function values
        eps: (n,) array of error terms
    
    Returns:
        Y: (n,) array of outcomes
    """
    
    if cfg.second_stage == "B1":
        if latent_h is None:
            raise ValueError("Second-stage B1 now requires latent_h draws.")
        h_arr = np.asarray(latent_h, dtype=float)
        if h_arr.shape != X.shape:
            raise ValueError("latent_h must have the same shape as X for B1.")
        eps_arr = np.asarray(eps, dtype=float)
        if eps_arr.shape != X.shape:
            raise ValueError("eps must have the same shape as X for B1.")
        m1 = cfg.beta1 * X + cfg.beta2 * (X ** 2)
        sigY = sigmaY_of_X(X, cfg)
        # Latent H drives endogeneity; eps supplies independent idiosyncratic noise.
        Y = m1 + sigY * (LATENT_H_WEIGHT * h_arr + eps_arr)
        return Y
    elif cfg.second_stage == "B2":
        if latent_h is None:
            raise ValueError("Second-stage B2 now requires latent_h draws.")
        # B2 always uses bimodal mixture of normals
        return simulate_bimodal_y(cfg, X, V_true, eps, latent_h)
    elif cfg.second_stage == "B3":
        if latent_h is None:
            raise ValueError("Second-stage B3 requires latent_h draws.")
        return X - 3.0 * latent_h + eps
    elif cfg.second_stage == "B4":
        if latent_h is None:
            raise ValueError("Second-stage B4 requires latent_h draws.")
        h_arr = np.asarray(latent_h, dtype=float)
        if h_arr.shape != X.shape:
            raise ValueError("latent_h must have same shape as X for B4.")
        eps_arr = np.asarray(eps, dtype=float)
        if eps_arr.shape != X.shape:
            raise ValueError("eps must have same shape as X for B4.")

        linear_branch = 0.2 * (5.5 + 2.0 * X + 3.0 * h_arr + eps_arr)
        # Ensure argument of log stays positive; clamp at a small epsilon.
        softplus_arg = (2.0 * X + h_arr) ** 2 + eps_arr ** 2
        safe_arg = np.maximum(softplus_arg, 1e-8)
        softplus_branch = np.log(safe_arg)
        return np.where(X <= 1.0, linear_branch, softplus_branch)
    elif cfg.second_stage == "B5":
        if latent_h is None:
            raise ValueError("Second-stage B5 requires latent_h draws.")
        h_arr = np.asarray(latent_h, dtype=float)
        if h_arr.shape != X.shape:
            raise ValueError("latent_h must have same shape as X for B5.")
        eps_arr = np.asarray(eps, dtype=float)
        if eps_arr.shape != X.shape:
            raise ValueError("eps must have same shape as X for B5.")
        return 3.0 * np.sin(2.0 * X) + 2.0 * X - 3.0 * h_arr + eps_arr
    elif cfg.second_stage == "B6":
        if latent_h is None:
            raise ValueError("Second-stage B6 requires latent_h draws.")
        h_arr = np.asarray(latent_h, dtype=float)
        if h_arr.shape != X.shape:
            raise ValueError("latent_h must have same shape as X for B6.")
        return 1.0 + 2.0 * np.cos(2.0 * X + h_arr) + X * h_arr
    elif cfg.second_stage == "B7":
        if latent_h is None:
            raise ValueError("Second-stage B7 requires latent_h draws.")
        h_arr = np.asarray(latent_h, dtype=float)
        if h_arr.shape != X.shape:
            raise ValueError("latent_h must have same shape as X for B7.")
        eps_arr = np.asarray(eps, dtype=float)
        if eps_arr.shape != X.shape:
            raise ValueError("eps must have same shape as X for B7.")

        # Tail-robust second stage:
        #   u = arctan(X / s) compresses heavy-tailed X to (-pi/2, pi/2)
        #   tanh(H) keeps the confounder effect bounded
        #   additive eps keeps variance stable in the tails (unlike X*eps in B6)
        u = np.arctan(np.asarray(X, dtype=float) / float(cfg.b7_x_scale))
        h_t = np.tanh(h_arr)
        return (
            cfg.b7_beta1 * u
            + cfg.b7_beta2 * np.sin(3.0 * u)
            + cfg.b7_h_weight * h_t * np.cos(2.0 * u)
            + cfg.b7_eps_scale * eps_arr
        )
    elif cfg.second_stage == "B8":
        if latent_h is None:
            raise ValueError("Second-stage B8 requires latent_h draws.")
        h_arr = np.asarray(latent_h, dtype=float)
        if h_arr.shape != X.shape:
            raise ValueError("latent_h must have same shape as X for B8.")
        eps_arr = np.asarray(eps, dtype=float)
        if eps_arr.shape != X.shape:
            raise ValueError("eps must have same shape as X for B8.")

        linear_branch = 1.0 + X + 2.0 * h_arr + eps_arr
        softplus_arg = 2.0 * (X + h_arr) ** 2 + eps_arr ** 2
        safe_arg = np.maximum(softplus_arg, 1e-8)
        softplus_branch = np.log(safe_arg)
        return np.where(X <= 1.0, linear_branch, softplus_branch)
    elif cfg.second_stage == "B9":
        if latent_h is None:
            raise ValueError("Second-stage B9 requires latent_h draws.")
        h_arr = np.asarray(latent_h, dtype=float)
        if h_arr.shape != X.shape:
            raise ValueError("latent_h must have same shape as X for B9.")
        eps_arr = np.asarray(eps, dtype=float)
        if eps_arr.shape != X.shape:
            raise ValueError("eps must have same shape as X for B9.")
        return (
            3.0 * np.sin(2.0 * X)
            + np.cos(0.5 * X)
            + 2.0 * X
            - 3.0 * h_arr
            + eps_arr
        )
    elif cfg.second_stage == "B10":
        if latent_h is None:
            raise ValueError("Second-stage B10 requires latent_h draws.")
        h_arr = np.asarray(latent_h, dtype=float)
        if h_arr.shape != X.shape:
            raise ValueError("latent_h must have same shape as X for B10.")
        eps_arr = np.asarray(eps, dtype=float)
        if eps_arr.shape != X.shape:
            raise ValueError("eps must have same shape as X for B10.")
        return b10_response(cfg, X, h_arr, eps_arr)
    elif cfg.second_stage == "B11":
        if latent_h is None:
            raise ValueError("Second-stage B11 requires latent_h draws.")
        h_arr = np.asarray(latent_h, dtype=float)
        if h_arr.shape != X.shape:
            raise ValueError("latent_h must have same shape as X for B11.")
        eps_arr = np.asarray(eps, dtype=float)
        if eps_arr.shape != X.shape:
            raise ValueError("eps must have same shape as X for B11.")
        return 1.0 + 2.0 * X + np.cos(2.0 * X) + X * h_arr - h_arr + eps_arr
    else:
        raise ValueError("Unknown second_stage")


def simulate_dataset(cfg: DGPConfig) -> Dict[str, np.ndarray]:
    """
    Main data generation function.
    
    Args:
        cfg: DGP configuration including sample size and random seed
    
    Returns:
        Dictionary containing:
            - Z: (n,) instrument values
            - X: (n,) endogenous regressor
            - Y: (n,) outcome
            - V_true: (n,) true control function (≡ η)
            - eps: (n,) second-stage shock ε_Y
            - eta: (n,) alias for V_true (kept for backward compatibility)
            - H: (n,) latent factor shared across stages
            - eps_x: (n,) first-stage idiosyncratic shock ε_X
            - eps_y: (n,) copy of ε for convenience when analysing components
    """
    set_seed(cfg.seed)
    n = cfg.n
    Z = np.random.uniform(0, 3, n)
    latent_h = np.random.randn(n)
    eps_x = np.random.randn(n)
    eps_y = np.random.randn(n)

    X, V_true = simulate_first_stage(cfg, Z, latent_h, eps_x)
    Y = simulate_second_stage(cfg, X, V_true, eps_y, latent_h=latent_h)

    data = {
        "Z": Z,
        "X": X,
        "Y": Y,
        "V_true": V_true,
        "eps": eps_y,
        "eta": V_true,
        "H": latent_h,
        "eps_x": eps_x,
        "eps_y": eps_y,
    }
    return data


# =========================================================
# Training and Test Data Generation Functions
# =========================================================
def _resolve_path(path_like: str | os.PathLike[str]) -> Path:
    """Resolve paths relative to the interv_mean root."""
    path = Path(path_like)
    if path.is_absolute():
        return path
    project_root = Path(__file__).resolve().parent.parent
    return project_root / path


def training_data_generation(
    cfg: DGPConfig,
    save_csv: bool = True,
    train_dir: str | os.PathLike[str] = "IV_datasets/train",
) -> Dict[str, np.ndarray]:
    """
    Generate training data dynamically for each run.
    
    This function should be called at the start of each training run to generate
    fresh training data with the specified sample size (typically 2000).
    
    Args:
        cfg: DGP configuration with training sample size and seed
        save_csv: If True, save data to CSV format in IV_datasets/train/
    
    Returns:
        Dictionary with training data arrays (Z, X, Y, V_true, eps, eta)
    """
    print(f"Generating training data: n={cfg.n}, seed={cfg.seed}")
    data = simulate_dataset(cfg)
    
    if save_csv:
        # Create training data directory (resolved relative to repo root by default)
        train_dir = _resolve_path(train_dir)
        os.makedirs(train_dir, exist_ok=True)

        # Create DataFrame and save as CSV
        base_cols = {
            'Z': data['Z'],
            'X': data['X'], 
            'Y': data['Y'],
            'V_true': data['V_true'],
            'eps': data['eps'],
            'eta': data['eta']
        }
        extra_cols = {k: v for k, v in data.items() if k not in base_cols}
        df = pd.DataFrame({**base_cols, **extra_cols})

        codes = f"{cfg.first_stage}_{cfg.second_stage}"
        sample_tag = f"_n{cfg.n}"
        seed_tag = f"_seed{cfg.seed}"
        csv_file = train_dir / f"train_data_{codes}{sample_tag}{seed_tag}.csv"
        df.to_csv(csv_file, index=False)
        print(f"  ✅ Training data saved to: {csv_file}")
    
    print(f"  ✅ Training data generated successfully")
    return data


def testdata_generation(
    cfg: DGPConfig,
    test_dir: str | os.PathLike[str] = "IV_datasets/test",
    test_seed: int = 999,
    force_regenerate: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Generate or load fixed test dataset.
    
    This function creates a large fixed test set (typically 10000 samples) and caches
    it to disk in CSV format. Subsequent calls will load the cached data unless force_regenerate=True.
    
    Args:
        cfg: DGP configuration (n will be overridden to 10000 for test set)
        test_dir: Directory to save/load test data
        test_seed: Fixed seed for test data reproducibility (default: 999)
        force_regenerate: If True, regenerate test data even if it exists
    
    Returns:
        Dictionary with test data arrays (Z, X, Y, V_true, eps, eta)
    """
    # Create test directory if it doesn't exist
    test_dir = _resolve_path(test_dir)
    os.makedirs(test_dir, exist_ok=True)

    # Test data file path (CSV format)
    test_data_file = test_dir / f"test_data_{cfg.first_stage}_{cfg.second_stage}.csv"

    # Check if test data already exists
    if os.path.exists(test_data_file) and not force_regenerate:
        print(f"Loading existing test data from: {test_data_file}")
        try:
            df = pd.read_csv(test_data_file)
            data = {
                'Z': df['Z'].values,
                'X': df['X'].values,
                'Y': df['Y'].values,
                'V_true': df['V_true'].values,
                'eps': df['eps'].values,
                'eta': df['eta'].values
            }
            print(f"  ✅ Test data loaded successfully (n={len(data['Z'])})")
            return data
        except Exception as e:
            print(f"  ⚠️  Failed to load test data: {e}")
            print(f"  Regenerating test data...")
    
    # Generate new test data
    print(f"Generating fixed test data: n=10000, seed={test_seed}")
    
    # Create test configuration with fixed parameters
    test_cfg = replace(cfg, n=10000, seed=test_seed)
    
    # Generate test data
    test_data = simulate_dataset(test_cfg)
    
    # Save test data to CSV
    try:
        base_cols = {
            'Z': test_data['Z'],
            'X': test_data['X'],
            'Y': test_data['Y'],
            'V_true': test_data['V_true'],
            'eps': test_data['eps'],
            'eta': test_data['eta']
        }
        extra_cols = {k: v for k, v in test_data.items() if k not in base_cols}
        df = pd.DataFrame({**base_cols, **extra_cols})
        df.to_csv(test_data_file, index=False)
        print(f"  ✅ Test data generated and saved to: {test_data_file}")
    except Exception as e:
        print(f"  ⚠️  Failed to save test data: {e}")
    
    return test_data


# =========================================================
# Utility Functions for Data Summary
# =========================================================
def print_data_summary(data: Dict[str, np.ndarray], cfg: DGPConfig, data_type: str = "Data"):
    """
    Print summary statistics of generated data.
    
    Args:
        data: Dictionary containing Z, X, Y, V_true arrays
        cfg: DGP configuration
        data_type: Type of data for display (e.g., "Training", "Test")
    """
    Z, X, Y, V_true = data["Z"], data["X"], data["Y"], data["V_true"]
    
    print("\n" + "="*60)
    print(f"{data_type.upper()} DATA SUMMARY")
    print("="*60)
    print(f"Sample size: {len(Z)}")
    print(f"DGP configuration: {cfg.first_stage}/{cfg.second_stage}")
    rho_display = "n/a" if cfg.rho is None else f"{cfg.rho}"
    print(f"Endogeneity correlation ρ: {rho_display}")
    print(f"Random seed: {cfg.seed}")
    
    print(f"\nVariable statistics:")
    print(f"  Z: mean={np.mean(Z):.3f}, std={np.std(Z):.3f}, range=[{np.min(Z):.3f}, {np.max(Z):.3f}]")
    print(f"  X: mean={np.mean(X):.3f}, std={np.std(X):.3f}, range=[{np.min(X):.3f}, {np.max(X):.3f}]")
    print(f"  Y: mean={np.mean(Y):.3f}, std={np.std(Y):.3f}, range=[{np.min(Y):.3f}, {np.max(Y):.3f}]")
    print(f"  V_true: mean={np.mean(V_true):.3f}, std={np.std(V_true):.3f}, range=[{np.min(V_true):.3f}, {np.max(V_true):.3f}]")


# =========================================================
# Main Execution (for testing)
# =========================================================
if __name__ == "__main__":
    """
    DGP CSV Data Generation
    Generates training and test data in CSV format
    """
    print("DGP CSV Data Generation")
    print("="*60)
    
    # Create configuration for training data
    train_cfg = DGPConfig(
        n=4000,
        seed=123,
        rho=0.6,
        first_stage="A1",
        second_stage="B2"
    )
    
    # Create configuration for test data  
    test_cfg = DGPConfig(
        n=10000,
        seed=999,
        rho=0.6,
        first_stage="A3",
        second_stage="B1"
    )
    
    print("Configuration:")
    print(f"  Training: n={train_cfg.n}, seed={train_cfg.seed}")
    print(f"  Test: n={test_cfg.n}, seed={test_cfg.seed}")
    print(f"  DGP: {train_cfg.first_stage}/{train_cfg.second_stage}, ρ={train_cfg.rho}")
    print()
    
    # Generate training data
    print("Generating training data...")
    train_data = training_data_generation(train_cfg, save_csv=True)
    print_data_summary(train_data, train_cfg, "Training")
    print()
    
    # Generate test data
    print("Generating test data...")
    test_data = testdata_generation(test_cfg, force_regenerate=True)
    print_data_summary(test_data, test_cfg, "Test")
    print()
    
    # Verify files were created
    train_dir = _resolve_path("IV_datasets/train")
    test_dir = _resolve_path("IV_datasets/test")
    
    print("Verifying generated files...")
    
    if train_dir.exists():
        train_files = [f for f in train_dir.glob("*.csv")]
        print(f"  Training CSV files: {len(train_files)}")
        for f in train_files:
            size = f.stat().st_size
            print(f"    {f.name}: {size:,} bytes")
    else:
        print("  ❌ Training directory not found")

    if test_dir.exists():
        test_files = [f for f in test_dir.glob("*.csv")]
        print(f"  Test CSV files: {len(test_files)}")
        for f in test_files:
            size = f.stat().st_size
            print(f"    {f.name}: {size:,} bytes")
    else:
        print("  ❌ Test directory not found")
    
    print()
    print("="*60)
    print("DGP CSV Generation Completed Successfully!")
    print("="*60)

#0
