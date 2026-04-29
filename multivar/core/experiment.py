#!/usr/bin/env python3
"""Single-run experiment logic for the multivariate benchmark plan."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, kstest, norm, spearmanr

from multivar.core.gaussian_copula import (
    clip_u,
    fit_gaussian_copula_rho,
    gaussian_copula_cdf_vectorized,
    implied_kendall_tau,
    implied_spearman_rho,
    sample_gaussian_copula,
    upper_tail_probability_from_copula,
)
from multivar.core.div_bridge import fit_div_marginal_estimators
from multivar.core.tabcf_bridge import fit_marginal_estimators
from multivar.core.tabcf_bridge import fit_tabpfn_naive_marginal_estimators
from multivar.core.dgp import (
    DGPConfig,
    normalize_dgp_code,
    sample_interventional_data,
    sample_observational_data,
    true_corr_under_do,
    true_joint_cdf_vectorized,
    true_joint_tail_probability,
    true_marginal_cdf,
    true_marginal_ppf,
)

MODEL_NAMES: tuple[str, ...] = ("estimated", "independence", "oracle")
TABCF_COMPAT_SOURCE_TO_BACKEND = {
    "deepcf": "tabpfn",
    "tabpfn_real": "tabpfn_real",
    "tabicl": "tabicl",
}
TABCF_COMPAT_SOURCES = frozenset(TABCF_COMPAT_SOURCE_TO_BACKEND)
EMPTY_BACKEND_METADATA = {
    "core_backend": "",
    "backend_package": "",
    "checkpoint_version": "",
    "model_path": "",
}


def _normalize_marginal_source(value: str) -> str:
    src = str(value).strip().lower()
    if src not in {*TABCF_COMPAT_SOURCES, "tabpfn-naive", "div", "oracle"}:
        raise ValueError(
            "marginal_source must be one of "
            "['deepcf', 'tabpfn_real', 'tabicl', 'tabpfn-naive', 'div', 'oracle'], "
            f"got '{value}'"
        )
    return src


def _normalize_quantile_levels(values: Tuple[float, ...], *, name: str) -> Tuple[float, ...]:
    levels = tuple(float(v) for v in values)
    if not levels:
        raise ValueError(f"{name} must contain at least one quantile level")
    for q in levels:
        if not 0.0 < q < 1.0:
            raise ValueError(f"{name} values must be in (0,1), got {q}")
    return levels


def _rho_for_model(model_name: str, rho_hat: float, rho_indep: float, rho_oracle: float) -> float:
    if model_name == "estimated":
        return float(rho_hat)
    if model_name == "independence":
        return float(rho_indep)
    if model_name == "oracle":
        return float(rho_oracle)
    raise ValueError(f"Unexpected model_name: {model_name}")


def _backend_metadata_for_source(
    source: str,
    fitted_marginals: object | None = None,
) -> dict[str, str]:
    if source in TABCF_COMPAT_SOURCES and fitted_marginals is not None:
        metadata = getattr(fitted_marginals, "metadata", None)
        if isinstance(metadata, dict):
            return {
                "core_backend": str(metadata.get("core_backend", "")),
                "backend_package": str(metadata.get("backend_package", "")),
                "checkpoint_version": str(metadata.get("checkpoint_version", "")),
                "model_path": str(metadata.get("model_path", "")),
            }
    return dict(EMPTY_BACKEND_METADATA)


@dataclass(frozen=True)
class ExperimentConfig:
    dgp_code: str
    n_train: int
    seed: int
    rho_eps: float
    x_grid_min: float = 0.0
    x_grid_max: float = 3.0
    x_grid_size: int = 13
    x_bins: int = 5
    clip_eps: float = 1e-6
    tail_x_ref: float | None = None
    y_quantile_levels: Tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9)
    marginal_source: str = "deepcf"
    deepcf_v_points: int = 101
    deepcf_eval_batch_size: int = 256
    save_distribution_output: bool = True
    dist_quantile_levels: Tuple[float, ...] = tuple(v / 100.0 for v in range(1, 100))
    dist_sample_size: int = 10000

    def normalize(self) -> "ExperimentConfig":
        clip_eps = float(self.clip_eps)
        if not 0.0 < clip_eps < 0.5:
            raise ValueError(f"clip_eps must be in (0, 0.5), got {clip_eps}")

        dist_sample_size = int(self.dist_sample_size)
        if dist_sample_size <= 0:
            raise ValueError(f"dist_sample_size must be positive, got {dist_sample_size}")

        return ExperimentConfig(
            dgp_code=normalize_dgp_code(self.dgp_code),
            n_train=int(self.n_train),
            seed=int(self.seed),
            rho_eps=float(self.rho_eps),
            x_grid_min=float(self.x_grid_min),
            x_grid_max=float(self.x_grid_max),
            x_grid_size=int(self.x_grid_size),
            x_bins=int(self.x_bins),
            clip_eps=clip_eps,
            tail_x_ref=None if self.tail_x_ref is None else float(self.tail_x_ref),
            y_quantile_levels=_normalize_quantile_levels(
                tuple(float(v) for v in self.y_quantile_levels),
                name="y_quantile_levels",
            ),
            marginal_source=_normalize_marginal_source(self.marginal_source),
            deepcf_v_points=int(self.deepcf_v_points),
            deepcf_eval_batch_size=int(self.deepcf_eval_batch_size),
            save_distribution_output=bool(self.save_distribution_output),
            dist_quantile_levels=_normalize_quantile_levels(
                tuple(float(v) for v in self.dist_quantile_levels),
                name="dist_quantile_levels",
            ),
            dist_sample_size=dist_sample_size,
        )


def _x_bin_diagnostics(
    x: np.ndarray,
    z1: np.ndarray,
    z2: np.ndarray,
    *,
    n_bins: int,
) -> pd.DataFrame:
    df = pd.DataFrame({"X": x, "z1": z1, "z2": z2})
    try:
        df["x_bin"] = pd.qcut(df["X"], q=n_bins, duplicates="drop")
    except ValueError:
        return pd.DataFrame(columns=["x_bin", "n_obs", "corr_z1_z2"])
    rows = []
    for bin_key, grp in df.groupby("x_bin", observed=True):
        corr = float(np.corrcoef(grp["z1"], grp["z2"])[0, 1]) if len(grp) >= 2 else np.nan
        rows.append({"x_bin": str(bin_key), "n_obs": int(len(grp)), "corr_z1_z2": corr})
    return pd.DataFrame(rows)


def _joint_cdf_mae_at_x(
    *,
    dgp_code: str,
    rho_eps: float,
    x: float,
    copula_rho: float,
    quantile_levels: tuple[float, ...],
    clip_eps: float,
    marginal_cdf_eval: Callable[[int, np.ndarray | float, np.ndarray | float], np.ndarray],
) -> float:
    quantiles = np.asarray(quantile_levels, dtype=float)
    y1_grid = np.asarray(true_marginal_ppf(dgp_code, rho_eps, 1, x, quantiles), dtype=float)
    y2_grid = np.asarray(true_marginal_ppf(dgp_code, rho_eps, 2, x, quantiles), dtype=float)
    u1_grid = clip_u(np.asarray(marginal_cdf_eval(1, x, y1_grid), dtype=float), eps=clip_eps)
    u2_grid = clip_u(np.asarray(marginal_cdf_eval(2, x, y2_grid), dtype=float), eps=clip_eps)

    y1_mesh, y2_mesh = np.meshgrid(y1_grid, y2_grid, indexing="ij")
    u1_mesh, u2_mesh = np.meshgrid(u1_grid, u2_grid, indexing="ij")
    cdf_hat = gaussian_copula_cdf_vectorized(u1_mesh, u2_mesh, copula_rho, eps=clip_eps)
    cdf_true = true_joint_cdf_vectorized(dgp_code, rho_eps, x, y1_mesh, y2_mesh)

    abs_errors = np.abs(np.asarray(cdf_hat, dtype=float) - np.asarray(cdf_true, dtype=float))
    return float(np.mean(abs_errors)) if abs_errors.size > 0 else np.nan


def _as_scalar(value: np.ndarray | float) -> float:
    arr = np.asarray(value, dtype=float)
    if arr.size != 1:
        raise ValueError(f"Expected scalar-compatible value, got shape {arr.shape}")
    return float(arr.reshape(-1)[0])


def _build_inverse_cdf_lookup(
    *,
    cfg: ExperimentConfig,
    component: int,
    x: float,
    marginal_cdf_eval: Callable[[int, np.ndarray | float, np.ndarray | float], np.ndarray],
    n_support: int = 513,
) -> Callable[[np.ndarray | float], np.ndarray]:
    p_low = cfg.clip_eps
    p_high = 1.0 - cfg.clip_eps
    p_grid = np.linspace(p_low, p_high, int(n_support), dtype=float)
    y_support = np.asarray(true_marginal_ppf(cfg.dgp_code, cfg.rho_eps, component, x, p_grid), dtype=float)
    x_support = np.full(y_support.shape, float(x), dtype=float)

    cdf_support = np.asarray(marginal_cdf_eval(component, x_support, y_support), dtype=float)
    if cdf_support.shape != y_support.shape:
        raise ValueError(
            "Unexpected marginal CDF shape while building inverse lookup: "
            f"expected {y_support.shape}, got {cdf_support.shape}"
        )

    cdf_support = np.clip(cdf_support, 0.0, 1.0)
    cdf_support = np.maximum.accumulate(cdf_support)
    cdf_support[0] = 0.0
    cdf_support[-1] = 1.0

    unique_cdf, unique_idx = np.unique(cdf_support, return_index=True)
    if unique_cdf.size < 2:
        median_y = float(true_marginal_ppf(cfg.dgp_code, cfg.rho_eps, component, x, 0.5))

        def inverse_constant(u: np.ndarray | float) -> np.ndarray:
            u_arr = np.asarray(u, dtype=float)
            out = np.full(u_arr.shape, median_y, dtype=float)
            return out

        return inverse_constant

    y_unique = y_support[unique_idx]

    def inverse_interp(u: np.ndarray | float) -> np.ndarray:
        u_arr = clip_u(np.asarray(u, dtype=float), eps=cfg.clip_eps)
        return np.interp(u_arr, unique_cdf, y_unique, left=y_unique[0], right=y_unique[-1])

    return inverse_interp


def _build_distributional_outputs(
    *,
    cfg: ExperimentConfig,
    x_grid: np.ndarray,
    rho_hat: float,
    rho_indep: float,
    marginal_cdf_eval: Callable[[int, np.ndarray | float, np.ndarray | float], np.ndarray],
    backend_metadata: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not cfg.save_distribution_output:
        return pd.DataFrame(), pd.DataFrame()

    dist_quantiles = np.asarray(cfg.dist_quantile_levels, dtype=float)
    q1_mesh, q2_mesh = np.meshgrid(dist_quantiles, dist_quantiles, indexing="ij")
    q1_flat = q1_mesh.reshape(-1)
    q2_flat = q2_mesh.reshape(-1)

    cdf_frames: list[pd.DataFrame] = []
    sample_frames: list[pd.DataFrame] = []
    rng = np.random.default_rng(cfg.seed + 99173)

    for x_val in x_grid:
        x_float = float(x_val)
        rho_true_x = true_corr_under_do(cfg.dgp_code, cfg.rho_eps, x=x_float)

        y1_grid = np.asarray(true_marginal_ppf(cfg.dgp_code, cfg.rho_eps, 1, x_float, dist_quantiles), dtype=float)
        y2_grid = np.asarray(true_marginal_ppf(cfg.dgp_code, cfg.rho_eps, 2, x_float, dist_quantiles), dtype=float)

        u1_grid = clip_u(np.asarray(marginal_cdf_eval(1, x_float, y1_grid), dtype=float), eps=cfg.clip_eps)
        u2_grid = clip_u(np.asarray(marginal_cdf_eval(2, x_float, y2_grid), dtype=float), eps=cfg.clip_eps)

        y1_mesh, y2_mesh = np.meshgrid(y1_grid, y2_grid, indexing="ij")
        u1_mesh, u2_mesh = np.meshgrid(u1_grid, u2_grid, indexing="ij")

        y1_flat = y1_mesh.reshape(-1)
        y2_flat = y2_mesh.reshape(-1)
        u1_flat = u1_mesh.reshape(-1)
        u2_flat = u2_mesh.reshape(-1)
        joint_true = true_joint_cdf_vectorized(cfg.dgp_code, cfg.rho_eps, x_float, y1_mesh, y2_mesh).reshape(-1)

        inv_y1 = _build_inverse_cdf_lookup(
            cfg=cfg,
            component=1,
            x=x_float,
            marginal_cdf_eval=marginal_cdf_eval,
        )
        inv_y2 = _build_inverse_cdf_lookup(
            cfg=cfg,
            component=2,
            x=x_float,
            marginal_cdf_eval=marginal_cdf_eval,
        )

        for model_name in MODEL_NAMES:
            if model_name == "oracle":
                rho_model = np.nan
                u1_model_flat = q1_flat
                u2_model_flat = q2_flat
                joint_hat = joint_true
            else:
                rho_model = _rho_for_model(model_name, rho_hat, rho_indep, rho_true_x)
                u1_model_flat = u1_flat
                u2_model_flat = u2_flat
                joint_hat = gaussian_copula_cdf_vectorized(u1_mesh, u2_mesh, rho_model, eps=cfg.clip_eps).reshape(-1)

            n_cdf_rows = joint_hat.size
            cdf_frames.append(
                pd.DataFrame(
                    {
                        "dgp_code": np.full(n_cdf_rows, cfg.dgp_code, dtype=object),
                        "n_train": np.full(n_cdf_rows, int(cfg.n_train), dtype=int),
                        "seed": np.full(n_cdf_rows, int(cfg.seed), dtype=int),
                        "rho_eps": np.full(n_cdf_rows, float(cfg.rho_eps), dtype=float),
                        "marginal_source": np.full(n_cdf_rows, cfg.marginal_source, dtype=object),
                        "core_backend": np.full(n_cdf_rows, backend_metadata["core_backend"], dtype=object),
                        "backend_package": np.full(n_cdf_rows, backend_metadata["backend_package"], dtype=object),
                        "checkpoint_version": np.full(n_cdf_rows, backend_metadata["checkpoint_version"], dtype=object),
                        "model_path": np.full(n_cdf_rows, backend_metadata["model_path"], dtype=object),
                        "deepcf_v_points": np.full(n_cdf_rows, int(cfg.deepcf_v_points), dtype=int),
                        "deepcf_eval_batch_size": np.full(n_cdf_rows, int(cfg.deepcf_eval_batch_size), dtype=int),
                        "x": np.full(n_cdf_rows, x_float, dtype=float),
                        "model": np.full(n_cdf_rows, model_name, dtype=object),
                        "copula_rho": np.full(n_cdf_rows, float(rho_model), dtype=float),
                        "q1": q1_flat,
                        "q2": q2_flat,
                        "y1": y1_flat,
                        "y2": y2_flat,
                        "u1_hat": u1_model_flat,
                        "u2_hat": u2_model_flat,
                        "joint_cdf_hat": joint_hat,
                        "joint_cdf_true": joint_true,
                        "joint_cdf_abs_error": np.abs(joint_hat - joint_true),
                    }
                )
            )

            if model_name == "oracle":
                oracle_samples = sample_interventional_data(
                    cfg.dgp_code,
                    cfg.rho_eps,
                    x_float,
                    cfg.dist_sample_size,
                    rng=rng,
                )
                y1_samples = oracle_samples["Y1"].to_numpy(dtype=float)
                y2_samples = oracle_samples["Y2"].to_numpy(dtype=float)
                u1_samples = clip_u(
                    np.asarray(true_marginal_cdf(cfg.dgp_code, cfg.rho_eps, 1, x_float, y1_samples), dtype=float),
                    eps=cfg.clip_eps,
                )
                u2_samples = clip_u(
                    np.asarray(true_marginal_cdf(cfg.dgp_code, cfg.rho_eps, 2, x_float, y2_samples), dtype=float),
                    eps=cfg.clip_eps,
                )
            else:
                copula_samples = sample_gaussian_copula(rho=rho_model, n=cfg.dist_sample_size, rng=rng)
                u1_samples = clip_u(copula_samples[:, 0], eps=cfg.clip_eps)
                u2_samples = clip_u(copula_samples[:, 1], eps=cfg.clip_eps)
                y1_samples = np.asarray(inv_y1(u1_samples), dtype=float)
                y2_samples = np.asarray(inv_y2(u2_samples), dtype=float)
            n_samples = len(u1_samples)

            sample_frames.append(
                pd.DataFrame(
                    {
                        "dgp_code": np.full(n_samples, cfg.dgp_code, dtype=object),
                        "n_train": np.full(n_samples, int(cfg.n_train), dtype=int),
                        "seed": np.full(n_samples, int(cfg.seed), dtype=int),
                        "rho_eps": np.full(n_samples, float(cfg.rho_eps), dtype=float),
                        "marginal_source": np.full(n_samples, cfg.marginal_source, dtype=object),
                        "core_backend": np.full(n_samples, backend_metadata["core_backend"], dtype=object),
                        "backend_package": np.full(n_samples, backend_metadata["backend_package"], dtype=object),
                        "checkpoint_version": np.full(n_samples, backend_metadata["checkpoint_version"], dtype=object),
                        "model_path": np.full(n_samples, backend_metadata["model_path"], dtype=object),
                        "deepcf_v_points": np.full(n_samples, int(cfg.deepcf_v_points), dtype=int),
                        "deepcf_eval_batch_size": np.full(n_samples, int(cfg.deepcf_eval_batch_size), dtype=int),
                        "x": np.full(n_samples, x_float, dtype=float),
                        "model": np.full(n_samples, model_name, dtype=object),
                        "copula_rho": np.full(n_samples, float(rho_model), dtype=float),
                        "sample_id": np.arange(n_samples, dtype=int),
                        "u1": u1_samples,
                        "u2": u2_samples,
                        "y1": y1_samples,
                        "y2": y2_samples,
                        "sample_weight": np.full(n_samples, 1.0 / float(cfg.dist_sample_size), dtype=float),
                    }
                )
            )

    dist_cdf_df = pd.concat(cdf_frames, ignore_index=True) if cdf_frames else pd.DataFrame()
    dist_samples_df = pd.concat(sample_frames, ignore_index=True) if sample_frames else pd.DataFrame()
    return dist_cdf_df, dist_samples_df


def run_single_experiment(
    config: ExperimentConfig,
) -> tuple[dict[str, float | str | int], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg = config.normalize()
    df_obs = sample_observational_data(
        DGPConfig(n=cfg.n_train, seed=cfg.seed, dgp_code=cfg.dgp_code, rho_eps=cfg.rho_eps)
    )

    x_obs = df_obs["X"].to_numpy(dtype=float)
    y1_obs = df_obs["Y1"].to_numpy(dtype=float)
    y2_obs = df_obs["Y2"].to_numpy(dtype=float)
    backend_metadata = dict(EMPTY_BACKEND_METADATA)

    if cfg.marginal_source == "oracle":

        def marginal_cdf_eval(component: int, x: np.ndarray | float, y: np.ndarray | float) -> np.ndarray:
            return np.asarray(true_marginal_cdf(cfg.dgp_code, cfg.rho_eps, component, x, y), dtype=float)

        u1_raw = marginal_cdf_eval(1, x_obs, y1_obs)
        u2_raw = marginal_cdf_eval(2, x_obs, y2_obs)
    else:
        if cfg.marginal_source in TABCF_COMPAT_SOURCES:
            marginal_model = fit_marginal_estimators(
                df_obs,
                v_points=cfg.deepcf_v_points,
                eval_batch_size=cfg.deepcf_eval_batch_size,
                backend_name=TABCF_COMPAT_SOURCE_TO_BACKEND[cfg.marginal_source],
                random_state=cfg.seed,
            )
            backend_metadata = _backend_metadata_for_source(cfg.marginal_source, marginal_model)
        elif cfg.marginal_source == "tabpfn-naive":
            marginal_model = fit_tabpfn_naive_marginal_estimators(df_obs)
        elif cfg.marginal_source == "div":
            marginal_model = fit_div_marginal_estimators(df_obs, seed=cfg.seed)
        else:
            raise ValueError(f"Unexpected marginal_source: {cfg.marginal_source}")

        def marginal_cdf_eval(component: int, x: np.ndarray | float, y: np.ndarray | float) -> np.ndarray:
            return np.asarray(marginal_model.cdf_pointwise(component, x, y), dtype=float)

        u1_raw = marginal_cdf_eval(1, x_obs, y1_obs)
        u2_raw = marginal_cdf_eval(2, x_obs, y2_obs)

    u1 = clip_u(u1_raw, eps=cfg.clip_eps)
    u2 = clip_u(u2_raw, eps=cfg.clip_eps)
    z1 = norm.ppf(u1)
    z2 = norm.ppf(u2)

    rho_hat = fit_gaussian_copula_rho(u1, u2, eps=cfg.clip_eps)
    rho_indep = 0.0

    pit_ks1 = kstest(u1, "uniform")
    pit_ks2 = kstest(u2, "uniform")
    tau_emp = kendalltau(u1, u2, nan_policy="omit").correlation
    rho_s_emp = spearmanr(u1, u2, nan_policy="omit").correlation
    tau_emp = float(tau_emp) if tau_emp is not None else np.nan
    rho_s_emp = float(rho_s_emp) if rho_s_emp is not None else np.nan

    x_bins_df = _x_bin_diagnostics(
        x_obs,
        z1,
        z2,
        n_bins=max(2, cfg.x_bins),
    )

    x_grid = np.linspace(cfg.x_grid_min, cfg.x_grid_max, cfg.x_grid_size)
    tail_x_ref = cfg.tail_x_ref if cfg.tail_x_ref is not None else float(np.median(x_grid))
    rho_true_ref = true_corr_under_do(cfg.dgp_code, cfg.rho_eps, x=tail_x_ref)
    # Keep thresholds defined by true marginals for comparability with existing results.
    a = float(true_marginal_ppf(cfg.dgp_code, cfg.rho_eps, 1, tail_x_ref, 0.5))
    b = float(true_marginal_ppf(cfg.dgp_code, cfg.rho_eps, 2, tail_x_ref, 0.5))

    x_rows = []
    for x_val in x_grid:
        x_float = float(x_val)
        true_tail = true_joint_tail_probability(cfg.dgp_code, cfg.rho_eps, x_float, a, b)
        corr_true_x = true_corr_under_do(cfg.dgp_code, cfg.rho_eps, x=x_float)
        f1_at_a = _as_scalar(marginal_cdf_eval(1, x_float, a))
        f2_at_b = _as_scalar(marginal_cdf_eval(2, x_float, b))

        for model_name in MODEL_NAMES:
            if model_name == "oracle":
                rho_model = np.nan
                tail_hat = float(true_tail)
                joint_mae = 0.0
                corr_hat = float(corr_true_x)
                tau_hat = implied_kendall_tau(corr_true_x)
            else:
                rho_model = _rho_for_model(model_name, rho_hat, rho_indep, corr_true_x)
                tail_hat = upper_tail_probability_from_copula(f1_at_a, f2_at_b, rho_model)
                joint_mae = _joint_cdf_mae_at_x(
                    dgp_code=cfg.dgp_code,
                    rho_eps=cfg.rho_eps,
                    x=x_float,
                    copula_rho=rho_model,
                    quantile_levels=cfg.y_quantile_levels,
                    clip_eps=cfg.clip_eps,
                    marginal_cdf_eval=marginal_cdf_eval,
                )
                corr_hat = float(rho_model)
                tau_hat = implied_kendall_tau(rho_model)
            x_rows.append(
                {
                    "dgp_code": cfg.dgp_code,
                    "n_train": cfg.n_train,
                    "seed": cfg.seed,
                    "rho_eps": cfg.rho_eps,
                    "marginal_source": cfg.marginal_source,
                    "core_backend": backend_metadata["core_backend"],
                    "backend_package": backend_metadata["backend_package"],
                    "checkpoint_version": backend_metadata["checkpoint_version"],
                    "model_path": backend_metadata["model_path"],
                    "deepcf_v_points": cfg.deepcf_v_points,
                    "deepcf_eval_batch_size": cfg.deepcf_eval_batch_size,
                    "x": x_float,
                    "model": model_name,
                    "copula_rho": float(rho_model),
                    "tail_true": float(true_tail),
                    "tail_hat": float(tail_hat),
                    "tail_abs_error": float(abs(tail_hat - true_tail)),
                    "corr_true": float(corr_true_x),
                    "corr_hat": corr_hat,
                    "tau_true": implied_kendall_tau(corr_true_x),
                    "tau_hat": tau_hat,
                    "joint_cdf_mae": float(joint_mae),
                }
            )
    x_metrics = pd.DataFrame(x_rows)

    dist_cdf_df, dist_samples_df = _build_distributional_outputs(
        cfg=cfg,
        x_grid=x_grid,
        rho_hat=rho_hat,
        rho_indep=rho_indep,
        marginal_cdf_eval=marginal_cdf_eval,
        backend_metadata=backend_metadata,
    )

    def _model_mean(col: str, model: str) -> float:
        subset = x_metrics[x_metrics["model"] == model]
        if subset.empty:
            return np.nan
        return float(subset[col].mean())

    xbin_corr_series = x_bins_df["corr_z1_z2"] if not x_bins_df.empty else pd.Series(dtype=float)

    summary = {
        "dgp_code": cfg.dgp_code,
        "n_train": int(cfg.n_train),
        "seed": int(cfg.seed),
        "rho_eps": float(cfg.rho_eps),
        "marginal_source": cfg.marginal_source,
        "core_backend": backend_metadata["core_backend"],
        "backend_package": backend_metadata["backend_package"],
        "checkpoint_version": backend_metadata["checkpoint_version"],
        "model_path": backend_metadata["model_path"],
        "deepcf_v_points": int(cfg.deepcf_v_points),
        "deepcf_eval_batch_size": int(cfg.deepcf_eval_batch_size),
        "save_distribution_output": bool(cfg.save_distribution_output),
        "dist_quantile_count": int(len(cfg.dist_quantile_levels)),
        "dist_sample_size": int(cfg.dist_sample_size),
        "rho_true": float(rho_true_ref),
        "rho_hat": float(rho_hat),
        "rho_abs_error": float(abs(rho_hat - rho_true_ref)),
        "tau_true": implied_kendall_tau(rho_true_ref),
        "tau_hat": implied_kendall_tau(rho_hat),
        "tau_abs_error": float(abs(implied_kendall_tau(rho_hat) - implied_kendall_tau(rho_true_ref))),
        "spearman_true": implied_spearman_rho(rho_true_ref),
        "spearman_hat": implied_spearman_rho(rho_hat),
        "spearman_abs_error": float(abs(implied_spearman_rho(rho_hat) - implied_spearman_rho(rho_true_ref))),
        "pseudo_tau_empirical": float(tau_emp),
        "pseudo_spearman_empirical": float(rho_s_emp),
        "pit_mean_u1": float(np.mean(u1)),
        "pit_mean_u2": float(np.mean(u2)),
        "pit_var_u1": float(np.var(u1)),
        "pit_var_u2": float(np.var(u2)),
        "pit_ks_stat_u1": float(pit_ks1.statistic),
        "pit_ks_pvalue_u1": float(pit_ks1.pvalue),
        "pit_ks_stat_u2": float(pit_ks2.statistic),
        "pit_ks_pvalue_u2": float(pit_ks2.pvalue),
        "xbin_corr_mean": float(xbin_corr_series.mean()) if not xbin_corr_series.empty else np.nan,
        "xbin_corr_std": float(xbin_corr_series.std()) if not xbin_corr_series.empty else np.nan,
        "xbin_corr_min": float(xbin_corr_series.min()) if not xbin_corr_series.empty else np.nan,
        "xbin_corr_max": float(xbin_corr_series.max()) if not xbin_corr_series.empty else np.nan,
        "x_grid_min": float(cfg.x_grid_min),
        "x_grid_max": float(cfg.x_grid_max),
        "x_grid_size": int(cfg.x_grid_size),
        "tail_x_ref": float(tail_x_ref),
        "tail_threshold_a": float(a),
        "tail_threshold_b": float(b),
        "estimated_tail_mae": _model_mean("tail_abs_error", "estimated"),
        "independence_tail_mae": _model_mean("tail_abs_error", "independence"),
        "oracle_tail_mae": _model_mean("tail_abs_error", "oracle"),
        "estimated_joint_cdf_mae": _model_mean("joint_cdf_mae", "estimated"),
        "independence_joint_cdf_mae": _model_mean("joint_cdf_mae", "independence"),
        "oracle_joint_cdf_mae": _model_mean("joint_cdf_mae", "oracle"),
    }
    return summary, x_metrics, x_bins_df, dist_cdf_df, dist_samples_df
