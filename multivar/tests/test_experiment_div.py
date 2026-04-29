from __future__ import annotations

import numpy as np
import pytest

from multivar.core import experiment as experiment_mod
from multivar.core.dgp import sample_interventional_data
from multivar.core.run_pipeline import build_parser


class _FakeDIVMarginalModel:
    def cdf_pointwise(self, component: int, x, y):
        x_b, y_b = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        shifted = y_b - 0.15 * component * x_b
        return 1.0 / (1.0 + np.exp(-shifted))


class _FakeCoreMarginalModel:
    def __init__(self, source: str):
        self.metadata = {
            "core_backend": source,
            "backend_package": "tabicl" if source == "tabicl" else "tabpfn",
            "checkpoint_version": (
                "tabicl-regressor-v2-20260212.ckpt"
                if source == "tabicl"
                else "tabpfn-v2.5-regressor-v2.5_real.ckpt"
            ),
            "model_path": "auto",
        }

    def cdf_pointwise(self, component: int, x, y):
        x_b, y_b = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        shifted = y_b - 0.1 * component * x_b
        return 1.0 / (1.0 + np.exp(-shifted))


def test_run_single_experiment_supports_div(monkeypatch):
    monkeypatch.setattr(
        experiment_mod,
        "fit_div_marginal_estimators",
        lambda df_obs, seed: _FakeDIVMarginalModel(),
    )

    cfg = experiment_mod.ExperimentConfig(
        dgp_code="DGP1_LINEAR",
        n_train=48,
        seed=3,
        rho_eps=0.3,
        marginal_source="div",
        x_grid_size=5,
        save_distribution_output=False,
    )
    summary, x_metrics, x_bins, dist_cdf, dist_samples = experiment_mod.run_single_experiment(cfg)

    assert summary["marginal_source"] == "div"
    assert not x_metrics.empty
    assert "marginal_source" in x_metrics.columns
    assert set(x_metrics["marginal_source"]) == {"div"}
    assert isinstance(x_bins.empty, bool)
    assert dist_cdf.empty
    assert dist_samples.empty


def test_run_batch_parser_accepts_div():
    args = build_parser().parse_args(["--marginal-source", "div", "--seeds", "1"])
    assert args.marginal_source == "div"


@pytest.mark.parametrize("source", ["tabpfn_real", "tabicl"])
def test_run_batch_parser_accepts_new_core_sources(source):
    args = build_parser().parse_args(["--marginal-source", source, "--seeds", "1"])
    assert args.marginal_source == source


@pytest.mark.parametrize("source", ["tabpfn_real", "tabicl"])
def test_run_single_experiment_writes_core_backend_metadata(monkeypatch, source):
    monkeypatch.setattr(
        experiment_mod,
        "fit_marginal_estimators",
        lambda *args, **kwargs: _FakeCoreMarginalModel(source),
    )

    cfg = experiment_mod.ExperimentConfig(
        dgp_code="DGP1_LINEAR",
        n_train=32,
        seed=2,
        rho_eps=0.3,
        marginal_source=source,
        x_grid_size=3,
        dist_quantile_levels=(0.25, 0.75),
        dist_sample_size=24,
        save_distribution_output=True,
    )

    summary, x_metrics, _x_bins, dist_cdf, dist_samples = experiment_mod.run_single_experiment(cfg)

    assert summary["marginal_source"] == source
    assert summary["core_backend"] == source
    assert summary["backend_package"] == ("tabicl" if source == "tabicl" else "tabpfn")
    assert summary["checkpoint_version"]
    assert set(x_metrics["core_backend"]) == {source}
    assert set(dist_cdf["core_backend"]) == {source}
    assert set(dist_samples["core_backend"]) == {source}


def test_run_batch_parser_rejects_unknown_marginal_source():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--marginal-source", "legacy-div"])


def test_sample_interventional_data_matches_structural_equations():
    df = sample_interventional_data("DGP1_LINEAR", rho_eps=0.3, x=1.25, n=8, seed=11)

    assert np.allclose(df["X"].to_numpy(dtype=float), 1.25)
    assert np.allclose(
        df["Y1"].to_numpy(dtype=float),
        1.25 - 3.0 * df["H"].to_numpy(dtype=float) + df["eps1"].to_numpy(dtype=float),
    )
    assert np.allclose(
        df["Y2"].to_numpy(dtype=float),
        0.5 * 1.25 - 1.0 * df["H"].to_numpy(dtype=float) + df["eps2"].to_numpy(dtype=float),
    )


def test_run_single_experiment_oracle_matches_true_joint_distribution():
    cfg = experiment_mod.ExperimentConfig(
        dgp_code="DGP2_NONLINEAR",
        n_train=64,
        seed=5,
        rho_eps=0.3,
        marginal_source="oracle",
        x_grid_size=3,
        y_quantile_levels=(0.25, 0.5, 0.75),
        dist_quantile_levels=(0.25, 0.75),
        dist_sample_size=32,
        save_distribution_output=True,
    )

    summary, x_metrics, x_bins, dist_cdf, dist_samples = experiment_mod.run_single_experiment(cfg)

    oracle_x = x_metrics[x_metrics["model"] == "oracle"].reset_index(drop=True)
    assert not oracle_x.empty
    assert np.allclose(oracle_x["tail_abs_error"].to_numpy(dtype=float), 0.0)
    assert np.allclose(oracle_x["joint_cdf_mae"].to_numpy(dtype=float), 0.0)
    assert oracle_x["copula_rho"].isna().all()

    oracle_cdf = dist_cdf[dist_cdf["model"] == "oracle"].reset_index(drop=True)
    assert not oracle_cdf.empty
    assert np.allclose(oracle_cdf["joint_cdf_abs_error"].to_numpy(dtype=float), 0.0)
    assert oracle_cdf["copula_rho"].isna().all()

    oracle_samples = dist_samples[dist_samples["model"] == "oracle"].reset_index(drop=True)
    assert not oracle_samples.empty
    assert oracle_samples["copula_rho"].isna().all()
    grouped_weights = oracle_samples.groupby("x", as_index=False)["sample_weight"].sum()
    assert np.allclose(grouped_weights["sample_weight"].to_numpy(dtype=float), 1.0)

    assert summary["oracle_tail_mae"] == pytest.approx(0.0)
    assert summary["oracle_joint_cdf_mae"] == pytest.approx(0.0)
    assert isinstance(x_bins.empty, bool)
