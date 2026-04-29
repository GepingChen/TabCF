from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from multivar.core import experiment as experiment_mod
from multivar.core.dgp import (
    DGP4_PIECEWISE,
    DGP5_SOFTPLUS,
    _piecewise_g1,
    normalize_dgp_code,
    sample_interventional_data,
    true_marginal_cdf,
    true_marginal_ppf,
)
from multivar.core.run_pipeline import build_parser


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_DEFAULT_DGPS = [
    "DGP1_LINEAR",
    "DGP2_NONLINEAR",
    "DGP3_PRE_ADDITIVE",
    "DGP4_PIECEWISE",
    "DGP5_SOFTPLUS",
]


def _softplus_g1_expected(x):
    x_arr = np.asarray(x, dtype=float)
    return np.logaddexp(0.0, 2.0 * x_arr)


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("piecewise", DGP4_PIECEWISE),
        ("D4", DGP4_PIECEWISE),
        ("DGP4", DGP4_PIECEWISE),
        ("softplus", DGP5_SOFTPLUS),
        ("D5", DGP5_SOFTPLUS),
        ("DGP5", DGP5_SOFTPLUS),
    ],
)
def test_normalize_dgp_code_accepts_new_aliases(code, expected):
    assert normalize_dgp_code(code) == expected


def test_piecewise_transform_covers_boundary_branches():
    w = np.array([-1e-8, 0.0, 0.4, 1.0, 1.0 + 1e-8], dtype=float)
    expected = np.array([-1e-8, 0.0, 0.8, 2.0, 2.0 + 0.5e-8], dtype=float)
    assert np.allclose(_piecewise_g1(w), expected)


@pytest.mark.parametrize(
    ("dgp_code", "g1_expected"),
    [
        (DGP4_PIECEWISE, _piecewise_g1),
        (DGP5_SOFTPLUS, _softplus_g1_expected),
    ],
)
def test_sample_interventional_data_matches_new_structural_equations(dgp_code, g1_expected):
    x_value = 0.75
    df = sample_interventional_data(dgp_code, rho_eps=0.3, x=x_value, n=12, seed=17)

    w1 = x_value + df["H"].to_numpy(dtype=float) + df["eps1"].to_numpy(dtype=float)
    w2 = x_value + df["H"].to_numpy(dtype=float) + df["eps2"].to_numpy(dtype=float)

    assert np.allclose(df["X"].to_numpy(dtype=float), x_value)
    assert np.allclose(df["Y1"].to_numpy(dtype=float), g1_expected(w1))
    assert np.allclose(df["Y2"].to_numpy(dtype=float), 0.5 * g1_expected(w2))


@pytest.mark.parametrize("dgp_code", [DGP4_PIECEWISE, DGP5_SOFTPLUS])
@pytest.mark.parametrize("component", [1, 2])
def test_new_dgp_marginal_ppf_roundtrip(dgp_code, component):
    x_value = 1.2
    u = np.array([0.1, 0.25, 0.5, 0.75, 0.9], dtype=float)

    y = true_marginal_ppf(dgp_code, rho_eps=0.3, component=component, x=x_value, u=u)
    cdf = true_marginal_cdf(dgp_code, rho_eps=0.3, component=component, x=x_value, y=y)

    assert np.all(np.isfinite(y))
    assert np.all((cdf >= 0.0) & (cdf <= 1.0))
    assert np.allclose(cdf, u, atol=5e-2)


@pytest.mark.parametrize("dgp_code", [DGP4_PIECEWISE, DGP5_SOFTPLUS])
def test_run_single_experiment_oracle_matches_true_joint_distribution_for_new_dgps(dgp_code):
    cfg = experiment_mod.ExperimentConfig(
        dgp_code=dgp_code,
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
    oracle_cdf = dist_cdf[dist_cdf["model"] == "oracle"].reset_index(drop=True)
    oracle_samples = dist_samples[dist_samples["model"] == "oracle"].reset_index(drop=True)

    assert not oracle_x.empty
    assert np.allclose(oracle_x["tail_abs_error"].to_numpy(dtype=float), 0.0)
    assert np.allclose(oracle_x["joint_cdf_mae"].to_numpy(dtype=float), 0.0)
    assert oracle_x["copula_rho"].isna().all()

    assert not oracle_cdf.empty
    assert np.allclose(oracle_cdf["joint_cdf_abs_error"].to_numpy(dtype=float), 0.0)
    assert oracle_cdf["copula_rho"].isna().all()

    assert not oracle_samples.empty
    assert oracle_samples["copula_rho"].isna().all()
    grouped_weights = oracle_samples.groupby("x", as_index=False)["sample_weight"].sum()
    assert np.allclose(grouped_weights["sample_weight"].to_numpy(dtype=float), 1.0)

    assert summary["oracle_tail_mae"] == pytest.approx(0.0)
    assert summary["oracle_joint_cdf_mae"] == pytest.approx(0.0)
    assert isinstance(x_bins.empty, bool)


def test_run_batch_parser_defaults_include_new_dgps():
    args = build_parser().parse_args([])
    assert args.dgp_codes == EXPECTED_DEFAULT_DGPS
    assert args.x_grid_size == 13


def test_presets_include_new_dgps():
    presets = json.loads((REPO_ROOT / "multivar" / "manifests" / "presets.yaml").read_text())
    assert presets["presets"]["pilot_s10"]["dgp_codes"] == EXPECTED_DEFAULT_DGPS
    assert presets["presets"]["paper_s50"]["dgp_codes"] == EXPECTED_DEFAULT_DGPS
    assert presets["presets"]["pilot_s10"]["x_grid_size"] == 13
    assert presets["presets"]["paper_s50"]["x_grid_size"] == 13


def test_submit_pipeline_dry_run_uses_new_default_dgps():
    script = REPO_ROOT / "multivar" / "slurm" / "submit" / "submit_multivar_pipeline.sh"
    if not script.exists():
        pytest.skip(f"Optional scheduler fixture is not present in this checkout: {script}")
    result = subprocess.run(
        ["bash", str(script), "--dry-run"],
        check=True,
        capture_output=True,
        text=True,
    )
    stdout = result.stdout

    assert "DGP4_PIECEWISE" in stdout
    assert "DGP5_SOFTPLUS" in stdout
    assert "X_GRID_SIZE=13" in stdout
