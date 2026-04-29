from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = REPO_ROOT / "interv_qtl" / "run_interv_quantile.py"
QUANTILE_PATH = REPO_ROOT / "interv_qtl" / "cdf_to_quantiles.py"
PLOT_PATH = REPO_ROOT / "interv_qtl" / "viz" / "plot_rmse_boxplot.py"


def _load_runner_module(monkeypatch):
    fake_stage1 = types.ModuleType("stage1_control")

    @dataclass
    class Stage1Config:
        random_state: int = 1
        backend_name: str = "tabpfn"
        softmax_temperature: float | None = None

    fake_stage1.Stage1Config = Stage1Config
    fake_stage1.run_stage1_experiment = lambda *args, **kwargs: None

    fake_stage2 = types.ModuleType("stage2_outcome")

    class ConditionalCDFEstimator:
        def __init__(self, *args, **kwargs):
            del args, kwargs

    fake_stage2.ConditionalCDFEstimator = ConditionalCDFEstimator
    fake_stage2.load_stage1_data = lambda *args, **kwargs: {}

    fake_dgp = types.ModuleType("dgp")

    @dataclass
    class DGPConfig:
        n: int = 0
        seed: int = 0
        first_stage: str = "A3"
        second_stage: str = "B3"

    fake_dgp.DGPConfig = DGPConfig
    fake_dgp.set_seed = lambda *args, **kwargs: None

    fake_foundation = types.ModuleType("foundation_backends")
    fake_foundation.TABPFN_BACKEND = "tabpfn"
    fake_foundation.backend_metadata = lambda backend_name, model_path: {
        "backend": backend_name,
        "backend_package": backend_name,
        "checkpoint_version": "",
        "model_path": model_path,
    }
    fake_foundation.normalize_backend_name = lambda backend_name: "tabpfn" if backend_name in (None, "") else str(backend_name)

    def stage1_output_filename(
        kind,
        codes,
        train_sample_size,
        seed,
        backend_name="tabpfn",
        softmax_temperature=None,
        timestamp=None,
    ):
        del kind, timestamp
        suffix = "" if backend_name == "tabpfn" else f"_{backend_name}"
        temp_suffix = ""
        if softmax_temperature is not None:
            normalized = format(float(softmax_temperature), "g").replace("-", "m").replace(".", "p")
            temp_suffix = f"_st{normalized}"
        return f"iv_stage1_train_{codes}_n{train_sample_size}_seed{seed}{suffix}{temp_suffix}.csv"

    fake_foundation.stage1_output_filename = stage1_output_filename

    fake_quantile = types.ModuleType("cdf_to_quantiles")
    fake_quantile.build_y_grid = lambda *args, **kwargs: None
    fake_quantile.compute_quantiles_on_grid = lambda *args, **kwargs: None
    fake_quantile.compute_true_quantiles = lambda *args, **kwargs: None

    monkeypatch.setitem(sys.modules, "stage1_control", fake_stage1)
    monkeypatch.setitem(sys.modules, "stage2_outcome", fake_stage2)
    monkeypatch.setitem(sys.modules, "dgp", fake_dgp)
    monkeypatch.setitem(sys.modules, "foundation_backends", fake_foundation)
    monkeypatch.setitem(sys.modules, "cdf_to_quantiles", fake_quantile)

    spec = importlib.util.spec_from_file_location("simq_runner_test_module", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def _load_quantile_module(monkeypatch):
    fake_stage2 = types.ModuleType("stage2_outcome")

    class ConditionalCDFEstimator:
        def __init__(self, *args, **kwargs):
            del args, kwargs

    def cdf_from_full_output(full_output, y_values, *, squeeze_last=True):
        x_vals = np.asarray(full_output["x"], dtype=float).reshape(-1, 1)
        y_arr = np.asarray(y_values, dtype=float).reshape(1, -1)
        cdf_vals = 1.0 / (1.0 + np.exp(-(y_arr - x_vals)))
        if squeeze_last and cdf_vals.shape[-1] == 1:
            return cdf_vals[:, 0]
        return cdf_vals

    fake_stage2.ConditionalCDFEstimator = ConditionalCDFEstimator
    fake_stage2._latest_matching_file = lambda *args, **kwargs: None
    fake_stage2.cdf_from_full_output = cdf_from_full_output
    fake_stage2.load_stage1_data = lambda *args, **kwargs: {}
    fake_stage2.monte_carlo_y_given_x = lambda *args, **kwargs: (np.zeros(1, dtype=float), None)

    fake_dgp = types.ModuleType("dgp")

    @dataclass
    class DGPConfig:
        n: int = 0
        seed: int = 0
        first_stage: str = "A3"
        second_stage: str = "B3"

    fake_dgp.DGPConfig = DGPConfig
    fake_dgp.set_seed = lambda *args, **kwargs: None

    monkeypatch.setitem(sys.modules, "stage2_outcome", fake_stage2)
    monkeypatch.setitem(sys.modules, "dgp", fake_dgp)

    spec = importlib.util.spec_from_file_location("simq_quantile_test_module", QUANTILE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def _load_plot_module():
    spec = importlib.util.spec_from_file_location("simq_plot_test_module", PLOT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_backend_aware_filename_helpers(monkeypatch):
    runner = _load_runner_module(monkeypatch)

    assert runner.expected_stage1_csv("A3", "B3", 1000, 1).name == "iv_stage1_train_A3_B3_n1000_seed1.csv"
    assert (
        runner.expected_stage1_csv("A3", "B3", 1000, 1, backend_name="tabpfn_real").name
        == "iv_stage1_train_A3_B3_n1000_seed1_tabpfn_real.csv"
    )
    assert (
        runner.expected_stage1_csv("A3", "B3", 1000, 1, softmax_temperature=1.0).name
        == "iv_stage1_train_A3_B3_n1000_seed1_st1.csv"
    )
    assert (
        runner.expected_stage2_predictions("A3", "B3", 1000, 1, backend_name="tabpfn_real").name
        == "s2q_tabpfn_real_A3_B3_n1000_seed1_predictions.csv"
    )
    assert (
        runner.expected_stage2_summary("A3", "B3", 1000, 1, backend_name="tabicl").name
        == "s2q_tabicl_A3_B3_n1000_seed1_summary.csv"
    )
    assert (
        runner.expected_stage2_summary(
            "A3",
            "B3",
            1000,
            1,
            output_dir=Path("/tmp/custom_stage2"),
            backend_name="tabpfn",
        )
        == Path("/tmp/custom_stage2") / "s2q_A3_B3_n1000_seed1_summary.csv"
    )


def test_runner_parse_args_defaults_to_gauss_legendre(monkeypatch):
    runner = _load_runner_module(monkeypatch)
    for key in (
        "DGP_CODES_OVERRIDE",
        "TRAIN_SIZES_OVERRIDE",
        "SEEDS_OVERRIDE",
        "MODEL_BACKEND_OVERRIDE",
        "SOFTMAX_TEMPERATURE_OVERRIDE",
        "SKIP_EXISTING",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_interv_quantile.py", "--train-sizes", "1000", "--seeds", "1"],
    )

    cfg = runner.parse_args()

    assert cfg.mu_integrator == "gauss_legendre"
    assert cfg.gauss_legendre_order == 18
    assert cfg.softmax_temperature is None


def test_runner_parse_args_accepts_softmax_temperature_from_cli(monkeypatch):
    runner = _load_runner_module(monkeypatch)
    monkeypatch.delenv("SOFTMAX_TEMPERATURE_OVERRIDE", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_interv_quantile.py", "--train-sizes", "1000", "--seeds", "1", "--softmax-temperature", "1"],
    )

    cfg = runner.parse_args()

    assert cfg.softmax_temperature == 1.0


def test_runner_parse_args_accepts_softmax_temperature_from_env(monkeypatch):
    runner = _load_runner_module(monkeypatch)
    monkeypatch.setenv("SOFTMAX_TEMPERATURE_OVERRIDE", "1")
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_interv_quantile.py", "--train-sizes", "1000", "--seeds", "1"],
    )

    cfg = runner.parse_args()

    assert cfg.softmax_temperature == 1.0


def test_build_x_grid_uses_trimmed_quantile_range(monkeypatch):
    runner = _load_runner_module(monkeypatch)

    train_data = {"X": np.array([-10.0, -5.0, 0.0, 5.0, 10.0], dtype=float)}
    test_df = pd.DataFrame({"X": np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)})

    test_grid = runner.build_x_grid(
        train_data,
        test_df,
        mode="test_quantile",
        points=5,
    )
    train_grid = runner.build_x_grid(
        train_data,
        test_df,
        mode="train_quantile",
        points=5,
    )

    expected_qs = np.linspace(0.05, 0.95, 5)
    assert np.allclose(test_grid, np.quantile(test_df["X"], expected_qs))
    assert np.allclose(train_grid, np.quantile(train_data["X"], expected_qs))
    assert np.isclose(test_grid[0], np.quantile(test_df["X"], 0.05))
    assert np.isclose(test_grid[-1], np.quantile(test_df["X"], 0.95))


def test_summary_matches_requested_integrator_flags_legacy_outputs(monkeypatch, tmp_path):
    runner = _load_runner_module(monkeypatch)
    legacy_path = tmp_path / "legacy_summary.csv"
    legacy_path.write_text("code,train_size,seed,tau,rmse\nA3_B3,1000,1,0.5,0.1\n", encoding="utf-8")

    assert not runner.summary_matches_requested_integrator(
        legacy_path,
        mu_integrator="gauss_legendre",
        gauss_legendre_order=18,
        softmax_temperature=1.0,
    )

    modern_path = tmp_path / "modern_summary.csv"
    modern_path.write_text(
        "code,train_size,seed,tau,mu_integrator,gauss_legendre_order,softmax_temperature,rmse\n"
        "A3_B3,1000,1,0.5,gauss_legendre,18,1,0.1\n",
        encoding="utf-8",
    )

    assert runner.summary_matches_requested_integrator(
        modern_path,
        mu_integrator="gauss_legendre",
        gauss_legendre_order=18,
        softmax_temperature=1.0,
    )
    assert not runner.summary_matches_requested_integrator(
        modern_path,
        mu_integrator="simpson",
        gauss_legendre_order=18,
        softmax_temperature=1.0,
    )
    assert not runner.summary_matches_requested_integrator(
        modern_path,
        mu_integrator="gauss_legendre",
        gauss_legendre_order=18,
        softmax_temperature=None,
    )


def test_quantile_config_defaults_to_gauss_legendre(monkeypatch):
    quantile = _load_quantile_module(monkeypatch)

    cfg = quantile.QuantileConfig()

    assert cfg.mu_integrator == "gauss_legendre"
    assert cfg.gauss_legendre_order == 18


def test_compute_quantiles_on_grid_matches_for_constant_v_cdf(monkeypatch):
    quantile = _load_quantile_module(monkeypatch)

    class FakeCDFModel:
        def predict_full_distribution(self, x, v):
            del v
            return {"x": np.asarray(x, dtype=float)}

    x_grid = np.array([-1.0, 0.0, 1.0], dtype=float)
    y_grid = np.linspace(-10.0, 10.0, 2001)
    taus = (0.25, 0.5, 0.75)

    simpson_quantiles = quantile.compute_quantiles_on_grid(
        FakeCDFModel(),
        x_grid,
        y_grid,
        taus,
        n_v_points=9,
        integrator_cfg=quantile.MuIntegratorConfig(method="simpson", n_v_points=9),
    )
    gauss_legendre_quantiles = quantile.compute_quantiles_on_grid(
        FakeCDFModel(),
        x_grid,
        y_grid,
        taus,
        n_v_points=9,
        integrator_cfg=quantile.MuIntegratorConfig(method="gauss_legendre", gauss_legendre_order=18),
    )

    assert np.allclose(simpson_quantiles, gauss_legendre_quantiles, atol=1e-5)


def test_parse_filename_supports_legacy_and_new_backends():
    plot = _load_plot_module()

    assert plot.parse_filename(Path("s2q_A3_B3_n1000_seed1_summary.csv")) == {
        "method": "tabpfn",
        "code": "A3_B3",
        "n": 1000,
        "seed": 1,
    }
    assert plot.parse_filename(Path("s2q_tabpfn_real_A3_B3_n1000_seed1_summary.csv")) == {
        "method": "tabpfn_real",
        "code": "A3_B3",
        "n": 1000,
        "seed": 1,
    }
    assert plot.parse_filename(Path("s2q_tabicl_A3_B3_n1000_seed1_summary.csv")) == {
        "method": "tabicl",
        "code": "A3_B3",
        "n": 1000,
        "seed": 1,
    }


def test_method_order_labels_and_colors_cover_new_backends():
    plot = _load_plot_module()

    assert plot.METHOD_ORDER[:3] == ("tabpfn", "tabpfn_real", "tabicl")
    assert plot.friendly_method("tabpfn") == "TabCF (TabPFNv2.5)"
    assert plot.friendly_method("tabpfn_real") == "TabCF (Real-TabPFNv2.5)"
    assert plot.friendly_method("tabicl") == "TabCF (TabICLv2)"
    assert plot.color_for_method("tabpfn_real") == "#8c564b"
    assert plot.color_for_method("tabicl") == "#17becf"
