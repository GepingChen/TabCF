from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "interv_qtl" / "benchmark_runtime.py"


def _load_runtime_module(monkeypatch):
    fake_stage1 = types.ModuleType("stage1_control")
    fake_stage2 = types.ModuleType("stage2_outcome")

    @dataclass
    class MuIntegratorConfig:
        method: str = "gauss_legendre"
        n_v_points: int = 101
        gauss_legendre_order: int = 18
        max_points_per_batch: int | None = None

    fake_mu = types.ModuleType("mu_integrators")
    fake_mu.DEFAULT_GAUSS_LEGENDRE_ORDER = 18
    fake_mu.DEFAULT_MU_INTEGRATOR = "gauss_legendre"
    fake_mu.MuIntegratorConfig = MuIntegratorConfig

    fake_quantile = types.ModuleType("cdf_to_quantiles")
    fake_quantile.build_y_grid = lambda *args, **kwargs: None
    fake_quantile.compute_quantiles_on_grid = lambda *args, **kwargs: None

    fake_runner = types.ModuleType("run_interv_quantile")
    fake_runner.DEFAULT_TAUS = (0.1, 0.5, 0.9)
    fake_runner.SIM_DATA_DIR = REPO_ROOT / "interv_qtl" / "IV_datasets"
    fake_runner.TEST_DIR = fake_runner.SIM_DATA_DIR / "test"
    fake_runner.TRAIN_DIR = fake_runner.SIM_DATA_DIR / "train"
    fake_runner.build_x_grid = lambda *args, **kwargs: None

    monkeypatch.setitem(sys.modules, "stage1_control", fake_stage1)
    monkeypatch.setitem(sys.modules, "stage2_outcome", fake_stage2)
    monkeypatch.setitem(sys.modules, "mu_integrators", fake_mu)
    monkeypatch.setitem(sys.modules, "cdf_to_quantiles", fake_quantile)
    monkeypatch.setitem(sys.modules, "run_interv_quantile", fake_runner)

    spec = importlib.util.spec_from_file_location("simq_runtime_test_module", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_build_method_specs_orders_and_dedupes(monkeypatch):
    runtime = _load_runtime_module(monkeypatch)

    specs = runtime.build_method_specs(["ivqr", "tabpfn", "div", "tabpfn"])

    assert [spec.method for spec in specs] == ["tabpfn", "div", "ivqr"]
    assert [spec.method_label for spec in specs] == ["TabCF", "DIV", "IVQR"]


def test_summarise_runtime_rows_skips_errors(monkeypatch):
    runtime = _load_runtime_module(monkeypatch)
    rows = [
        {"train_size": 1000, "method": "tabpfn", "method_label": "TabCF", "seconds": 1.0, "status": "ok"},
        {"train_size": 1000, "method": "tabpfn", "method_label": "TabCF", "seconds": 3.0, "status": "ok"},
        {"train_size": 1000, "method": "div", "method_label": "DIV", "seconds": 5.0, "status": "error"},
        {"train_size": 1000, "method": "div", "method_label": "DIV", "seconds": 7.0, "status": "ok"},
    ]

    summary = runtime.summarise_runtime_rows(rows)

    assert summary == [
        {
            "train_size": 1000,
            "method": "tabpfn",
            "method_label": "TabCF",
            "mean_seconds": 2.0,
            "std_seconds": 1.0,
            "n_runs": 2,
        },
        {
            "train_size": 1000,
            "method": "div",
            "method_label": "DIV",
            "mean_seconds": 7.0,
            "std_seconds": 0.0,
            "n_runs": 1,
        },
    ]


def test_build_r_runtime_command_uses_raw_paths(monkeypatch):
    runtime = _load_runtime_module(monkeypatch)
    run_ctx = runtime.RunContext(
        code="A3_B3",
        train_size=1000,
        seed=4,
        raw_train_path=Path("/tmp/train.csv"),
        raw_test_path=Path("/tmp/test.csv"),
    )

    cmd = runtime.build_r_runtime_command(
        rscript_bin="Rscript",
        script_path=Path("/tmp/run_div_baseline.r"),
        method="div",
        run_ctx=run_ctx,
        x_grid_mode="test_quantile",
        x_grid_points=200,
        taus=(0.1, 0.5, 0.9),
        timer_scope="predict_only",
    )

    assert cmd == [
        "Rscript",
        "/tmp/run_div_baseline.r",
        "--benchmark-runtime",
        "--benchmark-train",
        "/tmp/train.csv",
        "--benchmark-test",
        "/tmp/test.csv",
        "--benchmark-code",
        "A3_B3",
        "--benchmark-seed",
        "4",
        "--benchmark-x-grid-mode",
        "test_quantile",
        "--benchmark-x-grid-points",
        "200",
        "--benchmark-timer-scope",
        "predict_only",
        "--benchmark-taus",
        "0.1,0.5,0.9",
    ]


def test_default_timer_scope_is_fit_predict(monkeypatch):
    runtime = _load_runtime_module(monkeypatch)
    assert runtime.DEFAULT_TIMER_SCOPE == "fit_predict"


def test_python_timer_scopes_split_fit_and_predict(monkeypatch):
    runtime = _load_runtime_module(monkeypatch)
    counts = {"stage1": 0, "fit": 0, "predict": 0, "x_grid": 0, "y_grid": 0}

    class FakeStage1Config:
        def __init__(self, random_state, backend_name):
            self.random_state = random_state
            self.backend_name = backend_name

    def fake_build_stage1_training_frame(train_df, config, verbose=False):
        del train_df, config, verbose
        counts["stage1"] += 1
        return {"train": pd.DataFrame({"X": [1.0], "V_hat": [0.5], "Y": [2.0]})}

    class FakeEstimator:
        def __init__(self, use_tabpfn, backend_name, random_state):
            del use_tabpfn, backend_name, random_state

        def fit_full(self, X, V_hat, Y, verbose=False):
            del X, V_hat, Y, verbose
            counts["fit"] += 1

    def fake_build_x_grid(stage1_df, test_df, mode, points):
        del stage1_df, test_df, mode, points
        counts["x_grid"] += 1
        return np.array([0.1, 0.2])

    def fake_build_y_grid(y_values, padding, n_points):
        del y_values, padding, n_points
        counts["y_grid"] += 1
        return np.array([-1.0, 1.0])

    def fake_compute_quantiles_on_grid(cdf_model, x_grid, y_grid, taus, n_v_grid, max_points_per_batch=None, integrator_cfg=None):
        del cdf_model, x_grid, y_grid, taus, n_v_grid, max_points_per_batch, integrator_cfg
        counts["predict"] += 1
        return np.zeros((2, 2), dtype=float)

    runtime.stage1.Stage1Config = FakeStage1Config
    runtime.stage1.build_stage1_training_frame = fake_build_stage1_training_frame
    runtime.stage2.ConditionalCDFEstimator = FakeEstimator
    runtime.build_x_grid = fake_build_x_grid
    runtime.build_y_grid = fake_build_y_grid
    runtime.compute_quantiles_on_grid = fake_compute_quantiles_on_grid
    runtime._read_raw_frames = lambda run_ctx: (
        pd.DataFrame({"X": [0.0], "Y": [1.0], "Z": [2.0]}),
        pd.DataFrame({"X": [0.25, 0.75]}),
    )

    method_spec = runtime.RuntimeMethodSpec(method="tabpfn", method_label="TabCF", family="python")
    run_ctx = runtime.RunContext(
        code="A3_B3",
        train_size=1000,
        seed=1,
        raw_train_path=Path("/tmp/train.csv"),
        raw_test_path=Path("/tmp/test.csv"),
    )
    integrator_cfg = runtime.MuIntegratorConfig()

    row = runtime._run_python_method(
        method_spec,
        run_ctx,
        stage1_random_state=1,
        stage2_random_state=2,
        x_grid_mode="test_quantile",
        x_grid_points=200,
        y_grid_points=101,
        y_grid_padding=0.25,
        taus=(0.1, 0.9),
        n_v_grid=51,
        integrator_cfg=integrator_cfg,
        max_points_per_batch=None,
        timer_scope="fit_only",
    )
    assert row["status"] == "ok"
    assert counts == {"stage1": 1, "fit": 1, "predict": 0, "x_grid": 0, "y_grid": 0}
    assert row["stage1_seconds"] != ""
    assert row["stage2_seconds"] != ""

    counts = {"stage1": 0, "fit": 0, "predict": 0, "x_grid": 0, "y_grid": 0}
    row = runtime._run_python_method(
        method_spec,
        run_ctx,
        stage1_random_state=1,
        stage2_random_state=2,
        x_grid_mode="test_quantile",
        x_grid_points=200,
        y_grid_points=101,
        y_grid_padding=0.25,
        taus=(0.1, 0.9),
        n_v_grid=51,
        integrator_cfg=integrator_cfg,
        max_points_per_batch=None,
        timer_scope="fit_predict",
    )
    assert row["status"] == "ok"
    assert counts == {"stage1": 1, "fit": 1, "predict": 1, "x_grid": 1, "y_grid": 1}
    assert row["stage1_seconds"] != ""
    assert row["stage2_seconds"] != ""

    counts = {"stage1": 0, "fit": 0, "predict": 0, "x_grid": 0, "y_grid": 0}
    row = runtime._run_python_method(
        method_spec,
        run_ctx,
        stage1_random_state=1,
        stage2_random_state=2,
        x_grid_mode="test_quantile",
        x_grid_points=200,
        y_grid_points=101,
        y_grid_padding=0.25,
        taus=(0.1, 0.9),
        n_v_grid=51,
        integrator_cfg=integrator_cfg,
        max_points_per_batch=None,
        timer_scope="predict_only",
    )
    assert row["status"] == "ok"
    assert counts == {"stage1": 1, "fit": 1, "predict": 1, "x_grid": 1, "y_grid": 1}
    assert row["stage1_seconds"] == ""
    assert row["stage2_seconds"] != ""
    assert row["seconds"] == row["stage2_seconds"]
