from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")


REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_DIR = REPO_ROOT / "tabcf_core"
TABICL_SRC_DIR = REPO_ROOT / "tabicl" / "src"

for candidate in (CORE_DIR, TABICL_SRC_DIR, REPO_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import foundation_backends as fb
import stage1_control as stage1
import stage2_outcome as stage2
import run_pipeline as runner


class FakeCriterion:
    def __init__(self):
        self.borders = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32)

    def cdf(self, logits: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        del logits
        return torch.clamp((values + 1.0) / 2.0, min=0.0, max=1.0)

    def mean(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.full((logits.shape[0],), 0.25, dtype=torch.float32, device=logits.device)

    def variance(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.full((logits.shape[0],), 0.5, dtype=torch.float32, device=logits.device)

    def icdf(self, logits: torch.Tensor, alpha: float) -> torch.Tensor:
        return torch.full((logits.shape[0],), float(alpha), dtype=torch.float32, device=logits.device)


def test_make_regressor_backend_dispatches_by_backend(monkeypatch):
    calls: list[tuple[str, dict]] = []

    class FakeTabPFNRegressor:
        def __init__(self, **kwargs):
            calls.append(("tabpfn", dict(kwargs)))

    class FakeTabICLRegressor:
        def __init__(self, **kwargs):
            calls.append(("tabicl", dict(kwargs)))

    monkeypatch.setattr(fb, "import_tabpfn_regressor", lambda: FakeTabPFNRegressor)
    monkeypatch.setattr(fb, "import_tabicl_regressor", lambda: FakeTabICLRegressor)

    fb.make_regressor_backend("tabpfn", random_state=7, model_path="auto")
    fb.make_regressor_backend("tabpfn_real", random_state=11, model_path="auto")
    fb.make_regressor_backend("tabicl", random_state=13, model_path="auto", n_estimators=5, device="cpu")

    assert calls[0][0] == "tabpfn"
    assert calls[0][1]["random_state"] == 7
    assert "model_path" not in calls[0][1]

    assert calls[1][0] == "tabpfn"
    assert calls[1][1]["random_state"] == 11
    assert Path(calls[1][1]["model_path"]).name == fb.DEFAULT_TABPFN_REAL_REGRESSOR_CHECKPOINT

    assert calls[2][0] == "tabicl"
    assert calls[2][1]["random_state"] == 13
    assert calls[2][1]["checkpoint_version"] == fb.DEFAULT_TABICL_REGRESSOR_CHECKPOINT
    assert calls[2][1]["n_estimators"] == 5
    assert calls[2][1]["device"] == "cpu"
    assert "model_path" not in calls[2][1]


def test_make_regressor_backend_passes_softmax_temperature(monkeypatch):
    calls: list[dict] = []

    class FakeTabPFNRegressor:
        def __init__(self, **kwargs):
            calls.append(dict(kwargs))

    monkeypatch.setattr(fb, "import_tabpfn_regressor", lambda: FakeTabPFNRegressor)

    fb.make_regressor_backend(
        "tabpfn",
        random_state=7,
        model_path="auto",
        softmax_temperature=1.0,
    )

    assert calls[0]["softmax_temperature"] == 1.0


def test_make_regressor_backend_rejects_softmax_temperature_for_tabicl():
    try:
        fb.make_regressor_backend(
            "tabicl",
            random_state=13,
            model_path="auto",
            softmax_temperature=1.0,
        )
    except ValueError as exc:
        assert "softmax_temperature is only supported for TabPFN backends" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("tabicl should reject softmax_temperature")


def test_new_backends_fail_closed_without_fallback(monkeypatch):
    def raise_tabpfn_import():
        raise RuntimeError("tabpfn import failed")

    def raise_tabicl_import():
        raise RuntimeError("tabicl import failed")

    monkeypatch.setattr(fb, "import_tabpfn_regressor", raise_tabpfn_import)
    monkeypatch.setattr(fb, "import_tabicl_regressor", raise_tabicl_import)

    try:
        fb.make_regressor_backend("tabpfn_real", random_state=1, model_path="auto")
    except RuntimeError as exc:
        assert "tabpfn import failed" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("tabpfn_real should fail directly when TabPFN cannot be imported")

    try:
        fb.make_regressor_backend("tabicl", random_state=1, model_path="auto")
    except RuntimeError as exc:
        assert "tabicl import failed" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("tabicl should fail directly when TabICL cannot be imported")


def test_distribution_output_helpers_support_tabpfn_and_tabicl_shapes():
    logits = torch.zeros((3, 4), dtype=torch.float32)
    tabpfn_output = {"criterion": FakeCriterion(), "logits": logits}

    tabpfn_cdf_scalar = fb.cdf_from_distribution_output(tabpfn_output, 0.0)
    tabpfn_cdf_grid = fb.cdf_from_distribution_output(tabpfn_output, np.linspace(-0.5, 0.5, 5), squeeze_last=False)
    tabpfn_pdf_scalar = fb.pdf_from_distribution_output(tabpfn_output, 0.0)
    tabpfn_dist = fb.distribution_adapter_from_output(tabpfn_output)

    assert tabpfn_cdf_scalar.shape == (3,)
    assert tabpfn_cdf_grid.shape == (3, 5)
    assert np.allclose(tabpfn_pdf_scalar, 0.5, atol=1e-3)
    assert tabpfn_dist.mean().shape == (3,)
    assert tabpfn_dist.variance().shape == (3,)
    assert tabpfn_dist.icdf(np.array([0.1, 0.9])).shape == (3, 2)

    raw_quantiles = np.tile(np.linspace(-1.0, 1.0, 999, dtype=np.float32), (3, 1))
    tabicl_output = {"distribution": fb.TabICLDistributionAdapter(raw_quantiles=raw_quantiles)}
    tabicl_cdf_scalar = fb.cdf_from_distribution_output(tabicl_output, 0.0)
    tabicl_cdf_grid = fb.cdf_from_distribution_output(tabicl_output, np.linspace(-0.5, 0.5, 5), squeeze_last=False)
    tabicl_pdf_scalar = fb.pdf_from_distribution_output(tabicl_output, 0.0)
    tabicl_dist = fb.distribution_adapter_from_output(tabicl_output)

    assert tabicl_cdf_scalar.shape == (3,)
    assert tabicl_cdf_grid.shape == (3, 5)
    assert np.all((0.0 <= tabicl_cdf_grid) & (tabicl_cdf_grid <= 1.0))
    assert tabicl_pdf_scalar.shape == (3,)
    assert np.all(tabicl_pdf_scalar >= 0.0)
    assert tabicl_dist.mean().shape == (3,)
    assert tabicl_dist.variance().shape == (3,)
    assert tabicl_dist.icdf(np.array([0.1, 0.9])).shape == (3, 2)


def test_backend_aware_file_naming_and_legacy_defaults():
    stage1_cfg = stage1.Stage1Config(random_state=3)
    assert stage1_cfg.backend_name == fb.TABPFN_BACKEND

    m_model = stage2.FullDataStructuralFunctionModel(use_tabpfn=True)
    cdf_model = stage2.ConditionalCDFEstimator(use_tabpfn=True)
    assert m_model.backend_name == fb.TABPFN_BACKEND
    assert cdf_model.backend_name == fb.TABPFN_BACKEND

    assert runner.expected_stage1_csv("A3", "B3", 1000, 1).name == "iv_stage1_train_A3_B3_n1000_seed1.csv"
    assert runner.expected_stage1_csv(
        "A3",
        "B3",
        1000,
        1,
        backend_name=fb.TABPFN_REAL_BACKEND,
    ).name == "iv_stage1_train_A3_B3_n1000_seed1_tabpfn_real.csv"
    assert runner.expected_stage1_csv(
        "A3",
        "B3",
        1000,
        1,
        backend_name=fb.TABPFN_BACKEND,
        softmax_temperature=1.0,
    ).name == "iv_stage1_train_A3_B3_n1000_seed1_st1.csv"
    assert runner.expected_stage2_summary(
        "A3",
        "B3",
        1000,
        1,
        backend_name=fb.TABICL_BACKEND,
    ).name == "s2_A3_B3_n1000_seed1_tabicl_summary.csv"


def test_save_stage2_9_results_writes_backend_metadata(tmp_path):
    results = {
        "config": {
            "backend_name": fb.TABICL_BACKEND,
            "model_path": "auto",
            "train_seed": 5,
            "mu_integrator": "gauss_legendre",
            "gauss_legendre_order": 32,
        },
        "data_stats": {
            "n_train_samples": 2,
            "n_test_samples_raw": 2,
            "n_test_samples_selected": 2,
            "n_y_grid": 5,
            "n_v_integration_points": 3,
            "mu_integrator": "gauss_legendre",
            "gauss_legendre_order": 32,
        },
        "predictions": pd.DataFrame(
            {
                "X": [0.0, 1.0],
                "Y_clean_mean": [0.1, 0.2],
                "Y_do_pred": [0.11, 0.22],
                "mse_per_x": [0.01, 0.02],
                "X_quantile": [0.25, 0.75],
            }
        ),
        "metrics": {
            "mse_do_pred_vs_clean": 0.123,
            "iae_mean": 0.4,
            "iae_max": 0.5,
            "iae_per_x": [0.4, 0.5],
            "mse_mean_per_x": [0.01, 0.02],
            "y_clean_mc_samples": 100,
        },
        "kde": None,
        "metadata": {
            "train_csv": "train.csv",
            "test_csv": "test.csv",
            "codes": "A3_B3",
            "n_train_samples": 2,
            **fb.backend_metadata(fb.TABICL_BACKEND, "auto"),
        },
    }

    artifacts = stage2.save_stage2_9_results(results, tmp_path, use_timestamp=False)
    assert artifacts["summary"].name == "s2_A3_B3_n2_seed5_tabicl_summary.csv"

    summary_df = pd.read_csv(artifacts["summary"])
    summary_rows = dict(zip(summary_df["key"], summary_df["value"]))
    assert summary_rows["backend"] == fb.TABICL_BACKEND
    assert summary_rows["backend_package"] == "tabicl"
    assert summary_rows["checkpoint_version"] == fb.DEFAULT_TABICL_REGRESSOR_CHECKPOINT
    assert summary_rows["model_path"] == "auto"
    assert summary_rows["mu_integrator"] == "gauss_legendre"
    assert summary_rows["gauss_legendre_order"] == "32"


def test_summarise_results_groups_by_backend():
    df = pd.DataFrame(
        [
            {"backend": "tabpfn", "first_stage": "A3", "second_stage": "B3", "train_sample_size": 1000, "mse_do_pred_vs_clean": 1.0},
            {"backend": "tabpfn", "first_stage": "A3", "second_stage": "B3", "train_sample_size": 1000, "mse_do_pred_vs_clean": 3.0},
            {"backend": "tabicl", "first_stage": "A3", "second_stage": "B3", "train_sample_size": 1000, "mse_do_pred_vs_clean": 2.0},
        ]
    )

    summary = runner.summarise_results(df)
    assert list(summary["backend"]) == ["tabicl", "tabpfn"]
    assert list(summary["mean_mse"]) == [2.0, 2.0]
    assert list(summary["n_runs"]) == [1, 2]


def test_structural_function_predict_uses_explicit_mean_helper(monkeypatch):
    calls: list[tuple[str, tuple[int, int]]] = []

    def fake_predict_mean(model, features, *, backend_name):
        del model
        calls.append((backend_name, tuple(features.shape)))
        return np.arange(features.shape[0], dtype=float)

    monkeypatch.setattr(stage2, "predict_backend_mean", fake_predict_mean)

    model = stage2.FullDataStructuralFunctionModel(backend_name=fb.TABPFN_BACKEND, random_state=1)
    model.model = object()

    preds = model.predict(np.array([1.0, 2.0]), np.array([0.2, 0.8]))
    assert calls == [(fb.TABPFN_BACKEND, (2, 2))]
    assert np.allclose(preds, [0.0, 1.0])
