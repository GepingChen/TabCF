from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")


REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_DIR = REPO_ROOT / "tabcf_core"

for candidate in (CORE_DIR, REPO_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import foundation_backends as fb
import stage2_outcome as stage2
from local_context_backends import LocalContextConfig, local_context_metadata, resolve_local_k_neighbors


class FakeCriterion:
    def __init__(self):
        self.borders = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32)

    def _center(self, logits: torch.Tensor) -> torch.Tensor:
        return logits[:, :1]

    def cdf(self, logits: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(values - self._center(logits))

    def mean(self, logits: torch.Tensor) -> torch.Tensor:
        return logits[:, 0]

    def variance(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.full((logits.shape[0],), 0.5, dtype=logits.dtype, device=logits.device)

    def icdf(self, logits: torch.Tensor, alpha: float) -> torch.Tensor:
        return logits[:, 0] + float(alpha)


class FakeQuantileDistribution:
    def __init__(self, raw_quantiles: torch.Tensor):
        self.raw_quantiles = raw_quantiles

    def _center(self) -> torch.Tensor:
        return self.raw_quantiles.mean(dim=-1, keepdim=True)

    def cdf(self, values: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(values - self._center())

    def pdf(self, values: torch.Tensor) -> torch.Tensor:
        probs = self.cdf(values)
        return probs * (1.0 - probs)

    def icdf(self, alpha: torch.Tensor) -> torch.Tensor:
        alpha = alpha.reshape(-1)
        idx = torch.clamp(
            torch.round(alpha * (self.raw_quantiles.shape[1] - 1)).long(),
            min=0,
            max=self.raw_quantiles.shape[1] - 1,
        )
        return self.raw_quantiles[:, idx]

    def mean(self) -> torch.Tensor:
        return self.raw_quantiles.mean(dim=-1)

    def variance(self) -> torch.Tensor:
        return self.raw_quantiles.var(dim=-1, unbiased=False)


class FakeTabPFNRegressor:
    def __init__(self, **kwargs):
        self.kwargs = dict(kwargs)
        self.offset = 0.0

    def fit(self, X, y):
        del X
        self.offset = float(np.mean(np.asarray(y, dtype=float)))
        return self

    def predict(self, X, output_type="mean", quantiles=None):
        n_samples = len(np.asarray(X))
        if output_type == "mean":
            return np.full((n_samples,), self.offset, dtype=float)
        if output_type == "quantiles":
            if quantiles is None:
                raise ValueError("quantiles are required")
            return [np.full((n_samples,), self.offset + float(q), dtype=float) for q in quantiles]
        if output_type == "full":
            return {
                "criterion": FakeCriterion(),
                "logits": torch.full((n_samples, 1), self.offset, dtype=torch.float32),
            }
        raise ValueError(f"Unsupported output_type {output_type}")


class FakeTabICLRegressor:
    def __init__(self, **kwargs):
        self.kwargs = dict(kwargs)
        self.offset = 0.0

    def fit(self, X, y):
        del X
        self.offset = float(np.mean(np.asarray(y, dtype=float)))
        return self

    def predict(self, X, output_type="mean", alphas=None):
        n_samples = len(np.asarray(X))
        if output_type == "mean":
            return np.full((n_samples,), self.offset, dtype=float)
        if output_type == "quantiles":
            if alphas is None:
                raise ValueError("alphas are required")
            return np.column_stack([np.full((n_samples,), self.offset + float(alpha), dtype=float) for alpha in alphas])
        if output_type == "raw_quantiles":
            grid = np.linspace(self.offset - 1.0, self.offset + 1.0, 9, dtype=float)
            return np.tile(grid.reshape(1, -1), (n_samples, 1))
        raise ValueError(f"Unsupported output_type {output_type}")


def _install_fake_backends(monkeypatch):
    monkeypatch.setattr(fb, "import_tabpfn_regressor", lambda: FakeTabPFNRegressor)
    monkeypatch.setattr(fb, "import_tabicl_regressor", lambda: FakeTabICLRegressor)
    monkeypatch.setattr(fb, "import_tabicl_quantile_distribution", lambda: FakeQuantileDistribution)


def test_dynamic_local_k_matches_tabpfn_knn_rule():
    cfg = LocalContextConfig(strategy="local_knn")
    assert resolve_local_k_neighbors(cfg, 1000) == 316
    assert resolve_local_k_neighbors(cfg, 25) == 25
    assert resolve_local_k_neighbors(cfg, 40_000) == 1000


def test_local_knn_wrapper_supports_all_backends(monkeypatch):
    _install_fake_backends(monkeypatch)

    train_X = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [10.0, 10.0],
            [10.0, 11.0],
        ],
        dtype=float,
    )
    train_y = np.array([0.0, 1.0, 10.0, 11.0], dtype=float)
    query_X = np.array(
        [
            [10.0, 10.2],
            [0.0, 0.2],
            [0.0, 0.8],
        ],
        dtype=float,
    )
    expected_mean = np.array([10.5, 0.5, 0.5], dtype=float)
    local_cfg = LocalContextConfig(strategy="local_knn", k_neighbors=2, scale_features=False)

    for backend_name in (fb.TABPFN_BACKEND, fb.TABPFN_REAL_BACKEND, fb.TABICL_BACKEND):
        reg = fb.make_regressor_backend(
            backend_name,
            random_state=7,
            local_context=local_cfg,
        )
        reg.fit(train_X, train_y)

        mean_pred = fb.predict_mean(reg, query_X, backend_name=backend_name)
        assert np.allclose(mean_pred, expected_mean)

        quantile_pred = fb.predict_quantiles(reg, query_X, backend_name=backend_name, quantiles=(0.25, 0.75))
        assert quantile_pred.shape == (3, 2)
        assert np.allclose(quantile_pred[:, 0], expected_mean + 0.25)
        assert np.allclose(quantile_pred[:, 1], expected_mean + 0.75)

        full_output = fb.predict_distribution(reg, query_X, backend_name=backend_name, quantiles=(0.25, 0.75))
        assert full_output["local_k_neighbors_resolved"] == 2

        cdf_vals = fb.cdf_from_distribution_output(full_output, 0.0)
        pdf_vals = fb.pdf_from_distribution_output(full_output, 0.0)
        dist = fb.distribution_adapter_from_output(full_output)

        assert cdf_vals.shape == (3,)
        assert pdf_vals.shape == (3,)
        assert dist.mean().shape == (3,)
        assert dist.variance().shape == (3,)
        assert dist.icdf(np.array([0.1, 0.9])).shape == (3, 2)


def test_local_knn_groups_same_x_queries_across_v_values(monkeypatch):
    _install_fake_backends(monkeypatch)

    train_X = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [10.0, 0.0],
            [10.0, 1.0],
        ],
        dtype=float,
    )
    train_y = np.array([0.0, 1.0, 10.0, 11.0], dtype=float)
    query_X = np.array(
        [
            [0.0, 0.1],
            [0.0, 0.9],
            [10.0, 0.2],
        ],
        dtype=float,
    )

    reg = fb.make_regressor_backend(
        fb.TABPFN_BACKEND,
        random_state=7,
        local_context=LocalContextConfig(strategy="local_knn", k_neighbors=2, scale_features=False),
    )
    reg.fit(train_X, train_y)

    groups = reg._query_groups(query_X)
    assert len(groups) == 2
    grouped_rows = sorted(tuple(indices.tolist()) for _, indices in groups)
    assert grouped_rows == [(0, 1), (2,)]


def test_local_knn_wrapper_rebuilds_estimators_per_predict_call(monkeypatch):
    fit_calls: list[int] = []

    class CountingTabPFNRegressor(FakeTabPFNRegressor):
        def fit(self, X, y):
            fit_calls.append(int(len(np.asarray(y, dtype=float))))
            return super().fit(X, y)

    monkeypatch.setattr(fb, "import_tabpfn_regressor", lambda: CountingTabPFNRegressor)

    train_X = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [10.0, 10.0],
            [10.0, 11.0],
        ],
        dtype=float,
    )
    train_y = np.array([0.0, 1.0, 10.0, 11.0], dtype=float)
    query_X = np.array(
        [
            [10.0, 10.2],
            [0.0, 0.2],
            [0.0, 0.8],
        ],
        dtype=float,
    )
    reg = fb.make_regressor_backend(
        fb.TABPFN_BACKEND,
        random_state=7,
        local_context=LocalContextConfig(strategy="local_knn", k_neighbors=2, scale_features=False),
    )
    reg.fit(train_X, train_y)

    first_pred = fb.predict_mean(reg, query_X, backend_name=fb.TABPFN_BACKEND)
    second_pred = fb.predict_mean(reg, query_X, backend_name=fb.TABPFN_BACKEND)

    assert np.allclose(first_pred, second_pred)
    assert fit_calls == [2, 2, 2, 2]
    assert reg._estimator_cache == {}


def test_stage2_mean_only_experiment_supports_local_knn(monkeypatch):
    _install_fake_backends(monkeypatch)

    train_df = pd.DataFrame(
        {
            "X": [0.0, 0.0, 10.0, 10.0],
            "Y": [0.0, 1.0, 10.0, 11.0],
            "V_hat": [0.0, 1.0, 0.0, 1.0],
        }
    )
    test_df = pd.DataFrame({"X": [10.0, 0.0]})
    local_cfg = LocalContextConfig(strategy="local_knn", k_neighbors=2, scale_features=False)

    for backend_name in (fb.TABPFN_BACKEND, fb.TABPFN_REAL_BACKEND, fb.TABICL_BACKEND):
        results = stage2.run_stage2_9_mean_only_experiment(
            stage2.Stage2_9Config(
                backend_name=backend_name,
                random_state=7,
                n_v_integration_points=3,
                local_context=local_cfg,
            ),
            train_df=train_df,
            test_df=test_df,
            verbose=False,
        )

        pred_df = results["predictions"]
        assert list(pred_df["X"]) == [0.0, 10.0]
        assert np.allclose(pred_df["Y_do_pred"].to_numpy(), [0.5, 10.5])
        assert results["metadata"]["local_strategy"] == "local_knn"
        assert results["metadata"]["local_k_neighbors_resolved"] == 2
        assert results["metadata"]["local_retrieval_features"] == "x_only"


def test_save_stage2_results_adds_local_suffix_and_metadata(tmp_path):
    local_cfg = LocalContextConfig(strategy="local_knn", k_neighbors=2, scale_features=False)
    results = {
        "config": asdict(
            stage2.Stage2_9Config(
                backend_name=fb.TABPFN_BACKEND,
                random_state=7,
                n_train_samples=4,
                train_seed=3,
                local_context=local_cfg,
                mu_integrator="gauss_legendre",
                gauss_legendre_order=32,
            )
        ),
        "data_stats": {
            "n_train_samples": 4,
            "n_test_samples_raw": 2,
            "n_test_samples_selected": 2,
            "n_y_grid": 5,
            "n_v_integration_points": 3,
            "mu_integrator": "gauss_legendre",
            "gauss_legendre_order": 32,
            "test_x_trim_quantile_range": "0.05,0.95",
        },
        "predictions": pd.DataFrame(
            {
                "X": [0.0, 1.0],
                "Y_clean_mean": [0.1, 0.2],
                "Y_do_pred": [0.5, 10.5],
                "mse_per_x": [0.16, 106.09],
                "X_quantile": [0.25, 0.75],
            }
        ),
        "metrics": {
            "mse_do_pred_vs_clean": 0.123,
            "iae_mean": 0.4,
            "iae_max": 0.5,
            "iae_per_x": [0.4, 0.5],
            "mse_mean_per_x": [0.16, 106.09],
            "y_clean_mc_samples": 32,
        },
        "kde": None,
        "metadata": {
            "train_csv": "train.csv",
            "test_csv": "test.csv",
            "codes": "A3_B3",
            "n_train_samples": 4,
            **fb.backend_metadata(fb.TABPFN_BACKEND, "auto"),
            **local_context_metadata(local_cfg, n_train_samples=4, resolved_k=2),
        },
    }

    artifacts = stage2.save_stage2_9_results(results, tmp_path, use_timestamp=False)
    assert artifacts["summary"].name == "s2_A3_B3_n4_seed3_lknnx2_summary.csv"
    assert artifacts["predictions"].name == "s2_A3_B3_n4_seed3_lknnx2_predictions.csv"

    summary_df = pd.read_csv(artifacts["summary"])
    summary_rows = dict(zip(summary_df["key"], summary_df["value"]))
    assert summary_rows["local_strategy"] == "local_knn"
    assert summary_rows["local_k_neighbors_requested"] == "2"
    assert summary_rows["local_k_neighbors_resolved"] == "2"
    assert summary_rows["local_retrieval_features"] == "x_only"
    assert summary_rows["local_metric"] == "euclidean"
    assert summary_rows["local_scale_features"] == "False"
    assert summary_rows["mu_integrator"] == "gauss_legendre"
    assert summary_rows["gauss_legendre_order"] == "32"
    assert summary_rows["test_x_trim_quantile_range"] == "0.05,0.95"


def test_full_stage2_experiment_runs_with_local_knn(monkeypatch, tmp_path):
    _install_fake_backends(monkeypatch)

    stage1_dir = tmp_path / "stage1_output"
    dgp_dir = tmp_path / "dgp"
    output_dir = tmp_path / "stage2_output"
    test_dir = dgp_dir / "test"
    stage1_dir.mkdir()
    test_dir.mkdir(parents=True)
    output_dir.mkdir()

    train_df = pd.DataFrame(
        {
            "Z": [0.0, 1.0, 0.0, 1.0],
            "X": [0.0, 0.0, 10.0, 10.0],
            "Y": [0.0, 1.0, 10.0, 11.0],
            "V_true": [0.1, 0.9, 0.1, 0.9],
            "eps": [0.0, 0.0, 0.0, 0.0],
            "eta": [0.0, 0.0, 0.0, 0.0],
            "V_hat": [0.0, 1.0, 0.0, 1.0],
        }
    )
    test_df = pd.DataFrame(
        {
            "Z": [0.0, 1.0],
            "X": [0.0, 10.0],
            "Y": [0.2, 10.2],
            "V_true": [0.2, 0.8],
            "eps": [0.0, 0.0],
            "eta": [0.0, 0.0],
        }
    )

    train_df.to_csv(stage1_dir / "iv_stage1_train_A3_B3_n4_seed1.csv", index=False)
    test_df.to_csv(test_dir / "test_data_A3_B3.csv", index=False)

    cfg = stage2.Stage2_9Config(
        input_dir=str(stage1_dir),
        output_dir=str(output_dir),
        dgp_base_dir=str(dgp_dir),
        n_train_samples=4,
        train_seed=1,
        random_state=7,
        backend_name=fb.TABPFN_BACKEND,
        local_context=LocalContextConfig(strategy="local_knn", k_neighbors=2, scale_features=False),
        first_stage_code="A3",
        second_stage_code="B3",
        n_y_grid=5,
        n_v_integration_points=3,
        kde_sample_size=8,
        y_clean_mc_samples=16,
    )

    results = stage2.run_stage2_9_experiment(cfg)
    assert not results["predictions"].empty
    assert results["data_stats"]["n_test_samples_raw"] == 2
    assert results["data_stats"]["n_test_samples_selected"] == 2
    assert results["data_stats"]["test_x_trim_quantile_range"] == "0.05,0.95"
    assert results["metadata"]["local_strategy"] == "local_knn"
    assert results["metadata"]["local_k_neighbors_resolved"] == 2
    assert results["metadata"]["local_retrieval_features"] == "x_only"

    artifacts = stage2.save_stage2_9_results(results, output_dir, use_timestamp=False)
    assert artifacts["summary"].name.endswith("_lknnx2_summary.csv")
    summary_df = pd.read_csv(artifacts["summary"])
    summary_rows = dict(zip(summary_df["key"], summary_df["value"]))
    assert summary_rows["test_x_trim_quantile_range"] == "0.05,0.95"
