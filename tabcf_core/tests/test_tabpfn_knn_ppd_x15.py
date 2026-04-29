from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

torch = pytest.importorskip("torch")


REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_DIR = REPO_ROOT / "tabcf_core"

for candidate in (CORE_DIR, REPO_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import foundation_backends as fb
import run_local_tabpfn_ppd as ppd


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
            levels = list(quantiles or [])
            return [np.full((n_samples,), self.offset + float(level), dtype=float) for level in levels]
        if output_type == "full":
            return {
                "criterion": FakeCriterion(),
                "logits": torch.full((n_samples, 1), self.offset, dtype=torch.float32),
            }
        raise ValueError(f"Unsupported output_type {output_type}")


def _install_fake_tabpfn(monkeypatch):
    monkeypatch.setattr(fb, "import_tabpfn_regressor", lambda: FakeTabPFNRegressor)


def _write_stage1_csv(path: Path, train_df: pd.DataFrame) -> None:
    full_df = pd.DataFrame(
        {
            "Z": train_df["Z"],
            "X": train_df["X"],
            "Y": train_df["Y"],
            "V_true": train_df["V_hat"],
            "eps": np.zeros(len(train_df), dtype=float),
            "eta": np.zeros(len(train_df), dtype=float),
            "V_hat": train_df["V_hat"],
        }
    )
    full_df.to_csv(path, index=False)


def _write_test_csv(path: Path, *, y_values: np.ndarray, x_values: np.ndarray | None = None) -> None:
    x_arr = (
        np.full(len(y_values), 1.5, dtype=float)
        if x_values is None
        else np.asarray(x_values, dtype=float)
    )
    test_df = pd.DataFrame(
        {
            "Z": np.zeros(len(y_values), dtype=float),
            "X": x_arr,
            "Y": np.asarray(y_values, dtype=float),
            "V_true": np.linspace(0.1, 0.9, len(y_values)),
            "eps": np.zeros(len(y_values), dtype=float),
            "eta": np.zeros(len(y_values), dtype=float),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    test_df.to_csv(path, index=False)


def test_analytic_a3_b3_helpers_match_normal_formula():
    y_grid = np.array([-2.0, 1.5, 5.0], dtype=float)
    pdf = ppd.analytic_a3_b3_pdf(y_grid, 1.5)

    assert ppd.analytic_a3_b3_mean(1.5) == 1.5
    assert ppd.analytic_a3_b3_variance() == 10.0
    assert np.allclose(pdf, norm.pdf(y_grid, loc=1.5, scale=np.sqrt(10.0)))


def test_resolve_true_reference_uses_analytic_b4_linear_branch():
    config = ppd.SinglePointPPDConfig(
        x_value=0.1,
        first_stage_code="A3",
        second_stage_code="B4",
    )
    y_grid = np.array([-1.0, 0.0, 1.14, 2.0], dtype=float)
    reference = ppd.resolve_true_reference(config, y_grid)

    assert reference["true_method"] == "analytic_linear_branch"
    assert np.isclose(reference["mean_true"], 0.2 * (5.5 + 2.0 * 0.1))
    assert np.isclose(reference["var_true"], 0.4)
    assert np.allclose(
        reference["pdf_true"],
        norm.pdf(y_grid, loc=0.2 * (5.5 + 2.0 * 0.1), scale=np.sqrt(0.4)),
    )


def test_compute_seed_curves_returns_global_and_local_outputs(monkeypatch, tmp_path):
    _install_fake_tabpfn(monkeypatch)

    stage1_csv = tmp_path / "stage1" / "iv_stage1_train_A3_B3_n4_seed1.csv"
    stage1_csv.parent.mkdir(parents=True, exist_ok=True)
    train_df = pd.DataFrame(
        {
            "Z": [0.0, 0.0, 1.0, 1.0],
            "X": [0.0, 0.0, 1.5, 1.5],
            "Y": [0.0, 0.0, 5.0, 5.0],
            "V_hat": [0.0, 1.0, 0.0, 1.0],
        }
    )
    _write_stage1_csv(stage1_csv, train_df)

    config = ppd.SinglePointPPDConfig(
        train_size=4,
        seeds=(1,),
        x_value=1.5,
        local_k_neighbors=2,
        n_y_grid=21,
        n_v_integration_points=3,
        stage1_output_dir=stage1_csv.parent,
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "outputs",
    )
    y_grid = np.linspace(-4.0, 8.0, 21)
    curves_df, summary = ppd.compute_seed_curves(
        stage1_csv=stage1_csv,
        seed=1,
        config=config,
        y_grid=y_grid,
    )

    assert list(curves_df.columns) == ["seed", "y", "pdf_true", "pdf_global", "pdf_local"]
    assert len(curves_df) == len(y_grid)
    assert summary["seed"] == 1
    assert summary["local_k_neighbors_resolved"] == 2
    assert summary["mean_global"] != summary["mean_local"]


def test_build_y_grid_uses_trimmed_test_data_by_default(tmp_path):
    data_dir = tmp_path / "data"
    test_csv = data_dir / "test" / "test_data_A3_B3.csv"
    _write_test_csv(
        test_csv,
        x_values=np.arange(20, dtype=float),
        y_values=np.arange(20, dtype=float),
    )

    config = ppd.SinglePointPPDConfig(
        train_size=4,
        seeds=(1,),
        x_value=1.5,
        data_dir=data_dir,
        output_dir=tmp_path / "outputs",
    )

    y_grid = ppd.build_y_grid(config)

    assert np.isclose(y_grid[0], 1.0)
    assert np.isclose(y_grid[-1], 18.0)


def test_run_sideproject_writes_stable_outputs_and_aggregate(monkeypatch, tmp_path):
    _install_fake_tabpfn(monkeypatch)

    stage1_dir = tmp_path / "stage1"
    data_dir = tmp_path / "data"
    test_csv = data_dir / "test" / "test_data_A3_B3.csv"
    _write_test_csv(test_csv, y_values=np.linspace(-6.0, 10.0, 11))

    stage1_dir.mkdir(parents=True, exist_ok=True)
    train_seed1 = pd.DataFrame(
        {
            "Z": [0.0, 0.0, 1.0, 1.0],
            "X": [0.0, 0.0, 1.5, 1.5],
            "Y": [0.0, 0.0, 5.0, 5.0],
            "V_hat": [0.0, 1.0, 0.0, 1.0],
        }
    )
    train_seed2 = pd.DataFrame(
        {
            "Z": [0.0, 0.0, 1.0, 1.0],
            "X": [0.0, 0.0, 1.5, 1.5],
            "Y": [1.0, 1.0, 6.0, 6.0],
            "V_hat": [0.0, 1.0, 0.0, 1.0],
        }
    )
    _write_stage1_csv(stage1_dir / "iv_stage1_train_A3_B3_n4_seed1.csv", train_seed1)
    _write_stage1_csv(stage1_dir / "iv_stage1_train_A3_B3_n4_seed2.csv", train_seed2)

    config = ppd.SinglePointPPDConfig(
        train_size=4,
        seeds=(1, 2),
        x_value=1.5,
        local_k_neighbors=2,
        n_y_grid=25,
        n_v_integration_points=3,
        data_dir=data_dir,
        stage1_output_dir=stage1_dir,
        output_dir=tmp_path / "outputs",
    )
    output_paths = ppd.run_sideproject(config)

    assert output_paths["curves"].name == "curves_A3_B3_n4_x1p5.csv"
    assert output_paths["summary"].name == "summary_A3_B3_n4_x1p5.csv"
    assert output_paths["plot"].name == "plot_A3_B3_n4_x1p5.png"
    assert output_paths["plot"].exists()

    curves_df = pd.read_csv(output_paths["curves"])
    summary_df = pd.read_csv(output_paths["summary"])

    assert len(curves_df) == 2 * 25
    assert set(curves_df["seed"]) == {1, 2}
    assert "aggregate" in summary_df["seed"].astype(str).tolist()

    aggregate_row = summary_df.loc[summary_df["seed"].astype(str) == "aggregate"].iloc[0]
    assert float(aggregate_row["local_k_neighbors_resolved"]) == 2.0
    assert "var_gap_global" in summary_df.columns
    assert "var_gap_local" in summary_df.columns
    assert "mse_mean_global" in summary_df.columns
    assert "mse_mean_local" in summary_df.columns


def test_merge_worker_outputs_restores_all_seed_rows(monkeypatch, tmp_path):
    _install_fake_tabpfn(monkeypatch)

    stage1_dir = tmp_path / "stage1"
    data_dir = tmp_path / "data"
    test_csv = data_dir / "test" / "test_data_A3_B3.csv"
    _write_test_csv(test_csv, y_values=np.linspace(-6.0, 10.0, 11))

    stage1_dir.mkdir(parents=True, exist_ok=True)
    for seed, y_level in ((1, 5.0), (2, 6.0)):
        train_df = pd.DataFrame(
            {
                "Z": [0.0, 0.0, 1.0, 1.0],
                "X": [0.0, 0.0, 1.5, 1.5],
                "Y": [y_level - 1.0, y_level - 1.0, y_level, y_level],
                "V_hat": [0.0, 1.0, 0.0, 1.0],
            }
        )
        _write_stage1_csv(stage1_dir / f"iv_stage1_train_A3_B3_n4_seed{seed}.csv", train_df)

    worker_config_1 = ppd.SinglePointPPDConfig(
        train_size=4,
        seeds=(1,),
        x_value=1.5,
        local_k_neighbors=2,
        n_y_grid=25,
        n_v_integration_points=3,
        data_dir=data_dir,
        stage1_output_dir=stage1_dir,
        output_dir=tmp_path / "workers" / "worker_0",
    )
    worker_config_2 = ppd.SinglePointPPDConfig(
        train_size=4,
        seeds=(2,),
        x_value=1.5,
        local_k_neighbors=2,
        n_y_grid=25,
        n_v_integration_points=3,
        data_dir=data_dir,
        stage1_output_dir=stage1_dir,
        output_dir=tmp_path / "workers" / "worker_1",
    )
    ppd.run_sideproject(worker_config_1)
    ppd.run_sideproject(worker_config_2)

    final_config = ppd.SinglePointPPDConfig(
        train_size=4,
        seeds=(1, 2),
        x_value=1.5,
        local_k_neighbors=2,
        n_y_grid=25,
        n_v_integration_points=3,
        data_dir=data_dir,
        stage1_output_dir=stage1_dir,
        output_dir=tmp_path / "merged",
    )
    output_paths = ppd.merge_worker_outputs(
        [tmp_path / "workers" / "worker_0", tmp_path / "workers" / "worker_1"],
        final_config,
    )

    curves_df = pd.read_csv(output_paths["curves"])
    summary_df = pd.read_csv(output_paths["summary"])

    assert set(curves_df["seed"]) == {1, 2}
    assert summary_df["seed"].astype(str).tolist().count("aggregate") == 1
    assert {"1", "2", "aggregate"} == set(summary_df["seed"].astype(str))
    aggregate_row = summary_df.loc[summary_df["seed"].astype(str) == "aggregate"].iloc[0]
    per_seed_rows = summary_df.loc[summary_df["seed"].astype(str) != "aggregate"].copy()
    assert np.isclose(
        float(aggregate_row["mse_mean_global"]),
        float(per_seed_rows["mse_mean_global"].mean()),
    )
    assert np.isclose(
        float(aggregate_row["mse_mean_local"]),
        float(per_seed_rows["mse_mean_local"].mean()),
    )
