from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from empirical import run_empirical_mean as pipeline


def make_fulton_raw_df(n_rows: int = pipeline.EXPECTED_FULTON_ROWS) -> pd.DataFrame:
    rows = []
    for idx in range(1, n_rows + 1):
        rows.append(
            {
                "rownames": idx,
                "lavgprc": -0.5 + 0.01 * idx,
                "ltotqty": 7.0 + 0.02 * idx,
                "wave2": 2.0 + 0.05 * idx,
                "avgprc": 0.7 + 0.01 * idx,
                "totqty": 1000 + 10 * idx,
            }
        )
    return pd.DataFrame(rows)


def make_card_raw_df(n_rows: int = pipeline.EXPECTED_CARD_ROWS) -> pd.DataFrame:
    rows = []
    for idx in range(1, n_rows + 1):
        rows.append(
            {
                "rownames": idx,
                "nearc4": idx % 2,
                "educ": 1 + ((idx - 1) % 18),
                "lwage": 5.0 + 0.03 * (1 + ((idx - 1) % 18)) + 0.02 * (idx % 2),
            }
        )
    return pd.DataFrame(rows)


def make_cigarettes_raw_df() -> pd.DataFrame:
    states_1985 = [f"S{idx:02d}" for idx in range(1, 49)]
    states_1995 = [f"S{idx:02d}" for idx in range(1, 49)]
    rows = []
    for idx, state in enumerate(states_1985, start=1):
        rows.append(
            {
                "rownames": idx,
                "state": state,
                "year": 1985,
                "cpi": 1.05,
                "population": 1000000 + 1000 * idx,
                "packs": 90.0 + idx,
                "income": 30000000 + 10000 * idx,
                "tax": 30.0 + 0.25 * idx,
                "price": 100.0 + 0.75 * idx,
                "taxs": 35.0 + 0.50 * idx,
            }
        )
    for idx, state in enumerate(states_1995, start=49):
        rows.append(
            {
                "rownames": idx,
                "state": state,
                "year": 1995,
                "cpi": 1.50,
                "population": 1100000 + 1000 * idx,
                "packs": 80.0 + idx,
                "income": 35000000 + 10000 * idx,
                "tax": 40.0 + 0.20 * idx,
                "price": 120.0 + 0.60 * idx,
                "taxs": 46.0 + 0.35 * idx,
            }
        )
    return pd.DataFrame(rows)


def make_canonical_df(n_rows: int = 8) -> pd.DataFrame:
    x = np.linspace(-0.5, 0.5, n_rows)
    return pd.DataFrame(
        {
            "unit": [f"u{idx}" for idx in range(n_rows)],
            "X": x,
            "Z": np.linspace(2.0, 6.0, n_rows),
            "Y": 1.5 + 0.8 * x,
        }
    )


def fake_compute_tabcf_curve(
    df: pd.DataFrame,
    x_grid: np.ndarray,
    n_v_points: int,
    *,
    backend_name: str = pipeline.TABPFN_BACKEND,
) -> tuple[np.ndarray, np.ndarray]:
    del n_v_points
    offset = 0.0 if backend_name == pipeline.TABPFN_BACKEND else 0.75
    v_hat = np.linspace(0.1, 0.9, len(df))
    preds = 1.0 + 0.5 * np.asarray(x_grid, dtype=float) + offset
    return v_hat, preds


def fake_run_div_r_helper(
    *,
    data_csv: Path,
    grid_csv: Path,
    pred_csv: Path,
    r_script: Path,
    r_module: str,
    epochs: int,
    layers: int,
    lr: float,
    nsample: int,
    seed: int,
) -> None:
    del data_csv, r_script, r_module, epochs, layers, lr, nsample, seed
    grid_df = pd.read_csv(grid_csv)
    pred_df = pd.DataFrame({"X": grid_df["X"], "div_pred": 0.5 + 0.25 * grid_df["X"]})
    pred_df.to_csv(pred_csv, index=False)


def test_normalize_core_backends_defaults_and_dedupes():
    assert pipeline.normalize_core_backends(None) == (pipeline.TABPFN_BACKEND, pipeline.TABICL_BACKEND)
    assert pipeline.normalize_core_backends(["tabicl", "tabpfn", "tabicl"]) == (
        pipeline.TABICL_BACKEND,
        pipeline.TABPFN_BACKEND,
    )


def test_resolve_core_method_specs_uses_expanded_labels_for_multi_backend():
    specs = pipeline.resolve_core_method_specs(["tabpfn", "tabicl"])

    assert [spec.backend_name for spec in specs] == [pipeline.TABPFN_BACKEND, pipeline.TABICL_BACKEND]
    assert [spec.label for spec in specs] == ["TabCF (TabPFNv2.5)", "TabCF (TabICLv2)"]
    assert [spec.pred_column for spec in specs] == ["tabcf_pred_tabpfn", "tabcf_pred_tabicl"]


def test_resolve_core_method_specs_preserves_legacy_single_label():
    specs = pipeline.resolve_core_method_specs(["tabpfn"], legacy_core_label="TabCF")

    assert len(specs) == 1
    assert specs[0].label == "TabCF"
    assert specs[0].pred_column == "tabcf_pred_tabpfn"


def test_resolve_ajr_data_path_requires_public_or_explicit_path(tmp_path, monkeypatch):
    public_path = tmp_path / "empirical" / "data" / "manual" / "ajr_colonial_origins.dta"
    monkeypatch.setattr(pipeline, "DEFAULT_AJR_DATA_PATH", public_path)

    with pytest.raises(FileNotFoundError, match="ajr_colonial_origins.dta"):
        pipeline.resolve_ajr_data_path(public_path)


def test_load_fulton_data_maps_raw_schema(tmp_path):
    csv_path = tmp_path / "fish.csv"
    make_fulton_raw_df().to_csv(csv_path, index=False)

    df = pipeline.load_fulton_data(csv_path, allow_download=False)

    assert list(df.columns) == ["unit", "X", "Z", "Y"]
    assert len(df) == pipeline.EXPECTED_FULTON_ROWS
    assert df.iloc[0]["unit"] == "1"
    assert df.iloc[-1]["unit"] == str(pipeline.EXPECTED_FULTON_ROWS)
    assert df["X"].iloc[0] == pytest.approx(-0.49)
    assert df["Y"].iloc[0] == pytest.approx(7.02)
    assert df["Z"].iloc[0] == pytest.approx(2.05)


def test_load_card_data_maps_raw_schema(tmp_path):
    csv_path = tmp_path / "card.csv"
    make_card_raw_df().to_csv(csv_path, index=False)

    df = pipeline.load_card_data(csv_path, allow_download=False)

    assert list(df.columns) == ["unit", "X", "Z", "Y"]
    assert len(df) == pipeline.EXPECTED_CARD_ROWS
    assert df.iloc[0]["unit"] == "1"
    assert df.iloc[0]["X"] == pytest.approx(1.0)
    assert df.iloc[1]["Z"] == pytest.approx(0.0)
    assert sorted(df["X"].unique()) == list(range(1, 19))


def test_load_cigarettes_data_maps_raw_schema(tmp_path):
    csv_path = tmp_path / "cigarettes.csv"
    raw_df = make_cigarettes_raw_df()
    raw_df.to_csv(csv_path, index=False)

    df = pipeline.load_cigarettes_data(csv_path, allow_download=False)

    assert list(df.columns) == ["unit", "X", "Z", "Y"]
    assert len(df) == pipeline.EXPECTED_CIGARETTES_ROWS
    assert df.iloc[0]["unit"] == "S01_1985"
    assert df.iloc[-1]["unit"] == "S48_1995"
    expected_x0 = np.log(raw_df.iloc[0]["price"] / raw_df.iloc[0]["cpi"])
    expected_z0 = (raw_df.iloc[0]["taxs"] - raw_df.iloc[0]["tax"]) / raw_df.iloc[0]["cpi"]
    expected_y0 = np.log(raw_df.iloc[0]["packs"])
    assert df.iloc[0]["X"] == pytest.approx(expected_x0)
    assert df.iloc[0]["Z"] == pytest.approx(expected_z0)
    assert df.iloc[0]["Y"] == pytest.approx(expected_y0)


def test_load_fulton_data_missing_required_cols_raises(tmp_path):
    csv_path = tmp_path / "fish_missing.csv"
    df = make_fulton_raw_df().drop(columns=["wave2"])
    df.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="Missing required columns"):
        pipeline.load_fulton_data(csv_path, allow_download=False)


def test_load_cigarettes_data_validation_raises(tmp_path):
    csv_path = tmp_path / "cigarettes_missing.csv"
    wrong_rows_path = tmp_path / "cigarettes_wrong_rows.csv"
    wrong_years_path = tmp_path / "cigarettes_wrong_years.csv"

    make_cigarettes_raw_df().drop(columns=["taxs"]).to_csv(csv_path, index=False)
    make_cigarettes_raw_df().iloc[:-1].to_csv(wrong_rows_path, index=False)
    wrong_years_df = make_cigarettes_raw_df()
    wrong_years_df["year"] = 1985
    wrong_years_df.to_csv(wrong_years_path, index=False)

    with pytest.raises(ValueError, match="Missing required columns"):
        pipeline.load_cigarettes_data(csv_path, allow_download=False)
    with pytest.raises(ValueError, match="Unexpected CigarettesSW row count"):
        pipeline.load_cigarettes_data(wrong_rows_path, allow_download=False)
    with pytest.raises(ValueError, match="Unexpected CigarettesSW years"):
        pipeline.load_cigarettes_data(wrong_years_path, allow_download=False)


def test_fulton_spec_has_expected_outputs_and_metadata(tmp_path):
    spec = pipeline.DATASET_SPECS["fulton"]
    paths = pipeline.build_output_paths(spec, tmp_path)

    assert spec.x_label == "Log average fish price"
    assert spec.y_label == "Log total quantity sold"
    assert spec.source_url == pipeline.FULTON_SOURCE_URL
    assert paths["clean_csv"].name == "fulton_fish_analysis_data.csv"
    assert paths["merged_pred_csv"].name == "fulton_fish_model_predictions.csv"
    assert paths["png_path"].name == "fulton_fish_figure.png"
    assert paths["runtime_json"].name == "fulton_fish_runtime_summary.json"


def test_cigarettes_spec_has_expected_outputs_and_metadata(tmp_path):
    spec = pipeline.DATASET_SPECS["cigarettes"]
    paths = pipeline.build_output_paths(spec, tmp_path)

    assert spec.x_label == "Log real cigarette price"
    assert spec.y_label == "Log cigarette demand"
    assert spec.source_url == pipeline.CIGARETTES_SOURCE_URL
    assert paths["clean_csv"].name == "cigarettes_analysis_data.csv"
    assert paths["merged_pred_csv"].name == "cigarettes_model_predictions.csv"
    assert paths["png_path"].name == "cigarettes_sw_figure.png"
    assert paths["runtime_json"].name == "cigarettes_sw_runtime_summary.json"


def test_card_spec_uses_observed_support_grid(tmp_path):
    spec = pipeline.DATASET_SPECS["card"]
    df = pd.DataFrame(
        {
            "unit": ["u1", "u2", "u3", "u4", "u5"],
            "X": [12.0, 8.0, 12.0, 16.0, 10.0],
            "Z": [0.0, 1.0, 0.0, 1.0, 1.0],
            "Y": [6.1, 5.9, 6.2, 6.8, 6.0],
        }
    )

    x_grid = pipeline.build_x_grid(df, spec, requested_points=100)
    paths = pipeline.build_output_paths(spec, tmp_path)

    assert spec.grid_mode == "unique"
    assert spec.discrete_x is True
    assert x_grid.tolist() == [8.0, 10.0, 12.0, 16.0]
    assert paths["clean_csv"].name == "card_analysis_data.csv"
    assert paths["runtime_json"].name == "card_college_proximity_runtime_summary.json"


def test_main_fulton_smoke_writes_dual_backend_outputs(monkeypatch, tmp_path):
    csv_path = tmp_path / "fish.csv"
    out_dir = tmp_path / "out"
    make_fulton_raw_df().to_csv(csv_path, index=False)

    monkeypatch.setattr(pipeline, "compute_tabcf_curve", fake_compute_tabcf_curve)
    monkeypatch.setattr(pipeline, "run_div_r_helper", fake_run_div_r_helper)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_empirical_mean.py",
            "--dataset",
            "fulton",
            "--data-path",
            str(csv_path),
            "--out-dir",
            str(out_dir),
            "--grid-points",
            "9",
        ],
    )

    pipeline.main()

    expected = [
        out_dir / "fulton_fish_analysis_data.csv",
        out_dir / "fulton_fish_x_grid.csv",
        out_dir / "fulton_fish_model_predictions.csv",
        out_dir / "fulton_fish_figure.png",
        out_dir / "fulton_fish_figure.pdf",
        out_dir / "fulton_fish_method_runtimes.csv",
        out_dir / "fulton_fish_runtime_summary.json",
    ]
    for path in expected:
        assert path.exists(), f"Missing expected output: {path}"

    preds_df = pd.read_csv(out_dir / "fulton_fish_model_predictions.csv")
    assert list(preds_df.columns) == [
        "X",
        "ols_pred",
        "tsls_pred",
        "div_pred",
        "tabcf_pred_tabpfn",
        "tabcf_pred_tabicl",
    ]
    assert not np.allclose(preds_df["tabcf_pred_tabpfn"], preds_df["tabcf_pred_tabicl"])

    payload = json.loads((out_dir / "fulton_fish_runtime_summary.json").read_text())
    assert payload["dataset"] == "fulton"
    assert payload["source_url"] == pipeline.FULTON_SOURCE_URL
    assert payload["canonical_mapping"]["X"] == "lavgprc"
    assert payload["config"]["n_observations"] == pipeline.EXPECTED_FULTON_ROWS
    assert payload["config"]["core_backends"] == ["tabpfn", "tabicl"]
    assert payload["config"]["core_labels"] == {
        "tabpfn": "TabCF (TabPFNv2.5)",
        "tabicl": "TabCF (TabICLv2)",
    }
    runtime_df = pd.read_csv(out_dir / "fulton_fish_method_runtimes.csv")
    assert set(runtime_df["method"]) == {"OLS", "2SLS", "DIV", "TabCF (TabPFNv2.5)", "TabCF (TabICLv2)"}


def test_main_single_backend_preserves_legacy_label(monkeypatch, tmp_path):
    csv_path = tmp_path / "cigarettes.csv"
    out_dir = tmp_path / "cigarettes_out"
    make_cigarettes_raw_df().to_csv(csv_path, index=False)

    monkeypatch.setattr(pipeline, "compute_tabcf_curve", fake_compute_tabcf_curve)
    monkeypatch.setattr(pipeline, "run_div_r_helper", fake_run_div_r_helper)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_empirical_mean.py",
            "--dataset",
            "cigarettes",
            "--data-path",
            str(csv_path),
            "--out-dir",
            str(out_dir),
            "--grid-points",
            "11",
            "--core-backends",
            "tabpfn",
            "--core-label",
            "TabCF",
        ],
    )

    pipeline.main()

    preds_df = pd.read_csv(out_dir / "cigarettes_model_predictions.csv")
    assert "tabcf_pred_tabpfn" in preds_df.columns
    assert "tabcf_pred_tabicl" not in preds_df.columns
    assert "tabcf_pred" not in preds_df.columns

    payload = json.loads((out_dir / "cigarettes_sw_runtime_summary.json").read_text())
    assert payload["dataset"] == "cigarettes"
    assert payload["config"]["core_backends"] == ["tabpfn"]
    assert payload["config"]["core_labels"] == {"tabpfn": "TabCF"}
    runtime_df = pd.read_csv(out_dir / "cigarettes_sw_method_runtimes.csv")
    assert set(runtime_df["method"]) == {"OLS", "2SLS", "DIV", "TabCF"}


def test_main_card_smoke_uses_unique_support_grid_and_dual_backend(monkeypatch, tmp_path):
    out_dir = tmp_path / "card_out"
    card_df = pd.DataFrame(
        {
            "unit": [f"p{idx}" for idx in range(6)],
            "X": [8.0, 12.0, 12.0, 16.0, 10.0, 16.0],
            "Z": [0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
            "Y": [5.8, 6.2, 6.1, 6.8, 6.0, 6.7],
        }
    )

    monkeypatch.setattr(pipeline, "load_card_data", lambda path, allow_download: card_df)
    monkeypatch.setattr(pipeline, "compute_tabcf_curve", fake_compute_tabcf_curve)
    monkeypatch.setattr(pipeline, "run_div_r_helper", fake_run_div_r_helper)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_empirical_mean.py",
            "--dataset",
            "card",
            "--data-path",
            str(tmp_path / "card.csv"),
            "--out-dir",
            str(out_dir),
            "--grid-points",
            "25",
        ],
    )

    pipeline.main()

    grid_df = pd.read_csv(out_dir / "card_college_proximity_x_grid.csv")
    assert grid_df["X"].tolist() == [8.0, 10.0, 12.0, 16.0]

    preds_df = pd.read_csv(out_dir / "card_model_predictions.csv")
    assert "tabcf_pred_tabpfn" in preds_df.columns
    assert "tabcf_pred_tabicl" in preds_df.columns

    payload = json.loads((out_dir / "card_college_proximity_runtime_summary.json").read_text())
    assert payload["dataset"] == "card"
    assert payload["source_url"] == pipeline.CARD_SOURCE_URL
    assert payload["canonical_mapping"]["Z"] == "nearc4"
    assert payload["config"]["grid_points"] == 25
    assert payload["config"]["effective_grid_points"] == 4
    assert payload["config"]["x_grid_mode"] == "unique"


def test_main_ajr_smoke_uses_public_output_names(monkeypatch, tmp_path):
    out_dir = tmp_path / "ajr_out"

    monkeypatch.setattr(pipeline, "load_ajr_data", lambda path: make_canonical_df())
    monkeypatch.setattr(pipeline, "compute_tabcf_curve", fake_compute_tabcf_curve)
    monkeypatch.setattr(pipeline, "run_div_r_helper", fake_run_div_r_helper)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_empirical_mean.py",
            "--dataset",
            "ajr",
            "--data-path",
            str(tmp_path / "dummy.dta"),
            "--out-dir",
            str(out_dir),
            "--grid-points",
            "7",
        ],
    )

    pipeline.main()

    assert (out_dir / "ajr_analysis_data.csv").exists()
    assert (out_dir / "ajr_model_predictions.csv").exists()
    assert (out_dir / "colonial_origins_figure7_style.png").exists()
    assert (out_dir / "colonial_origins_runtime_summary.json").exists()
