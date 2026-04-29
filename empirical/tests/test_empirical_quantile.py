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

from empirical import run_empirical_quantile as pipeline


def make_fulton_raw_df(n_rows: int = pipeline.scalar_pipeline.EXPECTED_FULTON_ROWS) -> pd.DataFrame:
    rows = []
    for idx in range(1, n_rows + 1):
        rows.append(
            {
                "rownames": idx,
                "lavgprc": -0.6 + 0.012 * idx,
                "ltotqty": 7.4 + 0.015 * idx,
                "wave2": 2.0 + 0.06 * idx,
                "speed2": 10.0 + 0.2 * idx,
            }
        )
    return pd.DataFrame(rows)


def test_load_fulton_quantile_data_maps_schema_and_display_transforms(tmp_path):
    csv_path = tmp_path / "fish.csv"
    make_fulton_raw_df().to_csv(csv_path, index=False)

    df = pipeline.load_fulton_quantile_data(
        csv_path,
        instrument_spec=pipeline.FULTON_WAVE2_SPEC,
        allow_download=False,
    )

    assert list(df.columns) == ["unit", "X", "Z", "Y", "x_orig", "y_orig"]
    assert len(df) == pipeline.scalar_pipeline.EXPECTED_FULTON_ROWS
    assert df.iloc[0]["unit"] == "1"
    assert df.iloc[0]["X"] == pytest.approx(-0.588)
    assert df.iloc[0]["Z"] == pytest.approx(2.06)
    assert df.iloc[0]["Y"] == pytest.approx(7.415)
    assert df.iloc[0]["x_orig"] == pytest.approx(np.exp(df.iloc[0]["X"]))
    assert df.iloc[-1]["y_orig"] == pytest.approx(np.exp(df.iloc[-1]["Y"]))


def test_parse_args_defaults_to_fulton_wave2_and_paper_taus(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_empirical_quantile.py"])

    args = pipeline.parse_args()
    spec, _, _, _ = pipeline.resolve_dataset_args(args)
    instrument = pipeline.resolve_instrument_spec(spec, args.instrument_spec)
    taus = pipeline.parse_taus_arg(args.taus)

    assert args.dataset == "fulton"
    assert instrument.slug == "wave2"
    assert taus == pipeline.DEFAULT_TAUS
    assert pipeline.scalar_pipeline.normalize_core_backends(args.core_backends) == ("tabpfn",)


def test_main_smoke_writes_quantile_outputs_and_metadata(monkeypatch, tmp_path):
    csv_path = tmp_path / "fish.csv"
    out_dir = tmp_path / "quant_out"
    make_fulton_raw_df().to_csv(csv_path, index=False)

    def fake_compute_tabcf_quantile_curves(
        df: pd.DataFrame,
        x_grid: np.ndarray,
        taus: tuple[float, ...],
        *,
        n_v_points: int,
        y_grid_points: int,
        y_grid_padding: float,
        backend_name: str,
    ) -> pipeline.TabCFQuantileResult:
        del df, n_v_points, y_grid_points, y_grid_padding
        offset = 0.0 if backend_name == pipeline.scalar_pipeline.TABPFN_BACKEND else 0.35
        x_arr = np.asarray(x_grid, dtype=float).reshape(-1, 1)
        tau_arr = np.asarray(taus, dtype=float).reshape(1, -1)
        quantiles = 7.0 + 0.4 * x_arr + tau_arr + offset
        return pipeline.TabCFQuantileResult(
            backend_name=backend_name,
            v_hat=np.linspace(0.1, 0.9, pipeline.scalar_pipeline.EXPECTED_FULTON_ROWS),
            quantiles=quantiles,
            y_grid_min=6.0,
            y_grid_max=10.0,
        )

    def fake_run_quantile_r_helper(
        *,
        data_csv: Path,
        grid_csv: Path,
        curves_csv: Path,
        coefficients_csv: Path,
        runtime_csv: Path,
        r_script: Path,
        r_module: str,
        taus: tuple[float, ...],
        ivqr_grid_min: float | None,
        ivqr_grid_max: float | None,
        ivqr_grid_points: int,
        seed: int,
        div_epochs: int,
        div_layers: int,
        div_lr: float,
        div_nsample: int,
        div_seed: int,
    ) -> None:
        del (
            data_csv,
            r_script,
            r_module,
            ivqr_grid_min,
            ivqr_grid_max,
            ivqr_grid_points,
            seed,
            div_epochs,
            div_layers,
            div_lr,
            div_nsample,
            div_seed,
        )
        grid_df = pd.read_csv(grid_csv)
        x_vals = grid_df["x_log"].to_numpy(dtype=float)
        records = []
        for tau in taus:
            for x_val in x_vals:
                records.append(
                    {
                        "method": "QR",
                        "estimand_family": "conditional_quantile",
                        "tau": float(tau),
                        "x_log": float(x_val),
                        "q_pred_log": float(7.5 + 0.20 * x_val + tau),
                    }
                )
                records.append(
                    {
                        "method": "IVQR",
                        "estimand_family": "interventional_quantile",
                        "tau": float(tau),
                        "x_log": float(x_val),
                        "q_pred_log": float(7.8 + 0.35 * x_val + tau),
                    }
                )
                records.append(
                    {
                        "method": "DIV",
                        "estimand_family": "interventional_quantile",
                        "tau": float(tau),
                        "x_log": float(x_val),
                        "q_pred_log": float(7.65 + 0.28 * x_val + tau),
                    }
                )
        pd.DataFrame.from_records(records).to_csv(curves_csv, index=False)

        coef_df = pd.DataFrame(
            [
                {
                    "method": "QR",
                    "estimand_family": "conditional_quantile",
                    "tau": float(tau),
                    "intercept_log": 7.5 + float(tau),
                    "slope_log": 0.20,
                }
                for tau in taus
            ]
            + [
                {
                    "method": "IVQR",
                    "estimand_family": "interventional_quantile",
                    "tau": float(tau),
                    "intercept_log": 7.8 + float(tau),
                    "slope_log": 0.35,
                }
                for tau in taus
            ]
        )
        coef_df.to_csv(coefficients_csv, index=False)

        runtime_df = pd.DataFrame(
            [
                {
                    "method": "QR",
                    "backend_name": "",
                    "estimand_family": "conditional_quantile",
                    "seconds": 0.12,
                },
                {
                    "method": "IVQR",
                    "backend_name": "",
                    "estimand_family": "interventional_quantile",
                    "seconds": 0.45,
                },
                {
                    "method": "DIV",
                    "backend_name": "",
                    "estimand_family": "interventional_quantile",
                    "seconds": 0.85,
                },
            ]
        )
        runtime_df.to_csv(runtime_csv, index=False)

    monkeypatch.setattr(pipeline, "compute_tabcf_quantile_curves", fake_compute_tabcf_quantile_curves)
    monkeypatch.setattr(pipeline, "run_quantile_r_helper", fake_run_quantile_r_helper)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_empirical_quantile.py",
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
        out_dir / "fulton_fish_quantile_clean.csv",
        out_dir / "fulton_fish_quantile_x_grid.csv",
        out_dir / "fulton_fish_quantile_curves.csv",
        out_dir / "fulton_fish_quantile_coefficients.csv",
        out_dir / "fulton_fish_quantile_runtime.csv",
        out_dir / "fulton_fish_quantile_diagnostics.json",
        out_dir / "fulton_fish_quantile_figure.png",
        out_dir / "fulton_fish_quantile_figure.pdf",
    ]
    for path in expected:
        assert path.exists(), f"Missing expected output: {path}"

    curves_df = pd.read_csv(out_dir / "fulton_fish_quantile_curves.csv")
    assert list(curves_df.columns) == [
        "method",
        "backend_name",
        "estimand_family",
        "tau",
        "x_log",
        "x_orig",
        "q_pred_log",
        "q_pred_orig",
    ]
    assert set(curves_df["estimand_family"]) == {"conditional_quantile", "interventional_quantile"}
    conditional_methods = set(curves_df.loc[curves_df["estimand_family"] == "conditional_quantile", "method"])
    interventional_methods = set(curves_df.loc[curves_df["estimand_family"] == "interventional_quantile", "method"])
    assert conditional_methods == {"QR"}
    assert interventional_methods == {"IVQR", "DIV", "TabCF (TabPFNv2.5)"}

    coef_df = pd.read_csv(out_dir / "fulton_fish_quantile_coefficients.csv")
    assert set(coef_df["method"]) == {"QR", "IVQR"}
    assert set(coef_df["estimand_family"]) == {"conditional_quantile", "interventional_quantile"}

    runtime_df = pd.read_csv(out_dir / "fulton_fish_quantile_runtime.csv")
    assert set(runtime_df["method"]) == {"QR", "IVQR", "DIV", "TabCF (TabPFNv2.5)"}

    payload = json.loads((out_dir / "fulton_fish_quantile_diagnostics.json").read_text())
    assert payload["dataset"] == "fulton"
    assert payload["instrument_spec"] == "wave2"
    assert payload["canonical_mapping"]["Z"] == "wave2"
    assert payload["config"]["core_backends"] == ["tabpfn"]
    assert payload["config"]["n_observations"] == pipeline.scalar_pipeline.EXPECTED_FULTON_ROWS
    assert payload["tabcf_backend_summaries"]["tabpfn"]["method_label"] == "TabCF (TabPFNv2.5)"
    assert "tabicl" not in payload["tabcf_backend_summaries"]
    assert "truth" not in payload
    assert "rmse" not in json.dumps(payload).lower()
