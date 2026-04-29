from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from multivar.pipeline import evaluate_wasserstein as wvo


def _write_samples(
    per_run_dir: Path,
    *,
    marginal_source: str,
    models: dict[str, list[tuple[float, float]]],
    dgp_code: str = "DGP1_LINEAR",
    n_train: int = 20,
    seed: int = 1,
    rho_eps: float = 0.3,
) -> None:
    rows = []
    for model, points in models.items():
        weight = 1.0 / float(len(points))
        for sample_id, (y1, y2) in enumerate(points):
            rows.append(
                {
                    "dgp_code": dgp_code,
                    "n_train": n_train,
                    "seed": seed,
                    "rho_eps": rho_eps,
                    "marginal_source": marginal_source,
                    "x": 0.0,
                    "model": model,
                    "y1": y1,
                    "y2": y2,
                    "sample_weight": weight,
                }
            )
    rho_tag = str(rho_eps).replace(".", "p")
    path = per_run_dir / (
        f"multivar_{dgp_code}_ms{marginal_source}_n{n_train}_rho{rho_tag}_seed{seed}_distribution_samples.csv"
    )
    pd.DataFrame(rows).to_csv(path, index=False)


def _build_case_dir(tmp_path: Path, *, include_div: bool, include_extra_core: bool = False) -> Path:
    per_run_dir = tmp_path / "results" / "per_run"
    per_run_dir.mkdir(parents=True)
    _write_samples(
        per_run_dir,
        marginal_source="deepcf",
        models={
            "estimated": [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)],
            "independence": [(0.0, 1.0), (0.5, 0.5), (1.0, 0.0)],
        },
    )
    _write_samples(
        per_run_dir,
        marginal_source="tabpfn-naive",
        models={"estimated": [(0.1, 0.0), (0.6, 0.6), (1.2, 1.0)]},
    )
    if include_extra_core:
        _write_samples(
            per_run_dir,
            marginal_source="tabpfn_real",
            models={"estimated": [(0.05, 0.0), (0.55, 0.55), (1.05, 0.95)]},
        )
        _write_samples(
            per_run_dir,
            marginal_source="tabicl",
            models={"estimated": [(0.02, 0.0), (0.52, 0.5), (1.02, 0.9)]},
        )
    _write_samples(
        per_run_dir,
        marginal_source="oracle",
        models={"oracle": [(0.0, 0.0), (0.4, 0.4), (0.8, 0.8)]},
    )
    if include_div:
        _write_samples(
            per_run_dir,
            marginal_source="div",
            models={"estimated": [(0.0, 0.1), (0.45, 0.45), (0.9, 0.9)]},
        )
    return per_run_dir


def _build_filtered_case_dir(tmp_path: Path) -> Path:
    per_run_dir = tmp_path / "results" / "per_run"
    per_run_dir.mkdir(parents=True)

    cases = [
        ("DGP1_LINEAR", 1000, 1),
        ("DGP3_PRE_ADDITIVE", 2000, 2),
        ("DGP4_PIECEWISE", 2000, 3),
        ("DGP5_SOFTPLUS", 4000, 4),
    ]
    source_models = {
        "deepcf": {
            "estimated": [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)],
            "independence": [(0.0, 1.0), (0.5, 0.5), (1.0, 0.0)],
        },
        "tabpfn_real": {"estimated": [(0.05, 0.0), (0.55, 0.55), (1.05, 0.95)]},
        "tabicl": {"estimated": [(0.02, 0.0), (0.52, 0.5), (1.02, 0.9)]},
        "tabpfn-naive": {"estimated": [(0.1, 0.0), (0.6, 0.6), (1.2, 1.0)]},
        "oracle": {"oracle": [(0.0, 0.0), (0.4, 0.4), (0.8, 0.8)]},
    }

    for dgp_code, n_train, seed in cases:
        for marginal_source, models in source_models.items():
            _write_samples(
                per_run_dir,
                marginal_source=marginal_source,
                models=models,
                dgp_code=dgp_code,
                n_train=n_train,
                seed=seed,
            )

    return per_run_dir


def test_compute_seed_x_metrics_keeps_div_columns_optional(tmp_path):
    per_run_dir = _build_case_dir(tmp_path, include_div=False)
    case_files, skipped = wvo.discover_case_files(per_run_dir)
    assert skipped == 0

    detail_df = wvo.compute_seed_x_metrics(
        case_files,
        require_div=False,
        extra_core_sources=(),
        method="sliced",
        directions=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=float),
        max_samples_per_dist=None,
        random_seed=123,
    )

    assert not detail_df.empty
    assert "div_vs_oracle_wdist" in detail_df.columns
    assert detail_df["div_vs_oracle_wdist"].isna().all()
    assert "tabpfn_real_vs_oracle_wdist" not in detail_df.columns
    case_df = wvo.aggregate_case(detail_df)
    assert "div_vs_oracle_mean" in case_df.columns
    assert case_df["div_vs_oracle_mean"].isna().all()
    assert "tabicl_vs_oracle_mean" not in case_df.columns


def test_compute_seed_x_metrics_adds_extra_core_columns(tmp_path):
    per_run_dir = _build_case_dir(tmp_path, include_div=False, include_extra_core=True)
    case_files, skipped = wvo.discover_case_files(per_run_dir)
    assert skipped == 0

    detail_df = wvo.compute_seed_x_metrics(
        case_files,
        require_div=False,
        extra_core_sources=("tabpfn_real", "tabicl"),
        method="sliced",
        directions=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=float),
        max_samples_per_dist=None,
        random_seed=123,
    )

    assert "tabpfn_real_vs_oracle_wdist" in detail_df.columns
    assert "tabicl_vs_oracle_wdist" in detail_df.columns
    assert detail_df["tabpfn_real_vs_oracle_wdist"].notna().all()
    assert detail_df["tabicl_vs_oracle_wdist"].notna().all()
    case_df = wvo.aggregate_case(detail_df)
    assert "tabpfn_real_vs_oracle_mean" in case_df.columns
    assert "tabicl_vs_oracle_mean" in case_df.columns


def test_require_div_flag_fails_when_div_missing(tmp_path, monkeypatch):
    per_run_dir = _build_case_dir(tmp_path, include_div=False)
    aggregated_dir = tmp_path / "aggregated"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_wasserstein.py",
            "--results-dir",
            str(per_run_dir.parent),
            "--aggregated-dir",
            str(aggregated_dir),
            "--require-div",
            "--method",
            "sliced",
            "--n-projections",
            "2",
        ],
    )
    with pytest.raises(RuntimeError, match="Need deepcf \\+ tabpfn-naive \\+ div \\+ oracle"):
        wvo.main()


def test_wasserstein_main_writes_div_metrics_when_present(tmp_path, monkeypatch):
    per_run_dir = _build_case_dir(tmp_path, include_div=True)
    aggregated_dir = tmp_path / "aggregated"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_wasserstein.py",
            "--results-dir",
            str(per_run_dir.parent),
            "--aggregated-dir",
            str(aggregated_dir),
            "--require-div",
            "--method",
            "sliced",
            "--n-projections",
            "2",
            "--output-prefix",
            "unit_wass",
        ],
    )
    wvo.main()

    case_summary = pd.read_csv(aggregated_dir / "unit_wass_case_summary.csv")
    assert "div_vs_oracle_mean" in case_summary.columns
    assert case_summary["div_vs_oracle_mean"].notna().all()
    assert (aggregated_dir / "unit_wass_plots" / "wasserstein_case_DGP1_LINEAR_grid.png").exists()


def test_wasserstein_main_writes_extra_core_metrics_when_requested(tmp_path, monkeypatch):
    per_run_dir = _build_case_dir(tmp_path, include_div=False, include_extra_core=True)
    aggregated_dir = tmp_path / "aggregated_extra"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_wasserstein.py",
            "--results-dir",
            str(per_run_dir.parent),
            "--aggregated-dir",
            str(aggregated_dir),
            "--method",
            "sliced",
            "--n-projections",
            "2",
            "--extra-core-sources",
            "tabpfn_real",
            "tabicl",
            "--output-prefix",
            "unit_wass_extra",
        ],
    )
    wvo.main()

    case_summary = pd.read_csv(aggregated_dir / "unit_wass_extra_case_summary.csv")
    assert "tabpfn_real_vs_oracle_mean" in case_summary.columns
    assert "tabicl_vs_oracle_mean" in case_summary.columns
    assert case_summary["tabpfn_real_vs_oracle_mean"].notna().all()
    assert case_summary["tabicl_vs_oracle_mean"].notna().all()


def test_wasserstein_main_filters_by_train_size_and_dgp_codes(tmp_path, monkeypatch):
    per_run_dir = _build_filtered_case_dir(tmp_path)
    aggregated_dir = tmp_path / "filtered_aggregated"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_wasserstein.py",
            "--results-dir",
            str(per_run_dir.parent),
            "--aggregated-dir",
            str(aggregated_dir),
            "--method",
            "sliced",
            "--n-projections",
            "2",
            "--n-train",
            "2000",
            "--dgp-codes",
            "DGP3_PRE_ADDITIVE",
            "DGP4_PIECEWISE",
            "--output-prefix",
            "filtered_case",
        ],
    )
    wvo.main()

    case_summary = pd.read_csv(aggregated_dir / "filtered_case_case_summary.csv")
    assert set(case_summary["dgp_code"]) == {"DGP3_PRE_ADDITIVE", "DGP4_PIECEWISE"}
    assert set(case_summary["n_train"]) == {2000}
    assert (aggregated_dir / "filtered_case_plots" / "wasserstein_case_DGP3_PRE_ADDITIVE_grid.png").exists()
    assert (aggregated_dir / "filtered_case_plots" / "wasserstein_case_DGP4_PIECEWISE_grid.png").exists()
    assert not (aggregated_dir / "filtered_case_plots" / "wasserstein_case_DGP1_LINEAR_grid.png").exists()


def test_wasserstein_sharded_detail_and_merge_flow(tmp_path, monkeypatch):
    per_run_dir = _build_filtered_case_dir(tmp_path)
    aggregated_dir = tmp_path / "aggregated_parallel"

    for shard_index in (0, 1):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "evaluate_wasserstein.py",
                "--results-dir",
                str(per_run_dir.parent),
                "--aggregated-dir",
                str(aggregated_dir),
                "--method",
                "sliced",
                "--n-projections",
                "2",
                "--case-shard-count",
                "2",
                "--case-shard-index",
                str(shard_index),
                "--detail-only",
                "--output-prefix",
                f"unit_parallel_shard{shard_index:02d}of02",
            ],
        )
        wvo.main()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_wasserstein.py",
            "--aggregated-dir",
            str(aggregated_dir),
            "--method",
            "sliced",
            "--n-projections",
            "2",
            "--merge-detail-glob",
            str(aggregated_dir / "unit_parallel_shard*of02_seed_x.csv"),
            "--output-prefix",
            "unit_parallel",
        ],
    )
    wvo.main()

    case_summary = pd.read_csv(aggregated_dir / "unit_parallel_case_summary.csv")
    assert set(case_summary["dgp_code"]) == {
        "DGP1_LINEAR",
        "DGP3_PRE_ADDITIVE",
        "DGP4_PIECEWISE",
        "DGP5_SOFTPLUS",
    }
    assert (aggregated_dir / "unit_parallel_case_x.csv").exists()
    assert (aggregated_dir / "unit_parallel_plots" / "wasserstein_case_DGP1_LINEAR_grid.png").exists()
