from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]

for candidate in (
    REPO_ROOT / "tabcf_core",
    REPO_ROOT / "interv_mean" / "pipeline",
    REPO_ROOT,
):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import aggregate_results as agg
import plot_div_style as vis


def _write_summary(path: Path, metric: float) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["key", "value"])
        writer.writeheader()
        writer.writerow({"key": "backend", "value": path.stem.split("_")[-1]})
        writer.writerow({"key": "metric_mse_do_pred_vs_clean", "value": f"{metric:.6f}"})


def test_core_backend_records_and_merge(tmp_path):
    stage2_dir = tmp_path / "stage2_output"
    stage2_dir.mkdir()

    codes = ["A3_B3"]
    train_sizes = [1000, 4000]
    seeds = [1, 2]
    backends = ["tabpfn_real", "tabicl"]

    metric = 0.1
    for backend in backends:
        for train_size in train_sizes:
            for seed in seeds:
                summary_path = stage2_dir / f"s2_A3_B3_n{train_size}_seed{seed}_{backend}_summary.csv"
                _write_summary(summary_path, metric)
                metric += 0.1

    records = agg._core_backend_records(
        stage2_dir=stage2_dir,
        codes=codes,
        train_sizes=train_sizes,
        seeds=seeds,
        backends=backends,
    )
    summary_rows = agg._aggregate(records)

    assert len(summary_rows) == 4
    row_lookup = {(row["model"], row["train_size"]): row for row in summary_rows}
    assert row_lookup[("tabpfn_real", 1000)]["n_runs"] == 2
    assert row_lookup[("tabpfn_real", 1000)]["variant"] == "stage2_9"
    assert row_lookup[("tabicl", 4000)]["n_runs"] == 2

    merged = agg._merge_rows(
        [
            {
                "model": "tabpfn",
                "variant": "stage2_9",
                "code": "A3_B3",
                "scenario": "g_lin_f_lin",
                "train_size": "1000",
                "mean_mse": "0.2",
                "std_mse": "0.01",
                "n_runs": "10",
            }
        ],
        summary_rows,
    )
    assert len(merged) == 5
    assert merged[0]["model"] == "tabpfn"
    assert {"tabpfn_real", "tabicl"}.issubset({row["model"] for row in merged})


def test_build_method_specs_includes_new_backend_labels():
    records = [
        {"model": "tabpfn", "variant": "stage2_9"},
        {"model": "tabpfn_real", "variant": "stage2_9"},
        {"model": "tabicl", "variant": "stage2_9"},
    ]

    specs = vis.build_method_specs(records)

    assert [spec.label for spec in specs] == [
        "TabCF (TabPFNv2.5)",
        "TabCF (Real-TabPFNv2.5)",
        "TabCF (TabICLv2)",
    ]
    assert [spec.model for spec in specs] == ["tabpfn", "tabpfn_real", "tabicl"]


def test_local_core_backend_records_and_vis_order(tmp_path):
    stage2_dir = tmp_path / "stage2_output"
    stage2_dir.mkdir()

    codes = ["A3_B3"]
    train_sizes = [1000]
    seeds = [1, 2]
    backends = ["tabpfn", "tabpfn_real", "tabicl"]

    metric = 0.1
    for backend in backends:
        for seed in seeds:
            summary_path = agg._local_core_backend_summary_path(
                stage2_dir,
                code="A3_B3",
                train_size=1000,
                seed=seed,
                backend=backend,
                local_k_neighbors=128,
            )
            _write_summary(summary_path, metric)
            metric += 0.1

    records = agg._local_core_backend_records(
        stage2_dir=stage2_dir,
        codes=codes,
        train_sizes=train_sizes,
        seeds=seeds,
        backends=backends,
        local_k_neighbors=128,
    )
    summary_rows = agg._aggregate(records)
    row_lookup = {(row["model"], row["train_size"]): row for row in summary_rows}

    assert row_lookup[("tabpfn_local", 1000)]["variant"] == "stage2_9_local_knn"
    assert row_lookup[("tabpfn_real_local", 1000)]["n_runs"] == 2
    assert row_lookup[("tabicl_local", 1000)]["n_runs"] == 2

    specs = vis.build_method_specs(
        [
            {"model": "tabpfn", "variant": "stage2_9"},
            {"model": "tabpfn_local", "variant": "stage2_9_local_knn"},
            {"model": "tabpfn_real", "variant": "stage2_9"},
            {"model": "tabpfn_real_local", "variant": "stage2_9_local_knn"},
            {"model": "tabicl", "variant": "stage2_9"},
            {"model": "tabicl_local", "variant": "stage2_9_local_knn"},
        ]
    )
    assert [spec.model for spec in specs] == [
        "tabpfn",
        "tabpfn_local",
        "tabpfn_real",
        "tabpfn_real_local",
        "tabicl",
        "tabicl_local",
    ]
    assert [spec.label for spec in specs] == [
        "TabCF (TabPFNv2.5)",
        "Local-TabCF (TabPFNv2.5)",
        "TabCF (Real-TabPFNv2.5)",
        "Local-TabCF (Real-TabPFNv2.5)",
        "TabCF (TabICLv2)",
        "Local-TabCF (TabICLv2)",
    ]


def test_aggregate_results_fails_closed_without_local_k(monkeypatch, tmp_path):
    base_csv = tmp_path / "base.csv"
    with base_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "variant", "code", "scenario", "train_size", "mean_mse", "std_mse", "n_runs"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "model": "tabpfn",
                "variant": "stage2_9",
                "code": "A3_B3",
                "scenario": "g_lin_f_lin",
                "train_size": "1000",
                "mean_mse": "0.2",
                "std_mse": "0.01",
                "n_runs": "10",
            }
        )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "aggregate_results.py",
            "--base-csv",
            str(base_csv),
            "--append-local-core-backends",
            "tabpfn",
            "--codes",
            "A3_B3",
            "--append-seeds",
            "1",
        ],
    )
    with pytest.raises(SystemExit, match="--append-local-core-backends requires --local-k-neighbors."):
        agg.main()
