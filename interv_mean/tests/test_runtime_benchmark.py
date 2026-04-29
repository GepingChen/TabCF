from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_DIR = REPO_ROOT / "tabcf_core"
PIPELINE_DIR = REPO_ROOT / "interv_mean" / "pipeline"

for candidate in (CORE_DIR, PIPELINE_DIR, REPO_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import benchmark_runtime as bench
import compare_integrators as compare_stage2
from local_context_backends import LocalContextConfig


def _require_repo_script(path: Path) -> Path:
    if not path.exists():
        pytest.skip(f"Optional scheduler/script fixture is not present in this checkout: {path}")
    return path


def test_default_method_specs_match_b11_delivery_order():
    specs = bench.build_method_specs(bench.DEFAULT_METHODS)

    assert [spec.method for spec in specs] == [
        "tabpfn",
        "tabpfn_real",
        "tabicl",
        "linear_iv",
        "nonlinear_iv",
        "div_2",
        "deepgmm",
        "deepiv",
        "tabpfn_naive",
    ]
    assert {spec.method for spec in specs}.isdisjoint({"tabpfn_cf", "hsic", "div"})


def test_execute_method_benchmark_keeps_one_row_per_formal_run_with_warmup():
    method_spec = bench.RuntimeMethodSpec(
        method="tabpfn",
        variant="stage2_9",
        method_label="TabCF (TabPFNv2.5)",
        family="core",
    )
    contexts = [
        bench.RunContext(
            run={"code": "A3_B3", "train_size": 1000, "seed": seed},
            raw_train_path=Path(f"train_{seed}.csv"),
            raw_test_path=Path(f"test_{seed}.csv"),
            bridge_train_path=Path(f"bridge_train_{seed}.csv"),
            bridge_test_path=Path(f"bridge_test_{seed}.csv"),
        )
        for seed in (1, 2, 3)
    ]

    calls: list[int] = []

    def fake_runner(spec: bench.RuntimeMethodSpec, ctx: bench.RunContext) -> dict:
        del spec
        calls.append(int(ctx.run["seed"]))
        return {
            "code": ctx.run["code"],
            "train_size": ctx.run["train_size"],
            "seed": ctx.run["seed"],
            "method": method_spec.method,
            "method_label": method_spec.method_label,
            "family": method_spec.family,
            "seconds": 1.0,
            "stage1_seconds": "",
            "stage2_seconds": "",
            "status": "ok",
            "error": "",
        }

    rows = bench.execute_method_benchmark(method_spec, contexts, fake_runner, warmup=True, repeats=2)
    assert len(rows) == 6
    assert len(calls) == 7
    assert calls[0] == 1
    assert [row["seed"] for row in rows] == [1, 2, 3, 1, 2, 3]
    assert [row["repeat_index"] for row in rows] == [1, 1, 1, 2, 2, 2]


def test_summarise_runtime_rows_ignores_error_rows():
    rows = [
        {
            "code": "A3_B3",
            "train_size": 1000,
            "seed": 1,
            "method": "tabpfn",
            "method_label": "TabCF (TabPFNv2.5)",
            "family": "core",
            "seconds": 3.0,
            "stage1_seconds": 1.0,
            "stage2_seconds": 2.0,
            "status": "ok",
            "error": "",
        },
        {
            "code": "A3_B3",
            "train_size": 1000,
            "seed": 2,
            "method": "tabpfn",
            "method_label": "TabCF (TabPFNv2.5)",
            "family": "core",
            "seconds": 5.0,
            "stage1_seconds": 2.0,
            "stage2_seconds": 3.0,
            "status": "ok",
            "error": "",
        },
        {
            "code": "A3_B3",
            "train_size": 1000,
            "seed": 3,
            "method": "tabpfn",
            "method_label": "TabCF (TabPFNv2.5)",
            "family": "core",
            "seconds": 99.0,
            "stage1_seconds": "",
            "stage2_seconds": "",
            "status": "error",
            "error": "boom",
        },
    ]

    summary = bench.summarise_runtime_rows(rows)
    assert len(summary) == 1
    assert summary[0]["mean_seconds"] == 4.0
    assert summary[0]["n_runs"] == 2


def test_default_repeats_is_ten():
    assert bench.DEFAULT_REPEATS == 10


class _FakeMeanModel:
    def predict(self, x_values, v_values):
        x_arr = pd.Series(x_values, dtype=float).to_numpy()
        v_arr = pd.Series(v_values, dtype=float).to_numpy()
        return x_arr + 2.0 * v_arr


def test_mu_integrators_agree_on_linear_function():
    model = _FakeMeanModel()
    x_grid = [0.0, 1.0, 2.0]
    simpson = bench.compute_mu_c_with_integrator(
        model,
        x_grid,
        integrator_cfg=bench.MuIntegratorConfig(method="simpson", n_v_points=9),
    )
    gauss = bench.compute_mu_c_with_integrator(
        model,
        x_grid,
        integrator_cfg=bench.MuIntegratorConfig(method="gauss_legendre", gauss_legendre_order=4),
    )
    assert simpson == pytest.approx([1.0, 2.0, 3.0], abs=1e-5)
    assert gauss == pytest.approx([1.0, 2.0, 3.0], abs=1e-5)


def test_local_method_specs_and_core_resolution():
    specs = bench.build_method_specs(["tabpfn_local", "tabpfn_real_local", "tabicl_local"])
    assert [spec.method for spec in specs] == ["tabpfn_local", "tabpfn_real_local", "tabicl_local"]
    assert [spec.variant for spec in specs] == ["stage2_9_local_knn"] * 3
    assert [spec.family for spec in specs] == ["core", "core", "core"]
    assert [spec.method_label for spec in specs] == [
        "Local-TabCF (TabPFNv2.5)",
        "Local-TabCF (Real-TabPFNv2.5)",
        "Local-TabCF (TabICLv2)",
    ]

    tabpfn_local = bench.resolve_core_method_config("tabpfn_local", local_k_neighbors=128)
    assert tabpfn_local["backend_name"] == "tabpfn"
    assert tabpfn_local["use_tabpfn"] is True
    assert tabpfn_local["local_context"] == LocalContextConfig(strategy="local_knn", k_neighbors=128)

    tabicl_local = bench.resolve_core_method_config("tabicl_local")
    assert tabicl_local["backend_name"] == "tabicl"
    assert tabicl_local["use_tabpfn"] is False
    assert tabicl_local["local_context"] == LocalContextConfig(strategy="local_knn", k_neighbors=None)


def test_summary_rows_include_mu_integrator():
    rows = [
        {
            "code": "A3_B3",
            "train_size": 1000,
            "seed": 1,
            "method": "tabpfn",
            "method_label": "TabCF (TabPFNv2.5)",
            "family": "core",
            "mu_integrator": "gauss_legendre",
            "seconds": 2.0,
            "stage1_seconds": 0.1,
            "stage2_seconds": 1.9,
            "status": "ok",
            "error": "",
        }
    ]
    summary = bench.summarise_runtime_rows(rows)
    assert summary[0]["mu_integrator"] == "gauss_legendre"


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch unavailable in test environment")
def test_run_stage2_mean_only_avoids_full_diagnostics(monkeypatch):
    import stage2_outcome as stage2

    monkeypatch.setattr(stage2, "compute_y_clean", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not be called")))

    def fake_fit_full(self, X, V, Y, *, verbose=True):
        del self, X, V, Y, verbose
        return [0.0, 0.0]

    def fake_mu(model, x_grid, n_v_points, max_points_per_batch=None, *, integrator_cfg=None, verbose=True):
        del model, n_v_points, max_points_per_batch, integrator_cfg, verbose
        return [float(x) + 0.5 for x in x_grid]

    monkeypatch.setattr(stage2.FullDataStructuralFunctionModel, "fit_full", fake_fit_full)
    monkeypatch.setattr(stage2, "compute_mu_c_on_grid", fake_mu)

    train_df = pd.DataFrame({"X": [0.0, 1.0], "Y": [1.0, 2.0], "V_hat": [0.2, 0.8]})
    test_df = pd.DataFrame({"X": [2.0, -1.0]})
    result = stage2.run_stage2_9_mean_only_experiment(
        stage2.Stage2_9Config(backend_name="tabpfn", random_state=7),
        train_df=train_df,
        test_df=test_df,
        verbose=False,
    )

    preds = result["predictions"]
    assert list(preds["X"]) == [-1.0, 2.0]
    assert list(preds["Y_do_pred"]) == [-0.5, 2.5]


def test_build_r_runtime_command_uses_single_benchmark_model_flag():
    cmd = bench.build_r_runtime_command(
        rscript_bin="Rscript",
        script_path=Path("/tmp/run_r_baselines.r"),
        method="linear_iv",
        bridge_train_path=Path("/tmp/train.csv"),
        bridge_test_path=Path("/tmp/test.csv"),
        seed=11,
    )
    assert "--benchmark-model" in cmd
    assert "--models" not in cmd
    assert cmd[cmd.index("--benchmark-model") + 1] == "linear_iv"


def test_core_slurm_script_includes_local_override_passthrough():
    slurm_script = _require_repo_script(
        REPO_ROOT / "interv_mean" / "slurm" / "core" / "run_pipeline.slurm"
    ).read_text()
    assert "STAGE2_LOCAL_STRATEGY_OVERRIDE" in slurm_script
    assert "STAGE2_LOCAL_K_NEIGHBORS_OVERRIDE" in slurm_script
    assert "STAGE2_MU_INTEGRATOR_OVERRIDE" in slurm_script
    assert "STAGE2_GAUSS_LEGENDRE_ORDER_OVERRIDE" in slurm_script
    assert "STAGE2_OUTPUT_DIR_OVERRIDE" in slurm_script
    assert "--stage2-local-strategy" in slurm_script
    assert "--stage2-local-k-neighbors" in slurm_script
    assert "--stage2-mu-integrator" in slurm_script
    assert "--stage2-gauss-legendre-order" in slurm_script
    assert "--stage2-output-dir" in slurm_script
    assert "MAX_WORKERS_OVERRIDE" in slurm_script
    assert "defaulting to single-worker execution" in slurm_script


def test_submit_run_pipeline_supports_integrator_overrides():
    submit_script = _require_repo_script(
        REPO_ROOT / "interv_mean" / "slurm" / "core" / "submit_run_pipeline.sh"
    ).read_text()
    assert "--stage2-mu-integrator" in submit_script
    assert "--stage2-gauss-legendre-order" in submit_script
    assert "--stage2-output-dir" in submit_script
    assert "STAGE2_MU_INTEGRATOR_OVERRIDE" in submit_script
    assert "STAGE2_GAUSS_LEGENDRE_ORDER_OVERRIDE" in submit_script
    assert "STAGE2_OUTPUT_DIR_OVERRIDE" in submit_script


def test_compare_integrators_writes_expected_deltas(tmp_path):
    reference_dir = tmp_path / "reference"
    candidate_dir = tmp_path / "candidate"
    reference_dir.mkdir()
    candidate_dir.mkdir()

    def write_summary(path: Path, *, mse: float, iae_mean: float, iae_max: float, integrator: str | None, order: str | None) -> None:
        rows = [
            {"key": "metric_mse_do_pred_vs_clean", "value": f"{mse:.6f}"},
            {"key": "metric_iae_mean", "value": f"{iae_mean:.6f}"},
            {"key": "metric_iae_max", "value": f"{iae_max:.6f}"},
        ]
        if integrator is not None:
            rows.append({"key": "mu_integrator", "value": integrator})
        if order is not None:
            rows.append({"key": "gauss_legendre_order", "value": order})
        with path.open("w", newline="") as handle:
            writer = pd.DataFrame(rows)
            writer.to_csv(handle, index=False)

    for seed, ref_mse, cand_mse in ((1, 0.5, 0.4), (2, 0.7, 0.6)):
        ref_path = reference_dir / f"s2_A3_B3_n1000_seed{seed}_tabicl_summary.csv"
        cand_path = candidate_dir / f"s2_A3_B3_n1000_seed{seed}_tabicl_summary.csv"
        write_summary(ref_path, mse=ref_mse, iae_mean=0.2, iae_max=0.3, integrator=None, order=None)
        write_summary(cand_path, mse=cand_mse, iae_mean=0.15, iae_max=0.25, integrator="gauss_legendre", order="32")

    per_run_rows = compare_stage2.build_comparison_rows(
        reference_stage2_dir=reference_dir,
        candidate_stage2_dir=candidate_dir,
        codes=["A3_B3"],
        train_sizes=[1000],
        seeds=[1, 2],
        backend="tabicl",
    )
    assert len(per_run_rows) == 2
    assert per_run_rows[0]["reference_mu_integrator"] == "simpson"
    assert per_run_rows[0]["candidate_mu_integrator"] == "gauss_legendre"
    assert per_run_rows[0]["candidate_gauss_legendre_order"] == "32"
    assert per_run_rows[0]["delta_metric_mse_do_pred_vs_clean"] == pytest.approx(-0.1)

    aggregated_rows = compare_stage2.aggregate_comparison_rows(per_run_rows)
    assert len(aggregated_rows) == 1
    assert aggregated_rows[0]["mean_reference_metric_mse_do_pred_vs_clean"] == pytest.approx(0.6)
    assert aggregated_rows[0]["mean_candidate_metric_mse_do_pred_vs_clean"] == pytest.approx(0.5)
    assert aggregated_rows[0]["mean_delta_metric_mse_do_pred_vs_clean"] == pytest.approx(-0.1)

    output_dir = tmp_path / "comparison"
    completed = subprocess.run(
        [
            sys.executable,
            str(PIPELINE_DIR / "compare_integrators.py"),
            "--reference-stage2-dir",
            str(reference_dir),
            "--candidate-stage2-dir",
            str(candidate_dir),
            "--codes",
            "A3_B3",
            "--train-sizes",
            "1000",
            "--seeds",
            "1",
            "2",
            "--backend",
            "tabicl",
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "smoke_compare",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert (output_dir / "smoke_compare_per_run.csv").exists()
    assert (output_dir / "smoke_compare_aggregated.csv").exists()


def test_submit_pipeline_dry_run_stays_default_without_local():
    script_path = _require_repo_script(
        REPO_ROOT / "interv_mean" / "interv_mean" / "slurm" / "submit" / "submit_interv_mean_pipeline.sh"
    )
    completed = subprocess.run(
        ["bash", str(script_path), "--dry-run"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "stage_local_" not in completed.stdout
    assert "aggregate_local" not in completed.stdout
    assert "with_local_lknn" not in completed.stdout


def test_submit_pipeline_dry_run_adds_local_jobs_and_outputs():
    script_path = _require_repo_script(
        REPO_ROOT / "interv_mean" / "interv_mean" / "slurm" / "submit" / "submit_interv_mean_pipeline.sh"
    )
    completed = subprocess.run(
        [
            "bash",
            str(script_path),
            "--dry-run",
            "--local-core-backends",
            "tabpfn_real tabicl",
            "--local-k-neighbors",
            "128",
            "--with-div-style-vis",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    stdout = completed.stdout
    assert "stage_local_tabpfn_real" in stdout
    assert "stage_local_tabicl" in stdout
    assert "STAGE2_LOCAL_STRATEGY_OVERRIDE=local_knn" in stdout
    assert "STAGE2_LOCAL_K_NEIGHBORS_OVERRIDE=128" in stdout
    assert "aggregate_local" in stdout
    assert "aggregated_interv_mean_with_local_lknn128.csv" in stdout
    assert "visualize_div_style" in stdout
    assert "interv_mean_div_style_b11_with_local_lknn128.png" in stdout
