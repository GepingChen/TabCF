from __future__ import annotations

from pathlib import Path

from multivar.pipeline import monitor_official_s100_run as monitor


def _sample_manifest(tmp_path: Path) -> dict:
    return {
        "run_id": "unit_s100",
        "results_dir": str(tmp_path / "results"),
        "aggregated_dir": str(tmp_path / "agg" / "run"),
        "final_aggregated_dir": str(tmp_path / "agg"),
        "output_prefix": "evaluate_wasserstein_with_tabpfn_real_tabicl_s100",
        "official_plot_output": str(tmp_path / "official.png"),
        "post_slurm_script": str(Path("/repo") / "multivar" / "slurm" / "core" / "run_evaluate_wasserstein.slurm"),
        "upstream_jobs": [
            {
                "job_id": "101",
                "shard_tag": "s01",
                "marginal_source": "deepcf",
                "stdout_log": "/tmp/101.out",
                "stderr_log": "/tmp/101.err",
            },
            {
                "job_id": "102",
                "shard_tag": "s01",
                "marginal_source": "tabicl",
                "stdout_log": "/tmp/102.out",
                "stderr_log": "/tmp/102.err",
            },
        ],
    }


def test_parse_sacct_output_collapses_steps_and_array_children():
    parsed = monitor.parse_sacct_output(
        "\n".join(
            [
                "101|COMPLETED",
                "101.batch|COMPLETED",
                "102_0|RUNNING",
                "102_1|COMPLETED",
                "102.extern|COMPLETED",
            ]
        )
    )
    assert parsed["101"] == "COMPLETED"
    assert parsed["102"] == "RUNNING"


def test_failing_jobs_and_finalize_policy():
    statuses = [
        monitor.JobStatus(job_id="101", state="RUNNING", source="s01:deepcf"),
        monitor.JobStatus(job_id="102", state="COMPLETED", source="s01:tabicl"),
    ]
    assert monitor.failing_jobs(statuses) == []
    assert monitor.should_finalize(statuses) is True

    failed = statuses + [monitor.JobStatus(job_id="103", state="FAILED", source="s01:div")]
    assert [job.job_id for job in monitor.failing_jobs(failed)] == ["103"]
    assert monitor.should_finalize(failed) is False
    assert monitor.all_completed(statuses) is False
    assert monitor.all_completed([monitor.JobStatus(job_id="101", state="COMPLETED", source="s01:deepcf")]) is True


def test_build_final_postjob_command_uses_manifest_outputs(tmp_path):
    manifest = _sample_manifest(tmp_path)
    cmd = monitor.build_final_postjob_command(manifest, use_dependency=True)
    joined = " ".join(cmd)
    assert cmd[:2] == ["sbatch", "--parsable"]
    assert "--dependency=afterok:101:102" in joined
    assert "WASS_RESULTS_DIR=" + manifest["results_dir"] in joined
    assert "WASS_AGGREGATED_DIR=" + manifest["final_aggregated_dir"] in joined
    assert "WASS_OUTPUT_PREFIX=evaluate_wasserstein_with_tabpfn_real_tabicl_s100" in joined
    assert "WASS_RUN_OFFICIAL_PLOT=1" in joined
    assert "WASS_OFFICIAL_PLOT_OUTPUT=" + manifest["official_plot_output"] in joined
    assert "WASS_OFFICIAL_DGP_CODES=DGP1_LINEAR:DGP3_PRE_ADDITIVE:DGP4_PIECEWISE:DGP5_SOFTPLUS" in joined

    cmd_no_dep = monitor.build_final_postjob_command(manifest, use_dependency=False)
    assert all(not part.startswith("--dependency=") for part in cmd_no_dep)
