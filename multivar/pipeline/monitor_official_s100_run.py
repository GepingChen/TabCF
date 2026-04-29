#!/usr/bin/env python3
"""Monitor a sharded multivariate paper run and optionally submit the final post job."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import subprocess
from subprocess import CalledProcessError
import sys
import time
from pathlib import Path
from typing import Iterable


FAILURE_STATES = {"FAILED", "TIMEOUT", "OUT_OF_MEMORY", "CANCELLED", "NODE_FAIL"}
SAFE_PROGRESS_STATES = {"PENDING", "RUNNING", "COMPLETED", "CONFIGURING", "COMPLETING"}
FINAL_AGGREGATED_CASE_X_SUFFIX = "_case_x.csv"


@dataclass(frozen=True)
class JobStatus:
    job_id: str
    state: str
    source: str


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text())


def save_manifest(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest, indent=2) + "\n")


def normalize_slurm_state(raw_state: str) -> str:
    state = (raw_state or "").strip().upper()
    if not state:
        return "UNKNOWN"
    for prefix in ("CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "FAILED", "RUNNING", "PENDING", "COMPLETED", "CONFIGURING", "COMPLETING"):
        if state.startswith(prefix):
            return prefix
    return state


def parse_squeue_output(text: str) -> dict[str, str]:
    live_states: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        job_id, state = (part.strip() for part in line.split("|", 1))
        live_states[job_id] = normalize_slurm_state(state)
    return live_states


def _score_state(state: str) -> int:
    normalized = normalize_slurm_state(state)
    if normalized in FAILURE_STATES:
        return 100
    order = {
        "RUNNING": 40,
        "CONFIGURING": 35,
        "COMPLETING": 30,
        "PENDING": 20,
        "COMPLETED": 10,
        "UNKNOWN": 0,
    }
    return order.get(normalized, 5)


def parse_sacct_output(text: str) -> dict[str, str]:
    grouped: dict[str, list[str]] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        job_id_raw, state = (part.strip() for part in line.split("|", 1))
        job_root = job_id_raw.split(".")[0]
        job_root = job_root.split("_")[0]
        grouped.setdefault(job_root, []).append(normalize_slurm_state(state))

    merged: dict[str, str] = {}
    for job_id, states in grouped.items():
        merged[job_id] = max(states, key=_score_state)
    return merged


def query_job_states(job_ids: Iterable[str]) -> dict[str, str]:
    unique_ids = [str(job_id) for job_id in dict.fromkeys(job_ids)]
    if not unique_ids:
        return {}

    joined = ",".join(unique_ids)
    squeue_result = subprocess.run(
        ["squeue", "-h", "-j", joined, "-o", "%i|%T"],
        check=False,
        capture_output=True,
        text=True,
    )
    sacct_result = subprocess.run(
        ["sacct", "-n", "-P", "-j", joined, "--format", "JobIDRaw,State"],
        check=False,
        capture_output=True,
        text=True,
    )

    live_states = parse_squeue_output(squeue_result.stdout)
    acct_states = parse_sacct_output(sacct_result.stdout)
    merged: dict[str, str] = {}
    for job_id in unique_ids:
        if job_id in live_states:
            merged[job_id] = live_states[job_id]
        elif job_id in acct_states:
            merged[job_id] = acct_states[job_id]
        else:
            merged[job_id] = "UNKNOWN"
    return merged


def summarize_jobs(manifest: dict, state_map: dict[str, str]) -> list[JobStatus]:
    jobs = []
    for job in manifest.get("upstream_jobs", []):
        jobs.append(
            JobStatus(
                job_id=str(job["job_id"]),
                state=normalize_slurm_state(state_map.get(str(job["job_id"]), "UNKNOWN")),
                source=f'{job["shard_tag"]}:{job["marginal_source"]}',
            )
        )
    return jobs


def failing_jobs(statuses: Iterable[JobStatus]) -> list[JobStatus]:
    return [status for status in statuses if status.state in FAILURE_STATES]


def should_finalize(statuses: Iterable[JobStatus]) -> bool:
    status_list = list(statuses)
    if not status_list:
        return False
    return not failing_jobs(status_list) and all(status.state in SAFE_PROGRESS_STATES for status in status_list)


def all_completed(statuses: Iterable[JobStatus]) -> bool:
    status_list = list(statuses)
    return bool(status_list) and all(status.state == "COMPLETED" for status in status_list)


def build_final_postjob_command(manifest: dict, *, use_dependency: bool) -> list[str]:
    output_prefix = manifest["output_prefix"]
    final_case_x = str(Path(manifest["final_aggregated_dir"]) / f"{output_prefix}{FINAL_AGGREGATED_CASE_X_SUFFIX}")
    official_dgp_codes = "DGP1_LINEAR:DGP3_PRE_ADDITIVE:DGP4_PIECEWISE:DGP5_SOFTPLUS"
    export_items = [
        "ALL",
        f"REPO_DIR_OVERRIDE={Path(manifest['post_slurm_script']).resolve().parents[3]}",
        f"WASS_RESULTS_DIR={manifest['results_dir']}",
        f"WASS_AGGREGATED_DIR={manifest['final_aggregated_dir']}",
        f"WASS_OUTPUT_PREFIX={output_prefix}",
        "WASS_METHOD=sliced",
        "WASS_N_PROJECTIONS=128",
        "WASS_RANDOM_SEED=20260304",
        "WASS_REQUIRE_DIV=1",
        "WASS_EXTRA_CORE_SOURCES=tabpfn_real:tabicl",
        "WASS_RUN_OFFICIAL_PLOT=1",
        f"WASS_OFFICIAL_PLOT_INPUT={final_case_x}",
        f"WASS_OFFICIAL_PLOT_OUTPUT={manifest['official_plot_output']}",
        f"WASS_OFFICIAL_DGP_CODES={official_dgp_codes}",
        "WASS_OFFICIAL_N_TRAIN=2000",
        "WASS_OFFICIAL_RHO_EPS=0.6",
        "WASS_OFFICIAL_DPI=320",
    ]
    cmd = [
        "sbatch",
        "--parsable",
        f"--export={','.join(export_items)}",
        manifest["post_slurm_script"],
    ]
    if use_dependency:
        dependency_ids = [str(job["job_id"]) for job in manifest.get("upstream_jobs", [])]
        dependency = "afterok:" + ":".join(dependency_ids)
        cmd.insert(2, f"--dependency={dependency}")
    return cmd


def submit_final_postjob(manifest_path: Path, manifest: dict, *, dry_run: bool, use_dependency: bool) -> str:
    cmd = build_final_postjob_command(manifest, use_dependency=use_dependency)
    print("[final-postjob]", " ".join(cmd))
    if dry_run:
        return "dry-run"
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    job_id = result.stdout.strip()
    manifest["final_job"] = {
        "job_id": job_id,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "submission_mode": "dependency" if use_dependency else "no_dependency",
        "command": cmd,
    }
    save_manifest(manifest_path, manifest)
    print(f"Submitted final post job: {job_id}")
    return job_id


def monitor_run(manifest_path: Path, *, duration_seconds: int, interval_seconds: int, dry_run_final: bool) -> int:
    manifest = load_manifest(manifest_path)
    started = time.time()
    poll_count = 0
    while True:
        poll_count += 1
        state_map = query_job_states(job["job_id"] for job in manifest.get("upstream_jobs", []))
        statuses = summarize_jobs(manifest, state_map)
        counts = Counter(status.state for status in statuses)
        print(f"[poll {poll_count}] counts={dict(sorted(counts.items()))}")
        failures = failing_jobs(statuses)
        if failures:
            for failure in failures:
                job_meta = next(job for job in manifest["upstream_jobs"] if str(job["job_id"]) == failure.job_id)
                print(
                    f"failure job_id={failure.job_id} source={failure.source} "
                    f"stdout={job_meta['stdout_log']} stderr={job_meta['stderr_log']}"
                )
            manifest["monitor"] = {
                "last_checked_at": datetime.now(timezone.utc).isoformat(),
                "status_counts": dict(counts),
                "failed": [failure.__dict__ for failure in failures],
            }
            save_manifest(manifest_path, manifest)
            return 1

        elapsed = time.time() - started
        if elapsed >= duration_seconds:
            break
        time.sleep(interval_seconds)

    manifest["monitor"] = {
        "last_checked_at": datetime.now(timezone.utc).isoformat(),
        "status_counts": dict(Counter(status.state for status in statuses)),
        "failed": [],
    }
    save_manifest(manifest_path, manifest)

    if should_finalize(statuses):
        try:
            submit_final_postjob(manifest_path, manifest, dry_run=dry_run_final, use_dependency=True)
        except CalledProcessError as exc:
            manifest["monitor"]["final_submission_error"] = exc.stderr.strip() if exc.stderr else str(exc)
            save_manifest(manifest_path, manifest)
            raise
        return 0

    print("Run did not fail, but some jobs were not in a finalizable state.")
    return 2


def wait_until_complete_and_submit_final(
    manifest_path: Path,
    *,
    interval_seconds: int,
    dry_run_final: bool,
) -> int:
    manifest = load_manifest(manifest_path)
    while True:
        state_map = query_job_states(job["job_id"] for job in manifest.get("upstream_jobs", []))
        statuses = summarize_jobs(manifest, state_map)
        counts = Counter(status.state for status in statuses)
        print(f"[wait-until-complete] counts={dict(sorted(counts.items()))}")
        failures = failing_jobs(statuses)
        if failures:
            manifest["completion_wait"] = {
                "last_checked_at": datetime.now(timezone.utc).isoformat(),
                "status_counts": dict(counts),
                "failed": [failure.__dict__ for failure in failures],
            }
            save_manifest(manifest_path, manifest)
            return 1
        if all_completed(statuses):
            submit_final_postjob(manifest_path, manifest, dry_run=dry_run_final, use_dependency=False)
            return 0
        time.sleep(interval_seconds)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Monitor a sharded multivariate paper run manifest.")
    parser.add_argument("--manifest", type=Path, required=True, help="Path to the manifest JSON.")
    parser.add_argument(
        "--mode",
        choices=("monitor_window", "wait_for_completion"),
        default="monitor_window",
        help="monitor_window: watch for a fixed duration; wait_for_completion: keep polling until all upstream jobs complete, then submit the final post job without dependencies.",
    )
    parser.add_argument("--duration-seconds", type=int, default=3600, help="Total monitor duration.")
    parser.add_argument("--interval-seconds", type=int, default=300, help="Polling interval.")
    parser.add_argument("--dry-run-final", action="store_true", help="Print the final sbatch command without submitting it.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest_path = args.manifest.resolve()
    if args.mode == "wait_for_completion":
        exit_code = wait_until_complete_and_submit_final(
            manifest_path,
            interval_seconds=int(args.interval_seconds),
            dry_run_final=bool(args.dry_run_final),
        )
    else:
        exit_code = monitor_run(
            manifest_path,
            duration_seconds=int(args.duration_seconds),
            interval_seconds=int(args.interval_seconds),
            dry_run_final=bool(args.dry_run_final),
        )
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
