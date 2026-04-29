#!/usr/bin/env python3
"""Benchmark controlled runtime for interv_qtl methods."""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import numpy as np
import pandas as pd


CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
BATCH_DIR = REPO_ROOT / "interv_mean"
BATCH_CORE_DIR = REPO_ROOT / "tabcf_core"
for path in (CURRENT_DIR, REPO_ROOT, BATCH_DIR, BATCH_CORE_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import stage1_control as stage1
import stage2_outcome as stage2
from mu_integrators import (
    DEFAULT_GAUSS_LEGENDRE_ORDER,
    DEFAULT_MU_INTEGRATOR,
    MuIntegratorConfig,
)
from cdf_to_quantiles import build_y_grid, compute_quantiles_on_grid
from run_interv_quantile import (
    DEFAULT_TAUS,
    SIM_DATA_DIR,
    TEST_DIR,
    TRAIN_DIR,
    build_x_grid,
)


DEFAULT_CODES: List[str] = [
    "A3_B3",
    "A3_B4",
    "A3_B5",
    "A3_B11",
    "A9_B3",
    "A9_B4",
    "A9_B5",
    "A9_B11",
]
DEFAULT_TRAIN_SIZES: List[int] = [1000]
DEFAULT_SEEDS: List[int] = list(range(1, 11))
DEFAULT_METHODS: List[str] = ["tabpfn", "div", "ivqr"]
DEFAULT_OUTPUT_DIR = SIM_DATA_DIR / "runtime"
DEFAULT_TIMER_SCOPE = "fit_predict"

METHOD_LABELS: Dict[str, str] = {
    "tabpfn": "TabCF",
    "tabpfn_real": "TabCF (Real-TabPFN)",
    "tabicl": "TabICL",
    "div": "DIV",
    "ivqr": "IVQR",
}
METHOD_FAMILIES: Dict[str, str] = {
    "tabpfn": "python",
    "tabpfn_real": "python",
    "tabicl": "python",
    "div": "r",
    "ivqr": "r",
}
METHOD_ORDER: Dict[str, int] = {
    method: idx for idx, method in enumerate(["tabpfn", "tabpfn_real", "tabicl", "div", "ivqr"])
}


@dataclass(frozen=True)
class RuntimeMethodSpec:
    method: str
    method_label: str
    family: str


@dataclass(frozen=True)
class RunContext:
    code: str
    train_size: int
    seed: int
    raw_train_path: Path
    raw_test_path: Path


@contextlib.contextmanager
def _suppress_output():
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


def _write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _parse_csv_list(raw: str) -> List[str]:
    tokens: List[str] = []
    for chunk in str(raw).replace(":", ",").split(","):
        part = chunk.strip()
        if not part:
            continue
        tokens.extend(part.split())
    return tokens


def _parse_taus(raw: str) -> tuple[float, ...]:
    taus = tuple(float(item) for item in _parse_csv_list(raw))
    if not taus:
        raise SystemExit("At least one tau must be provided.")
    return taus


def _read_raw_frames(run_ctx: RunContext) -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_csv(run_ctx.raw_train_path), pd.read_csv(run_ctx.raw_test_path)


def build_method_specs(methods: Sequence[str]) -> List[RuntimeMethodSpec]:
    unknown = [method for method in methods if method not in METHOD_LABELS]
    if unknown:
        raise ValueError(f"Unknown methods requested: {unknown}")

    deduped = list(dict.fromkeys(methods))
    specs = [
        RuntimeMethodSpec(
            method=method,
            method_label=METHOD_LABELS[method],
            family=METHOD_FAMILIES[method],
        )
        for method in sorted(deduped, key=lambda item: METHOD_ORDER[item])
    ]
    return specs


def summarise_runtime_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[tuple[int, str, str], List[float]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        key = (
            int(row["train_size"]),
            str(row["method"]),
            str(row["method_label"]),
        )
        grouped.setdefault(key, []).append(float(row["seconds"]))

    summary_rows: List[Dict[str, object]] = []
    for (train_size, method, method_label), values in grouped.items():
        mean_val = sum(values) / len(values)
        var_val = sum((value - mean_val) ** 2 for value in values) / len(values)
        summary_rows.append(
            {
                "train_size": train_size,
                "method": method,
                "method_label": method_label,
                "mean_seconds": mean_val,
                "std_seconds": math.sqrt(var_val),
                "n_runs": len(values),
            }
        )

    summary_rows.sort(key=lambda row: (int(row["train_size"]), METHOD_ORDER[str(row["method"])]))
    return summary_rows


def _base_row(method_spec: RuntimeMethodSpec, run_ctx: RunContext) -> Dict[str, object]:
    return {
        "code": run_ctx.code,
        "train_size": run_ctx.train_size,
        "seed": run_ctx.seed,
        "repeat_index": "",
        "method": method_spec.method,
        "method_label": method_spec.method_label,
        "family": method_spec.family,
        "seconds": "",
        "stage1_seconds": "",
        "stage2_seconds": "",
        "status": "ok",
        "error": "",
    }


def _build_stage1_training_frame(
    method_spec: RuntimeMethodSpec,
    train_df: pd.DataFrame,
    *,
    stage1_random_state: int,
):
    return stage1.build_stage1_training_frame(
        train_df,
        stage1.Stage1Config(random_state=stage1_random_state, backend_name=method_spec.method),
        verbose=False,
    )


def _build_quantile_prediction_grids(
    stage1_train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    x_grid_mode: str,
    x_grid_points: int,
    y_grid_points: int,
    y_grid_padding: float,
) -> tuple[np.ndarray, np.ndarray]:
    x_grid = build_x_grid(
        stage1_train_df,
        test_df,
        mode=x_grid_mode,
        points=x_grid_points,
    )
    y_grid = build_y_grid(
        np.asarray(stage1_train_df["Y"], dtype=float),
        padding=y_grid_padding,
        n_points=y_grid_points,
    )
    return x_grid, y_grid


def _new_cdf_model(method_spec: RuntimeMethodSpec, *, stage2_random_state: int):
    return stage2.ConditionalCDFEstimator(
        use_tabpfn=method_spec.method != "tabicl",
        backend_name=method_spec.method,
        random_state=stage2_random_state,
    )


def _fit_cdf_model(cdf_model: object, stage1_train_df: pd.DataFrame) -> None:
    cdf_model.fit_full(
        np.asarray(stage1_train_df["X"], dtype=float),
        np.asarray(stage1_train_df["V_hat"], dtype=float),
        np.asarray(stage1_train_df["Y"], dtype=float),
        verbose=False,
    )


def _predict_quantiles(
    cdf_model: object,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    *,
    taus: tuple[float, ...],
    n_v_grid: int,
    integrator_cfg: MuIntegratorConfig,
    max_points_per_batch: int | None,
):
    return compute_quantiles_on_grid(
        cdf_model,
        x_grid,
        y_grid,
        taus,
        n_v_grid,
        max_points_per_batch=max_points_per_batch,
        integrator_cfg=integrator_cfg,
    )


def _run_python_method(
    method_spec: RuntimeMethodSpec,
    run_ctx: RunContext,
    *,
    stage1_random_state: int,
    stage2_random_state: int,
    x_grid_mode: str,
    x_grid_points: int,
    y_grid_points: int,
    y_grid_padding: float,
    taus: tuple[float, ...],
    n_v_grid: int,
    integrator_cfg: MuIntegratorConfig,
    max_points_per_batch: int | None,
    timer_scope: str,
) -> Dict[str, object]:
    row = _base_row(method_spec, run_ctx)
    train_df, test_df = _read_raw_frames(run_ctx)

    if timer_scope == "predict_only":
        try:
            with _suppress_output():
                stage1_result = _build_stage1_training_frame(
                    method_spec,
                    train_df,
                    stage1_random_state=stage1_random_state,
                )
            x_grid, y_grid = _build_quantile_prediction_grids(
                stage1_result["train"],
                test_df,
                x_grid_mode=x_grid_mode,
                x_grid_points=x_grid_points,
                y_grid_points=y_grid_points,
                y_grid_padding=y_grid_padding,
            )
            cdf_model = _new_cdf_model(method_spec, stage2_random_state=stage2_random_state)
            with _suppress_output():
                _fit_cdf_model(cdf_model, stage1_result["train"])
        except Exception as exc:
            row["status"] = "error"
            row["error"] = str(exc)
            return row

        predict_start = time.perf_counter()
        try:
            with _suppress_output():
                _ = _predict_quantiles(
                    cdf_model,
                    x_grid,
                    y_grid,
                    taus=taus,
                    n_v_grid=n_v_grid,
                    integrator_cfg=integrator_cfg,
                    max_points_per_batch=max_points_per_batch,
                )
            row["stage2_seconds"] = time.perf_counter() - predict_start
            row["seconds"] = float(row["stage2_seconds"])
            return row
        except Exception as exc:
            row["stage2_seconds"] = time.perf_counter() - predict_start
            row["seconds"] = float(row["stage2_seconds"])
            row["status"] = "error"
            row["error"] = str(exc)
            return row

    stage1_start = time.perf_counter()
    try:
        with _suppress_output():
            stage1_result = _build_stage1_training_frame(
                method_spec,
                train_df,
                stage1_random_state=stage1_random_state,
            )
        row["stage1_seconds"] = time.perf_counter() - stage1_start
    except Exception as exc:
        row["seconds"] = time.perf_counter() - stage1_start
        row["status"] = "error"
        row["error"] = str(exc)
        return row

    x_grid = y_grid = None
    if timer_scope == "fit_predict":
        try:
            x_grid, y_grid = _build_quantile_prediction_grids(
                stage1_result["train"],
                test_df,
                x_grid_mode=x_grid_mode,
                x_grid_points=x_grid_points,
                y_grid_points=y_grid_points,
                y_grid_padding=y_grid_padding,
            )
        except Exception as exc:
            row["seconds"] = float(row["stage1_seconds"])
            row["status"] = "error"
            row["error"] = str(exc)
            return row

    stage2_start = time.perf_counter()
    try:
        cdf_model = _new_cdf_model(method_spec, stage2_random_state=stage2_random_state)
        with _suppress_output():
            _fit_cdf_model(cdf_model, stage1_result["train"])
            if timer_scope == "fit_predict":
                assert x_grid is not None and y_grid is not None
                _ = _predict_quantiles(
                    cdf_model,
                    x_grid,
                    y_grid,
                    taus=taus,
                    n_v_grid=n_v_grid,
                    integrator_cfg=integrator_cfg,
                    max_points_per_batch=max_points_per_batch,
                )
        row["stage2_seconds"] = time.perf_counter() - stage2_start
        row["seconds"] = float(row["stage1_seconds"]) + float(row["stage2_seconds"])
        return row
    except Exception as exc:
        row["stage2_seconds"] = time.perf_counter() - stage2_start
        row["seconds"] = float(row["stage1_seconds"]) + float(row["stage2_seconds"])
        row["status"] = "error"
        row["error"] = str(exc)
        return row


def build_r_runtime_command(
    *,
    rscript_bin: str,
    script_path: Path,
    method: str,
    run_ctx: RunContext,
    x_grid_mode: str,
    x_grid_points: int,
    taus: tuple[float, ...],
    timer_scope: str,
) -> List[str]:
    return [
        rscript_bin,
        str(script_path),
        "--benchmark-runtime",
        "--benchmark-train",
        str(run_ctx.raw_train_path),
        "--benchmark-test",
        str(run_ctx.raw_test_path),
        "--benchmark-code",
        run_ctx.code,
        "--benchmark-seed",
        str(run_ctx.seed),
        "--benchmark-x-grid-mode",
        x_grid_mode,
        "--benchmark-x-grid-points",
        str(x_grid_points),
        "--benchmark-timer-scope",
        timer_scope,
        "--benchmark-taus",
        ",".join(str(tau) for tau in taus),
    ]


def _parse_json_from_stdout(stdout: str) -> Dict[str, object]:
    for line in reversed(stdout.splitlines()):
        candidate = line.strip()
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise ValueError("No JSON payload found in R helper stdout.")


def _run_r_method(
    method_spec: RuntimeMethodSpec,
    run_ctx: RunContext,
    *,
    rscript_bin: str,
    x_grid_mode: str,
    x_grid_points: int,
    taus: tuple[float, ...],
    timer_scope: str,
) -> Dict[str, object]:
    row = _base_row(method_spec, run_ctx)
    script_name = "run_div_baseline.r" if method_spec.method == "div" else "run_ivqr_baseline.r"
    script_path = CURRENT_DIR / "baselines" / script_name
    cmd = build_r_runtime_command(
        rscript_bin=rscript_bin,
        script_path=script_path,
        method=method_spec.method,
        run_ctx=run_ctx,
        x_grid_mode=x_grid_mode,
        x_grid_points=x_grid_points,
        taus=taus,
        timer_scope=timer_scope,
    )
    try:
        completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
    except Exception as exc:
        row["status"] = "error"
        row["error"] = str(exc)
        return row

    if completed.returncode != 0:
        message = (completed.stderr or completed.stdout or f"Rscript exited with {completed.returncode}").strip()
        try:
            payload = _parse_json_from_stdout(completed.stdout)
            if payload.get("error"):
                message = str(payload["error"])
        except Exception:
            pass
        row["status"] = "error"
        row["error"] = message
        return row

    try:
        payload = _parse_json_from_stdout(completed.stdout)
    except Exception as exc:
        row["status"] = "error"
        row["error"] = f"Unable to parse R runtime payload: {exc}"
        return row

    row["seconds"] = float(payload.get("seconds", 0.0))
    row["stage1_seconds"] = ""
    row["stage2_seconds"] = ""
    if str(payload.get("status", "ok")) != "ok":
        row["status"] = "error"
        row["error"] = str(payload.get("error", "Unknown R benchmark error"))
    return row


def execute_method_benchmark(
    method_spec: RuntimeMethodSpec,
    run_contexts: Sequence[RunContext],
    runner: Callable[[RuntimeMethodSpec, RunContext], Dict[str, object]],
    *,
    warmup: bool,
    repeats: int,
) -> List[Dict[str, object]]:
    contexts = list(run_contexts)
    if warmup and contexts:
        _ = runner(method_spec, contexts[0])
    rows: List[Dict[str, object]] = []
    for repeat_index in range(1, int(repeats) + 1):
        for run_ctx in contexts:
            row = runner(method_spec, run_ctx)
            row["repeat_index"] = repeat_index
            rows.append(row)
    return rows


def _resolve_run_contexts(
    *,
    codes: Sequence[str],
    train_sizes: Sequence[int],
    seeds: Sequence[int],
) -> List[RunContext]:
    contexts: List[RunContext] = []
    missing_paths: List[Path] = []
    for code in codes:
        for train_size in train_sizes:
            for seed in seeds:
                train_path = TRAIN_DIR / f"train_data_{code}_n{int(train_size)}_seed{int(seed)}.csv"
                test_path = TEST_DIR / f"test_data_{code}.csv"
                if not train_path.exists():
                    missing_paths.append(train_path)
                    continue
                if not test_path.exists():
                    missing_paths.append(test_path)
                    continue
                contexts.append(
                    RunContext(
                        code=code,
                        train_size=int(train_size),
                        seed=int(seed),
                        raw_train_path=train_path,
                        raw_test_path=test_path,
                    )
                )
    if missing_paths:
        display = "\n".join(str(path) for path in missing_paths[:10])
        if len(missing_paths) > 10:
            display += f"\n... plus {len(missing_paths) - 10} more"
        raise FileNotFoundError(f"Missing required quantile benchmark inputs:\n{display}")
    contexts.sort(key=lambda ctx: (ctx.code, ctx.train_size, ctx.seed))
    return contexts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark controlled runtime for interv_qtl methods.")
    parser.add_argument("--codes", nargs="+", default=DEFAULT_CODES, help="DGP codes to benchmark.")
    parser.add_argument("--train-sizes", nargs="+", type=int, default=DEFAULT_TRAIN_SIZES, help="Train sizes.")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS, help="Seeds to benchmark.")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=DEFAULT_METHODS,
        choices=sorted(METHOD_LABELS),
        help="Subset of methods to benchmark.",
    )
    parser.add_argument("--repeats", type=int, default=1, help="Number of timed repeats per context.")
    parser.add_argument("--warmup", dest="warmup", action="store_true", help="Run one untimed warm-up per method.")
    parser.add_argument("--no-warmup", dest="warmup", action="store_false", help="Disable untimed warm-up.")
    parser.set_defaults(warmup=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for runtime CSVs.")
    parser.add_argument("--rscript-bin", default="Rscript", help="Rscript executable for R runtime helpers.")
    parser.add_argument(
        "--timer-scope",
        choices=["fit_only", "fit_predict", "predict_only"],
        default=DEFAULT_TIMER_SCOPE,
        help="Timed scope for each method.",
    )
    parser.add_argument("--stage1-random-state", type=int, default=1, help="Stage 1 random state for Python methods.")
    parser.add_argument("--stage2-random-state", type=int, default=1, help="Stage 2 random state for Python methods.")
    parser.add_argument(
        "--taus",
        type=str,
        default=",".join(str(tau) for tau in DEFAULT_TAUS),
        help="Comma-separated quantile levels.",
    )
    parser.add_argument("--x-grid-mode", type=str, default="test_quantile", help="Quantile benchmark x-grid mode.")
    parser.add_argument("--x-grid-points", type=int, default=200, help="Number of x-grid points.")
    parser.add_argument("--y-grid-points", type=int, default=1201, help="Number of y-grid points.")
    parser.add_argument("--y-grid-padding", type=float, default=0.25, help="Y-grid padding fraction.")
    parser.add_argument("--n-v-grid", type=int, default=101, help="Uniform-grid fallback V integration points.")
    parser.add_argument(
        "--stage2-mu-integrator",
        choices=["simpson", "gauss_legendre"],
        default=DEFAULT_MU_INTEGRATOR,
        help="Stage-2 integration rule for Python quantile methods.",
    )
    parser.add_argument(
        "--stage2-gauss-legendre-order",
        type=int,
        default=DEFAULT_GAUSS_LEGENDRE_ORDER,
        help="Quadrature order when --stage2-mu-integrator=gauss_legendre.",
    )
    parser.add_argument(
        "--max-points-per-batch",
        type=int,
        default=None,
        help="Optional chunk size for batched CDF prediction over (x, v).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if int(args.repeats) <= 0:
        raise SystemExit("--repeats must be positive.")
    if int(args.x_grid_points) <= 1:
        raise SystemExit("--x-grid-points must be greater than 1.")
    if int(args.y_grid_points) <= 9:
        raise SystemExit("--y-grid-points must be at least 10.")
    if int(args.n_v_grid) <= 1:
        raise SystemExit("--n-v-grid must be greater than 1.")
    if int(args.stage2_gauss_legendre_order) <= 0:
        raise SystemExit("--stage2-gauss-legendre-order must be positive.")
    taus = _parse_taus(args.taus)
    method_specs = build_method_specs(args.methods)
    run_contexts = _resolve_run_contexts(codes=args.codes, train_sizes=args.train_sizes, seeds=args.seeds)
    integrator_cfg = MuIntegratorConfig(
        method=args.stage2_mu_integrator,
        n_v_points=args.n_v_grid,
        gauss_legendre_order=args.stage2_gauss_legendre_order,
        max_points_per_batch=args.max_points_per_batch,
    )

    print(
        "Benchmarking interv_qtl runtime with "
        f"{len(run_contexts)} contexts, methods={','.join(spec.method for spec in method_specs)}, "
        f"repeats={args.repeats}, warmup={args.warmup}",
        flush=True,
    )

    run_rows: List[Dict[str, object]] = []
    for method_spec in method_specs:
        print(f"Benchmarking {method_spec.method_label} ({method_spec.method})", flush=True)
        if method_spec.family == "python":
            rows = execute_method_benchmark(
                method_spec,
                run_contexts,
                lambda spec, ctx: _run_python_method(
                    spec,
                    ctx,
                    stage1_random_state=args.stage1_random_state,
                    stage2_random_state=args.stage2_random_state,
                    x_grid_mode=args.x_grid_mode,
                    x_grid_points=args.x_grid_points,
                    y_grid_points=args.y_grid_points,
                    y_grid_padding=args.y_grid_padding,
                    taus=taus,
                    n_v_grid=args.n_v_grid,
                    integrator_cfg=integrator_cfg,
                    max_points_per_batch=args.max_points_per_batch,
                    timer_scope=args.timer_scope,
                ),
                warmup=args.warmup,
                repeats=args.repeats,
            )
        else:
            rows = execute_method_benchmark(
                method_spec,
                run_contexts,
                lambda spec, ctx: _run_r_method(
                    spec,
                    ctx,
                    rscript_bin=args.rscript_bin,
                    x_grid_mode=args.x_grid_mode,
                    x_grid_points=args.x_grid_points,
                    taus=taus,
                    timer_scope=args.timer_scope,
                ),
                warmup=args.warmup,
                repeats=args.repeats,
            )
        run_rows.extend(rows)
        for row in rows:
            if row["status"] == "ok":
                print(
                    f"  code={row['code']} n={row['train_size']} seed={row['seed']}: "
                    f"{float(row['seconds']):.6f}s",
                    flush=True,
                )
            else:
                print(
                    f"  code={row['code']} n={row['train_size']} seed={row['seed']}: "
                    f"ERROR {row['error']}",
                    flush=True,
                )

    summary_rows = summarise_runtime_rows(run_rows)
    run_csv = Path(args.output_dir) / "quantile_runtime_runs.csv"
    summary_csv = Path(args.output_dir) / "quantile_runtime_summary.csv"
    _write_csv(
        run_csv,
        run_rows,
        [
            "code",
            "train_size",
            "seed",
            "repeat_index",
            "method",
            "method_label",
            "family",
            "seconds",
            "stage1_seconds",
            "stage2_seconds",
            "status",
            "error",
        ],
    )
    _write_csv(
        summary_csv,
        summary_rows,
        [
            "train_size",
            "method",
            "method_label",
            "mean_seconds",
            "std_seconds",
            "n_runs",
        ],
    )
    print(f"Saved run-level runtime CSV to {run_csv}")
    print(f"Saved summary runtime CSV to {summary_csv}")


if __name__ == "__main__":
    main()
