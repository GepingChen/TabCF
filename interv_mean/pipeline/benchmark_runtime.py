#!/usr/bin/env python3
"""Benchmark pure method runtime for sec5.1 delivery methods on a single DGP."""

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
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CORE_DIR = REPO_ROOT / "tabcf_core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from interv_mean.pipeline import run_py_baselines, run_tabpfn_naive, utils, plot_div_style
from local_context_backends import LocalContextConfig
from mu_integrators import (
    DEFAULT_GAUSS_LEGENDRE_ORDER,
    DEFAULT_MU_INTEGRATOR,
    MuIntegratorConfig,
    integrate_mean_function_over_v,
)


DEFAULT_MANIFEST = utils.DEFAULT_MANIFEST_GENERATED_DIR / "a3_s10.json"
DEFAULT_OUTPUT_DIR = utils.DEFAULT_IO_ROOT / "runtime"
DEFAULT_CODE = "A3_B3"
DEFAULT_TIMER_SCOPE = "fit_predict"
DEFAULT_REPEATS = 10

METHOD_VARIANTS: Dict[str, str] = {
    "tabpfn": "stage2_9",
    "tabpfn_local": "stage2_9_local_knn",
    "tabpfn_real": "stage2_9",
    "tabpfn_real_local": "stage2_9_local_knn",
    "tabicl": "stage2_9",
    "tabicl_local": "stage2_9_local_knn",
    "linear_iv": "control_function_linear",
    "nonlinear_iv": "control_function_spline_df5",
    "div_2": "div_cran_layer3_epoch1000",
    "deepgmm": "deepgmm",
    "deepiv": "deepiv_restarts10",
    "tabpfn_naive": "tabpfn_naive_y_on_x",
}
METHOD_FAMILIES: Dict[str, str] = {
    "tabpfn": "core",
    "tabpfn_local": "core",
    "tabpfn_real": "core",
    "tabpfn_real_local": "core",
    "tabicl": "core",
    "tabicl_local": "core",
    "linear_iv": "r",
    "nonlinear_iv": "r",
    "div_2": "r",
    "deepgmm": "python",
    "deepiv": "python",
    "tabpfn_naive": "python",
}
DEFAULT_METHODS: List[str] = [
    pair[0]
    for pair in plot_div_style.PREFERRED_METHOD_ORDER
    if pair[0] in METHOD_VARIANTS
    and METHOD_VARIANTS[pair[0]] == pair[1]
    and not pair[0].endswith("_local")
]
METHOD_ORDER: Dict[str, int] = {
    method: idx
    for idx, method in enumerate(
        [
            pair[0]
            for pair in plot_div_style.PREFERRED_METHOD_ORDER
            if pair[0] in METHOD_VARIANTS and METHOD_VARIANTS[pair[0]] == pair[1]
        ]
    )
}


@dataclass(frozen=True)
class RuntimeMethodSpec:
    method: str
    variant: str
    method_label: str
    family: str


@dataclass(frozen=True)
class RunContext:
    run: Dict[str, object]
    raw_train_path: Path
    raw_test_path: Path
    bridge_train_path: Path
    bridge_test_path: Path


def resolve_core_method_config(method: str, local_k_neighbors: int | None = None) -> Dict[str, object]:
    backend_name = str(method)
    local_context = LocalContextConfig()
    if method == "tabpfn_local":
        backend_name = "tabpfn"
        local_context = LocalContextConfig(strategy="local_knn", k_neighbors=local_k_neighbors)
    elif method == "tabpfn_real_local":
        backend_name = "tabpfn_real"
        local_context = LocalContextConfig(strategy="local_knn", k_neighbors=local_k_neighbors)
    elif method == "tabicl_local":
        backend_name = "tabicl"
        local_context = LocalContextConfig(strategy="local_knn", k_neighbors=local_k_neighbors)

    return {
        "backend_name": backend_name,
        "use_tabpfn": backend_name != "tabicl",
        "local_context": local_context,
    }


def build_method_specs(methods: Sequence[str]) -> List[RuntimeMethodSpec]:
    unknown = [method for method in methods if method not in METHOD_VARIANTS]
    if unknown:
        raise ValueError(f"Unknown methods requested: {unknown}")

    deduped = list(dict.fromkeys(methods))
    specs: List[RuntimeMethodSpec] = []
    for method in sorted(deduped, key=lambda item: METHOD_ORDER[item]):
        pair = (method, METHOD_VARIANTS[method])
        style = plot_div_style.KNOWN_METHODS.get(pair)
        if style is None:
            raise KeyError(f"No visualization label registered for {pair}")
        specs.append(
            RuntimeMethodSpec(
                method=method,
                variant=METHOD_VARIANTS[method],
                method_label=str(style["label"]),
                family=METHOD_FAMILIES[method],
            )
        )
    return specs


def summarise_runtime_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[tuple[str, int, str, str, str], List[float]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        key = (
            str(row["code"]),
            int(row["train_size"]),
            str(row["method"]),
            str(row["method_label"]),
            str(row.get("mu_integrator", "")),
        )
        grouped.setdefault(key, []).append(float(row["seconds"]))

    summary_rows: List[Dict[str, object]] = []
    for (code, train_size, method, method_label, mu_integrator), values in grouped.items():
        mean_val = sum(values) / len(values)
        var_val = sum((value - mean_val) ** 2 for value in values) / len(values)
        summary_rows.append(
            {
                "code": code,
                "train_size": train_size,
                "method": method,
                "method_label": method_label,
                "mu_integrator": mu_integrator,
                "mean_seconds": mean_val,
                "std_seconds": math.sqrt(var_val),
                "n_runs": len(values),
            }
        )
    summary_rows.sort(key=lambda row: (METHOD_ORDER[str(row["method"])], int(row["train_size"])))
    return summary_rows


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
        runner(method_spec, contexts[0])
    rows: List[Dict[str, object]] = []
    for repeat_index in range(1, int(repeats) + 1):
        for ctx in contexts:
            row = runner(method_spec, ctx)
            row["repeat_index"] = repeat_index
            rows.append(row)
    return rows


@contextlib.contextmanager
def _suppress_output():
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


def _read_raw_frames(run_ctx: RunContext) -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_csv(run_ctx.raw_train_path), pd.read_csv(run_ctx.raw_test_path)


def _read_bridge_frames(run_ctx: RunContext):
    return run_py_baselines.load_bridged_csvs(run_ctx.bridge_train_path, run_ctx.bridge_test_path)


def _sorted_test_x_grid(test_df: pd.DataFrame) -> np.ndarray:
    if "X" not in test_df.columns:
        raise ValueError(f"Test frame missing column 'X'; available columns: {list(test_df.columns)}")
    return np.sort(np.asarray(test_df["X"], dtype=float))


def compute_mu_c_with_integrator(
    m_model: object,
    x_grid: np.ndarray,
    *,
    integrator_cfg: MuIntegratorConfig,
) -> np.ndarray:
    return integrate_mean_function_over_v(
        m_model.predict,
        np.asarray(x_grid, dtype=float),
        integrator_cfg=integrator_cfg,
    )


def _fit_structural_model_for_runtime(stage2_module, train_df: pd.DataFrame, *, core_cfg: Dict[str, object], seed: int):
    m_model = stage2_module.FullDataStructuralFunctionModel(
        use_tabpfn=bool(core_cfg["use_tabpfn"]),
        backend_name=str(core_cfg["backend_name"]),
        random_state=seed,
        local_context=core_cfg["local_context"],
    )
    X = np.asarray(train_df["X"], dtype=float).reshape(-1, 1)
    V = np.asarray(train_df["V_hat"], dtype=float).reshape(-1, 1)
    Y = np.asarray(train_df["Y"], dtype=float)
    features = np.hstack([X, V])
    reg = m_model._new_regressor()
    reg.fit(features, Y)
    m_model.model = reg
    return m_model


def compare_core_mu_integrators(
    method_spec: RuntimeMethodSpec,
    run_ctx: RunContext,
    *,
    selected_integrator: MuIntegratorConfig,
    reference_integrator: MuIntegratorConfig,
    local_k_neighbors: int | None = None,
) -> Dict[str, object]:
    import stage1_control as stage1
    import stage2_outcome as stage2

    if METHOD_FAMILIES.get(method_spec.method) != "core":
        raise ValueError("Integrator comparison only supports core methods.")

    train_df, test_df = _read_raw_frames(run_ctx)
    seed = int(run_ctx.run["seed"])
    core_cfg = resolve_core_method_config(method_spec.method, local_k_neighbors=local_k_neighbors)

    with _suppress_output():
        stage1_start = time.perf_counter()
        stage1_result = stage1.build_stage1_training_frame(
            train_df,
            stage1.Stage1Config(random_state=seed, backend_name=str(core_cfg["backend_name"])),
            verbose=False,
        )
        stage1_seconds = time.perf_counter() - stage1_start

        fit_start = time.perf_counter()
        m_model = _fit_structural_model_for_runtime(stage2, stage1_result["train"], core_cfg=core_cfg, seed=seed)
        fit_seconds = time.perf_counter() - fit_start

    x_grid = _sorted_test_x_grid(test_df)

    with _suppress_output():
        ref_start = time.perf_counter()
        ref_mu = compute_mu_c_with_integrator(m_model, x_grid, integrator_cfg=reference_integrator)
        ref_integration_seconds = time.perf_counter() - ref_start

        selected_start = time.perf_counter()
        selected_mu = compute_mu_c_with_integrator(m_model, x_grid, integrator_cfg=selected_integrator)
        selected_integration_seconds = time.perf_counter() - selected_start

    diff = np.asarray(selected_mu - ref_mu, dtype=float)
    metrics = {
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "max_abs": float(np.max(np.abs(diff))),
    }
    return {
        "method": method_spec.method,
        "method_label": method_spec.method_label,
        "code": str(run_ctx.run["code"]),
        "train_size": int(run_ctx.run["train_size"]),
        "seed": seed,
        "stage1_seconds": float(stage1_seconds),
        "stage2_fit_seconds": float(fit_seconds),
        "reference_integrator": reference_integrator.method,
        "selected_integrator": selected_integrator.method,
        "reference_integration_seconds": float(ref_integration_seconds),
        "selected_integration_seconds": float(selected_integration_seconds),
        "reference_total_seconds": float(stage1_seconds + fit_seconds + ref_integration_seconds),
        "selected_total_seconds": float(stage1_seconds + fit_seconds + selected_integration_seconds),
        "n_test_points": int(len(x_grid)),
        "error_metrics": metrics,
        "x_grid": x_grid.tolist(),
        "reference_mu": ref_mu.tolist(),
        "selected_mu": selected_mu.tolist(),
    }


def _base_row(method_spec: RuntimeMethodSpec, run_ctx: RunContext) -> Dict[str, object]:
    run = run_ctx.run
    return {
        "code": str(run["code"]),
        "train_size": int(run["train_size"]),
        "seed": int(run["seed"]),
        "repeat_index": "",
        "method": method_spec.method,
        "method_label": method_spec.method_label,
        "family": method_spec.family,
        "mu_integrator": "",
        "seconds": "",
        "stage1_seconds": "",
        "stage2_seconds": "",
        "status": "ok",
        "error": "",
    }


def _run_core_method(
    method_spec: RuntimeMethodSpec,
    run_ctx: RunContext,
    *,
    local_k_neighbors: int | None = None,
    mu_integrator: MuIntegratorConfig | None = None,
) -> Dict[str, object]:
    row = _base_row(method_spec, run_ctx)
    try:
        import stage1_control as stage1
        import stage2_outcome as stage2
    except Exception as exc:
        row["status"] = "error"
        row["error"] = str(exc)
        return row

    train_df, test_df = _read_raw_frames(run_ctx)
    seed = int(run_ctx.run["seed"])
    core_cfg = resolve_core_method_config(method_spec.method, local_k_neighbors=local_k_neighbors)
    integrator_cfg = MuIntegratorConfig() if mu_integrator is None else mu_integrator

    stage1_start = time.perf_counter()
    try:
        with _suppress_output():
            stage1_result = stage1.build_stage1_training_frame(
                train_df,
                stage1.Stage1Config(random_state=seed, backend_name=str(core_cfg["backend_name"])),
                verbose=False,
            )
        row["stage1_seconds"] = time.perf_counter() - stage1_start
    except Exception as exc:
        row["seconds"] = time.perf_counter() - stage1_start
        row["status"] = "error"
        row["error"] = str(exc)
        return row

    stage2_start = time.perf_counter()
    try:
        with _suppress_output():
            m_model = _fit_structural_model_for_runtime(
                stage2,
                stage1_result["train"],
                core_cfg=core_cfg,
                seed=seed,
            )
            _ = compute_mu_c_with_integrator(
                m_model,
                _sorted_test_x_grid(test_df),
                integrator_cfg=integrator_cfg,
            )
        row["stage2_seconds"] = time.perf_counter() - stage2_start
        row["seconds"] = float(row["stage1_seconds"]) + float(row["stage2_seconds"])
        row["mu_integrator"] = integrator_cfg.method
        return row
    except Exception as exc:
        row["stage2_seconds"] = time.perf_counter() - stage2_start
        row["seconds"] = float(row["stage1_seconds"]) + float(row["stage2_seconds"])
        row["status"] = "error"
        row["error"] = str(exc)
        return row


def _run_python_method(method_spec: RuntimeMethodSpec, run_ctx: RunContext) -> Dict[str, object]:
    row = _base_row(method_spec, run_ctx)
    train_df, test_df = _read_bridge_frames(run_ctx)
    seed = int(run_ctx.run["seed"])
    x_test = test_df["Xint"].to_numpy()

    start = time.perf_counter()
    try:
        with _suppress_output():
            if method_spec.method == "deepiv":
                predictor, _variant = run_py_baselines._deepiv_predictor(train_df, seed=seed, n_restarts=10)
                _ = predictor(x_test)
            elif method_spec.method == "deepgmm":
                predictor, variant = run_py_baselines._deepgmm_predictor(train_df, val_fraction=0.1, seed=seed)
                if variant == "linear_iv_fallback":
                    raise RuntimeError("DeepGMM benchmark resolved to linear_iv_fallback; refusing fallback timing.")
                _ = predictor(x_test)
            elif method_spec.method == "tabpfn_naive":
                predictor, _variant = run_tabpfn_naive._tabpfn_naive_predictor(train_df, seed=seed)
                _ = predictor(x_test)
            else:  # pragma: no cover
                raise ValueError(f"Unsupported python benchmark method: {method_spec.method}")
        row["seconds"] = time.perf_counter() - start
        return row
    except Exception as exc:
        row["seconds"] = time.perf_counter() - start
        row["status"] = "error"
        row["error"] = str(exc)
        return row


def build_r_runtime_command(
    *,
    rscript_bin: str,
    script_path: Path,
    method: str,
    bridge_train_path: Path,
    bridge_test_path: Path,
    seed: int,
) -> List[str]:
    return [
        rscript_bin,
        str(script_path),
        "--benchmark-runtime",
        "--benchmark-model",
        method,
        "--benchmark-train",
        str(bridge_train_path),
        "--benchmark-test",
        str(bridge_test_path),
        "--benchmark-seed",
        str(seed),
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


def _run_r_method(method_spec: RuntimeMethodSpec, run_ctx: RunContext, *, rscript_bin: str, script_path: Path) -> Dict[str, object]:
    row = _base_row(method_spec, run_ctx)
    cmd = build_r_runtime_command(
        rscript_bin=rscript_bin,
        script_path=script_path,
        method=method_spec.method,
        bridge_train_path=run_ctx.bridge_train_path,
        bridge_test_path=run_ctx.bridge_test_path,
        seed=int(run_ctx.run["seed"]),
    )
    try:
        completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
    except Exception as exc:
        row["status"] = "error"
        row["error"] = str(exc)
        return row

    if completed.returncode != 0:
        row["status"] = "error"
        row["error"] = (completed.stderr or completed.stdout or f"Rscript exited with {completed.returncode}").strip()
        return row

    try:
        payload = _parse_json_from_stdout(completed.stdout)
        row["seconds"] = float(payload["seconds"])
        return row
    except Exception as exc:
        row["status"] = "error"
        row["error"] = f"Unable to parse R runtime payload: {exc}"
        return row


def _select_run_contexts(
    *,
    manifest: Mapping[str, object],
    code: str,
    train_sizes: Optional[Sequence[int]],
    seeds: Optional[Sequence[int]],
    bridge_dir: Path,
) -> List[RunContext]:
    runs = manifest.get("runs", [])
    if not isinstance(runs, list):
        raise ValueError("Manifest missing runs list.")

    size_filter = {int(size) for size in train_sizes} if train_sizes else None
    seed_filter = {int(seed) for seed in seeds} if seeds else None

    selected: List[RunContext] = []
    for run in runs:
        if str(run.get("code")) != code:
            continue
        if size_filter and int(run["train_size"]) not in size_filter:
            continue
        if seed_filter and int(run["seed"]) not in seed_filter:
            continue
        resolved = utils.resolve_run_paths(run, manifest=dict(manifest), bridge_dir=bridge_dir)
        selected.append(
            RunContext(
                run=run,
                raw_train_path=resolved["train"],
                raw_test_path=resolved["test"],
                bridge_train_path=resolved["bridge_train"],
                bridge_test_path=resolved["bridge_test"],
            )
        )

    selected.sort(key=lambda ctx: (int(ctx.run["train_size"]), int(ctx.run["seed"])))
    if not selected:
        raise ValueError(f"No runs found for code={code} with the requested filters.")
    return selected


def _write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark pure method runtime for sec5.1 delivery methods.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="sec5.1 manifest to benchmark.")
    parser.add_argument("--code", default=DEFAULT_CODE, help="Single DGP code to benchmark.")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=DEFAULT_METHODS,
        choices=sorted(METHOD_VARIANTS),
        help="Subset of delivery methods to benchmark.",
    )
    parser.add_argument("--train-sizes", nargs="+", type=int, default=None, help="Optional subset of train sizes.")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Optional subset of seeds.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for runtime CSVs.")
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help="Number of repeated formal runs per method/context for averaging.",
    )
    parser.add_argument(
        "--timer-scope",
        default=DEFAULT_TIMER_SCOPE,
        choices=[DEFAULT_TIMER_SCOPE],
        help="Pure-timing scope. Currently only fit_predict is supported.",
    )
    parser.add_argument("--warmup", dest="warmup", action="store_true", help="Run one untimed warm-up per method.")
    parser.add_argument("--no-warmup", dest="warmup", action="store_false", help="Disable untimed warm-up.")
    parser.set_defaults(warmup=True)
    parser.add_argument(
        "--bridge-dir",
        type=Path,
        default=utils.DEFAULT_BRIDGE_DIR,
        help="Directory containing bridged sec5.1 CSVs.",
    )
    parser.add_argument("--rscript-bin", default="Rscript", help="Rscript executable for R baseline benchmarking.")
    parser.add_argument(
        "--mu-integrator",
        choices=["simpson", "gauss_legendre"],
        default=DEFAULT_MU_INTEGRATOR,
        help="Integration rule used to compute mu_c(x) inside core-method timing.",
    )
    parser.add_argument(
        "--n-v-points",
        type=int,
        default=100,
        help="Number of V grid points for Simpson integration in timing-only core runs.",
    )
    parser.add_argument(
        "--gauss-legendre-order",
        type=int,
        default=DEFAULT_GAUSS_LEGENDRE_ORDER,
        help="Quadrature order when --mu-integrator=gauss_legendre.",
    )
    parser.add_argument(
        "--max-points-per-batch",
        type=int,
        default=None,
        help="Optional explicit batch size for X chunks during mu_c(x) integration.",
    )
    parser.add_argument(
        "--compare-core-integrators",
        action="store_true",
        help="Run one core-method comparison between Simpson and the selected --mu-integrator, then exit.",
    )
    parser.add_argument(
        "--local-k-neighbors",
        type=int,
        default=None,
        help="Optional fixed k for local core methods. Defaults to the Stage-2 heuristic when omitted.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if int(args.repeats) <= 0:
        raise SystemExit("--repeats must be positive.")
    if int(args.n_v_points) <= 1:
        raise SystemExit("--n-v-points must be greater than 1.")
    if int(args.gauss_legendre_order) <= 0:
        raise SystemExit("--gauss-legendre-order must be positive.")
    manifest = utils.load_manifest(args.manifest)
    method_specs = build_method_specs(args.methods)
    run_contexts = _select_run_contexts(
        manifest=manifest,
        code=args.code,
        train_sizes=args.train_sizes,
        seeds=args.seeds,
        bridge_dir=args.bridge_dir,
    )

    r_script_path = Path(__file__).resolve().with_name("run_r_baselines.r")
    run_rows: List[Dict[str, object]] = []
    selected_integrator = MuIntegratorConfig(
        method=args.mu_integrator,
        n_v_points=args.n_v_points,
        gauss_legendre_order=args.gauss_legendre_order,
        max_points_per_batch=args.max_points_per_batch,
    )
    reference_integrator = MuIntegratorConfig(
        method="simpson",
        n_v_points=args.n_v_points,
        gauss_legendre_order=args.gauss_legendre_order,
        max_points_per_batch=args.max_points_per_batch,
    )

    if args.compare_core_integrators:
        if len(method_specs) != 1 or method_specs[0].family != "core":
            raise SystemExit("--compare-core-integrators requires exactly one core method.")
        if len(run_contexts) != 1:
            raise SystemExit("--compare-core-integrators requires exactly one (code, train_size, seed) context.")
        if selected_integrator.method == reference_integrator.method:
            raise SystemExit("--compare-core-integrators requires --mu-integrator different from simpson.")

        comparison = compare_core_mu_integrators(
            method_specs[0],
            run_contexts[0],
            selected_integrator=selected_integrator,
            reference_integrator=reference_integrator,
            local_k_neighbors=args.local_k_neighbors,
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        stem = (
            f"mu_integrator_compare_{method_specs[0].method}_{args.code}"
            f"_n{int(run_contexts[0].run['train_size'])}_seed{int(run_contexts[0].run['seed'])}"
        )
        json_path = args.output_dir / f"{stem}.json"
        csv_path = args.output_dir / f"{stem}.csv"
        json_path.write_text(json.dumps(comparison, indent=2))
        rows = [
            {
                "x": x,
                "mu_reference": ref_mu,
                "mu_selected": sel_mu,
                "abs_diff": abs(sel_mu - ref_mu),
            }
            for x, ref_mu, sel_mu in zip(
                comparison["x_grid"],
                comparison["reference_mu"],
                comparison["selected_mu"],
            )
        ]
        _write_csv(csv_path, rows, ["x", "mu_reference", "mu_selected", "abs_diff"])
        print(f"Saved integrator comparison JSON to {json_path}")
        print(f"Saved integrator comparison CSV to {csv_path}")
        print(
            f"{comparison['reference_integrator']} total={comparison['reference_total_seconds']:.6f}s, "
            f"{comparison['selected_integrator']} total={comparison['selected_total_seconds']:.6f}s, "
            f"rmse={comparison['error_metrics']['rmse']:.6e}, "
            f"max_abs={comparison['error_metrics']['max_abs']:.6e}"
        )
        return

    for method_spec in method_specs:
        print(f"Benchmarking {method_spec.method_label} ({method_spec.method})", flush=True)
        if method_spec.family == "core":
            rows = execute_method_benchmark(
                method_spec,
                run_contexts,
                lambda spec, ctx: _run_core_method(
                    spec,
                    ctx,
                    local_k_neighbors=args.local_k_neighbors,
                    mu_integrator=selected_integrator,
                ),
                warmup=args.warmup,
                repeats=args.repeats,
            )
        elif method_spec.family == "python":
            rows = execute_method_benchmark(
                method_spec,
                run_contexts,
                _run_python_method,
                warmup=args.warmup,
                repeats=args.repeats,
            )
        elif method_spec.family == "r":
            rows = execute_method_benchmark(
                method_spec,
                run_contexts,
                lambda spec, ctx: _run_r_method(spec, ctx, rscript_bin=args.rscript_bin, script_path=r_script_path),
                warmup=args.warmup,
                repeats=args.repeats,
            )
        else:  # pragma: no cover
            raise ValueError(f"Unsupported method family: {method_spec.family}")
        run_rows.extend(rows)
        for row in rows:
            status = row["status"]
            seconds = row["seconds"]
            size = row["train_size"]
            seed = row["seed"]
            if status == "ok":
                print(f"  n={size} seed={seed}: {seconds:.6f}s", flush=True)
            else:
                print(f"  n={size} seed={seed}: ERROR {row['error']}", flush=True)

    summary_rows = summarise_runtime_rows(run_rows)
    stem = f"interv_mean_runtime_{args.code}"
    run_csv = args.output_dir / f"{stem}_runs.csv"
    summary_csv = args.output_dir / f"{stem}_summary.csv"

    run_fields = [
        "code",
        "train_size",
        "seed",
        "repeat_index",
        "method",
        "method_label",
        "family",
        "mu_integrator",
        "seconds",
        "stage1_seconds",
        "stage2_seconds",
        "status",
        "error",
    ]
    summary_fields = [
        "code",
        "train_size",
        "method",
        "method_label",
        "mu_integrator",
        "mean_seconds",
        "std_seconds",
        "n_runs",
    ]
    _write_csv(run_csv, run_rows, run_fields)
    _write_csv(summary_csv, summary_rows, summary_fields)
    print(f"Saved run-level runtime CSV to {run_csv}")
    print(f"Saved summary runtime CSV to {summary_csv}")


if __name__ == "__main__":
    main()
