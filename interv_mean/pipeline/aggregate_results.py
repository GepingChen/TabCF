#!/usr/bin/env python3
"""Aggregate sec5.1 baseline MSEs and optional core-backend references."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
import sys
from typing import Dict, Iterable, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from interv_mean.pipeline import utils


_BASELINE_JSON_NAME_RE = re.compile(
    r"^(?P<model>.+?)_(?P<code>A\d+_B\d+)_n(?P<train_size>\d+)_seed(?P<seed>\d+)\.json$"
)


def _load_metrics_json(path: Path) -> Dict[str, object]:
    text = path.read_text()
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        if exc.msg != "Extra data":
            raise
        decoder = json.JSONDecoder()
        data, _end = decoder.raw_decode(text)
        if not isinstance(data, dict):
            raise
        return data


def _load_baseline_metrics(
    results_dir: Path,
    *,
    models: Sequence[str] | None = None,
    codes: Sequence[str] | None = None,
    train_sizes: Sequence[int] | None = None,
    seeds: Sequence[int] | None = None,
) -> List[Dict[str, object]]:
    model_allowlist = set(models) if models is not None else None
    code_allowlist = set(codes) if codes is not None else None
    size_allowlist = {int(size) for size in train_sizes} if train_sizes is not None else None
    seed_allowlist = {int(seed) for seed in seeds} if seeds is not None else None

    records: List[Dict[str, object]] = []
    for path in results_dir.glob("*.json"):
        try:
            match = _BASELINE_JSON_NAME_RE.match(path.name)
            if match is not None:
                model = match.group("model")
                code = match.group("code")
                train_size = int(match.group("train_size"))
                seed = int(match.group("seed"))
                if model_allowlist is not None and model not in model_allowlist:
                    continue
                if code_allowlist is not None and code not in code_allowlist:
                    continue
                if size_allowlist is not None and train_size not in size_allowlist:
                    continue
                if seed_allowlist is not None and seed not in seed_allowlist:
                    continue

            data = _load_metrics_json(path)
            data["metrics_path"] = str(path)
            records.append(data)
        except Exception as exc:
            print(f"Skipping {path}: {exc}")
    return records


def _manifest_run_keys(manifest_runs: List[Dict[str, object]]) -> List[Tuple[str, int, int]]:
    keys: List[Tuple[str, int, int]] = []
    for run in manifest_runs:
        try:
            keys.append((str(run["code"]), int(run["train_size"]), int(run["seed"])))
        except Exception:
            continue
    return keys


def _read_tabpfn_mse(summary_path: Path) -> float:
    with summary_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("key") == "metric_mse_do_pred_vs_clean":
                return float(row["value"])
    raise KeyError(f"metric_mse_do_pred_vs_clean not found in {summary_path}")


def _read_summary_kv(summary_path: Path) -> Dict[str, str]:
    with summary_path.open() as f:
        reader = csv.DictReader(f)
        return {str(row["key"]): str(row["value"]) for row in reader}


def _tabpfn_records(manifest: Dict[str, object], manifest_runs: List[Dict[str, object]]) -> List[Dict[str, object]]:
    seen = set()
    records: List[Dict[str, object]] = []
    for run in manifest_runs:
        resolved = utils.resolve_run_paths(run, manifest=manifest)
        summary_path = resolved["stage2_summary"]
        if summary_path is None:
            continue
        key = (str(run["code"]), int(run["train_size"]), int(run["seed"]))
        if key in seen:
            continue
        seen.add(key)
        mse_val = _read_tabpfn_mse(summary_path)
        records.append(
            {
                "model": "tabpfn",
                "variant": "stage2_9",
                "code": str(run["code"]),
                "scenario": str(run.get("scenario", "")),
                "train_size": int(run["train_size"]),
                "seed": int(run["seed"]),
                "mse_vs_mean_int": mse_val,
                "metrics_path": str(summary_path),
            }
        )
    return records


def _aggregate(records: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str, str, int], List[float]] = {}
    scenario_map: Dict[Tuple[str, str, str, int], str] = {}
    for rec in records:
        key = (str(rec["model"]), str(rec.get("variant", "")), str(rec["code"]), int(rec["train_size"]))
        grouped.setdefault(key, []).append(float(rec["mse_vs_mean_int"]))
        scenario_map[key] = str(rec.get("scenario", ""))

    summary_rows: List[Dict[str, object]] = []
    for key, values in grouped.items():
        model, variant, code, size = key
        n = len(values)
        mean_val = sum(values) / n if n else math.nan
        var = sum((v - mean_val) ** 2 for v in values) / n if n else math.nan
        std_val = math.sqrt(var) if n else math.nan
        summary_rows.append(
            {
                "model": model,
                "variant": variant,
                "code": code,
                "scenario": scenario_map.get(key, ""),
                "train_size": size,
                "mean_mse": mean_val,
                "std_mse": std_val,
                "n_runs": n,
            }
        )
    summary_rows.sort(key=lambda r: (r["model"], r["code"], r["train_size"]))
    return summary_rows


def _load_aggregated_rows(path: Path) -> List[Dict[str, object]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _core_backend_summary_path(
    stage2_dir: Path,
    *,
    code: str,
    train_size: int,
    seed: int,
    backend: str,
) -> Path:
    return stage2_dir / f"s2_{code}_n{train_size}_seed{seed}_{backend}_summary.csv"


def _local_core_backend_summary_path(
    stage2_dir: Path,
    *,
    code: str,
    train_size: int,
    seed: int,
    backend: str,
    local_k_neighbors: int,
) -> Path:
    prefix = (
        f"s2_{code}"
        f"_n{int(train_size)}"
        f"_seed{int(seed)}"
        f"_{backend}"
        f"_lknnx{int(local_k_neighbors)}"
    )
    return stage2_dir / f"{prefix}_summary.csv"


def _core_backend_prediction_path(
    stage2_dir: Path,
    *,
    code: str,
    train_size: int,
    seed: int,
    backend: str,
) -> Path:
    backend_suffix = "" if backend == "tabpfn" else f"_{backend}"
    return stage2_dir / f"s2_{code}_n{train_size}_seed{seed}{backend_suffix}_predictions.csv"


def _read_trimmed_mse_from_predictions(
    predictions_path: Path,
    *,
    trim_quantile_range: Tuple[float, float] | None,
) -> Tuple[float, int]:
    mse_values: List[float] = []
    lower: float | None
    upper: float | None
    if trim_quantile_range is None:
        lower = None
        upper = None
    else:
        lower, upper = trim_quantile_range

    with predictions_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if lower is not None and upper is not None:
                x_quantile = float(row["X_quantile"])
                if x_quantile < lower or x_quantile > upper:
                    continue
            mse_values.append(float(row["mse_per_x"]))

    if not mse_values:
        raise ValueError(f"No retained prediction rows found in {predictions_path}")
    return (sum(mse_values) / len(mse_values), len(mse_values))


def _core_backend_records(
    *,
    stage2_dir: Path,
    codes: Sequence[str],
    train_sizes: Sequence[int],
    seeds: Sequence[int],
    backends: Sequence[str],
) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for backend in backends:
        for code in codes:
            scenario = utils.CODE_SCENARIO_MAP[code]["scenario"]
            for train_size in train_sizes:
                for seed in seeds:
                    summary_path = _core_backend_summary_path(
                        stage2_dir,
                        code=code,
                        train_size=int(train_size),
                        seed=int(seed),
                        backend=backend,
                    )
                    if not summary_path.exists():
                        raise FileNotFoundError(f"Missing backend summary: {summary_path}")
                    summary = _read_summary_kv(summary_path)
                    metric = summary.get("metric_mse_do_pred_vs_clean")
                    if metric is None:
                        raise KeyError(f"metric_mse_do_pred_vs_clean not found in {summary_path}")
                    records.append(
                        {
                            "model": backend,
                            "variant": "stage2_9",
                            "code": code,
                            "scenario": scenario,
                            "train_size": int(train_size),
                            "seed": int(seed),
                            "mse_vs_mean_int": float(metric),
                            "metrics_path": str(summary_path),
                        }
                    )
    return records


def _posthoc_trimmed_core_backend_records(
    *,
    stage2_dir: Path,
    codes: Sequence[str],
    train_sizes: Sequence[int],
    seeds: Sequence[int],
    backends: Sequence[str],
    trim_quantile_range: Tuple[float, float] | None,
) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for backend in backends:
        for code in codes:
            scenario = utils.CODE_SCENARIO_MAP[code]["scenario"]
            for train_size in train_sizes:
                for seed in seeds:
                    predictions_path = _core_backend_prediction_path(
                        stage2_dir,
                        code=code,
                        train_size=int(train_size),
                        seed=int(seed),
                        backend=backend,
                    )
                    if not predictions_path.exists():
                        raise FileNotFoundError(f"Missing backend predictions: {predictions_path}")
                    mse_val, retained_n = _read_trimmed_mse_from_predictions(
                        predictions_path,
                        trim_quantile_range=trim_quantile_range,
                    )
                    records.append(
                        {
                            "model": backend,
                            "variant": "stage2_9",
                            "code": code,
                            "scenario": scenario,
                            "train_size": int(train_size),
                            "seed": int(seed),
                            "mse_vs_mean_int": float(mse_val),
                            "metrics_path": str(predictions_path),
                            "n_test": int(retained_n),
                        }
                    )
    return records


def _local_model_name_for_backend(backend: str) -> str:
    return f"{backend}_local"


def _local_core_backend_records(
    *,
    stage2_dir: Path,
    codes: Sequence[str],
    train_sizes: Sequence[int],
    seeds: Sequence[int],
    backends: Sequence[str],
    local_k_neighbors: int,
) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for backend in backends:
        for code in codes:
            scenario = utils.CODE_SCENARIO_MAP[code]["scenario"]
            for train_size in train_sizes:
                for seed in seeds:
                    summary_path = _local_core_backend_summary_path(
                        stage2_dir,
                        code=code,
                        train_size=int(train_size),
                        seed=int(seed),
                        backend=backend,
                        local_k_neighbors=int(local_k_neighbors),
                    )
                    if not summary_path.exists():
                        raise FileNotFoundError(f"Missing local backend summary: {summary_path}")
                    summary = _read_summary_kv(summary_path)
                    metric = summary.get("metric_mse_do_pred_vs_clean")
                    if metric is None:
                        raise KeyError(f"metric_mse_do_pred_vs_clean not found in {summary_path}")
                    records.append(
                        {
                            "model": _local_model_name_for_backend(backend),
                            "variant": "stage2_9_local_knn",
                            "code": code,
                            "scenario": scenario,
                            "train_size": int(train_size),
                            "seed": int(seed),
                            "mse_vs_mean_int": float(metric),
                            "metrics_path": str(summary_path),
                        }
                    )
    return records


def _merge_rows(
    base_rows: Sequence[Dict[str, object]],
    appended_rows: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    merged_rows = list(base_rows)
    merged_rows.extend(appended_rows)
    return merged_rows


def _save_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate sec5.1 baseline MSEs and core-backend references.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=utils.DEFAULT_MANIFEST_PATH,
        help="Manifest produced by manifest_manager.py / generate_manifest.py.",
    )
    parser.add_argument(
        "--filter-baselines-to-manifest",
        action="store_true",
        help="Restrict baseline metrics to runs present in the manifest.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=utils.DEFAULT_RESULTS_DIR,
        help="Directory containing baseline metrics JSON files.",
    )
    parser.add_argument("--include-tabpfn", action="store_true", help="Include TabPFN Stage2 MSE reference.")
    parser.add_argument(
        "--base-csv",
        type=Path,
        default=None,
        help="Optional existing aggregated CSV to extend without recomputing legacy rows.",
    )
    parser.add_argument(
        "--append-core-backends",
        nargs="+",
        default=[],
        help="Append aggregated rows from core Stage2 summaries for these backends.",
    )
    parser.add_argument(
        "--append-core-backends-posthoc-trim",
        nargs="+",
        default=[],
        help=(
            "Append aggregated rows from core Stage2 prediction CSVs after posthoc trimming "
            "by X quantile for these backends."
        ),
    )
    parser.add_argument(
        "--append-local-core-backends",
        nargs="+",
        default=[],
        help="Append aggregated rows from local-knn Stage2 summaries for these backends.",
    )
    parser.add_argument(
        "--stage2-dir",
        type=Path,
        default=utils.DEFAULT_STAGE2_DIR,
        help="Directory containing core Stage2 summary CSVs.",
    )
    parser.add_argument(
        "--train-sizes",
        nargs="+",
        type=int,
        default=utils.TRAIN_SIZES,
        help="Training sizes to aggregate when appending core backends.",
    )
    parser.add_argument(
        "--append-seeds",
        nargs="+",
        type=int,
        default=None,
        help="Seed list to aggregate when appending core backends.",
    )
    parser.add_argument(
        "--posthoc-trim-quantiles",
        nargs=2,
        type=float,
        default=None,
        metavar=("LOWER", "UPPER"),
        help="Optional X-quantile range retained when aggregating from Stage2 prediction CSVs.",
    )
    parser.add_argument(
        "--local-k-neighbors",
        type=int,
        default=None,
        help="Required when appending local core backends, to resolve the _lknn{k} summary filenames.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "deepiv",
            "deepgmm",
            "hsic",
            "tabpfn_cf",
            "tabpfn_naive",
            "linear_iv",
            "nonlinear_iv",
            "div",
            "div_2",
        ],
        help="Filter to these baseline models.",
    )
    parser.add_argument("--codes", nargs="+", default=None, help="Optional subset of DGP codes.")
    parser.add_argument(
        "--output",
        type=Path,
        default=utils.DEFAULT_AGGREGATED_DIR / "aggregated_interv_mean.csv",
        help="Path to save aggregated CSV.",
    )
    args = parser.parse_args()

    if args.base_csv is not None:
        if not args.base_csv.exists():
            raise SystemExit(f"Base aggregated CSV not found: {args.base_csv}")
        if (
            not args.append_core_backends
            and not args.append_core_backends_posthoc_trim
            and not args.append_local_core_backends
        ):
            raise SystemExit(
                "--base-csv requires --append-core-backends, --append-core-backends-posthoc-trim, "
                "and/or --append-local-core-backends."
            )
        if not args.codes:
            raise SystemExit("--base-csv merge mode requires --codes.")
        if args.append_seeds is None:
            raise SystemExit("--base-csv merge mode requires --append-seeds.")
        if args.append_local_core_backends and args.local_k_neighbors is None:
            raise SystemExit("--append-local-core-backends requires --local-k-neighbors.")

        base_rows = _load_aggregated_rows(args.base_csv)
        append_records: List[Dict[str, object]] = []
        if args.append_core_backends:
            append_records.extend(
                _core_backend_records(
                    stage2_dir=args.stage2_dir,
                    codes=args.codes,
                    train_sizes=args.train_sizes,
                    seeds=args.append_seeds,
                    backends=args.append_core_backends,
                )
            )
        if args.append_core_backends_posthoc_trim:
            append_records.extend(
                _posthoc_trimmed_core_backend_records(
                    stage2_dir=args.stage2_dir,
                    codes=args.codes,
                    train_sizes=args.train_sizes,
                    seeds=args.append_seeds,
                    backends=args.append_core_backends_posthoc_trim,
                    trim_quantile_range=(
                        tuple(float(v) for v in args.posthoc_trim_quantiles)
                        if args.posthoc_trim_quantiles is not None
                        else None
                    ),
                )
            )
        if args.append_local_core_backends:
            append_records.extend(
                _local_core_backend_records(
                    stage2_dir=args.stage2_dir,
                    codes=args.codes,
                    train_sizes=args.train_sizes,
                    seeds=args.append_seeds,
                    backends=args.append_local_core_backends,
                    local_k_neighbors=int(args.local_k_neighbors),
                )
            )
        append_rows = _aggregate(append_records)
        merged_rows = _merge_rows(base_rows, append_rows)

        print("=== Appended core backend rows ===")
        for row in append_rows:
            print(
                f"{row['model']:11s} {row['variant']:15s} {row['code']:6s} "
                f"n={row['train_size']:5d} | mean={row['mean_mse']:.6f} std={row['std_mse']:.6f} (n={row['n_runs']})"
            )

        _save_csv(args.output, merged_rows)
        print(f"\nMerged aggregated CSV saved to {args.output}")
        return

    manifest = utils.load_manifest(args.manifest)
    runs = manifest.get("runs", []) if isinstance(manifest, dict) else []

    requested_models = list(args.models)
    baseline_records = _load_baseline_metrics(
        args.results_dir,
        models=requested_models,
        codes=args.codes,
        train_sizes=args.train_sizes,
        seeds=args.append_seeds,
    )
    if args.append_core_backends_posthoc_trim:
        if not args.codes:
            raise SystemExit("--append-core-backends-posthoc-trim requires --codes.")
        if args.append_seeds is None:
            raise SystemExit("--append-core-backends-posthoc-trim requires --append-seeds.")
        baseline_records.extend(
            _posthoc_trimmed_core_backend_records(
                stage2_dir=args.stage2_dir,
                codes=args.codes,
                train_sizes=args.train_sizes,
                seeds=args.append_seeds,
                backends=args.append_core_backends_posthoc_trim,
                trim_quantile_range=(
                    tuple(float(v) for v in args.posthoc_trim_quantiles)
                    if args.posthoc_trim_quantiles is not None
                    else None
                ),
            )
        )

    if args.filter_baselines_to_manifest and runs:
        allowed_run_keys = set(_manifest_run_keys(runs))
        if allowed_run_keys:
            filtered: List[Dict[str, object]] = []
            for rec in baseline_records:
                try:
                    key = (str(rec["code"]), int(rec["train_size"]), int(rec["seed"]))
                except Exception:
                    continue
                if key in allowed_run_keys:
                    filtered.append(rec)
            baseline_records = filtered

    if args.include_tabpfn and runs:
        baseline_records.extend(_tabpfn_records(manifest, runs))

    if args.codes:
        code_allowlist = set(args.codes)
        baseline_records = [rec for rec in baseline_records if rec.get("code") in code_allowlist]

    model_allowlist = set(args.models)
    if args.include_tabpfn:
        model_allowlist.add("tabpfn")
    model_allowlist.update(args.append_core_backends_posthoc_trim)
    baseline_records = [rec for rec in baseline_records if rec.get("model") in model_allowlist]

    if not baseline_records:
        raise SystemExit("No baseline metrics found to aggregate.")

    summary_rows = _aggregate(baseline_records)

    print("=== Aggregated MSE (mean/std) ===")
    for row in summary_rows:
        print(
            f"{row['model']:8s} {row['variant']:15s} {row['code']:6s} "
            f"n={row['train_size']:5d} | mean={row['mean_mse']:.6f} std={row['std_mse']:.6f} (n={row['n_runs']})"
        )

    _save_csv(args.output, summary_rows)
    print(f"\nAggregated CSV saved to {args.output}")


if __name__ == "__main__":
    main()
