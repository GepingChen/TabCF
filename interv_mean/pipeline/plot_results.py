#!/usr/bin/env python3
"""
Visualize sec5.1 aggregated results as a grid (scenario x training size).

The script expects the aggregated CSV under interv_mean/io/aggregated
and compares baseline methods plus TabPFN. It includes R baselines
(linear IV, nonlinear IV, DIV (CRAN)) when present.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import namedtuple
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from interv_mean.pipeline import utils


Method = namedtuple("Method", ["label", "model", "variant"])

# Variants to plot (labels kept short for the x-axis).
METHODS: List[Method] = [
    Method("TabCF", "tabpfn", "stage2_9"),
    Method("DIV", "div_2", "div_cran_layer3_epoch1000"),
    Method("DeepGMM", "deepgmm", "deepgmm"),
    Method("DeepIV", "deepiv", "deepiv_restarts10"),
    Method("Linear CF", "linear_iv", "control_function_linear"),
    Method("Nonlinear CF", "nonlinear_iv", "control_function_spline_df5"),
    Method("TabPFN-CF", "tabpfn_cf", "tabpfn_cf_resid0"),
    Method("TabPFN-Naive", "tabpfn_naive", "tabpfn_naive_y_on_x"),
]

# Distinct shapes/colors for the legend.
METHOD_STYLES: Dict[str, Dict[str, object]] = {
    "DeepGMM": {"marker": "s", "color": "#6ab187", "edgecolor": "white"},
    "DeepIV": {"marker": "*", "color": "#f1c232", "edgecolor": "#f1c232"},
    "Linear CF": {"marker": "v", "color": "#3b8adb", "edgecolor": "white"},
    "Nonlinear CF": {"marker": "D", "color": "#ff7f0e", "edgecolor": "white"},
    "DIV": {"marker": "p", "color": "#9c755f", "edgecolor": "white"},
    "TabPFN-CF": {"marker": "X", "color": "#9b59b6", "edgecolor": "white"},
    "TabPFN-Naive": {"marker": "P", "color": "#7f8c8d", "edgecolor": "white"},
    "TabCF": {"marker": "o", "color": "#4c88ff", "edgecolor": "white"},
}

DEFAULT_CSV = utils.DEFAULT_AGGREGATED_DIR / "aggregated_interv_mean.csv"
DEFAULT_OUT = utils.DEFAULT_VISUALIZATION_DIR / "mean_benchmark_figure.png"
DEFAULT_TRAIN_SIZES = [1000, 4000, 10000]
DEFAULT_CODES = [
    "A3_B3",
    "A3_B4",
    "A3_B5",
    "A3_B6",
    "A5_B3",
    "A5_B4",
    "A5_B5",
    "A5_B6",
    "A9_B3",
    "A9_B4",
    "A9_B5",
]


def load_records(csv_path: Path) -> Tuple[List[dict], Dict[str, str]]:
    """Load all rows from the aggregated CSV."""
    records: List[dict] = []
    scenario_title: Dict[str, str] = {}

    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                train_size = int(row["train_size"])
                mean_mse = float(row["mean_mse"])
                std_mse = float(row["std_mse"])
            except (ValueError, KeyError):
                continue

            record = {
                "model": row.get("model", ""),
                "variant": row.get("variant", ""),
                "code": row.get("code", ""),
                "scenario": row.get("scenario", ""),
                "train_size": train_size,
                "mean_mse": mean_mse,
                "std_mse": std_mse,
            }
            # Store the scenario string once per code for labeling.
            scenario_title.setdefault(record["code"], record["scenario"])
            records.append(record)

    return records, scenario_title


def _read_stage2_mse(summary_path: Path) -> float:
    with summary_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("key") == "metric_mse_do_pred_vs_clean":
                return float(row["value"])
    raise KeyError(f"metric_mse_do_pred_vs_clean not found in {summary_path}")


def _supplement_tabcf_records(
    records: List[dict],
    scenario_title: Dict[str, str],
    *,
    codes: Optional[Sequence[str]] = None,
    train_sizes: Optional[Sequence[int]] = None,
    stage2_dir: Path = utils.DEFAULT_STAGE2_DIR,
) -> List[dict]:
    """Backfill missing TabCF rows from stage2 summary CSVs when the aggregated CSV omitted them."""
    existing_keys = {
        (str(rec["code"]), int(rec["train_size"]))
        for rec in records
        if rec.get("model") == "tabpfn" and rec.get("variant") == "stage2_9"
    }
    code_filter = set(codes) if codes else None
    size_filter = {int(size) for size in train_sizes} if train_sizes else None

    grouped: Dict[Tuple[str, int], List[float]] = {}
    indexed = utils._index_stage2_summaries(stage2_dir)
    for (code, size, _seed), summary_path in indexed.items():
        if code_filter and code not in code_filter:
            continue
        if size_filter and size not in size_filter:
            continue
        if (code, size) in existing_keys:
            continue
        try:
            grouped.setdefault((code, size), []).append(_read_stage2_mse(summary_path))
        except Exception as exc:
            print(f"Skipping TabCF fallback {summary_path}: {exc}")

    supplemented: List[dict] = []
    for (code, size), values in sorted(grouped.items()):
        if not values:
            continue
        mean_val = sum(values) / len(values)
        var_val = sum((value - mean_val) ** 2 for value in values) / len(values)
        scenario = scenario_title.get(code) or utils.CODE_SCENARIO_MAP.get(code, {}).get("scenario", "")
        scenario_title.setdefault(code, scenario)
        supplemented.append(
            {
                "model": "tabpfn",
                "variant": "stage2_9",
                "code": code,
                "scenario": scenario,
                "train_size": size,
                "mean_mse": mean_val,
                "std_mse": math.sqrt(var_val),
            }
        )
    return supplemented


def build_lookup(records: Iterable[dict]) -> Dict[Tuple[str, int, str, str], dict]:
    """Create a quick lookup by (code, train_size, model, variant)."""
    lookup: Dict[Tuple[str, int, str, str], dict] = {}
    for rec in records:
        key = (rec["code"], rec["train_size"], rec["model"], rec["variant"])
        lookup[key] = rec
    return lookup


def _expand_codes(codes: Iterable[str]) -> List[str]:
    """Allow shorthand like `A3`/`A5` to mean `A3_B3..B6`."""
    expanded: List[str] = []
    for raw in codes:
        token = raw.strip()
        if not token:
            continue
        if token.lower() == "all":
            return ["all"]
        if token.upper() in {"A3", "A4", "A5"}:
            a = token.upper()
            for b in ("B3", "B4", "B5", "B6"):
                expanded.append(f"{a}_{b}")
        else:
            expanded.append(token)
    # de-duplicate while preserving order
    seen = set()
    deduped: List[str] = []
    for code in expanded:
        if code not in seen:
            seen.add(code)
            deduped.append(code)
    return deduped


def _normalize_method_token(token: str) -> str:
    """Normalize method tokens for CLI matching (case/space/punct insensitive)."""
    return re.sub(r"[^a-z0-9]+", "", token.strip().lower())


def _filter_methods(methods: Sequence[Method], drop_tokens: Sequence[str]) -> List[Method]:
    tokens: Set[str] = {_normalize_method_token(t) for t in drop_tokens if t.strip()}
    if not tokens:
        return list(methods)

    kept: List[Method] = []
    for method in methods:
        candidates = {
            _normalize_method_token(method.label),
            _normalize_method_token(method.model),
            _normalize_method_token(method.variant),
        }
        if candidates & tokens:
            continue
        kept.append(method)

    if not kept:
        raise ValueError(f"All methods removed by --drop-methods {list(drop_tokens)}")
    return kept


def plot_grid(
    records: List[dict],
    scenario_title: Dict[str, str],
    output_path: Path,
    *,
    codes: Optional[List[str]] = None,
    methods: Optional[Sequence[Method]] = None,
    dpi: int = 300,
) -> None:
    """Render the 3x3 grid and save to disk."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to run this visualization script.") from exc

    train_sizes = sorted({rec["train_size"] for rec in records})
    if not codes:
        codes = sorted({rec["code"] for rec in records})
    if methods is None:
        methods = METHODS
    lookup = build_lookup(records)

    # Keep single-row exports compact; multi-row grids can stay roomier.
    n_rows = len(codes)
    n_cols = len(train_sizes)
    width = max(10.0, 3.6 * n_cols)
    if n_rows == 1:
        height = 3.1
    elif n_rows == 2:
        height = 5.4
    else:
        height = max(8.0, 2.6 * n_rows)

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(width, height), sharex=True, sharey="row", squeeze=False
    )

    missing: List[Tuple[str, int, str]] = []
    for row_idx, code in enumerate(codes):
        for col_idx, train_size in enumerate(train_sizes):
            ax = axes[row_idx][col_idx]

            for method_idx, method in enumerate(methods):
                key = (code, train_size, method.model, method.variant)
                rec = lookup.get(key)
                if rec is None:
                    missing.append((code, train_size, method.label))
                    continue

                style = METHOD_STYLES[method.label]
                ax.scatter(
                    method_idx,
                    rec["mean_mse"],
                    s=120,
                    marker=style["marker"],
                    color=style["color"],
                    edgecolor=style["edgecolor"],
                    linewidth=1.0,
                    label=method.label,
                    zorder=3,
                )

            ax.set_xlim(-0.6, len(methods) - 0.4)
            ax.grid(True, axis="y", linestyle="--", alpha=0.35, zorder=0)

            if row_idx == len(codes) - 1:
                ax.set_xticks(range(len(methods)))
                ax.set_xticklabels([m.label for m in methods], rotation=25, ha="right")
            else:
                ax.set_xticks([])

            if col_idx == 0:
                scenario_text = scenario_title.get(code, code)
                label = scenario_text if scenario_text == code else f"{code}\n{scenario_text}"
                ax.set_ylabel(f"{label}\nmean MSE")

            if row_idx == 0:
                ax.set_title(f"n = {train_size}")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)

    if missing:
        # Collapsed list of any gaps to make it easy to spot missing runs.
        print("Missing combinations (not plotted):")
        for code, size, method in missing:
            print(f"  {code}, n={size}: {method}")
    print(f"Saved figure to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot sec5.1 aggregated results (scenario x training size)."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help="Path to aggregated_interv_mean_vis.csv (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUT,
        help="Output image path (default: %(default)s)",
    )
    parser.add_argument(
        "--dpi", type=int, default=300, help="Figure DPI for the saved image."
    )
    parser.add_argument(
        "--train-sizes",
        nargs="+",
        type=int,
        default=DEFAULT_TRAIN_SIZES,
        help=(
            "Training sizes to plot. Default: 1000 4000 10000. "
            "Example: --train-sizes 4000 10000"
        ),
    )
    parser.add_argument(
        "--codes",
        nargs="+",
        default=DEFAULT_CODES,
        help=(
            "Which DGP codes to plot. Default plots A3_B3..B6 and A5_B3..B6. "
            "You can also pass `A3`/`A5` shorthand or `all`."
        ),
    )
    parser.add_argument(
        "--drop-methods",
        nargs="+",
        default=[],
        help=(
            "Remove one or more methods from the plot by label/model/variant "
            "(case/space/punct insensitive). Example: --drop-methods linear_cf"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records, scenario_title = load_records(args.csv)
    if not records:
        raise SystemExit(f"No rows loaded from {args.csv}")
    train_sizes = sorted({int(size) for size in args.train_sizes})
    if not train_sizes:
        raise SystemExit("No training sizes requested.")
    records = [rec for rec in records if int(rec["train_size"]) in set(train_sizes)]
    if not records:
        raise SystemExit(
            f"No rows in {args.csv} match the requested train sizes {train_sizes}"
        )
    codes = _expand_codes(args.codes)
    if len(codes) == 1 and codes[0].lower() == "all":
        codes = None
    supplemented = _supplement_tabcf_records(
        records,
        scenario_title,
        codes=codes,
        train_sizes=train_sizes,
    )
    if supplemented:
        records.extend(supplemented)
        print(f"Supplemented {len(supplemented)} TabCF aggregate rows from {utils.DEFAULT_STAGE2_DIR}")
    methods = _filter_methods(METHODS, args.drop_methods)
    plot_grid(records, scenario_title, args.output, codes=codes, methods=methods, dpi=args.dpi)


if __name__ == "__main__":
    main()
