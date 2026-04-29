#!/usr/bin/env python3
"""Run real-data interventional quantile comparisons for shared benchmarks."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.lines import Line2D


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]
CORE_DIR = REPO_ROOT / "tabcf_core"

os.environ.setdefault("TABPFN_MODEL_VERSION", "v2.5")
os.environ.setdefault("TABPFN_MODEL_CACHE_DIR", str(REPO_ROOT / "tabpfn_home_config" / "models"))

if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from empirical import run_empirical_mean as scalar_pipeline


DEFAULT_R_SCRIPT = SCRIPT_PATH.with_name("run_empirical_quantile_baselines.R")
DEFAULT_TAUS: tuple[float, ...] = (0.15, 0.25, 0.50, 0.75, 0.85)
DEFAULT_Y_GRID_POINTS = 1201
DEFAULT_Y_GRID_PADDING = 0.25
DEFAULT_IVQR_GRID_POINTS = 201
DEFAULT_R_MODULE = "r/4.4.3-py311-xspgsan"
DEFAULT_DIV_EPOCHS = 1000
DEFAULT_DIV_LAYERS = 4
DEFAULT_DIV_LR = 1e-4
DEFAULT_DIV_NSAMPLE = 1000
DEFAULT_DIV_SEED = 1
PLOT_FONT_FAMILY = "Latin Modern Roman"
FONT_SCALE = 1.42


def _register_latin_modern_fonts() -> None:
    tectonic_cache = Path.home() / ".cache" / "Tectonic" / "bundles" / "data"
    if not tectonic_cache.exists():
        return
    for font_path in sorted(tectonic_cache.glob("*/lmroman*.otf")):
        font_manager.fontManager.addfont(str(font_path))


def _apply_official_plot_style() -> None:
    _register_latin_modern_fonts()
    font_path = font_manager.findfont(PLOT_FONT_FAMILY, fallback_to_default=False)
    plt.rcParams.update(
        {
            "font.family": PLOT_FONT_FAMILY,
            "font.serif": [PLOT_FONT_FAMILY],
            "mathtext.fontset": "cm",
            "axes.formatter.use_mathtext": True,
            "axes.unicode_minus": False,
            "font.size": 11.0 * FONT_SCALE,
            "axes.labelsize": 13.5 * FONT_SCALE,
            "axes.titlesize": 13.0 * FONT_SCALE,
            "xtick.labelsize": 10.0 * FONT_SCALE,
            "ytick.labelsize": 10.0 * FONT_SCALE,
            "legend.fontsize": 10.5 * FONT_SCALE,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    print(f"Using plot font: {PLOT_FONT_FAMILY} ({font_path})")


def _style_data_panel(ax: plt.Axes) -> None:
    ax.set_facecolor("#ffffff")
    ax.grid(True, axis="y", color="#d9ddd6", linewidth=0.8, alpha=0.95)
    ax.grid(True, axis="x", color="#eceee9", linewidth=0.75, alpha=0.95)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=9.8 * FONT_SCALE, length=0, pad=6, colors="#4a4a46")
    ax.tick_params(axis="x", labelsize=9.8 * FONT_SCALE, length=0, pad=6, colors="#4a4a46")
    for spine in ax.spines.values():
        spine.set_visible(False)


@dataclass(frozen=True)
class QuantileInstrumentSpec:
    slug: str
    display_name: str
    source_column: str


@dataclass(frozen=True)
class QuantileDatasetSpec:
    slug: str
    display_name: str
    default_data_path: Path
    default_out_dir: Path
    output_prefix: str
    source_url: str | None
    canonical_mapping: Mapping[str, str]
    x_label_log: str
    y_label_log: str
    x_label_orig: str
    y_label_orig: str
    x_transform: Callable[[np.ndarray], np.ndarray]
    y_transform: Callable[[np.ndarray], np.ndarray]
    instrument_specs: Mapping[str, QuantileInstrumentSpec]
    default_instrument_spec: str
    grid_mode: str = "linspace"
    discrete_x: bool = False


@dataclass(frozen=True)
class TabCFQuantileResult:
    backend_name: str
    v_hat: np.ndarray
    quantiles: np.ndarray
    y_grid_min: float
    y_grid_max: float


def _exp_transform(values: np.ndarray) -> np.ndarray:
    return np.exp(np.asarray(values, dtype=float))


FULTON_WAVE2_SPEC = QuantileInstrumentSpec(
    slug="wave2",
    display_name="Wave height (wave2)",
    source_column="wave2",
)

DATASET_SPECS: dict[str, QuantileDatasetSpec] = {
    "fulton": QuantileDatasetSpec(
        slug="fulton",
        display_name="Fulton Fish Market",
        default_data_path=scalar_pipeline.DEFAULT_FULTON_DATA_PATH,
        default_out_dir=REPO_ROOT / "empirical" / "outputs_fulton_fish_quantile",
        output_prefix="fulton_fish",
        source_url=scalar_pipeline.FULTON_SOURCE_URL,
        canonical_mapping={
            "unit": "rownames or market_day_<id>",
            "X": "lavgprc",
            "Z": "wave2",
            "Y": "ltotqty",
        },
        x_label_log="Log average fish price",
        y_label_log="Log total quantity sold",
        x_label_orig="Average fish price",
        y_label_orig="Total quantity sold",
        x_transform=_exp_transform,
        y_transform=_exp_transform,
        instrument_specs={"wave2": FULTON_WAVE2_SPEC},
        default_instrument_spec="wave2",
    ),
}


def parse_taus_arg(raw_value: str | None) -> tuple[float, ...]:
    if raw_value is None or not raw_value.strip():
        return DEFAULT_TAUS

    values: list[float] = []
    for token in raw_value.split(","):
        text = token.strip()
        if not text:
            continue
        tau = float(text)
        if tau <= 0.0 or tau >= 1.0:
            raise ValueError(f"Invalid tau '{tau}'. Quantiles must lie strictly between 0 and 1.")
        values.append(tau)

    if not values:
        raise ValueError("At least one quantile level is required.")
    return tuple(values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASET_SPECS), default="fulton")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Dataset-specific default path is used when omitted.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Dataset-specific default output directory is used when omitted.",
    )
    parser.add_argument(
        "--instrument-spec",
        type=str,
        default=None,
        help="Dataset-specific instrument choice. Fulton currently supports: wave2.",
    )
    parser.add_argument(
        "--taus",
        type=str,
        default=",".join(f"{tau:.2f}" for tau in DEFAULT_TAUS),
        help="Comma-separated quantile levels. Default: 0.15,0.25,0.50,0.75,0.85",
    )
    parser.add_argument("--grid-points", type=int, default=100)
    parser.add_argument("--n-v-points", type=int, default=101)
    parser.add_argument("--y-grid-points", type=int, default=DEFAULT_Y_GRID_POINTS)
    parser.add_argument("--y-grid-padding", type=float, default=DEFAULT_Y_GRID_PADDING)
    parser.add_argument(
        "--core-backends",
        nargs="+",
        default=[scalar_pipeline.TABPFN_BACKEND],
        help="Core backends to fit. Supported in this quantile pipeline: tabpfn.",
    )
    parser.add_argument("--r-script", type=Path, default=DEFAULT_R_SCRIPT)
    parser.add_argument("--r-module", type=str, default=DEFAULT_R_MODULE)
    parser.add_argument("--ivqr-grid-min", type=float, default=None)
    parser.add_argument("--ivqr-grid-max", type=float, default=None)
    parser.add_argument("--ivqr-grid-points", type=int, default=DEFAULT_IVQR_GRID_POINTS)
    parser.add_argument("--ivqr-seed", type=int, default=1)
    parser.add_argument("--div-epochs", type=int, default=DEFAULT_DIV_EPOCHS)
    parser.add_argument("--div-layers", type=int, default=DEFAULT_DIV_LAYERS)
    parser.add_argument("--div-lr", type=float, default=DEFAULT_DIV_LR)
    parser.add_argument("--div-nsample", type=int, default=DEFAULT_DIV_NSAMPLE)
    parser.add_argument("--div-seed", type=int, default=DEFAULT_DIV_SEED)
    return parser.parse_args()


def resolve_dataset_args(
    args: argparse.Namespace,
) -> tuple[QuantileDatasetSpec, Path, Path, bool]:
    spec = DATASET_SPECS[args.dataset]
    data_path = args.data_path.resolve() if args.data_path is not None else spec.default_data_path.resolve()
    out_dir = args.out_dir.resolve() if args.out_dir is not None else spec.default_out_dir.resolve()
    allow_download = args.data_path is None and spec.source_url is not None
    return spec, data_path, out_dir, allow_download


def resolve_instrument_spec(
    spec: QuantileDatasetSpec,
    instrument_slug: str | None,
) -> QuantileInstrumentSpec:
    selected = spec.default_instrument_spec if instrument_slug is None else instrument_slug.strip().lower()
    if selected not in spec.instrument_specs:
        valid = ", ".join(sorted(spec.instrument_specs))
        raise ValueError(
            f"Unsupported instrument spec '{instrument_slug}' for dataset '{spec.slug}'. Valid options: {valid}."
        )
    return spec.instrument_specs[selected]


def transform_values(
    transform: Callable[[np.ndarray], np.ndarray],
    values: Sequence[float] | np.ndarray,
) -> np.ndarray:
    return np.asarray(transform(np.asarray(values, dtype=float)), dtype=float)


def build_output_paths(spec: QuantileDatasetSpec, out_dir: Path) -> dict[str, Path]:
    prefix = f"{spec.output_prefix}_quantile"
    return {
        "clean_csv": out_dir / f"{prefix}_clean.csv",
        "grid_csv": out_dir / f"{prefix}_x_grid.csv",
        "curves_csv": out_dir / f"{prefix}_curves.csv",
        "coefficients_csv": out_dir / f"{prefix}_coefficients.csv",
        "runtime_csv": out_dir / f"{prefix}_runtime.csv",
        "diagnostics_json": out_dir / f"{prefix}_diagnostics.json",
        "png_path": out_dir / f"{prefix}_figure.png",
        "pdf_path": out_dir / f"{prefix}_figure.pdf",
    }


def load_fulton_quantile_data(
    path: Path,
    *,
    instrument_spec: QuantileInstrumentSpec,
    allow_download: bool,
) -> pd.DataFrame:
    if not path.exists():
        if allow_download:
            print(f"Downloading Fulton Fish Market data to: {path}")
            scalar_pipeline.download_url_to_file(scalar_pipeline.FULTON_SOURCE_URL, path)
        else:
            raise FileNotFoundError(f"Data file not found: {path}")

    raw_df = pd.read_csv(path)
    scalar_pipeline.ensure_columns(
        raw_df,
        ("lavgprc", "ltotqty", instrument_spec.source_column),
        context="Fulton Fish CSV",
    )
    if len(raw_df) != scalar_pipeline.EXPECTED_FULTON_ROWS:
        raise ValueError(
            f"Unexpected Fulton Fish row count in {path}: "
            f"expected {scalar_pipeline.EXPECTED_FULTON_ROWS}, found {len(raw_df)}."
        )

    before = len(raw_df)
    canonical = pd.DataFrame(
        {
            "unit": scalar_pipeline._fulton_units(raw_df),
            "X": raw_df["lavgprc"],
            "Z": raw_df[instrument_spec.source_column],
            "Y": raw_df["ltotqty"],
        }
    )
    canonical = canonical.dropna(subset=["unit", "X", "Z", "Y"]).reset_index(drop=True)
    finite_mask = np.isfinite(canonical[["X", "Z", "Y"]].to_numpy(dtype=float)).all(axis=1)
    canonical = canonical.loc[finite_mask].reset_index(drop=True)
    dropped = before - len(canonical)
    if dropped:
        print(f"Dropped {dropped} Fulton rows with missing or non-finite values in unit/X/Z/Y.")

    if canonical.empty:
        raise ValueError("No usable Fulton Fish rows remained after canonical preprocessing.")

    canonical["x_orig"] = transform_values(_exp_transform, canonical["X"].to_numpy(dtype=float))
    canonical["y_orig"] = transform_values(_exp_transform, canonical["Y"].to_numpy(dtype=float))
    scalar_pipeline.summarize_loaded_data(canonical.loc[:, ["unit", "X", "Z", "Y"]], display_name=DATASET_SPECS["fulton"].display_name)
    return canonical


def load_dataset_frame(
    spec: QuantileDatasetSpec,
    path: Path,
    *,
    instrument_spec: QuantileInstrumentSpec,
    allow_download: bool,
) -> pd.DataFrame:
    if spec.slug == "fulton":
        return load_fulton_quantile_data(path, instrument_spec=instrument_spec, allow_download=allow_download)
    raise ValueError(f"Unsupported dataset: {spec.slug}")


def build_x_grid(df: pd.DataFrame, spec: QuantileDatasetSpec, requested_points: int) -> np.ndarray:
    x_values = df["X"].to_numpy(dtype=float)
    if spec.grid_mode == "unique":
        x_grid = np.sort(np.unique(x_values))
        if x_grid.size < 2:
            raise ValueError(f"{spec.display_name} needs at least two distinct X values; found {x_grid.size}.")
        return x_grid

    if requested_points < 2:
        raise ValueError("--grid-points must be at least 2.")
    return np.linspace(x_values.min(), x_values.max(), int(requested_points))


def compute_tabcf_quantile_curves(
    df: pd.DataFrame,
    x_grid: np.ndarray,
    taus: tuple[float, ...],
    *,
    n_v_points: int,
    y_grid_points: int,
    y_grid_padding: float,
    backend_name: str,
) -> TabCFQuantileResult:
    from stage1_control import CondCDFModel
    from stage2_outcome import ConditionalCDFEstimator
    from interv_qtl.cdf_to_quantiles import build_y_grid, compute_quantiles_on_grid

    backend = scalar_pipeline.normalize_backend_name(backend_name)
    x_train = df["X"].to_numpy(dtype=float)
    z_train = df["Z"].to_numpy(dtype=float)
    y_train = df["Y"].to_numpy(dtype=float)

    stage1_model = CondCDFModel(quantiles=(), backend_name=backend)
    stage1_model.fit(z_train, x_train, verbose=False)
    v_hat = np.clip(stage1_model.predict(z_train, x_train), 0.0, 1.0)
    if np.any(~np.isfinite(v_hat)):
        raise ValueError(f"Stage-1 V_hat contains non-finite values for backend '{backend}'.")

    cdf_model = ConditionalCDFEstimator(
        use_tabpfn=(backend != scalar_pipeline.TABICL_BACKEND),
        backend_name=backend,
    )
    cdf_model.fit_full(x_train, v_hat, y_train, verbose=False)

    y_grid = build_y_grid(
        y_train,
        padding=float(y_grid_padding),
        n_points=int(y_grid_points),
    )
    quantiles = compute_quantiles_on_grid(
        cdf_model,
        np.asarray(x_grid, dtype=float),
        y_grid,
        tuple(taus),
        int(n_v_points),
    )
    if np.any(~np.isfinite(quantiles)):
        raise ValueError(f"TabCF quantile curves contain non-finite values for backend '{backend}'.")

    return TabCFQuantileResult(
        backend_name=backend,
        v_hat=np.asarray(v_hat, dtype=float),
        quantiles=np.asarray(quantiles, dtype=float),
        y_grid_min=float(np.min(y_grid)),
        y_grid_max=float(np.max(y_grid)),
    )


def _run_shell_command(shell_cmd: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-lc", shell_cmd],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )


def run_quantile_r_helper(
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
    if not r_script.exists():
        raise FileNotFoundError(f"R helper not found: {r_script}")

    arg_parts = [
        "Rscript",
        str(r_script),
        "--data-csv",
        str(data_csv),
        "--grid-csv",
        str(grid_csv),
        "--curves-csv",
        str(curves_csv),
        "--coefficients-csv",
        str(coefficients_csv),
        "--runtime-csv",
        str(runtime_csv),
        "--taus",
        ",".join(f"{tau:g}" for tau in taus),
        "--ivqr-grid-points",
        str(int(ivqr_grid_points)),
        "--seed",
        str(int(seed)),
        "--div-num-epochs",
        str(int(div_epochs)),
        "--div-num-layer",
        str(int(div_layers)),
        "--div-lr",
        str(float(div_lr)),
        "--div-nsample",
        str(int(div_nsample)),
        "--div-seed",
        str(int(div_seed)),
    ]
    if ivqr_grid_min is not None:
        arg_parts.extend(["--ivqr-grid-min", str(float(ivqr_grid_min))])
    if ivqr_grid_max is not None:
        arg_parts.extend(["--ivqr-grid-max", str(float(ivqr_grid_max))])

    rscript_cmd = " ".join(shlex.quote(part) for part in arg_parts)
    if r_module:
        shell_cmd = (
            "source /etc/profile >/dev/null 2>&1 && "
            f"module load {shlex.quote(r_module)} >/dev/null 2>&1 && {rscript_cmd}"
        )
    else:
        shell_cmd = rscript_cmd

    print(f"Running QR/IVQR quantile baselines via R helper: {r_script}")
    completed = _run_shell_command(shell_cmd)
    if completed.returncode != 0 and r_module and shutil.which("Rscript") is not None:
        completed = _run_shell_command(rscript_cmd)

    if completed.stdout.strip():
        print(completed.stdout.strip())
    if completed.returncode != 0:
        raise RuntimeError(
            "Quantile R helper failed.\n"
            f"Command: {shell_cmd}\n"
            f"Stdout:\n{completed.stdout}\n"
            f"Stderr:\n{completed.stderr}"
        )


def validate_curves_frame(curves: pd.DataFrame) -> None:
    required = [
        "method",
        "backend_name",
        "estimand_family",
        "tau",
        "x_log",
        "x_orig",
        "q_pred_log",
        "q_pred_orig",
    ]
    missing = [column for column in required if column not in curves.columns]
    if missing:
        raise ValueError(f"Curves frame is missing required columns: {missing}")

    numeric = curves.loc[:, ["tau", "x_log", "x_orig", "q_pred_log", "q_pred_orig"]].to_numpy(dtype=float)
    if np.isnan(numeric).any() or not np.isfinite(numeric).all():
        raise ValueError("Curves frame contains NA or non-finite numeric values.")

    allowed = {"conditional_quantile", "interventional_quantile"}
    families = set(curves["estimand_family"].astype(str))
    if not families.issubset(allowed):
        raise ValueError(f"Unexpected estimand families in curves frame: {sorted(families - allowed)}")


def validate_coefficients_frame(coefficients: pd.DataFrame) -> None:
    required = ["method", "estimand_family", "tau", "intercept_log", "slope_log"]
    missing = [column for column in required if column not in coefficients.columns]
    if missing:
        raise ValueError(f"Coefficient frame is missing required columns: {missing}")

    numeric = coefficients.loc[:, ["tau", "intercept_log", "slope_log"]].to_numpy(dtype=float)
    if np.isnan(numeric).any() or not np.isfinite(numeric).all():
        raise ValueError("Coefficient frame contains NA or non-finite values.")


def validate_runtime_frame(runtime_df: pd.DataFrame) -> None:
    required = ["method", "backend_name", "estimand_family", "seconds"]
    missing = [column for column in required if column not in runtime_df.columns]
    if missing:
        raise ValueError(f"Runtime frame is missing required columns: {missing}")
    seconds = runtime_df["seconds"].to_numpy(dtype=float)
    if np.isnan(seconds).any() or np.any(seconds < 0.0):
        raise ValueError("Runtime frame contains invalid method durations.")


def _tau_color_map(taus: Sequence[float]) -> dict[float, tuple[float, float, float, float]]:
    unique_taus = sorted(float(tau) for tau in set(taus))
    color_values = plt.cm.viridis(np.linspace(0.12, 0.88, len(unique_taus)))
    return {tau: tuple(color) for tau, color in zip(unique_taus, color_values)}


def _method_style_map(methods: Sequence[str]) -> dict[str, str]:
    styles = ["-", "-.", ":"]
    mapping: dict[str, str] = {}
    remaining_styles = iter(styles)
    for method in methods:
        if method == "IVQR":
            mapping[method] = "--"
            continue
        try:
            style = next(remaining_styles)
        except StopIteration:
            style = "-"
        mapping[method] = style
    return mapping


def plot_quantile_curves(
    df: pd.DataFrame,
    curves: pd.DataFrame,
    *,
    spec: QuantileDatasetSpec,
    png_path: Path,
    pdf_path: Path,
) -> None:
    structural_curves = curves[curves["estimand_family"] == "interventional_quantile"].copy()
    tau_colors = _tau_color_map(curves["tau"].astype(float).unique())
    preferred_order = ("TabCF (TabPFNv2.5)", "DIV", "IVQR")
    available_methods = set(structural_curves["method"].astype(str))
    ordered_methods = [method for method in preferred_order if method in available_methods]
    ordered_methods.extend(
        method for method in structural_curves["method"].astype(str).drop_duplicates().tolist() if method not in ordered_methods
    )

    _apply_official_plot_style()

    fig, axes = plt.subplots(1, len(ordered_methods), figsize=(16.2, 6.0), sharex=True, sharey=True)
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes], dtype=object)

    scatter_kwargs = {"color": "#b3b3b3", "alpha": 0.55, "s": 34, "linewidths": 0}
    for ax, method in zip(axes, ordered_methods):
        method_df = structural_curves[structural_curves["method"] == method].copy()
        ax.scatter(df["X"], df["Y"], **scatter_kwargs)
        for tau, tau_df in method_df.groupby("tau", sort=True):
            color = tau_colors[float(tau)]
            ax.plot(
                tau_df["x_log"],
                tau_df["q_pred_log"],
                color=color,
                linewidth=2.2,
            )
        ax.set_title(method)
        ax.set_xlabel(spec.x_label_log)
        _style_data_panel(ax)

    axes[0].set_ylabel(spec.y_label_log)

    tau_handles = [
        Line2D([0], [0], color=tau_colors[float(tau)], linewidth=2.4, label=rf"$\tau={float(tau):.2f}$")
        for tau in sorted(tau_colors)
    ]

    fig.subplots_adjust(top=0.76, bottom=0.18, left=0.08, right=0.99, wspace=0.14)
    fig.legend(
        handles=tau_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=min(5, max(1, len(tau_handles))),
        frameon=True,
        fancybox=False,
        edgecolor="#5d5d57",
        borderpad=0.72,
        labelspacing=0.72,
        columnspacing=1.0,
    )
    legend = fig.legends[-1]
    legend.get_frame().set_alpha(1.0)
    legend.get_frame().set_linewidth(0.85)
    legend.get_frame().set_facecolor("#ffffff")

    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def compute_first_stage_relevance_summary(df: pd.DataFrame) -> dict[str, float]:
    x = df["X"].to_numpy(dtype=float)
    z = df["Z"].to_numpy(dtype=float)
    design = np.column_stack([np.ones_like(z), z])
    beta, *_ = np.linalg.lstsq(design, x, rcond=None)
    fitted = design @ beta
    sse = float(np.sum((x - fitted) ** 2))
    sst = float(np.sum((x - np.mean(x)) ** 2))
    r_squared = 0.0 if sst <= 0.0 else float(1.0 - (sse / sst))
    df_num = max(len(x) - design.shape[1], 1)
    ssr = max(sst - sse, 0.0)
    numerator_df = max(design.shape[1] - 1, 1)
    f_statistic = float((ssr / numerator_df) / (sse / df_num)) if sse > 0.0 else float("inf")

    return {
        "intercept": float(beta[0]),
        "slope": float(beta[1]),
        "r_squared": r_squared,
        "f_statistic": f_statistic,
    }


def compute_support_summary(
    df: pd.DataFrame,
    x_grid: np.ndarray,
    *,
    spec: QuantileDatasetSpec,
) -> dict[str, object]:
    x = df["X"].to_numpy(dtype=float)
    y = df["Y"].to_numpy(dtype=float)
    z = df["Z"].to_numpy(dtype=float)
    x_orig = transform_values(spec.x_transform, x)
    y_orig = transform_values(spec.y_transform, y)
    x_grid_orig = transform_values(spec.x_transform, x_grid)
    return {
        "n_observations": int(len(df)),
        "x_log_range": [float(np.min(x)), float(np.max(x))],
        "y_log_range": [float(np.min(y)), float(np.max(y))],
        "z_range": [float(np.min(z)), float(np.max(z))],
        "x_orig_range": [float(np.min(x_orig)), float(np.max(x_orig))],
        "y_orig_range": [float(np.min(y_orig)), float(np.max(y_orig))],
        "x_grid_log_range": [float(np.min(x_grid)), float(np.max(x_grid))],
        "x_grid_orig_range": [float(np.min(x_grid_orig)), float(np.max(x_grid_orig))],
        "x_grid_points": int(len(x_grid)),
    }


def build_tabcf_curves_frame(
    spec: QuantileDatasetSpec,
    x_grid: np.ndarray,
    taus: tuple[float, ...],
    *,
    method_label: str,
    backend_name: str,
    quantiles: np.ndarray,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    x_orig = transform_values(spec.x_transform, x_grid)
    q_pred_orig = transform_values(spec.y_transform, quantiles)
    for row_idx, x_value in enumerate(np.asarray(x_grid, dtype=float)):
        for tau_idx, tau in enumerate(taus):
            records.append(
                {
                    "method": method_label,
                    "backend_name": backend_name,
                    "estimand_family": "interventional_quantile",
                    "tau": float(tau),
                    "x_log": float(x_value),
                    "x_orig": float(x_orig[row_idx]),
                    "q_pred_log": float(quantiles[row_idx, tau_idx]),
                    "q_pred_orig": float(q_pred_orig[row_idx, tau_idx]),
                }
            )
    return pd.DataFrame.from_records(records)


def augment_baseline_curves(
    spec: QuantileDatasetSpec,
    curves_df: pd.DataFrame,
) -> pd.DataFrame:
    required = ["method", "estimand_family", "tau", "x_log", "q_pred_log"]
    missing = [column for column in required if column not in curves_df.columns]
    if missing:
        raise ValueError(f"R helper curves are missing required columns: {missing}")

    out = curves_df.copy()
    if "backend_name" not in out.columns:
        out["backend_name"] = ""
    out["backend_name"] = out["backend_name"].fillna("").astype(str)
    out["x_log"] = out["x_log"].astype(float)
    out["q_pred_log"] = out["q_pred_log"].astype(float)
    out["x_orig"] = transform_values(spec.x_transform, out["x_log"].to_numpy(dtype=float))
    out["q_pred_orig"] = transform_values(spec.y_transform, out["q_pred_log"].to_numpy(dtype=float))
    return out.loc[:, ["method", "backend_name", "estimand_family", "tau", "x_log", "x_orig", "q_pred_log", "q_pred_orig"]]


def main() -> None:
    pipeline_start = time.perf_counter()
    args = parse_args()
    taus = parse_taus_arg(args.taus)

    spec, data_path, out_dir, allow_download = resolve_dataset_args(args)
    instrument_spec = resolve_instrument_spec(spec, args.instrument_spec)
    core_method_specs = scalar_pipeline.resolve_core_method_specs(
        args.core_backends,
        legacy_core_label=scalar_pipeline.CORE_BACKEND_LABELS[scalar_pipeline.TABPFN_BACKEND],
    )
    unsupported_backends = [spec.backend_name for spec in core_method_specs if spec.backend_name != scalar_pipeline.TABPFN_BACKEND]
    if unsupported_backends:
        raise ValueError(
            "This real-data quantile pipeline currently supports only the TabPFN core backend. "
            f"Unsupported backend(s): {unsupported_backends}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    output_paths = build_output_paths(spec, out_dir)

    df = load_dataset_frame(
        spec,
        data_path,
        instrument_spec=instrument_spec,
        allow_download=allow_download,
    )
    x_grid = build_x_grid(df, spec, int(args.grid_points))
    grid_df = pd.DataFrame(
        {
            "x_log": np.asarray(x_grid, dtype=float),
            "x_orig": transform_values(spec.x_transform, x_grid),
        }
    )
    df.to_csv(output_paths["clean_csv"], index=False)
    grid_df.to_csv(output_paths["grid_csv"], index=False)

    method_times: dict[str, float] = {}
    tabcf_backend_summaries: dict[str, dict[str, object]] = {}
    tabcf_curve_frames: list[pd.DataFrame] = []

    for method_spec in core_method_specs:
        method_start = time.perf_counter()
        result = compute_tabcf_quantile_curves(
            df,
            np.asarray(x_grid, dtype=float),
            taus,
            n_v_points=int(args.n_v_points),
            y_grid_points=int(args.y_grid_points),
            y_grid_padding=float(args.y_grid_padding),
            backend_name=method_spec.backend_name,
        )
        elapsed = time.perf_counter() - method_start
        method_times[method_spec.label] = elapsed
        tabcf_backend_summaries[method_spec.backend_name] = {
            "method_label": method_spec.label,
            "v_hat_range": [float(np.min(result.v_hat)), float(np.max(result.v_hat))],
            "y_grid_range": [float(result.y_grid_min), float(result.y_grid_max)],
        }
        tabcf_curve_frames.append(
            build_tabcf_curves_frame(
                spec,
                np.asarray(x_grid, dtype=float),
                taus,
                method_label=method_spec.label,
                backend_name=method_spec.backend_name,
                quantiles=result.quantiles,
            )
        )
        print(
            f"{method_spec.label} quantiles: "
            f"backend={method_spec.backend_name}, "
            f"V_hat_range=({result.v_hat.min():.6f}, {result.v_hat.max():.6f}), "
            f"y_grid_range=({result.y_grid_min:.6f}, {result.y_grid_max:.6f}), "
            f"runtime={elapsed:.6f}s"
        )

    with tempfile.TemporaryDirectory(prefix="empirical_quantile_") as tmp_dir_raw:
        tmp_dir = Path(tmp_dir_raw)
        baseline_curves_path = tmp_dir / "baseline_curves.csv"
        baseline_coefficients_path = tmp_dir / "baseline_coefficients.csv"
        baseline_runtime_path = tmp_dir / "baseline_runtime.csv"

        run_quantile_r_helper(
            data_csv=output_paths["clean_csv"],
            grid_csv=output_paths["grid_csv"],
            curves_csv=baseline_curves_path,
            coefficients_csv=baseline_coefficients_path,
            runtime_csv=baseline_runtime_path,
            r_script=args.r_script.resolve(),
            r_module=args.r_module,
            taus=taus,
            ivqr_grid_min=args.ivqr_grid_min,
            ivqr_grid_max=args.ivqr_grid_max,
            ivqr_grid_points=int(args.ivqr_grid_points),
            seed=int(args.ivqr_seed),
            div_epochs=int(args.div_epochs),
            div_layers=int(args.div_layers),
            div_lr=float(args.div_lr),
            div_nsample=int(args.div_nsample),
            div_seed=int(args.div_seed),
        )

        baseline_curves = augment_baseline_curves(spec, pd.read_csv(baseline_curves_path))
        baseline_coefficients = pd.read_csv(baseline_coefficients_path)
        baseline_runtime = pd.read_csv(baseline_runtime_path)

    validate_coefficients_frame(baseline_coefficients)

    if "backend_name" not in baseline_runtime.columns:
        baseline_runtime["backend_name"] = ""
    baseline_runtime["backend_name"] = baseline_runtime["backend_name"].fillna("").astype(str)
    baseline_runtime["seconds"] = baseline_runtime["seconds"].astype(float)
    validate_runtime_frame(
        baseline_runtime.loc[:, ["method", "backend_name", "estimand_family", "seconds"]]
    )
    for _, row in baseline_runtime.iterrows():
        method_times[str(row["method"])] = float(row["seconds"])

    curves = pd.concat([baseline_curves, *tabcf_curve_frames], ignore_index=True)
    curves["backend_name"] = curves["backend_name"].fillna("").astype(str)
    curves["tau"] = curves["tau"].astype(float)
    curves = curves.sort_values(
        ["estimand_family", "method", "backend_name", "tau", "x_log"],
        kind="mergesort",
    ).reset_index(drop=True)
    validate_curves_frame(curves)

    coefficients = baseline_coefficients.copy()
    coefficients["tau"] = coefficients["tau"].astype(float)
    coefficients = coefficients.sort_values(["method", "tau"], kind="mergesort").reset_index(drop=True)

    runtime_rows = baseline_runtime.loc[:, ["method", "backend_name", "estimand_family", "seconds"]].to_dict("records")
    for method_spec in core_method_specs:
        runtime_rows.append(
            {
                "method": method_spec.label,
                "backend_name": method_spec.backend_name,
                "estimand_family": "interventional_quantile",
                "seconds": float(method_times[method_spec.label]),
            }
        )
    runtime_df = pd.DataFrame.from_records(runtime_rows)
    runtime_df["backend_name"] = runtime_df["backend_name"].fillna("").astype(str)
    runtime_df["seconds"] = runtime_df["seconds"].astype(float)
    runtime_df = runtime_df.sort_values("seconds", ascending=False, kind="mergesort").reset_index(drop=True)
    validate_runtime_frame(runtime_df)

    curves.to_csv(output_paths["curves_csv"], index=False)
    coefficients.to_csv(output_paths["coefficients_csv"], index=False)
    runtime_df.to_csv(output_paths["runtime_csv"], index=False)
    plot_quantile_curves(
        df,
        curves,
        spec=spec,
        png_path=output_paths["png_path"],
        pdf_path=output_paths["pdf_path"],
    )

    total_seconds = time.perf_counter() - pipeline_start
    diagnostics_payload = {
        "dataset": spec.slug,
        "source_url": spec.source_url,
        "instrument_spec": instrument_spec.slug,
        "instrument_display_name": instrument_spec.display_name,
        "canonical_mapping": {
            **dict(spec.canonical_mapping),
            "Z": instrument_spec.source_column,
        },
        "config": {
            "dataset": spec.slug,
            "data_path": str(data_path),
            "out_dir": str(out_dir),
            "taus": [float(tau) for tau in taus],
            "grid_points": int(args.grid_points),
            "effective_grid_points": int(len(x_grid)),
            "x_grid_mode": spec.grid_mode,
            "n_v_points": int(args.n_v_points),
            "y_grid_points": int(args.y_grid_points),
            "y_grid_padding": float(args.y_grid_padding),
            "core_backends": [method.backend_name for method in core_method_specs],
            "core_labels": {method.backend_name: method.label for method in core_method_specs},
            "ivqr_grid_min": None if args.ivqr_grid_min is None else float(args.ivqr_grid_min),
            "ivqr_grid_max": None if args.ivqr_grid_max is None else float(args.ivqr_grid_max),
            "ivqr_grid_points": int(args.ivqr_grid_points),
            "ivqr_seed": int(args.ivqr_seed),
            "div_epochs": int(args.div_epochs),
            "div_layers": int(args.div_layers),
            "div_lr": float(args.div_lr),
            "div_nsample": int(args.div_nsample),
            "div_seed": int(args.div_seed),
            "n_observations": int(len(df)),
        },
        "first_stage_relevance": compute_first_stage_relevance_summary(df),
        "support_summary": compute_support_summary(df, np.asarray(x_grid, dtype=float), spec=spec),
        "tabcf_backend_summaries": tabcf_backend_summaries,
        "method_times_seconds": {str(key): float(value) for key, value in method_times.items()},
        "total_pipeline_seconds": float(total_seconds),
    }
    output_paths["diagnostics_json"].write_text(json.dumps(diagnostics_payload, indent=2))

    print(f"Saved cleaned data to: {output_paths['clean_csv']}")
    print(f"Saved X grid to: {output_paths['grid_csv']}")
    print(f"Saved quantile curves to: {output_paths['curves_csv']}")
    print(f"Saved quantile coefficients to: {output_paths['coefficients_csv']}")
    print(f"Saved runtime CSV to: {output_paths['runtime_csv']}")
    print(f"Saved diagnostics JSON to: {output_paths['diagnostics_json']}")
    print(f"Saved figure PNG to: {output_paths['png_path']}")
    print(f"Saved figure PDF to: {output_paths['pdf_path']}")
    print(f"Total pipeline runtime: {total_seconds:.6f}s")


if __name__ == "__main__":
    main()
