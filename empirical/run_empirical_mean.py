#!/usr/bin/env python3
"""Run scalar-IV real-data comparisons for the shared real-data benchmarks."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]
CORE_DIR = REPO_ROOT / "tabcf_core"

os.environ.setdefault("TABPFN_MODEL_VERSION", "v2.5")
os.environ.setdefault("TABPFN_MODEL_CACHE_DIR", str(REPO_ROOT / "tabpfn_home_config" / "models"))

if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TABPFN_BACKEND = "tabpfn"
TABICL_BACKEND = "tabicl"
SUPPORTED_BACKENDS = (TABPFN_BACKEND, TABICL_BACKEND)


def normalize_backend_name(backend_name: str | None) -> str:
    candidate = str(backend_name or "").strip().lower()
    if not candidate:
        return TABPFN_BACKEND
    if candidate not in SUPPORTED_BACKENDS:
        raise ValueError(f"Unsupported backend '{backend_name}'. Expected one of {SUPPORTED_BACKENDS}.")
    return candidate


DEFAULT_R_SCRIPT = SCRIPT_PATH.with_name("run_div_baseline.R")
PUBLIC_AJR_DATA_PATH = REPO_ROOT / "empirical" / "data" / "manual" / "ajr_colonial_origins.dta"
DEFAULT_AJR_DATA_PATH = PUBLIC_AJR_DATA_PATH
DEFAULT_AJR_OUTPUT_DIR = REPO_ROOT / "empirical" / "outputs"
DEFAULT_FULTON_DATA_PATH = REPO_ROOT / "empirical" / "downloads" / "fulton_fish_wooldridge.csv"
DEFAULT_FULTON_OUTPUT_DIR = REPO_ROOT / "empirical" / "outputs_fulton_fish"
DEFAULT_CARD_DATA_PATH = REPO_ROOT / "empirical" / "downloads" / "card_wooldridge.csv"
DEFAULT_CARD_OUTPUT_DIR = REPO_ROOT / "empirical" / "outputs_card"
DEFAULT_CIGARETTES_DATA_PATH = REPO_ROOT / "empirical" / "downloads" / "cigarettes_sw.csv"
DEFAULT_CIGARETTES_OUTPUT_DIR = REPO_ROOT / "empirical" / "outputs_cigarettes_sw"
FULTON_SOURCE_URL = "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/fish.csv"
CARD_SOURCE_URL = "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/card.csv"
CIGARETTES_SOURCE_URL = "https://vincentarelbundock.github.io/Rdatasets/csv/AER/CigarettesSW.csv"
EXPECTED_FULTON_ROWS = 97
EXPECTED_CARD_ROWS = 3010
EXPECTED_CIGARETTES_ROWS = 96

DEFAULT_CORE_BACKENDS = (TABPFN_BACKEND, TABICL_BACKEND)
SUPPORTED_REAL_DATA_CORE_BACKENDS = DEFAULT_CORE_BACKENDS

BASELINE_COLORS = {
    "DIV": "#66a61e",
    "2SLS": "#e67e22",
    "OLS": "#4c78a8",
}
CORE_BACKEND_LABELS = {
    TABPFN_BACKEND: "TabCF (TabPFNv2.5)",
    TABICL_BACKEND: "TabCF (TabICLv2)",
}
CORE_BACKEND_COLORS = {
    TABPFN_BACKEND: "#d1495b",
    TABICL_BACKEND: "#17becf",
}


@dataclass(frozen=True)
class DatasetSpec:
    slug: str
    display_name: str
    default_data_path: Path
    default_out_dir: Path
    output_prefix: str
    figure_basename: str
    x_label: str
    y_label: str
    source_url: str | None
    canonical_mapping: Mapping[str, str]
    grid_mode: str = "linspace"
    discrete_x: bool = False


@dataclass(frozen=True)
class CoreMethodSpec:
    backend_name: str
    label: str
    pred_column: str
    color: str


DATASET_SPECS: dict[str, DatasetSpec] = {
    "ajr": DatasetSpec(
        slug="ajr",
        display_name="AJR colonial origins",
        default_data_path=DEFAULT_AJR_DATA_PATH,
        default_out_dir=DEFAULT_AJR_OUTPUT_DIR,
        output_prefix="colonial_origins",
        figure_basename="colonial_origins_figure7_style",
        x_label="Average protection against expropriation risk, 1985-95",
        y_label="Log GDP per capita, 1995",
        source_url=None,
        canonical_mapping={
            "unit": "shortnam",
            "X": "risk",
            "Z": "logmort0",
            "Y": "loggdp",
        },
    ),
    "fulton": DatasetSpec(
        slug="fulton",
        display_name="Fulton Fish Market",
        default_data_path=DEFAULT_FULTON_DATA_PATH,
        default_out_dir=DEFAULT_FULTON_OUTPUT_DIR,
        output_prefix="fulton_fish",
        figure_basename="fulton_fish_figure",
        x_label="Log average fish price",
        y_label="Log total quantity sold",
        source_url=FULTON_SOURCE_URL,
        canonical_mapping={
            "unit": "rownames or market_day_<id>",
            "X": "lavgprc",
            "Z": "wave2",
            "Y": "ltotqty",
        },
    ),
    "card": DatasetSpec(
        slug="card",
        display_name="Card college proximity",
        default_data_path=DEFAULT_CARD_DATA_PATH,
        default_out_dir=DEFAULT_CARD_OUTPUT_DIR,
        output_prefix="card_college_proximity",
        figure_basename="card_college_proximity_figure",
        x_label="Years of education (1976)",
        y_label="Log wage (1976)",
        source_url=CARD_SOURCE_URL,
        canonical_mapping={
            "unit": "rownames or person_<id>",
            "X": "educ",
            "Z": "nearc4",
            "Y": "lwage",
        },
        grid_mode="unique",
        discrete_x=True,
    ),
    "cigarettes": DatasetSpec(
        slug="cigarettes",
        display_name="CigarettesSW cigarette demand",
        default_data_path=DEFAULT_CIGARETTES_DATA_PATH,
        default_out_dir=DEFAULT_CIGARETTES_OUTPUT_DIR,
        output_prefix="cigarettes_sw",
        figure_basename="cigarettes_sw_figure",
        x_label="Log real cigarette price",
        y_label="Log cigarette demand",
        source_url=CIGARETTES_SOURCE_URL,
        canonical_mapping={
            "unit": "state_year",
            "X": "log(price / cpi)",
            "Z": "(taxs - tax) / cpi",
            "Y": "log(packs)",
        },
    ),
}


PUBLIC_OUTPUT_NAMES: dict[str, dict[str, str]] = {
    "ajr": {
        "clean_csv": "ajr_analysis_data.csv",
        "merged_pred_csv": "ajr_model_predictions.csv",
    },
    "fulton": {
        "clean_csv": "fulton_fish_analysis_data.csv",
        "merged_pred_csv": "fulton_fish_model_predictions.csv",
    },
    "card": {
        "clean_csv": "card_analysis_data.csv",
        "merged_pred_csv": "card_model_predictions.csv",
    },
    "cigarettes": {
        "clean_csv": "cigarettes_analysis_data.csv",
        "merged_pred_csv": "cigarettes_model_predictions.csv",
    },
}


def prediction_column_for_backend(backend_name: str) -> str:
    return f"tabcf_pred_{normalize_backend_name(backend_name)}"


def _tokenize_core_backend_args(values: Sequence[str] | None) -> list[str]:
    if not values:
        return list(DEFAULT_CORE_BACKENDS)

    tokens: list[str] = []
    for value in values:
        for token in str(value).replace(",", " ").split():
            normalized = token.strip()
            if normalized:
                tokens.append(normalized)
    return tokens or list(DEFAULT_CORE_BACKENDS)


def normalize_core_backends(values: Sequence[str] | None) -> tuple[str, ...]:
    tokens = _tokenize_core_backend_args(values)
    normalized: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        backend = normalize_backend_name(token)
        if backend not in SUPPORTED_REAL_DATA_CORE_BACKENDS:
            valid = ", ".join(SUPPORTED_REAL_DATA_CORE_BACKENDS)
            raise ValueError(
                f"Unsupported real-data core backend '{token}'. Valid backends: {valid}."
            )
        if backend not in seen:
            normalized.append(backend)
            seen.add(backend)
    return tuple(normalized)


def resolve_core_method_specs(
    core_backends: Sequence[str] | None,
    *,
    legacy_core_label: str = "TabCF",
) -> tuple[CoreMethodSpec, ...]:
    normalized = normalize_core_backends(core_backends)
    multi_backend = len(normalized) > 1
    legacy_label = legacy_core_label.strip() or "TabCF"

    specs: list[CoreMethodSpec] = []
    for backend_name in normalized:
        label = CORE_BACKEND_LABELS[backend_name] if multi_backend else legacy_label
        specs.append(
            CoreMethodSpec(
                backend_name=backend_name,
                label=label,
                pred_column=prediction_column_for_backend(backend_name),
                color=CORE_BACKEND_COLORS[backend_name],
            )
        )
    return tuple(specs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASET_SPECS), default="ajr")
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
        "--core-backends",
        nargs="+",
        default=list(DEFAULT_CORE_BACKENDS),
        help="Core backends to fit. Supported: tabpfn tabicl.",
    )
    parser.add_argument(
        "--core-label",
        type=str,
        default="TabCF",
        help="Legacy single-backend display label. Ignored when multiple core backends are used.",
    )
    parser.add_argument("--grid-points", type=int, default=100)
    parser.add_argument("--n-v-points", type=int, default=101)
    parser.add_argument("--r-script", type=Path, default=DEFAULT_R_SCRIPT)
    parser.add_argument("--r-module", type=str, default="r/4.4.3-py311-xspgsan")
    parser.add_argument("--div-epochs", type=int, default=1000)
    parser.add_argument("--div-layers", type=int, default=4)
    parser.add_argument("--div-lr", type=float, default=1e-4)
    parser.add_argument("--div-nsample", type=int, default=1000)
    parser.add_argument("--div-seed", type=int, default=1)
    return parser.parse_args()


def ensure_columns(df: pd.DataFrame, required: Tuple[str, ...], *, context: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {context}: {missing}")


def download_url_to_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(destination.suffix + ".tmp")
    request = urllib.request.Request(url, headers={"User-Agent": "TabCF real-data downloader"})

    try:
        with urllib.request.urlopen(request, timeout=60) as response, tmp_path.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        tmp_path.replace(destination)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def summarize_loaded_data(df: pd.DataFrame, *, display_name: str) -> None:
    print(
        f"Loaded {display_name} data: "
        f"n={len(df)}, "
        f"X_range=({df['X'].min():.3f}, {df['X'].max():.3f}), "
        f"Z_range=({df['Z'].min():.3f}, {df['Z'].max():.3f}), "
        f"Y_range=({df['Y'].min():.3f}, {df['Y'].max():.3f})"
    )


def resolve_ajr_data_path(path: Path) -> Path:
    if path.exists():
        return path
    raise FileNotFoundError(
        "AJR data file not found. Place the AJR Stata file at "
        f"{DEFAULT_AJR_DATA_PATH} or pass --data-path explicitly. Requested path: {path}"
    )


def load_ajr_data(path: Path) -> pd.DataFrame:
    path = resolve_ajr_data_path(path)

    df = pd.read_stata(path)
    ensure_columns(df, ("shortnam", "risk", "logmort0", "loggdp"), context="AJR Stata file")
    before = len(df)
    canonical = df.loc[:, ["shortnam", "risk", "logmort0", "loggdp"]].rename(
        columns={
            "shortnam": "unit",
            "risk": "X",
            "logmort0": "Z",
            "loggdp": "Y",
        }
    )
    canonical = canonical.dropna(subset=["unit", "X", "Z", "Y"]).reset_index(drop=True)
    dropped = before - len(canonical)
    if dropped:
        print(f"Dropped {dropped} rows with missing values in unit/X/Z/Y.")

    if canonical.empty:
        raise ValueError("No usable AJR rows remained after dropna on unit/X/Z/Y.")

    summarize_loaded_data(canonical, display_name=DATASET_SPECS["ajr"].display_name)
    return canonical


def _fulton_units(df: pd.DataFrame) -> list[str]:
    if "rownames" not in df.columns:
        return [f"market_day_{idx}" for idx in range(1, len(df) + 1)]

    units: list[str] = []
    for idx, raw_value in enumerate(df["rownames"], start=1):
        if pd.isna(raw_value):
            units.append(f"market_day_{idx}")
            continue
        text = str(raw_value).strip()
        units.append(text if text else f"market_day_{idx}")
    return units


def load_fulton_data(path: Path, *, allow_download: bool) -> pd.DataFrame:
    if not path.exists():
        if allow_download:
            print(f"Downloading Fulton Fish Market data to: {path}")
            download_url_to_file(FULTON_SOURCE_URL, path)
        else:
            raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path)
    ensure_columns(df, ("lavgprc", "ltotqty", "wave2"), context="Fulton Fish CSV")
    if len(df) != EXPECTED_FULTON_ROWS:
        raise ValueError(
            f"Unexpected Fulton Fish row count in {path}: expected {EXPECTED_FULTON_ROWS}, found {len(df)}."
        )

    before = len(df)
    canonical = pd.DataFrame(
        {
            "unit": _fulton_units(df),
            "X": df["lavgprc"],
            "Z": df["wave2"],
            "Y": df["ltotqty"],
        }
    )
    canonical = canonical.dropna(subset=["unit", "X", "Z", "Y"]).reset_index(drop=True)
    dropped = before - len(canonical)
    if dropped:
        print(f"Dropped {dropped} rows with missing values in unit/X/Z/Y.")

    if canonical.empty:
        raise ValueError("No usable Fulton Fish rows remained after dropna on unit/X/Z/Y.")

    summarize_loaded_data(canonical, display_name=DATASET_SPECS["fulton"].display_name)
    return canonical


def _card_units(df: pd.DataFrame) -> list[str]:
    if "rownames" not in df.columns:
        return [f"person_{idx}" for idx in range(1, len(df) + 1)]

    units: list[str] = []
    for idx, raw_value in enumerate(df["rownames"], start=1):
        if pd.isna(raw_value):
            units.append(f"person_{idx}")
            continue
        text = str(raw_value).strip()
        units.append(text if text else f"person_{idx}")
    return units


def load_card_data(path: Path, *, allow_download: bool) -> pd.DataFrame:
    if not path.exists():
        if allow_download:
            print(f"Downloading Card data to: {path}")
            download_url_to_file(CARD_SOURCE_URL, path)
        else:
            raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path)
    ensure_columns(df, ("nearc4", "educ", "lwage"), context="Card CSV")
    if len(df) != EXPECTED_CARD_ROWS:
        raise ValueError(
            f"Unexpected Card row count in {path}: expected {EXPECTED_CARD_ROWS}, found {len(df)}."
        )

    before = len(df)
    canonical = pd.DataFrame(
        {
            "unit": _card_units(df),
            "X": df["educ"],
            "Z": df["nearc4"],
            "Y": df["lwage"],
        }
    )
    canonical = canonical.dropna(subset=["unit", "X", "Z", "Y"]).reset_index(drop=True)
    dropped = before - len(canonical)
    if dropped:
        print(f"Dropped {dropped} rows with missing values in unit/X/Z/Y.")

    if canonical.empty:
        raise ValueError("No usable Card rows remained after dropna on unit/X/Z/Y.")

    summarize_loaded_data(canonical, display_name=DATASET_SPECS["card"].display_name)
    return canonical


def _cigarettes_units(df: pd.DataFrame) -> list[str | float]:
    units: list[str | float] = []
    for state, year in zip(df["state"], df["year"]):
        if pd.isna(state) or pd.isna(year):
            units.append(np.nan)
            continue
        units.append(f"{str(state).strip()}_{int(year)}")
    return units


def load_cigarettes_data(path: Path, *, allow_download: bool) -> pd.DataFrame:
    if not path.exists():
        if allow_download:
            print(f"Downloading CigarettesSW data to: {path}")
            download_url_to_file(CIGARETTES_SOURCE_URL, path)
        else:
            raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path)
    ensure_columns(
        df,
        ("state", "year", "cpi", "packs", "tax", "price", "taxs"),
        context="CigarettesSW CSV",
    )
    if len(df) != EXPECTED_CIGARETTES_ROWS:
        raise ValueError(
            f"Unexpected CigarettesSW row count in {path}: expected {EXPECTED_CIGARETTES_ROWS}, found {len(df)}."
        )

    if df["year"].isna().any():
        raise ValueError("CigarettesSW CSV contains missing values in year.")
    observed_years = set(pd.Series(df["year"]).astype(int).unique().tolist())
    expected_years = {1985, 1995}
    if observed_years != expected_years:
        raise ValueError(
            f"Unexpected CigarettesSW years in {path}: expected {sorted(expected_years)}, found {sorted(observed_years)}."
        )

    before = len(df)
    canonical = pd.DataFrame(
        {
            "unit": _cigarettes_units(df),
            "X": np.log(df["price"] / df["cpi"]),
            "Z": (df["taxs"] - df["tax"]) / df["cpi"],
            "Y": np.log(df["packs"]),
        }
    )
    canonical = canonical.dropna(subset=["unit", "X", "Z", "Y"]).reset_index(drop=True)
    finite_mask = np.isfinite(canonical[["X", "Z", "Y"]].to_numpy(dtype=float)).all(axis=1)
    canonical = canonical.loc[finite_mask].reset_index(drop=True)
    dropped = before - len(canonical)
    if dropped:
        print(f"Dropped {dropped} rows with missing or non-finite values in unit/X/Z/Y.")

    if canonical.empty:
        raise ValueError("No usable CigarettesSW rows remained after canonical preprocessing.")

    summarize_loaded_data(canonical, display_name=DATASET_SPECS["cigarettes"].display_name)
    return canonical


def load_dataset_frame(spec: DatasetSpec, path: Path, *, allow_download: bool) -> pd.DataFrame:
    if spec.slug == "ajr":
        return load_ajr_data(path)
    if spec.slug == "fulton":
        return load_fulton_data(path, allow_download=allow_download)
    if spec.slug == "card":
        return load_card_data(path, allow_download=allow_download)
    if spec.slug == "cigarettes":
        return load_cigarettes_data(path, allow_download=allow_download)
    raise ValueError(f"Unsupported dataset: {spec.slug}")


def fit_linear_regression(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    design = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ beta
    return beta, fitted


def fit_2sls(z: np.ndarray, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    stage1_design = np.column_stack([np.ones_like(z), z])
    beta_stage1, *_ = np.linalg.lstsq(stage1_design, x, rcond=None)
    x_hat = stage1_design @ beta_stage1

    stage2_design = np.column_stack([np.ones_like(x_hat), x_hat])
    beta_stage2, *_ = np.linalg.lstsq(stage2_design, y, rcond=None)
    return beta_stage1, beta_stage2


def compute_tabcf_curve(
    df: pd.DataFrame,
    x_grid: np.ndarray,
    n_v_points: int,
    *,
    backend_name: str = TABPFN_BACKEND,
) -> tuple[np.ndarray, np.ndarray]:
    from stage1_control import CondCDFModel
    from stage2_outcome import FullDataStructuralFunctionModel, compute_mu_c_on_grid

    backend = normalize_backend_name(backend_name)
    x_train = df["X"].to_numpy(dtype=float)
    z_train = df["Z"].to_numpy(dtype=float)
    y_train = df["Y"].to_numpy(dtype=float)

    cdf_model = CondCDFModel(quantiles=(), backend_name=backend)
    cdf_model.fit(z_train, x_train)
    v_hat = np.clip(cdf_model.predict(z_train, x_train), 0.0, 1.0)

    if np.any(~np.isfinite(v_hat)):
        raise ValueError(f"Stage-1 V_hat contains non-finite values for backend '{backend}'.")

    m_model = FullDataStructuralFunctionModel(
        use_tabpfn=(backend != TABICL_BACKEND),
        backend_name=backend,
    )
    _ = m_model.fit_full(x_train, v_hat, y_train)
    tabcf_pred = compute_mu_c_on_grid(m_model, np.asarray(x_grid, dtype=float), int(n_v_points))

    if np.any(~np.isfinite(tabcf_pred)):
        raise ValueError(f"TabCF curve contains non-finite values for backend '{backend}'.")

    return v_hat, tabcf_pred


def _run_shell_command(shell_cmd: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-lc", shell_cmd],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )


def run_div_r_helper(
    *,
    data_csv: Path,
    grid_csv: Path,
    pred_csv: Path,
    r_script: Path,
    r_module: str,
    epochs: int,
    layers: int,
    lr: float,
    nsample: int,
    seed: int,
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
        "--pred-csv",
        str(pred_csv),
        "--num-epochs",
        str(epochs),
        "--num-layer",
        str(layers),
        "--lr",
        str(lr),
        "--nsample",
        str(nsample),
        "--seed",
        str(seed),
    ]
    rscript_cmd = " ".join(shlex.quote(part) for part in arg_parts)
    if r_module:
        shell_cmd = (
            "source /etc/profile >/dev/null 2>&1 && "
            f"module load {shlex.quote(r_module)} >/dev/null 2>&1 && {rscript_cmd}"
        )
    else:
        shell_cmd = rscript_cmd

    print(f"Running DIV baseline via R helper: {r_script}")
    completed = _run_shell_command(shell_cmd)
    if completed.returncode != 0 and r_module and shutil.which("Rscript") is not None:
        completed = _run_shell_command(rscript_cmd)

    if completed.stdout.strip():
        print(completed.stdout.strip())
    if completed.returncode != 0:
        raise RuntimeError(
            "DIV R helper failed.\n"
            f"Command: {shell_cmd}\n"
            f"Stdout:\n{completed.stdout}\n"
            f"Stderr:\n{completed.stderr}"
        )


def plot_curves(
    df: pd.DataFrame,
    preds: pd.DataFrame,
    *,
    spec: DatasetSpec,
    core_method_specs: Sequence[CoreMethodSpec],
    png_path: Path,
    pdf_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    ax.scatter(df["X"], df["Y"], color="grey", alpha=0.5, s=28, linewidths=0, label="_scatter")
    line_kwargs = {"marker": "o", "markersize": 4.0} if spec.discrete_x else {}

    ax.plot(preds["X"], preds["div_pred"], color=BASELINE_COLORS["DIV"], linewidth=2.2, label="DIV", **line_kwargs)
    ax.plot(
        preds["X"],
        preds["tsls_pred"],
        color=BASELINE_COLORS["2SLS"],
        linewidth=2.0,
        label="2SLS",
        **line_kwargs,
    )
    ax.plot(
        preds["X"],
        preds["ols_pred"],
        color=BASELINE_COLORS["OLS"],
        linewidth=2.0,
        label="OLS",
        **line_kwargs,
    )
    for method_spec in core_method_specs:
        ax.plot(
            preds["X"],
            preds[method_spec.pred_column],
            color=method_spec.color,
            linewidth=2.2,
            label=method_spec.label,
            **line_kwargs,
        )

    ax.set_xlabel(spec.x_label)
    ax.set_ylabel(spec.y_label)
    ax.set_xlim(float(preds["X"].min()), float(preds["X"].max()))
    if spec.discrete_x:
        ax.set_xticks(preds["X"].to_numpy(dtype=float))
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.14),
        ncol=3 + len(core_method_specs),
        frameon=True,
        fancybox=False,
        edgecolor="black",
    )
    ax.grid(False)
    fig.tight_layout()

    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def build_output_paths(spec: DatasetSpec, out_dir: Path) -> dict[str, Path]:
    prefix = spec.output_prefix
    public_names = PUBLIC_OUTPUT_NAMES.get(spec.slug, {})
    return {
        "clean_csv": out_dir / public_names.get("clean_csv", f"{prefix}_clean.csv"),
        "grid_csv": out_dir / f"{prefix}_x_grid.csv",
        "div_pred_csv": out_dir / f"{prefix}_div_predictions.csv",
        "merged_pred_csv": out_dir / public_names.get("merged_pred_csv", f"{prefix}_predictions.csv"),
        "png_path": out_dir / f"{spec.figure_basename}.png",
        "pdf_path": out_dir / f"{spec.figure_basename}.pdf",
        "runtime_csv": out_dir / f"{prefix}_method_runtimes.csv",
        "runtime_json": out_dir / f"{prefix}_runtime_summary.json",
    }


def build_x_grid(df: pd.DataFrame, spec: DatasetSpec, requested_points: int) -> np.ndarray:
    x_values = df["X"].to_numpy(dtype=float)
    if spec.grid_mode == "unique":
        x_grid = np.sort(np.unique(x_values))
        if x_grid.size < 2:
            raise ValueError(
                f"{spec.display_name} needs at least two distinct X values; found {x_grid.size}."
            )
        return x_grid

    if requested_points < 2:
        raise ValueError("--grid-points must be at least 2.")
    return np.linspace(x_values.min(), x_values.max(), int(requested_points))


def save_runtime_outputs(
    *,
    runtime_csv: Path,
    runtime_json: Path,
    dataset: DatasetSpec,
    method_times: dict[str, float],
    total_seconds: float,
    config: dict[str, object],
) -> tuple[Path, Path]:
    runtime_df = (
        pd.DataFrame(
            [{"method": method, "seconds": float(seconds)} for method, seconds in method_times.items()]
        )
        .sort_values("seconds", ascending=False, kind="mergesort")
        .reset_index(drop=True)
    )
    runtime_df.to_csv(runtime_csv, index=False)

    payload = {
        "dataset": dataset.slug,
        "source_url": dataset.source_url,
        "canonical_mapping": dict(dataset.canonical_mapping),
        "config": config,
        "method_times_seconds": {key: float(value) for key, value in method_times.items()},
        "total_pipeline_seconds": float(total_seconds),
    }
    runtime_json.write_text(json.dumps(payload, indent=2))
    return runtime_csv, runtime_json


def resolve_dataset_args(args: argparse.Namespace) -> tuple[DatasetSpec, Path, Path, bool]:
    spec = DATASET_SPECS[args.dataset]
    data_path = args.data_path.resolve() if args.data_path is not None else spec.default_data_path.resolve()
    out_dir = args.out_dir.resolve() if args.out_dir is not None else spec.default_out_dir.resolve()
    allow_download = args.data_path is None and spec.source_url is not None
    return spec, data_path, out_dir, allow_download


def validate_predictions_frame(preds: pd.DataFrame, core_method_specs: Sequence[CoreMethodSpec]) -> None:
    required_columns = ["X", "ols_pred", "tsls_pred", "div_pred"]
    required_columns.extend(spec.pred_column for spec in core_method_specs)
    missing = [column for column in required_columns if column not in preds.columns]
    if missing:
        raise ValueError(f"Merged predictions are missing required columns: {missing}")
    numeric = preds.loc[:, required_columns].to_numpy(dtype=float)
    if np.isnan(numeric).any() or not np.isfinite(numeric).all():
        raise ValueError("Merged predictions contain NA or non-finite values.")


def main() -> None:
    pipeline_start = time.perf_counter()
    args = parse_args()

    spec, data_path, out_dir, allow_download = resolve_dataset_args(args)
    core_method_specs = resolve_core_method_specs(args.core_backends, legacy_core_label=args.core_label)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_paths = build_output_paths(spec, out_dir)

    df = load_dataset_frame(spec, data_path, allow_download=allow_download)
    x_grid = build_x_grid(df, spec, int(args.grid_points))
    method_times: dict[str, float] = {}

    df.to_csv(output_paths["clean_csv"], index=False)
    pd.DataFrame({"X": x_grid}).to_csv(output_paths["grid_csv"], index=False)

    x = df["X"].to_numpy(dtype=float)
    z = df["Z"].to_numpy(dtype=float)
    y = df["Y"].to_numpy(dtype=float)

    method_start = time.perf_counter()
    ols_beta, _ = fit_linear_regression(x, y)
    ols_pred = ols_beta[0] + ols_beta[1] * x_grid
    method_times["OLS"] = time.perf_counter() - method_start

    method_start = time.perf_counter()
    _, tsls_beta = fit_2sls(z, x, y)
    tsls_pred = tsls_beta[0] + tsls_beta[1] * x_grid
    method_times["2SLS"] = time.perf_counter() - method_start

    print(f"OLS coefficients: intercept={ols_beta[0]:.6f}, slope={ols_beta[1]:.6f}")
    print(f"2SLS coefficients: intercept={tsls_beta[0]:.6f}, slope={tsls_beta[1]:.6f}")
    print(f"OLS runtime: {method_times['OLS']:.6f}s")
    print(f"2SLS runtime: {method_times['2SLS']:.6f}s")

    core_predictions: dict[str, np.ndarray] = {}
    for method_spec in core_method_specs:
        method_start = time.perf_counter()
        v_hat, core_pred = compute_tabcf_curve(
            df,
            x_grid,
            args.n_v_points,
            backend_name=method_spec.backend_name,
        )
        method_times[method_spec.label] = time.perf_counter() - method_start
        core_predictions[method_spec.pred_column] = core_pred
        print(
            f"{method_spec.label} control function summary: "
            f"backend={method_spec.backend_name}, "
            f"V_hat_range=({v_hat.min():.6f}, {v_hat.max():.6f}), "
            f"pred_range=({core_pred.min():.6f}, {core_pred.max():.6f})"
        )
        print(f"{method_spec.label} runtime: {method_times[method_spec.label]:.6f}s")

    method_start = time.perf_counter()
    run_div_r_helper(
        data_csv=output_paths["clean_csv"],
        grid_csv=output_paths["grid_csv"],
        pred_csv=output_paths["div_pred_csv"],
        r_script=args.r_script.resolve(),
        r_module=args.r_module,
        epochs=args.div_epochs,
        layers=args.div_layers,
        lr=args.div_lr,
        nsample=args.div_nsample,
        seed=args.div_seed,
    )
    method_times["DIV"] = time.perf_counter() - method_start
    print(f"DIV runtime: {method_times['DIV']:.6f}s")

    div_df = pd.read_csv(output_paths["div_pred_csv"])
    if list(div_df.columns) != ["X", "div_pred"]:
        raise ValueError(
            f"Unexpected DIV prediction schema in {output_paths['div_pred_csv']}: {list(div_df.columns)}"
        )
    if len(div_df) != len(x_grid):
        raise ValueError(f"DIV predictions length {len(div_df)} does not match grid size {len(x_grid)}.")

    preds_payload: dict[str, np.ndarray] = {
        "X": x_grid,
        "ols_pred": ols_pred,
        "tsls_pred": tsls_pred,
        "div_pred": div_df["div_pred"].to_numpy(dtype=float),
    }
    preds_payload.update(core_predictions)
    preds = pd.DataFrame(preds_payload).sort_values("X", kind="mergesort").reset_index(drop=True)
    validate_predictions_frame(preds, core_method_specs)

    preds.to_csv(output_paths["merged_pred_csv"], index=False)
    plot_curves(
        df,
        preds,
        spec=spec,
        core_method_specs=core_method_specs,
        png_path=output_paths["png_path"],
        pdf_path=output_paths["pdf_path"],
    )
    total_seconds = time.perf_counter() - pipeline_start
    runtime_csv, runtime_json = save_runtime_outputs(
        runtime_csv=output_paths["runtime_csv"],
        runtime_json=output_paths["runtime_json"],
        dataset=spec,
        method_times=method_times,
        total_seconds=total_seconds,
        config={
            "dataset": spec.slug,
            "data_path": str(data_path),
            "out_dir": str(out_dir),
            "core_backends": [method.backend_name for method in core_method_specs],
            "core_labels": {method.backend_name: method.label for method in core_method_specs},
            "core_prediction_columns": {
                method.backend_name: method.pred_column for method in core_method_specs
            },
            "legacy_core_label": args.core_label,
            "grid_points": int(args.grid_points),
            "effective_grid_points": int(len(x_grid)),
            "x_grid_mode": spec.grid_mode,
            "n_v_points": int(args.n_v_points),
            "div_epochs": int(args.div_epochs),
            "div_layers": int(args.div_layers),
            "div_lr": float(args.div_lr),
            "div_nsample": int(args.div_nsample),
            "div_seed": int(args.div_seed),
            "n_observations": int(len(df)),
        },
    )

    print(f"Saved cleaned data to: {output_paths['clean_csv']}")
    print(f"Saved X grid to: {output_paths['grid_csv']}")
    print(f"Saved merged predictions to: {output_paths['merged_pred_csv']}")
    print(f"Saved figure PNG to: {output_paths['png_path']}")
    print(f"Saved figure PDF to: {output_paths['pdf_path']}")
    print(f"Saved runtime CSV to: {runtime_csv}")
    print(f"Saved runtime JSON to: {runtime_json}")
    print(f"Total pipeline runtime: {total_seconds:.6f}s")


if __name__ == "__main__":
    main()
