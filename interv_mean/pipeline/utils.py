"""
Shared utilities for Section 5.1 pipeline.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from tabcf_core.dgp import DGPConfig  # pragma: no cover


# --------------------------------------------------------------------------------------
# Static configuration
# --------------------------------------------------------------------------------------

CODE_SCENARIO_MAP: Dict[str, Dict[str, str]] = {
    "A3_B3": {"first_stage": "A3", "second_stage": "B3", "scenario": "g_lin_f_lin", "instrument": "Zcont"},
    "A3_B4": {"first_stage": "A3", "second_stage": "B4", "scenario": "g_lin_f_log_case", "instrument": "Zcont"},
    "A3_B5": {"first_stage": "A3", "second_stage": "B5", "scenario": "g_lin_f_sin_lin", "instrument": "Zcont"},
    "A3_B6": {"first_stage": "A3", "second_stage": "B6", "scenario": "g_lin_f_sin_xh", "instrument": "Zcont"},
    "A3_B7": {"first_stage": "A3", "second_stage": "B7", "scenario": "g_lin_f_atan_trig", "instrument": "Zcont"},
    "A3_B8": {"first_stage": "A3", "second_stage": "B8", "scenario": "g_lin_f_log_case_b8", "instrument": "Zcont"},
    "A3_B9": {"first_stage": "A3", "second_stage": "B9", "scenario": "g_lin_f_b9star", "instrument": "Zcont"},
    "A3_B10": {
        "first_stage": "A3",
        "second_stage": "B10",
        "scenario": "g_lin_f_periodic_nonadditive_error",
        "instrument": "Zcont",
    },
    "A3_B11": {
        "first_stage": "A3",
        "second_stage": "B11",
        "scenario": "g_lin_f_cos2x_xh_minus_h",
        "instrument": "Zcont",
    },
    "A4_B3": {"first_stage": "A4", "second_stage": "B3", "scenario": "g_z1_hscale_f_lin", "instrument": "Zcont"},
    "A4_B4": {"first_stage": "A4", "second_stage": "B4", "scenario": "g_z1_hscale_f_log_case", "instrument": "Zcont"},
    "A4_B5": {"first_stage": "A4", "second_stage": "B5", "scenario": "g_z1_hscale_f_sin_lin", "instrument": "Zcont"},
    "A4_B6": {"first_stage": "A4", "second_stage": "B6", "scenario": "g_z1_hscale_f_sin_xh", "instrument": "Zcont"},
    "A4_B8": {"first_stage": "A4", "second_stage": "B8", "scenario": "g_z1_hscale_f_log_case_b8", "instrument": "Zcont"},
    "A5_B3": {"first_stage": "A5", "second_stage": "B3", "scenario": "g_zquad_hscale_f_lin", "instrument": "Zcont"},
    "A5_B4": {"first_stage": "A5", "second_stage": "B4", "scenario": "g_zquad_hscale_f_log_case", "instrument": "Zcont"},
    "A5_B5": {"first_stage": "A5", "second_stage": "B5", "scenario": "g_zquad_hscale_f_sin_lin", "instrument": "Zcont"},
    "A5_B6": {"first_stage": "A5", "second_stage": "B6", "scenario": "g_zquad_hscale_f_sin_xh", "instrument": "Zcont"},
    "A6_B3": {"first_stage": "A6", "second_stage": "B3", "scenario": "g_zquad_sin_hscale_f_lin", "instrument": "Zcont"},
    "A6_B4": {"first_stage": "A6", "second_stage": "B4", "scenario": "g_zquad_sin_hscale_f_log_case", "instrument": "Zcont"},
    "A6_B5": {"first_stage": "A6", "second_stage": "B5", "scenario": "g_zquad_sin_hscale_f_sin_lin", "instrument": "Zcont"},
    "A6_B6": {"first_stage": "A6", "second_stage": "B6", "scenario": "g_zquad_sin_hscale_f_sin_xh", "instrument": "Zcont"},
    "A6_B7": {"first_stage": "A6", "second_stage": "B7", "scenario": "g_zquad_sin_hscale_f_atan_trig", "instrument": "Zcont"},
    "A7_B3": {"first_stage": "A7", "second_stage": "B3", "scenario": "g_zquad_exp_hscale_f_lin", "instrument": "Zcont"},
    "A7_B4": {"first_stage": "A7", "second_stage": "B4", "scenario": "g_zquad_exp_hscale_f_log_case", "instrument": "Zcont"},
    "A7_B5": {"first_stage": "A7", "second_stage": "B5", "scenario": "g_zquad_exp_hscale_f_sin_lin", "instrument": "Zcont"},
    "A7_B6": {"first_stage": "A7", "second_stage": "B6", "scenario": "g_zquad_exp_hscale_f_sin_xh", "instrument": "Zcont"},
    "A7_B7": {"first_stage": "A7", "second_stage": "B7", "scenario": "g_zquad_exp_hscale_f_atan_trig", "instrument": "Zcont"},
    "A8_B3": {"first_stage": "A8", "second_stage": "B3", "scenario": "g_z1_addh_hscale_f_lin", "instrument": "Zcont"},
    "A8_B4": {"first_stage": "A8", "second_stage": "B4", "scenario": "g_z1_addh_hscale_f_log_case", "instrument": "Zcont"},
    "A8_B5": {"first_stage": "A8", "second_stage": "B5", "scenario": "g_z1_addh_hscale_f_sin_lin", "instrument": "Zcont"},
    "A8_B6": {"first_stage": "A8", "second_stage": "B6", "scenario": "g_z1_addh_hscale_f_sin_xh", "instrument": "Zcont"},
    "A8_B8": {"first_stage": "A8", "second_stage": "B8", "scenario": "g_z1_addh_hscale_f_log_case_b8", "instrument": "Zcont"},
    "A9_B3": {"first_stage": "A9", "second_stage": "B3", "scenario": "g_quadtrend_linscale_f_lin", "instrument": "Zcont"},
    "A9_B4": {"first_stage": "A9", "second_stage": "B4", "scenario": "g_quadtrend_linscale_f_log_case", "instrument": "Zcont"},
    "A9_B5": {"first_stage": "A9", "second_stage": "B5", "scenario": "g_quadtrend_linscale_f_sin_lin", "instrument": "Zcont"},
    "A9_B9": {"first_stage": "A9", "second_stage": "B9", "scenario": "g_quadtrend_linscale_f_b9star", "instrument": "Zcont"},
    "A9_B10": {
        "first_stage": "A9",
        "second_stage": "B10",
        "scenario": "g_quadtrend_linscale_f_periodic_nonadditive_error",
        "instrument": "Zcont",
    },
    "A9_B11": {
        "first_stage": "A9",
        "second_stage": "B11",
        "scenario": "g_quadtrend_linscale_f_cos2x_xh_minus_h",
        "instrument": "Zcont",
    },
}

TRAIN_SIZES = [1000, 4000, 10000]
SEEDS = list(range(1, 100)) + [123]

PIPELINE_DIR = Path(__file__).resolve().parent
SEC5_1_ROOT = PIPELINE_DIR.parent
BATCH_ROOT = SEC5_1_ROOT.parent
REPO_ROOT = BATCH_ROOT.parent

CORE_ROOT = BATCH_ROOT / "core"
DATA_ROOT = BATCH_ROOT / "IV_datasets"
DEFAULT_STAGE2_DIR = DATA_ROOT / "stage2_output"
DEFAULT_TRAIN_DIR = DATA_ROOT / "train"
DEFAULT_TEST_DIR = DATA_ROOT / "test"

DEFAULT_IO_ROOT = SEC5_1_ROOT / "io"
DEFAULT_BRIDGE_DIR = DEFAULT_IO_ROOT / "bridge"
DEFAULT_RESULTS_DIR = DEFAULT_IO_ROOT / "results"
DEFAULT_AGGREGATED_DIR = DEFAULT_IO_ROOT / "aggregated"
DEFAULT_VISUALIZATION_DIR = DEFAULT_IO_ROOT / "visualization"

DEFAULT_MANIFESTS_DIR = SEC5_1_ROOT / "manifests"
DEFAULT_MANIFEST_GENERATED_DIR = DEFAULT_MANIFESTS_DIR / "generated"
DEFAULT_MANIFEST_PATH = DEFAULT_MANIFEST_GENERATED_DIR / "paper_main_s10.json"

SUMMARY_REGEX = re.compile(
    r"s2_(?P<code>A\d+_B\d+)_n(?P<size>\d+)_seed(?P<seed>\d+)_summary(?:_(?P<timestamp>\d+))?\.csv$"
)

B4_SOFTPLUS_EPS = 1e-8


# --------------------------------------------------------------------------------------
# Manifest helpers
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class RunSpec:
    code: str
    scenario: str
    instrument: str
    train_size: int
    seed: int
    train_rel: str
    test_rel: str
    stage2_summary_rel: Optional[str]
    bridge_train_name: str
    bridge_test_name: str


def _to_repo_rel(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT.resolve()))


def _from_repo_rel(path_rel: str) -> Path:
    return (REPO_ROOT / path_rel).resolve()


def bridge_filenames(code: str, scenario: str, instrument: str, size: int, seed: int) -> Tuple[str, str]:
    del code
    train_name = f"train_{instrument}_{scenario}_n{size}_seed{seed}.csv"
    test_name = f"test_{instrument}_{scenario}_n{size}_seed{seed}.csv"
    return train_name, test_name


def _index_stage2_summaries(stage2_dir: Path) -> Dict[Tuple[str, int, int], Path]:
    summary_map: Dict[Tuple[str, int, int], Path] = {}
    if not stage2_dir.exists():
        return summary_map

    for csv_path in stage2_dir.glob("s2_*_summary*.csv"):
        match = SUMMARY_REGEX.match(csv_path.name)
        if not match:
            continue
        key = (match.group("code"), int(match.group("size")), int(match.group("seed")))
        prev = summary_map.get(key)
        if prev is None or csv_path.stat().st_mtime > prev.stat().st_mtime:
            summary_map[key] = csv_path
    return summary_map


def build_runs(
    train_sizes: Iterable[int] = TRAIN_SIZES,
    seeds: Iterable[int] = SEEDS,
    train_dir: Path = DEFAULT_TRAIN_DIR,
    test_dir: Path = DEFAULT_TEST_DIR,
    stage2_dir: Path = DEFAULT_STAGE2_DIR,
    require_stage2: bool = True,
    codes: Iterable[str] | None = None,
) -> List[RunSpec]:
    stage2_index = _index_stage2_summaries(stage2_dir)
    selected_codes = list(codes) if codes is not None else sorted(CODE_SCENARIO_MAP.keys())

    runs: List[RunSpec] = []
    for code in selected_codes:
        if code not in CODE_SCENARIO_MAP:
            raise KeyError(f"Unknown code: {code}")
        meta = CODE_SCENARIO_MAP[code]
        for size in train_sizes:
            for seed in seeds:
                train_path = train_dir / f"train_data_{code}_n{size}_seed{seed}.csv"
                test_path = test_dir / f"test_data_{code}.csv"
                summary_path = stage2_index.get((code, size, seed))

                if require_stage2 and summary_path is None:
                    raise FileNotFoundError(
                        f"Stage2 summary missing for {code}, n={size}, seed={seed} in {stage2_dir}"
                    )
                if not train_path.exists():
                    raise FileNotFoundError(f"Train CSV missing: {train_path}")
                if not test_path.exists():
                    raise FileNotFoundError(f"Test CSV missing: {test_path}")

                bridge_train_name, bridge_test_name = bridge_filenames(
                    code, meta["scenario"], meta["instrument"], size, seed
                )

                runs.append(
                    RunSpec(
                        code=code,
                        scenario=meta["scenario"],
                        instrument=meta["instrument"],
                        train_size=int(size),
                        seed=int(seed),
                        train_rel=_to_repo_rel(train_path),
                        test_rel=_to_repo_rel(test_path),
                        stage2_summary_rel=_to_repo_rel(summary_path) if summary_path else None,
                        bridge_train_name=bridge_train_name,
                        bridge_test_name=bridge_test_name,
                    )
                )
    return runs


def save_manifest(
    runs: List[RunSpec],
    path: Path,
    *,
    train_dir: Path = DEFAULT_TRAIN_DIR,
    test_dir: Path = DEFAULT_TEST_DIR,
    stage2_dir: Path = DEFAULT_STAGE2_DIR,
    default_bridge_dir: Path = DEFAULT_BRIDGE_DIR,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    code_values = sorted({run.code for run in runs})
    size_values = sorted({int(run.train_size) for run in runs})
    seed_values = sorted({int(run.seed) for run in runs})

    payload = {
        "meta": {
            "schema_version": "interv_mean_manifest_v2",
            "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "codes": code_values,
            "train_sizes": size_values,
            "seeds": seed_values,
            "train_dir_rel": _to_repo_rel(train_dir),
            "test_dir_rel": _to_repo_rel(test_dir),
            "stage2_dir_rel": _to_repo_rel(stage2_dir),
            "default_bridge_dir_rel": _to_repo_rel(default_bridge_dir),
        },
        "runs": [run.__dict__ for run in runs],
    }
    path.write_text(json.dumps(payload, indent=2))


def load_manifest(path: Path = DEFAULT_MANIFEST_PATH) -> Dict[str, object]:
    return json.loads(path.read_text())


def manifest_bridge_dir(manifest: Dict[str, object]) -> Path:
    meta = manifest.get("meta", {}) if isinstance(manifest, dict) else {}
    if isinstance(meta, dict) and isinstance(meta.get("default_bridge_dir_rel"), str):
        return _from_repo_rel(meta["default_bridge_dir_rel"])
    return DEFAULT_BRIDGE_DIR


def resolve_run_paths(
    run: Dict[str, object],
    *,
    manifest: Optional[Dict[str, object]] = None,
    bridge_dir: Optional[Path] = None,
) -> Dict[str, Optional[Path]]:
    effective_bridge_dir = bridge_dir or (manifest_bridge_dir(manifest) if manifest else DEFAULT_BRIDGE_DIR)

    # v2 schema (relative paths + bridge file names)
    if "train_rel" in run and "test_rel" in run:
        train_path = _from_repo_rel(str(run["train_rel"]))
        test_path = _from_repo_rel(str(run["test_rel"]))
        stage2_rel = run.get("stage2_summary_rel")
        stage2_path = _from_repo_rel(str(stage2_rel)) if stage2_rel else None
        bridge_train_name = str(run.get("bridge_train_name", ""))
        bridge_test_name = str(run.get("bridge_test_name", ""))
        if not bridge_train_name or not bridge_test_name:
            code = str(run["code"])
            scenario = str(run["scenario"])
            instrument = str(run["instrument"])
            size = int(run["train_size"])
            seed = int(run["seed"])
            bridge_train_name, bridge_test_name = bridge_filenames(code, scenario, instrument, size, seed)
        return {
            "train": train_path,
            "test": test_path,
            "stage2_summary": stage2_path,
            "bridge_train": effective_bridge_dir / bridge_train_name,
            "bridge_test": effective_bridge_dir / bridge_test_name,
        }

    # Legacy schema fallback
    train_path = Path(str(run["train_path"]))
    test_path = Path(str(run["test_path"]))
    stage2_summary = run.get("stage2_summary")
    stage2_path = Path(str(stage2_summary)) if stage2_summary else None

    legacy_bridge_train = Path(str(run.get("bridge_train_path", ""))) if run.get("bridge_train_path") else None
    legacy_bridge_test = Path(str(run.get("bridge_test_path", ""))) if run.get("bridge_test_path") else None

    if legacy_bridge_train and legacy_bridge_test:
        bridge_train = effective_bridge_dir / legacy_bridge_train.name
        bridge_test = effective_bridge_dir / legacy_bridge_test.name
    else:
        code = str(run["code"])
        scenario = str(run["scenario"])
        instrument = str(run["instrument"])
        size = int(run["train_size"])
        seed = int(run["seed"])
        bridge_train_name, bridge_test_name = bridge_filenames(code, scenario, instrument, size, seed)
        bridge_train = effective_bridge_dir / bridge_train_name
        bridge_test = effective_bridge_dir / bridge_test_name

    return {
        "train": train_path,
        "test": test_path,
        "stage2_summary": stage2_path,
        "bridge_train": bridge_train,
        "bridge_test": bridge_test,
    }


# --------------------------------------------------------------------------------------
# Ground-truth interventional mean via Monte Carlo
# --------------------------------------------------------------------------------------
def _simulate_y_given_x(
    cfg,
    x_value: float,
    eps_draws: np.ndarray,
    h_draws: Optional[np.ndarray],
    *,
    rng: Optional[np.random.Generator] = None,
    sigmaY_fn=None,
    latent_h_weight: Optional[float] = None,
) -> np.ndarray:
    if sigmaY_fn is None or latent_h_weight is None:
        from tabcf_core.dgp import LATENT_H_WEIGHT, sigmaY_of_X

        sigmaY_fn = sigmaY_of_X
        latent_h_weight = LATENT_H_WEIGHT
    rng = np.random.default_rng() if rng is None else rng
    eps_arr = np.asarray(eps_draws, dtype=float)
    x_arr = np.full_like(eps_arr, float(x_value), dtype=float)

    if cfg.second_stage == "B3":
        if h_draws is None:
            raise ValueError("B3 requires latent H draws")
        h_arr = np.asarray(h_draws, dtype=float)
        return x_arr - 3.0 * h_arr + eps_arr
    if cfg.second_stage == "B4":
        if h_draws is None:
            raise ValueError("B4 requires latent H draws")
        h_arr = np.asarray(h_draws, dtype=float)
        linear_branch = 0.2 * (5.5 + 2.0 * x_arr + 3.0 * h_arr + eps_arr)
        softplus_arg = (2.0 * x_arr + h_arr) ** 2 + eps_arr**2
        safe_arg = np.maximum(softplus_arg, B4_SOFTPLUS_EPS)
        softplus_branch = np.log(safe_arg)
        return np.where(x_arr <= 1.0, linear_branch, softplus_branch)
    if cfg.second_stage == "B5":
        if h_draws is None:
            raise ValueError("B5 requires latent H draws")
        h_arr = np.asarray(h_draws, dtype=float)
        return 3.0 * np.sin(2.0 * x_arr) + 2.0 * x_arr - 3.0 * h_arr + eps_arr
    if cfg.second_stage == "B1":
        if h_draws is None:
            raise ValueError("B1 requires latent H draws")
        h_arr = np.asarray(h_draws, dtype=float)
        sigma_y = sigmaY_fn(x_arr, cfg)
        m1 = cfg.beta1 * x_arr + cfg.beta2 * (x_arr**2)
        return m1 + sigma_y * (latent_h_weight * h_arr + eps_arr)
    if cfg.second_stage == "B2":
        if h_draws is None:
            raise ValueError("B2 requires latent H draws")
        h_arr = np.asarray(h_draws, dtype=float)
        n = len(eps_arr)
        mixture_indicators = rng.binomial(1, cfg.b2_mixture_weight, size=n)
        mu1 = np.sin(x_arr) + 0.3 * x_arr * h_arr
        mu2 = np.sin(x_arr + cfg.b2_beta_offset) + cfg.b2_peak_separation + 0.3 * x_arr * h_arr
        sigma1 = cfg.b2_sigma1 * (1.0 + 0.2 * np.abs(x_arr))
        sigma2 = cfg.b2_sigma2 * (1.0 + 0.3 * np.abs(x_arr))
        y1 = mu1 + sigma1 * rng.standard_normal(n)
        y2 = mu2 + sigma2 * rng.standard_normal(n)
        return mixture_indicators * y1 + (1 - mixture_indicators) * y2
    if cfg.second_stage == "B6":
        if h_draws is None:
            raise ValueError("B6 requires latent H draws")
        h_arr = np.asarray(h_draws, dtype=float)
        if h_arr.shape != eps_arr.shape:
            raise ValueError("Shape mismatch between eps_draws and h_draws for B6.")
        return 1.0 + 2.0 * np.cos(2.0 * x_arr + h_arr) + x_arr * h_arr
    if cfg.second_stage == "B7":
        if h_draws is None:
            raise ValueError("B7 requires latent H draws")
        h_arr = np.asarray(h_draws, dtype=float)
        if h_arr.shape != eps_arr.shape:
            raise ValueError("Shape mismatch between eps_draws and h_draws for B7.")
        u = np.arctan(x_arr / float(cfg.b7_x_scale))
        h_t = np.tanh(h_arr)
        return (
            cfg.b7_beta1 * u
            + cfg.b7_beta2 * np.sin(3.0 * u)
            + cfg.b7_h_weight * h_t * np.cos(2.0 * u)
            + cfg.b7_eps_scale * eps_arr
        )
    if cfg.second_stage == "B8":
        if h_draws is None:
            raise ValueError("B8 requires latent H draws")
        h_arr = np.asarray(h_draws, dtype=float)
        if h_arr.shape != eps_arr.shape:
            raise ValueError("Shape mismatch between eps_draws and h_draws for B8.")
        linear_branch = 1.0 + x_arr + 2.0 * h_arr + eps_arr
        softplus_arg = 2.0 * (x_arr + h_arr) ** 2 + eps_arr ** 2
        safe_arg = np.maximum(softplus_arg, B4_SOFTPLUS_EPS)
        softplus_branch = np.log(safe_arg)
        return np.where(x_arr <= 1.0, linear_branch, softplus_branch)
    if cfg.second_stage == "B9":
        if h_draws is None:
            raise ValueError("B9 requires latent H draws")
        h_arr = np.asarray(h_draws, dtype=float)
        if h_arr.shape != eps_arr.shape:
            raise ValueError("Shape mismatch between eps_draws and h_draws for B9.")
        return (
            3.0 * np.sin(2.0 * x_arr)
            + np.cos(0.5 * x_arr)
            + 2.0 * x_arr
            - 3.0 * h_arr
            + eps_arr
        )
    if cfg.second_stage == "B10":
        if h_draws is None:
            raise ValueError("B10 requires latent H draws")
        h_arr = np.asarray(h_draws, dtype=float)
        if h_arr.shape != eps_arr.shape:
            raise ValueError("Shape mismatch between eps_draws and h_draws for B10.")
        from tabcf_core.dgp import b10_response

        return b10_response(cfg, x_arr, h_arr, eps_arr)
    if cfg.second_stage == "B11":
        if h_draws is None:
            raise ValueError("B11 requires latent H draws")
        h_arr = np.asarray(h_draws, dtype=float)
        if h_arr.shape != eps_arr.shape:
            raise ValueError("Shape mismatch between eps_draws and h_draws for B11.")
        return 1.0 + 2.0 * x_arr + np.cos(2.0 * x_arr) + x_arr * h_arr - h_arr + eps_arr
    raise ValueError(f"Unsupported second_stage: {cfg.second_stage}")


def compute_y_clean_monte_carlo(
    cfg: DGPConfig,
    x: np.ndarray,
    *,
    n_samples: int = 5000,
    rng_seed: int = 1,
    force_h_zero: bool = False,
) -> np.ndarray:
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")

    rng = np.random.default_rng(rng_seed)
    from tabcf_core.dgp import LATENT_H_WEIGHT, sigmaY_of_X

    x_arr = np.atleast_1d(np.asarray(x, dtype=float))
    means = np.empty_like(x_arr, dtype=float)

    needs_h = cfg.second_stage in {"B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10", "B11"}
    for idx, x_val in enumerate(x_arr):
        eps_draws = rng.standard_normal(n_samples)
        if force_h_zero and needs_h:
            h_draws = np.zeros(n_samples)
        elif needs_h:
            h_draws = rng.standard_normal(n_samples)
        else:
            h_draws = None
        y_samples = _simulate_y_given_x(
            cfg,
            float(x_val),
            eps_draws,
            h_draws,
            rng=rng,
            sigmaY_fn=sigmaY_of_X,
            latent_h_weight=LATENT_H_WEIGHT,
        )
        means[idx] = float(np.mean(y_samples))
    return means


def make_dgp_config(code: str, seed: int | None = None) -> DGPConfig:
    if code not in CODE_SCENARIO_MAP:
        raise KeyError(f"Unknown DGP code: {code}")
    from tabcf_core.dgp import DGPConfig

    meta = CODE_SCENARIO_MAP[code]
    cfg = DGPConfig(first_stage=meta["first_stage"], second_stage=meta["second_stage"])
    if seed is not None:
        cfg.seed = seed
    return cfg
