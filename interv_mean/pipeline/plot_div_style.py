#!/usr/bin/env python3
"""
Visualize sec5.1 aggregated results with DIV-style faceting.

Layout:
- outer facets by treatment model (first-stage code, e.g. A3/A9)
- inner facets by outcome model (second-stage code, e.g. B3/B4/B5/B11)
- each inner facet plots exactly three training sizes as vertical result columns
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from interv_mean.pipeline import utils


DEFAULT_OFFICIAL_CSV = utils.DEFAULT_AGGREGATED_DIR / "aggregated_a3a9_b4b5b11_s100_xtrim0595_official.csv"
DEFAULT_BASE_CSV = utils.DEFAULT_AGGREGATED_DIR / "aggregated_a3a9_b3b4b5b11_s10.csv"
DEFAULT_MERGED_CSV = utils.DEFAULT_AGGREGATED_DIR / "aggregated_a3a9_b3b4b5b11_s10_tabicl.csv"
if DEFAULT_OFFICIAL_CSV.exists():
    DEFAULT_CSV = DEFAULT_OFFICIAL_CSV
elif DEFAULT_MERGED_CSV.exists():
    DEFAULT_CSV = DEFAULT_MERGED_CSV
else:
    DEFAULT_CSV = DEFAULT_BASE_CSV
DEFAULT_OUTPUT_STEM = utils.DEFAULT_VISUALIZATION_DIR / "interv_mean_div_style_b11_final_global_tabicl_s100_xg0595_20260421"
DEFAULT_TRAIN_SIZES = [1000, 4000, 10000]
DEFAULT_FIRST_STAGE_MAP = {"A3": "A", "A9": "B"}
DEFAULT_SECOND_STAGE_MAP = {"B3": "1", "B4": "2", "B5": "3", "B11": "4"}
DEFAULT_FIRST_STAGE_TABLE_LABELS = {
    "A3": "Linear,\nAdditive",
    "A9": "Nonlinear,\nNonadditive",
}
DEFAULT_SECOND_STAGE_TABLE_LABELS = {
    "B3": "Linear",
    "B4": "Piecewise",
    "B5": "Periodic",
    "B9": "Periodic,\nNonadditive",
    "B11": "Periodic,\nNonadditive",
}
SUPPORTED_OUTPUT_FORMATS = {".png": "png", ".pdf": "pdf"}
PLOT_FONT_FAMILY = "Latin Modern Roman"
EXCLUDED_METHODS = {
    ("tabpfn_cf", "tabpfn_cf_resid0"),
    ("hsic", "hsicx_notebook_runs10"),
    ("div", "div_layer4_epoch10000"),
}
EXCLUDED_SECOND_STAGES = {"B3"}


@dataclass(frozen=True)
class MethodSpec:
    label: str
    model: str
    variant: str
    marker: str
    color: str
    edgecolor: str
    size: float
    zorder: int


KNOWN_METHODS: Dict[Tuple[str, str], Dict[str, object]] = {
    ("tabpfn", "stage2_9"): {
        "label": "TabCF (TabPFNv2.5)",
        "marker": "h",
        "color": "#d62728",
        "edgecolor": "white",
        "size": 180.0,
        "zorder": 5,
    },
    ("tabpfn_local", "stage2_9_local_knn"): {
        "label": "Local-TabCF (TabPFNv2.5)",
        "marker": "D",
        "color": "#ff9896",
        "edgecolor": "#d62728",
        "size": 172.0,
        "zorder": 5,
    },
    ("tabpfn_real", "stage2_9"): {
        "label": "TabCF (Real-TabPFNv2.5)",
        "marker": "H",
        "color": "#8c564b",
        "edgecolor": "white",
        "size": 185.0,
        "zorder": 5,
    },
    ("tabpfn_real_local", "stage2_9_local_knn"): {
        "label": "Local-TabCF (Real-TabPFNv2.5)",
        "marker": "d",
        "color": "#c49c94",
        "edgecolor": "#8c564b",
        "size": 178.0,
        "zorder": 5,
    },
    ("tabicl", "stage2_9"): {
        "label": "TabCF (TabICLv2)",
        "marker": "^",
        "color": "#17becf",
        "edgecolor": "white",
        "size": 170.0,
        "zorder": 5,
    },
    ("tabicl_local", "stage2_9_local_knn"): {
        "label": "Local-TabCF (TabICLv2)",
        "marker": "v",
        "color": "#9edae5",
        "edgecolor": "#17becf",
        "size": 166.0,
        "zorder": 5,
    },
    ("linear_iv", "control_function_linear"): {
        "label": "Linear CF",
        "marker": "D",
        "color": "#6f8fc6",
        "edgecolor": "white",
        "size": 132.0,
        "zorder": 3,
    },
    ("nonlinear_iv", "control_function_spline_df5"): {
        "label": "Nonlinear CF",
        "marker": "o",
        "color": "#c76aa9",
        "edgecolor": "white",
        "size": 160.0,
        "zorder": 3,
    },
    ("div_2", "div_cran_layer3_epoch1000"): {
        "label": "DIV",
        "marker": "P",
        "color": "#4d8c57",
        "edgecolor": "white",
        "size": 148.0,
        "zorder": 4,
    },
    ("deepgmm", "deepgmm"): {
        "label": "DeepGMM",
        "marker": "s",
        "color": "#9bcf53",
        "edgecolor": "white",
        "size": 150.0,
        "zorder": 3,
    },
    ("deepiv", "deepiv_restarts10"): {
        "label": "DeepIV",
        "marker": (8, 2, 0),
        "color": "#ffcf20",
        "edgecolor": "#ffcf20",
        "size": 340.0,
        "zorder": 4,
    },
    ("tabpfn_naive", "tabpfn_naive_y_on_x"): {
        "label": "TabPFN-Naive",
        "marker": "X",
        "color": "#7f8c8d",
        "edgecolor": "white",
        "size": 155.0,
        "zorder": 2,
    },
}

PREFERRED_METHOD_ORDER = [
    ("tabpfn", "stage2_9"),
    ("tabpfn_local", "stage2_9_local_knn"),
    ("tabpfn_real", "stage2_9"),
    ("tabpfn_real_local", "stage2_9_local_knn"),
    ("tabicl", "stage2_9"),
    ("tabicl_local", "stage2_9_local_knn"),
    ("linear_iv", "control_function_linear"),
    ("nonlinear_iv", "control_function_spline_df5"),
    ("div_2", "div_cran_layer3_epoch1000"),
    ("deepgmm", "deepgmm"),
    ("deepiv", "deepiv_restarts10"),
    ("tabpfn_naive", "tabpfn_naive_y_on_x"),
]

UNKNOWN_METHOD_MARKERS = ["v", "<", ">", "p", "h", "X", "d"]
UNKNOWN_METHOD_COLORS = [
    "#4e79a7",
    "#f28e2b",
    "#e15759",
    "#76b7b2",
    "#59a14f",
    "#edc948",
    "#b07aa1",
]


def _natural_stage_key(token: str) -> Tuple[str, int, str]:
    match = re.match(r"([A-Za-z]+)(\d+)$", token)
    if not match:
        return (token, math.inf, token)
    return (match.group(1), int(match.group(2)), token)


def _safe_float(value: str) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: str) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def load_records(csv_path: Path) -> List[dict]:
    records: List[dict] = []
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            train_size = _safe_int(row.get("train_size", ""))
            mean_mse = _safe_float(row.get("mean_mse", ""))
            std_mse = _safe_float(row.get("std_mse", ""))
            if train_size is None or mean_mse is None or std_mse is None:
                continue
            records.append(
                {
                    "model": row.get("model", "").strip(),
                    "variant": row.get("variant", "").strip(),
                    "code": row.get("code", "").strip(),
                    "scenario": row.get("scenario", "").strip(),
                    "train_size": train_size,
                    "mean_mse": mean_mse,
                    "std_mse": std_mse,
                    "n_runs": _safe_int(row.get("n_runs", "")),
                }
            )
    return records


def _parse_code_stages(code: str) -> Tuple[str, str]:
    meta = utils.CODE_SCENARIO_MAP.get(code)
    if meta:
        return meta["first_stage"], meta["second_stage"]

    match = re.match(r"^(A\d+)_?(B\d+)$", code)
    if match:
        return match.group(1), match.group(2)

    raise KeyError(f"Unable to infer first/second stage from code '{code}'")


def _format_stage_label(stage_token: str, mapping: Mapping[str, str]) -> str:
    return mapping.get(stage_token, stage_token)


def _format_panel_label(first_stage: str, second_stage: str, first_map: Mapping[str, str], second_map: Mapping[str, str]) -> str:
    return f"{_format_stage_label(first_stage, first_map)}{_format_stage_label(second_stage, second_map)}"


def _table_first_stage_label(
    first_stage: str,
    labels: Optional[Mapping[str, str]] = None,
) -> str:
    table_labels = DEFAULT_FIRST_STAGE_TABLE_LABELS if labels is None else labels
    return table_labels.get(first_stage, first_stage)


def _table_second_stage_label(
    second_stage: str,
    labels: Optional[Mapping[str, str]] = None,
) -> str:
    table_labels = DEFAULT_SECOND_STAGE_TABLE_LABELS if labels is None else labels
    return table_labels.get(second_stage, second_stage)


def _format_vertical_first_stage_label(
    first_stage: str,
    labels: Optional[Mapping[str, str]] = None,
) -> str:
    return _table_first_stage_label(first_stage, labels).replace("\n", " ")


def _label_unknown_method(model: str, variant: str) -> str:
    base = model.replace("_", " ").strip()
    if not base:
        base = variant.replace("_", " ").strip()
    parts = [token.capitalize() for token in base.split() if token]
    label = " ".join(parts) if parts else "Unknown"
    if variant and variant != model:
        label = f"{label} [{variant}]"
    return label


def build_method_specs(records: Iterable[dict]) -> List[MethodSpec]:
    seen_pairs = {
        (str(record["model"]), str(record["variant"]))
        for record in records
        if (str(record["model"]), str(record["variant"])) not in EXCLUDED_METHODS
    }

    ordered_pairs: List[Tuple[str, str]] = []
    for pair in PREFERRED_METHOD_ORDER:
        if pair in seen_pairs:
            ordered_pairs.append(pair)
    unknown_pairs = sorted(pair for pair in seen_pairs if pair not in set(ordered_pairs))
    ordered_pairs.extend(unknown_pairs)

    specs: List[MethodSpec] = []
    for index, pair in enumerate(ordered_pairs):
        style = KNOWN_METHODS.get(pair)
        if style is None:
            style = {
                "label": _label_unknown_method(pair[0], pair[1]),
                "marker": UNKNOWN_METHOD_MARKERS[index % len(UNKNOWN_METHOD_MARKERS)],
                "color": UNKNOWN_METHOD_COLORS[index % len(UNKNOWN_METHOD_COLORS)],
                "edgecolor": "white",
                "size": 145.0,
                "zorder": 2,
            }
        specs.append(
            MethodSpec(
                label=str(style["label"]),
                model=pair[0],
                variant=pair[1],
                marker=style["marker"],
                color=str(style["color"]),
                edgecolor=str(style["edgecolor"]),
                size=float(style["size"]),
                zorder=int(style["zorder"]),
            )
        )
    return specs


def _parse_key_value_pairs(items: Sequence[str], arg_name: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"{arg_name} entries must use KEY=VALUE syntax: got '{item}'")
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise ValueError(f"{arg_name} entries must use non-empty KEY=VALUE syntax: got '{item}'")
        mapping[key] = value
    return mapping


def parse_stage_maps(args: argparse.Namespace) -> Tuple[Dict[str, str], Dict[str, str]]:
    first_stage_map = dict(DEFAULT_FIRST_STAGE_MAP)
    first_stage_map.update(_parse_key_value_pairs(args.first_stage_map, "--first-stage-map"))

    second_stage_map = dict(DEFAULT_SECOND_STAGE_MAP)
    second_stage_map.update(_parse_key_value_pairs(args.second_stage_map, "--second-stage-map"))
    return first_stage_map, second_stage_map


def group_records(
    records: Iterable[dict],
    train_sizes: Sequence[int],
    first_stage_map: Mapping[str, str],
    second_stage_map: Mapping[str, str],
) -> Dict[str, object]:
    selected_sizes = [int(size) for size in train_sizes]
    size_set = set(selected_sizes)
    grouped_lookup: MutableMapping[Tuple[str, str, int, str, str], dict] = {}
    treatment_order: List[str] = []
    outcome_order_by_treatment: Dict[str, List[str]] = {}

    for record in records:
        if int(record["train_size"]) not in size_set:
            continue
        pair = (str(record["model"]), str(record["variant"]))
        if pair in EXCLUDED_METHODS:
            continue

        first_stage, second_stage = _parse_code_stages(str(record["code"]))
        if second_stage in EXCLUDED_SECOND_STAGES:
            continue
        if first_stage not in outcome_order_by_treatment:
            treatment_order.append(first_stage)
            outcome_order_by_treatment[first_stage] = []
        if second_stage not in outcome_order_by_treatment[first_stage]:
            outcome_order_by_treatment[first_stage].append(second_stage)

        key = (first_stage, second_stage, int(record["train_size"]), str(record["model"]), str(record["variant"]))
        grouped_lookup[key] = record

    treatment_order.sort(key=_natural_stage_key)
    for first_stage, values in outcome_order_by_treatment.items():
        values.sort(key=_natural_stage_key)

    panel_labels = {
        (first_stage, second_stage): _format_panel_label(first_stage, second_stage, first_stage_map, second_stage_map)
        for first_stage in treatment_order
        for second_stage in outcome_order_by_treatment.get(first_stage, [])
    }
    return {
        "lookup": grouped_lookup,
        "treatment_order": treatment_order,
        "outcome_order_by_treatment": outcome_order_by_treatment,
        "panel_labels": panel_labels,
        "train_sizes": selected_sizes,
    }


def _build_legend_handles(method_specs: Sequence[MethodSpec]):
    from matplotlib.lines import Line2D

    handles = []
    for spec in method_specs:
        markersize = max(9.0, math.sqrt(spec.size))
        handles.append(
            Line2D(
                [0],
                [0],
                marker=spec.marker,
                linestyle="None",
                markerfacecolor=spec.color,
                markeredgecolor=spec.edgecolor,
                markeredgewidth=1.1,
                color="none",
                markersize=markersize,
                label=spec.label,
            )
        )
    return handles


def _scatter_kwargs(spec: MethodSpec) -> Dict[str, object]:
    kwargs: Dict[str, object] = {
        "s": spec.size,
        "marker": spec.marker,
        "color": spec.color,
        "linewidth": 1.0,
        "zorder": spec.zorder,
    }
    # Matplotlib warns when edgecolor is passed for tuple-style unfilled markers
    # (used here for the DeepIV "mosquito" glyph). Omit the edgecolor in that case.
    if not isinstance(spec.marker, tuple):
        kwargs["edgecolor"] = spec.edgecolor
    return kwargs


def _infer_output_format(output_path: Path) -> str:
    suffix = output_path.suffix.lower()
    fmt = SUPPORTED_OUTPUT_FORMATS.get(suffix)
    if fmt is None:
        supported = ", ".join(sorted(SUPPORTED_OUTPUT_FORMATS))
        raise ValueError(
            f"Unsupported output extension '{output_path.suffix}'. "
            f"Use one of: {supported}"
        )
    return fmt


def _normalize_output_stem(output_path: Path) -> Path:
    if output_path.suffix.lower() in SUPPORTED_OUTPUT_FORMATS:
        return output_path.with_suffix("")
    return output_path


def _output_paths_for_stem(output_stem: Path) -> List[Path]:
    stem = _normalize_output_stem(output_stem)
    return [stem.with_suffix(".png"), stem.with_suffix(".pdf")]


def _register_latin_modern_fonts(font_manager: object) -> None:
    tectonic_cache = Path.home() / ".cache" / "Tectonic" / "bundles" / "data"
    if not tectonic_cache.exists():
        return
    for font_path in sorted(tectonic_cache.glob("*/lmroman*.otf")):
        font_manager.fontManager.addfont(str(font_path))


def _collect_missing(
    grouped: Mapping[str, object],
    method_specs: Sequence[MethodSpec],
) -> List[Tuple[str, int, str]]:
    lookup = grouped["lookup"]
    treatment_order = grouped["treatment_order"]
    outcome_order_by_treatment = grouped["outcome_order_by_treatment"]
    panel_labels = grouped["panel_labels"]
    train_sizes = grouped["train_sizes"]

    missing: List[Tuple[str, int, str]] = []
    for first_stage in treatment_order:
        for second_stage in outcome_order_by_treatment[first_stage]:
            panel_label = panel_labels[(first_stage, second_stage)]
            for train_size in train_sizes:
                for spec in method_specs:
                    key = (first_stage, second_stage, train_size, spec.model, spec.variant)
                    if key not in lookup:
                        missing.append((panel_label, train_size, spec.label))
    return missing


def plot_div_style(
    grouped: Mapping[str, object],
    method_specs: Sequence[MethodSpec],
    output_stem: Path,
    *,
    dpi: int = 300,
    first_stage_table_labels: Optional[Mapping[str, str]] = None,
    second_stage_table_labels: Optional[Mapping[str, str]] = None,
) -> List[Path]:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
        from matplotlib import font_manager
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to run this visualization script.") from exc

    font_scale = 1.42
    _register_latin_modern_fonts(font_manager)
    font_path = font_manager.findfont(PLOT_FONT_FAMILY, fallback_to_default=False)

    # Publication-font note:
    # - TMLR figures are kept consistent with LaTeX defaults, i.e. Computer Modern / Latin Modern.
    # - NeurIPS figures are commonly prepared with Times-like serif text.
    # Use the Latin Modern OTFs from the local Tectonic cache, without fallback.
    plt.rcParams.update(
        {
            "font.family": PLOT_FONT_FAMILY,
            "font.serif": [PLOT_FONT_FAMILY],
            "mathtext.fontset": "cm",
            "axes.formatter.use_mathtext": True,
            "axes.unicode_minus": False,
            "axes.labelsize": 13.5 * font_scale,
            "axes.titlesize": 13.0 * font_scale,
            "xtick.labelsize": 10.0 * font_scale,
            "ytick.labelsize": 10.0 * font_scale,
            "legend.fontsize": 10.5 * font_scale,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    print(f"Using plot font: {PLOT_FONT_FAMILY} ({font_path})")

    treatment_order = grouped["treatment_order"]
    outcome_order_by_treatment = grouped["outcome_order_by_treatment"]
    lookup = grouped["lookup"]
    panel_labels = grouped["panel_labels"]
    train_sizes = grouped["train_sizes"]

    if not treatment_order:
        raise ValueError("No treatment groups available after filtering records.")

    n_treatments = len(treatment_order)
    ordered_outcomes = sorted(
        {outcome for first_stage in treatment_order for outcome in outcome_order_by_treatment[first_stage]},
        key=_natural_stage_key,
    )
    n_outcomes = len(ordered_outcomes)
    # Keep the canvas closer to the final manuscript width so the rendered
    # fonts do not get shrunk too aggressively when LaTeX includes the PDF.
    width = max(12.15, 1.55 + 3.55 * n_outcomes)
    height = max(10.25, 3.0 + 4.25 * n_treatments)

    fig = plt.figure(figsize=(width, height), facecolor="white")
    fig.patch.set_edgecolor("none")
    fig.patch.set_linewidth(0.0)
    grid = fig.add_gridspec(
        n_treatments + 1,
        n_outcomes + 1,
        left=0.04,
        right=0.994,
        bottom=0.108,
        top=0.865,
        hspace=0.085,
        wspace=0.105,
        width_ratios=[0.78] + [4.55] * n_outcomes,
        height_ratios=[0.44] + [5.2] * n_treatments,
    )
    x_positions = list(range(len(train_sizes)))
    if len(method_specs) == 1:
        offsets = [0.0]
    else:
        spread = 0.34
        step = spread / max(len(method_specs) - 1, 1)
        offsets = [(-spread / 2.0) + idx * step for idx in range(len(method_specs))]

    def _style_header_cell(ax: object, *, show_border: bool, facecolor: str = "#ffffff") -> None:
        ax.set_facecolor(facecolor)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(show_border)
            if show_border:
                spine.set_color("#d7d7d1")
                spine.set_linewidth(0.85)

    corner_ax = fig.add_subplot(grid[0, 0])
    _style_header_cell(corner_ax, show_border=False, facecolor="#ffffff")

    for outcome_index, second_stage in enumerate(ordered_outcomes, start=1):
        header_ax = fig.add_subplot(grid[0, outcome_index])
        _style_header_cell(header_ax, show_border=False)
        header_ax.text(
            0.5,
            -0.5,
            _table_second_stage_label(second_stage, second_stage_table_labels),
            ha="center",
            va="bottom",
            fontsize=12.2 * font_scale,
            fontweight="bold",
            color="#2f2f2f",
            wrap=True,
        )

    for treatment_index, first_stage in enumerate(treatment_order, start=1):
        row_header_ax = fig.add_subplot(grid[treatment_index, 0])
        _style_header_cell(row_header_ax, show_border=False)
        row_header_ax.text(
            0.42,
            0.5,
            _format_vertical_first_stage_label(first_stage, first_stage_table_labels),
            ha="center",
            va="center",
            fontsize=11.8 * font_scale,
            fontweight="bold",
            color="#2f2f2f",
            rotation=90,
        )

        for outcome_index, second_stage in enumerate(ordered_outcomes, start=1):
            ax = fig.add_subplot(grid[treatment_index, outcome_index])
            # Keep panel backgrounds visually white without letting later axes
            # cover neighboring tick labels in tight column layouts.
            ax.set_facecolor("none")
            ax.set_zorder(n_outcomes - outcome_index + 1)
            panel_y_values: List[float] = []
            for method_index, spec in enumerate(method_specs):
                xs: List[float] = []
                ys: List[float] = []
                for train_pos, train_size in enumerate(train_sizes):
                    record = lookup.get((first_stage, second_stage, train_size, spec.model, spec.variant))
                    if record is None:
                        continue
                    xs.append(x_positions[train_pos] + offsets[method_index])
                    ys.append(float(record["mean_mse"]))
                if xs:
                    panel_y_values.extend(ys)
                    ax.scatter(xs, ys, **_scatter_kwargs(spec))

            # Reserve a bit more in-panel room for the edge tick labels
            # (especially "10000"), so they stay inside the panel bbox.
            ax.set_xlim(-0.8, len(train_sizes) - 0.2)
            ax.set_xticks(x_positions)
            ax.set_xticklabels([str(size) for size in train_sizes], fontsize=9.8 * font_scale)
            ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4, min_n_ticks=3))
            ax.tick_params(axis="y", labelsize=9.8 * font_scale, length=0, pad=6, colors="#4a4a46")
            ax.tick_params(axis="x", labelsize=9.8 * font_scale, length=0, pad=5, colors="#4a4a46")
            ax.grid(True, axis="y", color="#d9ddd6", linewidth=0.8, alpha=0.95)
            ax.grid(True, axis="x", color="#eceee9", linewidth=0.75, alpha=0.95)
            ax.set_axisbelow(True)
            ax.margins(x=0.05, y=0.16)
            if panel_y_values and min(panel_y_values) >= 0:
                _, current_top = ax.get_ylim()
                ax.set_ylim(bottom=-0.035 * current_top, top=current_top)
            for spine in ax.spines.values():
                spine.set_visible(False)

    legend_handles = _build_legend_handles(method_specs)
    legend = fig.legend(
        handles=legend_handles,
        labels=[spec.label for spec in method_specs],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=min(len(method_specs), 4),
        frameon=True,
        fancybox=False,
        edgecolor="#5d5d57",
        fontsize=10.2 * font_scale,
        borderpad=0.72,
        labelspacing=0.72,
        columnspacing=1.0,
        handletextpad=0.58,
    )
    legend.get_frame().set_alpha(1.0)
    legend.get_frame().set_linewidth(0.85)
    legend.get_frame().set_facecolor("#ffffff")

    fig.supylabel("MSE", fontsize=13.5 * font_scale, x=0.0295, color="#2f2f2f")
    fig.supxlabel("Training Size", fontsize=13.5 * font_scale, y=0.058)

    output_paths = _output_paths_for_stem(output_stem)
    output_paths[0].parent.mkdir(parents=True, exist_ok=True)
    for output_path in output_paths:
        output_format = _infer_output_format(output_path)
        savefig_kwargs: Dict[str, object] = {
            "format": output_format,
            "bbox_inches": "tight",
            "pad_inches": 0.02,
            "facecolor": "white",
            "edgecolor": "none",
        }
        if output_format != "pdf":
            savefig_kwargs["dpi"] = dpi
        fig.savefig(output_path, **savefig_kwargs)
    plt.close(fig)
    return output_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot sec5.1 aggregated results using DIV-style faceting."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help="Path to aggregated CSV (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_STEM,
        help=(
            "Output path stem for the visualization. If .png or .pdf is supplied, "
            "the suffix is stripped and both formats are written (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure DPI for the saved image.",
    )
    parser.add_argument(
        "--train-sizes",
        nargs="+",
        type=int,
        default=DEFAULT_TRAIN_SIZES,
        help="Exactly three training sizes to show as vertical result columns.",
    )
    parser.add_argument(
        "--first-stage-map",
        nargs="*",
        default=[],
        help="Custom first-stage labels using KEY=VALUE syntax, e.g. A3=A A9=B.",
    )
    parser.add_argument(
        "--second-stage-map",
        nargs="*",
        default=[],
        help="Custom second-stage labels using KEY=VALUE syntax, e.g. B3=1 B4=2.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.train_sizes) != 3:
        raise SystemExit("--train-sizes must contain exactly 3 values.")

    train_sizes = [int(size) for size in args.train_sizes]
    first_stage_map, second_stage_map = parse_stage_maps(args)
    records = load_records(args.csv)
    if not records:
        raise SystemExit(f"No rows loaded from {args.csv}")

    filtered_records = [record for record in records if int(record["train_size"]) in set(train_sizes)]
    if not filtered_records:
        raise SystemExit(
            f"No rows in {args.csv} match the requested train sizes {train_sizes}"
        )

    grouped = group_records(filtered_records, train_sizes, first_stage_map, second_stage_map)
    method_specs = build_method_specs(filtered_records)
    if not method_specs:
        raise SystemExit(f"No plottable methods found in {args.csv}")

    output_paths = plot_div_style(grouped, method_specs, args.output, dpi=args.dpi)
    missing = _collect_missing(grouped, method_specs)
    if missing:
        print("Missing combinations (not plotted):")
        for panel_label, train_size, method_label in missing:
            print(f"  {panel_label}, n={train_size}: {method_label}")
    for output_path in output_paths:
        print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    main()
