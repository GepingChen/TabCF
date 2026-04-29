from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from empirical import run_empirical_mean as pipeline
from empirical import plot_results as vis


def _clean_df_for_spec(spec: pipeline.DatasetSpec) -> pd.DataFrame:
    if spec.discrete_x:
        x = np.array([8.0, 10.0, 12.0, 16.0], dtype=float)
    else:
        x = np.linspace(-0.5, 0.5, 6)
    return pd.DataFrame(
        {
            "unit": [f"u{idx}" for idx in range(len(x))],
            "X": x,
            "Z": np.linspace(0.0, 1.0, len(x)),
            "Y": 1.5 + 0.8 * x,
        }
    )


def _preds_df_for_clean(clean_df: pd.DataFrame) -> pd.DataFrame:
    x = clean_df["X"].to_numpy(dtype=float)
    return pd.DataFrame(
        {
            "X": x,
            "ols_pred": 1.0 + 0.4 * x,
            "tsls_pred": 1.1 + 0.35 * x,
            "div_pred": 0.9 + 0.3 * x,
            "tabcf_pred_tabpfn": 1.2 + 0.45 * x,
        }
    )


def _write_panel_dir(base_dir: Path, dataset_key: str) -> Path:
    spec = pipeline.DATASET_SPECS[dataset_key]
    out_dir = base_dir / dataset_key
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = pipeline.build_output_paths(spec, out_dir)
    clean_df = _clean_df_for_spec(spec)
    preds_df = _preds_df_for_clean(clean_df)
    clean_df.to_csv(paths["clean_csv"], index=False)
    preds_df.to_csv(paths["merged_pred_csv"], index=False)
    return out_dir


def test_load_panel_accepts_legacy_single_tabcf_column(tmp_path):
    out_dir = _write_panel_dir(tmp_path, "ajr")
    spec = pipeline.DATASET_SPECS["ajr"]
    paths = pipeline.build_output_paths(spec, out_dir)
    preds_df = pd.read_csv(paths["merged_pred_csv"]).rename(columns={"tabcf_pred_tabpfn": "tabcf_pred"})
    preds_df.to_csv(paths["merged_pred_csv"], index=False)

    task = vis.OfficialTaskSpec("ajr", "AJR colonial origins", out_dir)
    panel = vis.load_panel(task)
    assert "tabcf_pred_tabpfn" in panel.preds_df.columns


def test_make_figure_builds_shared_legend_and_panel_titles(tmp_path):
    tasks = (
        vis.OfficialTaskSpec("ajr", "AJR colonial origins", _write_panel_dir(tmp_path, "ajr")),
        vis.OfficialTaskSpec("fulton", "Fulton Fish Market", _write_panel_dir(tmp_path, "fulton")),
        vis.OfficialTaskSpec("card", "Card college proximity", _write_panel_dir(tmp_path, "card")),
        vis.OfficialTaskSpec("cigarettes", "CigarettesSW", _write_panel_dir(tmp_path, "cigarettes")),
    )

    panels = vis.load_panels(tasks)
    fig, axes, legend = vis.make_figure(panels)
    try:
        assert len(axes) == 4
        assert [ax.get_title() for ax in axes] == [
            "AJR colonial origins",
            "Fulton Fish Market",
            "Card college proximity",
            "CigarettesSW",
        ]
        assert [text.get_text() for text in legend.get_texts()] == [
            "TabCF (TabPFNv2.5)",
            "DIV",
            "2SLS",
            "OLS",
        ]
        card_axis = axes[2]
        assert np.allclose(card_axis.get_xticks(), [8.0, 10.0, 12.0, 16.0])
    finally:
        plt.close(fig)


def test_main_writes_png_and_pdf(tmp_path):
    output_stem = tmp_path / "official" / "empirical_comparison_grid"
    tasks = (
        vis.OfficialTaskSpec("ajr", "AJR colonial origins", _write_panel_dir(tmp_path, "ajr")),
        vis.OfficialTaskSpec("fulton", "Fulton Fish Market", _write_panel_dir(tmp_path, "fulton")),
        vis.OfficialTaskSpec("card", "Card college proximity", _write_panel_dir(tmp_path, "card")),
        vis.OfficialTaskSpec("cigarettes", "CigarettesSW", _write_panel_dir(tmp_path, "cigarettes")),
    )

    panels = vis.load_panels(tasks)
    fig, _axes, _legend = vis.make_figure(panels)
    try:
        png_path, pdf_path = vis.save_figure(fig, output_stem)
    finally:
        plt.close(fig)

    assert png_path.exists()
    assert pdf_path.exists()
