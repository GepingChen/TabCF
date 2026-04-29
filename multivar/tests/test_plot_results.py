from __future__ import annotations

import sys

import pandas as pd

from multivar.pipeline import plot_results


def test_series_for_frame_prioritizes_new_core_models():
    df = pd.DataFrame(
        {
            "dgp_code": ["DGP1_LINEAR"],
            "n_train": [2000],
            "rho_eps": [0.6],
            "x": [0.0],
            "core_vs_oracle_mean": [0.2],
            "core_vs_oracle_std": [0.01],
            "tabpfn_real_vs_oracle_mean": [0.18],
            "tabpfn_real_vs_oracle_std": [0.01],
            "tabicl_vs_oracle_mean": [0.17],
            "tabicl_vs_oracle_std": [0.01],
            "tabpfn_naive_vs_oracle_mean": [0.25],
            "tabpfn_naive_vs_oracle_std": [0.01],
            "independence_vs_oracle_mean": [0.3],
            "independence_vs_oracle_std": [0.01],
            "div_vs_oracle_mean": [0.22],
            "div_vs_oracle_std": [0.01],
        }
    )

    series = plot_results._series_for_frame(df)
    labels = [entry[2] for entry in series]
    assert labels[:3] == [
        "TabCF (TabPFNv2.5)",
        "TabCF (TabICLv2)",
        "TabPFN-naive",
    ]


def test_make_figure_uses_header_row_and_horizontal_panels():
    rows = []
    for dgp_code, offset in [
        ("DGP1_LINEAR", 0.00),
        ("DGP3_PRE_ADDITIVE", 0.05),
        ("DGP4_PIECEWISE", 0.10),
        ("DGP5_SOFTPLUS", 0.15),
    ]:
        for x_value, step in [(0.0, 0.00), (0.5, 0.03), (1.0, 0.06)]:
            rows.append(
                {
                    "dgp_code": dgp_code,
                    "n_train": 2000,
                    "rho_eps": 0.6,
                    "x": x_value,
                    "core_vs_oracle_mean": 0.20 + offset + step,
                    "core_vs_oracle_std": 0.01,
                    "tabpfn_real_vs_oracle_mean": 0.18 + offset + step,
                    "tabpfn_real_vs_oracle_std": 0.01,
                    "tabicl_vs_oracle_mean": 0.17 + offset + step,
                    "tabicl_vs_oracle_std": 0.01,
                    "tabpfn_naive_vs_oracle_mean": 0.25 + offset + step,
                    "tabpfn_naive_vs_oracle_std": 0.01,
                    "independence_vs_oracle_mean": 0.30 + offset + step,
                    "independence_vs_oracle_std": 0.01,
                    "div_vs_oracle_mean": 0.22 + offset + step,
                    "div_vs_oracle_std": 0.01,
                }
            )

    df = pd.DataFrame(rows)

    fig, axes = plot_results.make_figure(
        df,
        dgp_codes=tuple(plot_results.DEFAULT_DGP_CODES),
        n_train=2000,
        rho_eps=0.6,
    )

    assert len(axes) == 4
    assert len(fig.axes) == 8
    assert [ax.get_title() for ax in axes] == ["", "", "", ""]
    assert axes[0].get_ylim()[0] == 0.0
    assert axes[1].get_ylim()[0] > 0.0
    assert axes[2].get_ylim()[0] == 0.0
    assert axes[3].get_ylim()[0] == 0.0

    header_labels = [ax.texts[0].get_text() for ax in fig.axes if ax.texts]
    assert header_labels == ["Linear", "Nonlinear", "Piecewise", "Softplus"]
    assert fig._supylabel is not None
    supylabel_x = fig._supylabel.get_position()[0]
    left_axis_x0 = axes[0].get_position().x0
    assert left_axis_x0 - supylabel_x >= 0.03

    plot_results.plt.close(fig)


def test_main_writes_png_and_pdf_for_s100_named_output(tmp_path, monkeypatch):
    rows = []
    for dgp_code, offset in [
        ("DGP1_LINEAR", 0.00),
        ("DGP3_PRE_ADDITIVE", 0.05),
        ("DGP4_PIECEWISE", 0.10),
        ("DGP5_SOFTPLUS", 0.15),
    ]:
        for x_value, step in [(0.0, 0.00), (0.5, 0.03), (1.0, 0.06)]:
            rows.append(
                {
                    "dgp_code": dgp_code,
                    "n_train": 2000,
                    "rho_eps": 0.6,
                    "x": x_value,
                    "core_vs_oracle_mean": 0.20 + offset + step,
                    "core_vs_oracle_std": 0.01,
                    "tabicl_vs_oracle_mean": 0.17 + offset + step,
                    "tabicl_vs_oracle_std": 0.01,
                    "tabpfn_naive_vs_oracle_mean": 0.25 + offset + step,
                    "tabpfn_naive_vs_oracle_std": 0.01,
                    "independence_vs_oracle_mean": 0.30 + offset + step,
                    "independence_vs_oracle_std": 0.01,
                    "div_vs_oracle_mean": 0.22 + offset + step,
                    "div_vs_oracle_std": 0.01,
                }
            )
    input_csv = tmp_path / "evaluate_wasserstein_with_tabpfn_real_tabicl_s100_case_x.csv"
    pd.DataFrame(rows).to_csv(input_csv, index=False)
    output_png = tmp_path / "wasserstein_official_dgp1345_n2000_rho0p6_with_tabpfn_real_tabicl_s100.png"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plot_results.py",
            "--input",
            str(input_csv),
            "--output",
            str(output_png),
        ],
    )

    plot_results.main()

    assert output_png.exists()
    assert output_png.with_suffix(".pdf").exists()
