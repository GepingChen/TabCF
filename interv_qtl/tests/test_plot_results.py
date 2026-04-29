from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from interv_qtl.viz import plot_results


def test_load_report_and_save(tmp_path: Path):
    csv_path = tmp_path / "report.csv"
    pd.DataFrame(
        {
            "code": ["A3_B4", "A3_B4", "A9_B5", "A9_B5"],
            "tau": [0.1, 0.5, 0.1, 0.5],
            "rmse_new_median_seed": [0.3, 0.2, 0.4, 0.35],
        }
    ).to_csv(csv_path, index=False)

    df = plot_results.load_report(csv_path, codes=("A3_B4", "A9_B5"))
    fig, _axes = plot_results.make_figure(df, codes=("A3_B4", "A9_B5"))
    png_path, pdf_path = plot_results.save_figure(fig, tmp_path / "official_report", dpi=100)

    assert png_path.exists()
    assert pdf_path.exists()
