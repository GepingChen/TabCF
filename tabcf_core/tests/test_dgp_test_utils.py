from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_DIR = REPO_ROOT / "tabcf_core"

for candidate in (CORE_DIR, REPO_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from dgp_test_utils import trim_test_dataframe_by_x


def _make_test_frame(x_values: np.ndarray) -> pd.DataFrame:
    x_arr = np.asarray(x_values, dtype=float)
    return pd.DataFrame(
        {
            "Z": np.linspace(0.0, 1.0, len(x_arr)),
            "X": x_arr,
            "Y": 100.0 + x_arr,
            "V_true": np.linspace(0.1, 0.9, len(x_arr)),
            "eps": np.zeros(len(x_arr), dtype=float),
            "eta": np.linspace(0.1, 0.9, len(x_arr)),
        }
    )


def test_trim_test_dataframe_by_x_default_range_preserves_original_order():
    df = _make_test_frame(
        np.array([10.0, 0.0, 9.0, 1.0, 8.0, 2.0, 7.0, 3.0, 6.0, 4.0, 5.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0])
    )

    trimmed = trim_test_dataframe_by_x(df)

    assert len(trimmed) == 18
    assert list(trimmed["X"]) == [10.0, 9.0, 1.0, 8.0, 2.0, 7.0, 3.0, 6.0, 4.0, 5.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0]
    assert list(trimmed["Y"]) == [110.0, 109.0, 101.0, 108.0, 102.0, 107.0, 103.0, 106.0, 104.0, 105.0, 111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0]


def test_trim_test_dataframe_by_x_small_inputs_stay_non_empty():
    single = trim_test_dataframe_by_x(_make_test_frame(np.array([3.0])))
    pair = trim_test_dataframe_by_x(_make_test_frame(np.array([3.0, -1.0])))

    assert list(single["X"]) == [3.0]
    assert list(pair["X"]) == [3.0, -1.0]


def test_load_dgp_data_trims_only_test_split_by_default(tmp_path):
    stage1 = pytest.importorskip("stage1_control")

    train_dir = tmp_path / "train"
    test_dir = tmp_path / "test"
    train_dir.mkdir()
    test_dir.mkdir()

    train_df = _make_test_frame(np.array([0.0, 1.0, 2.0]))
    test_df = _make_test_frame(np.arange(20, dtype=float))

    train_df.to_csv(train_dir / "train_data_A3_B3_n3_seed1.csv", index=False)
    test_df.to_csv(test_dir / "test_data_A3_B3.csv", index=False)

    loaded_train, loaded_test = stage1.load_dgp_data(
        "A3",
        "B3",
        train_sample_size=3,
        seed=1,
        base_dir=tmp_path,
    )

    assert len(loaded_train) == 3
    assert list(loaded_train["X"]) == [0.0, 1.0, 2.0]
    assert len(loaded_test) == 18
    assert float(loaded_test["X"].min()) == 1.0
    assert float(loaded_test["X"].max()) == 18.0
