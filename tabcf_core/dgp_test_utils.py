from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
import pandas as pd


DEFAULT_TEST_X_TRIM_QUANTILE_RANGE: tuple[float, float] = (0.05, 0.95)


def normalize_test_x_trim_quantile_range(
    quantile_range: tuple[float, float] | list[float] | None,
) -> tuple[float, float] | None:
    """Validate and normalize an optional [lower, upper] X-trim range."""
    if quantile_range is None:
        return None
    if len(quantile_range) != 2:
        raise ValueError("test_x_trim_quantile_range must contain exactly two values.")
    lower = float(quantile_range[0])
    upper = float(quantile_range[1])
    if not (0.0 <= lower < upper <= 1.0):
        raise ValueError(
            "test_x_trim_quantile_range must satisfy 0.0 <= lower < upper <= 1.0."
        )
    return (lower, upper)


def select_test_row_indices_by_x(
    x_values: np.ndarray,
    quantile_range: tuple[float, float] | list[float] | None = DEFAULT_TEST_X_TRIM_QUANTILE_RANGE,
) -> np.ndarray:
    """
    Select test rows by X rank and return retained positions in original row order.

    Rows are ranked by ascending X using a stable sort. The retained ranked slice is:
      start = floor(n * lower_q)
      stop = ceil(n * upper_q)
    """
    x_arr = np.asarray(x_values, dtype=float).reshape(-1)
    n_rows = int(x_arr.size)
    if n_rows == 0:
        return np.empty(0, dtype=int)

    normalized_range = normalize_test_x_trim_quantile_range(quantile_range)
    if normalized_range is None:
        return np.arange(n_rows, dtype=int)

    lower_q, upper_q = normalized_range
    start = max(0, min(int(math.floor(n_rows * lower_q)), n_rows))
    stop = max(start, min(int(math.ceil(n_rows * upper_q)), n_rows))

    if start == 0 and stop == n_rows:
        return np.arange(n_rows, dtype=int)

    ranked_positions = np.argsort(x_arr, kind="mergesort")
    retained_positions = ranked_positions[start:stop]
    if retained_positions.size == n_rows:
        return np.arange(n_rows, dtype=int)
    return np.sort(retained_positions, kind="mergesort")


def trim_test_dataframe_by_x(
    df: pd.DataFrame,
    quantile_range: tuple[float, float] | list[float] | None = DEFAULT_TEST_X_TRIM_QUANTILE_RANGE,
    *,
    x_col: str = "X",
) -> pd.DataFrame:
    """Trim a test DataFrame by X rank while preserving original row order."""
    if x_col not in df.columns:
        raise ValueError(f"Test DataFrame is missing required X column '{x_col}'.")
    selected_positions = select_test_row_indices_by_x(df[x_col].to_numpy(), quantile_range)
    if len(selected_positions) == len(df):
        return df.reset_index(drop=True)
    return df.iloc[selected_positions].reset_index(drop=True)


def trim_test_array_dict_by_x(
    data: Dict[str, Optional[np.ndarray]],
    quantile_range: tuple[float, float] | list[float] | None = DEFAULT_TEST_X_TRIM_QUANTILE_RANGE,
    *,
    x_key: str = "X",
) -> Dict[str, Optional[np.ndarray]]:
    """Trim array-valued test data entries by X rank while preserving row alignment."""
    if x_key not in data or data[x_key] is None:
        raise ValueError(f"Test data is missing required X entry '{x_key}'.")

    x_arr = np.asarray(data[x_key], dtype=float).reshape(-1)
    selected_positions = select_test_row_indices_by_x(x_arr, quantile_range)
    if len(selected_positions) == len(x_arr):
        return dict(data)

    trimmed: Dict[str, Optional[np.ndarray]] = {}
    for key, value in data.items():
        if value is None:
            trimmed[key] = None
            continue
        arr = np.asarray(value)
        if arr.ndim == 0 or len(arr) != len(x_arr):
            trimmed[key] = value
            continue
        trimmed[key] = arr[selected_positions]
    return trimmed
