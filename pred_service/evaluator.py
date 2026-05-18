"""
Evaluation metrics for time-series forecasts.

All functions handle NaN values and edge cases (e.g. zero denominators for MAPE).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def evaluate(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    """
    Compute MAE, RMSE, MAPE, and R2 between actual and predicted series.

    Both series are aligned on their index before computing metrics.
    NaN values in either series are dropped pairwise.

    Returns a dict with keys: mae, rmse, mape, r2.
    """
    # Align on common index
    actual, predicted = actual.align(predicted, join="inner")

    # Drop NaN pairs
    mask = actual.notna() & predicted.notna()
    actual = actual[mask]
    predicted = predicted[mask]

    if len(actual) == 0:
        logger.warning("No valid (non-NaN) pairs to evaluate — returning NaN metrics.")
        return {"mae": float("nan"), "rmse": float("nan"), "mape": float("nan"), "r2": float("nan")}

    a = actual.to_numpy(dtype=float)
    p = predicted.to_numpy(dtype=float)

    mae = float(np.mean(np.abs(a - p)))
    rmse = float(np.sqrt(np.mean((a - p) ** 2)))
    r2 = _r2(a, p)
    mape = _mape(a, p)

    return {"mae": mae, "rmse": rmse, "mape": mape, "r2": r2}


def _r2(actual: np.ndarray, predicted: np.ndarray) -> float:
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    if ss_tot == 0.0:
        # Constant series: R2 is 1 if predictions are perfect, else undefined
        return 1.0 if ss_res == 0.0 else float("nan")
    return float(1.0 - ss_res / ss_tot)


def _mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Mean Absolute Percentage Error.

    Rows where actual == 0 are excluded to avoid division by zero.
    Returns NaN if all actual values are zero.
    """
    nonzero = actual != 0.0
    if not np.any(nonzero):
        logger.warning("All actual values are zero — MAPE is undefined, returning NaN.")
        return float("nan")
    return float(np.mean(np.abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero])) * 100.0)
