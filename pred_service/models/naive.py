"""
Naive baseline forecast models.

These require no external dependencies beyond pandas/numpy.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class NaiveConstantModel:
    """
    Predicts the mean of the last N observed values (flat line forecast).

    Parameters
    ----------
    n:
        Number of recent observations to average. Default is 10.
    """

    name: str = "naive_constant"

    def __init__(self, n: int = 10) -> None:
        self._n = n
        self._level: float = float("nan")
        self._last_timestamp: pd.Timestamp | None = None

    def fit(self, series: pd.Series) -> "NaiveConstantModel":
        if series.empty:
            raise ValueError("NaiveConstantModel.fit: series is empty.")
        tail = series.dropna().iloc[-self._n :]
        self._level = float(tail.mean()) if not tail.empty else float("nan")
        self._last_timestamp = series.index[-1]
        logger.debug("%s fitted: level=%.4f (from last %d points)", self.name, self._level, len(tail))
        return self

    def predict(self, horizon: int, freq: str) -> pd.Series:
        if self._last_timestamp is None:
            raise RuntimeError("Model must be fitted before calling predict().")
        idx = pd.date_range(
            start=self._last_timestamp + pd.tseries.frequencies.to_offset(freq),  # type: ignore[operator]
            periods=horizon,
            freq=freq,
            tz="UTC",
        )
        return pd.Series(self._level, index=idx, name="forecast")


class SeasonalNaiveModel:
    """
    Repeats the last complete seasonal period as the forecast.

    Parameters
    ----------
    period_steps:
        Number of steps in one seasonal period (e.g. 288 for a daily cycle at 5-min resolution).
    """

    name: str = "seasonal_naive"

    def __init__(self, period_steps: int = 288) -> None:
        self._period = period_steps
        self._season: pd.Series | None = None
        self._last_timestamp: pd.Timestamp | None = None
        self._freq: str | None = None

    def fit(self, series: pd.Series) -> "SeasonalNaiveModel":
        if series.empty:
            raise ValueError("SeasonalNaiveModel.fit: series is empty.")
        series = series.dropna()
        if len(series) < self._period:
            logger.warning(
                "%s: series length (%d) < period (%d). Using all available data as the season.",
                self.name,
                len(series),
                self._period,
            )
            self._season = series.copy()
        else:
            self._season = series.iloc[-self._period :].copy()
        self._last_timestamp = series.index[-1]
        # Infer freq from the series index
        if series.index.freq is not None:
            self._freq = str(series.index.freq)
        logger.debug("%s fitted: using %d-step season.", self.name, len(self._season))
        return self

    def predict(self, horizon: int, freq: str) -> pd.Series:
        if self._season is None or self._last_timestamp is None:
            raise RuntimeError("Model must be fitted before calling predict().")
        idx = pd.date_range(
            start=self._last_timestamp + pd.tseries.frequencies.to_offset(freq),  # type: ignore[operator]
            periods=horizon,
            freq=freq,
            tz="UTC",
        )
        season_values = self._season.to_numpy(dtype=float)
        # Tile the season to cover the horizon
        tiled = np.tile(season_values, int(np.ceil(horizon / len(season_values))))
        values = tiled[:horizon]
        return pd.Series(values, index=idx, name="forecast")


class DriftModel:
    """
    Linear extrapolation (drift) from the last N points.

    Fits a line through the first and last of the N recent observations
    and extrapolates that trend forward.

    Parameters
    ----------
    n:
        Number of recent observations to use for trend estimation.
    """

    name: str = "drift"

    def __init__(self, n: int = 60) -> None:
        self._n = n
        self._slope: float = 0.0
        self._last_value: float = float("nan")
        self._last_timestamp: pd.Timestamp | None = None

    def fit(self, series: pd.Series) -> "DriftModel":
        if series.empty:
            raise ValueError("DriftModel.fit: series is empty.")
        tail = series.dropna().iloc[-self._n :]
        if len(tail) < 2:
            logger.warning("%s: not enough points to estimate slope; using 0.", self.name)
            self._slope = 0.0
        else:
            self._slope = (tail.iloc[-1] - tail.iloc[0]) / (len(tail) - 1)
        self._last_value = float(tail.iloc[-1])
        self._last_timestamp = series.index[-1]
        logger.debug("%s fitted: slope=%.6f, last_value=%.4f", self.name, self._slope, self._last_value)
        return self

    def predict(self, horizon: int, freq: str) -> pd.Series:
        if self._last_timestamp is None:
            raise RuntimeError("Model must be fitted before calling predict().")
        idx = pd.date_range(
            start=self._last_timestamp + pd.tseries.frequencies.to_offset(freq),  # type: ignore[operator]
            periods=horizon,
            freq=freq,
            tz="UTC",
        )
        steps = np.arange(1, horizon + 1, dtype=float)
        values = self._last_value + self._slope * steps
        return pd.Series(values, index=idx, name="forecast")
