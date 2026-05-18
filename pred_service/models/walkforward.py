"""
Walk-forward (recursive multi-step) forecast model.

Creates lag + time features from a time series, fits an sklearn-compatible
estimator, then does recursive prediction one step at a time.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Supported estimator names and their lazy import paths
_ESTIMATOR_REGISTRY: dict[str, tuple[str, str]] = {
    "ridge": ("sklearn.linear_model", "Ridge"),
    "lasso": ("sklearn.linear_model", "Lasso"),
    "linear": ("sklearn.linear_model", "LinearRegression"),
    "random_forest": ("sklearn.ensemble", "RandomForestRegressor"),
    "extra_trees": ("sklearn.ensemble", "ExtraTreesRegressor"),
    "hist_gradient_boosting": ("sklearn.ensemble", "HistGradientBoostingRegressor"),
    "lightgbm": ("lightgbm", "LGBMRegressor"),
    "xgboost": ("xgboost", "XGBRegressor"),
}

# Base lags always used
_BASE_LAGS = [1, 2, 3, 6, 12, 24, 48]
# Minimum series length to add the daily seasonality lag (288 steps)
_SEASONAL_LAG = 288
_SEASONAL_LAG_MIN_LEN = _SEASONAL_LAG + 50


def _import_estimator(name: str, params: dict[str, Any]) -> Any:
    """Lazily import and instantiate an estimator by name."""
    if name not in _ESTIMATOR_REGISTRY:
        raise ValueError(
            f"Unknown estimator '{name}'. Available: {list(_ESTIMATOR_REGISTRY.keys())}"
        )
    module_name, class_name = _ESTIMATOR_REGISTRY[name]
    try:
        import importlib
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
    except ImportError:
        raise ImportError(
            f"Estimator '{name}' requires '{module_name}' which is not installed. "
            f"Install it or choose a different estimator."
        )
    return cls(**params)


def _make_time_features(index: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    """Cyclical hour and day-of-week features plus is_weekend flag."""
    hour = index.hour.to_numpy(dtype=float)
    dow = index.dayofweek.to_numpy(dtype=float)
    return {
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "dow_sin": np.sin(2 * np.pi * dow / 7),
        "dow_cos": np.cos(2 * np.pi * dow / 7),
        "is_weekend": (dow >= 5).astype(float),
    }


def _build_features(values: np.ndarray, index: pd.DatetimeIndex, lags: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the feature matrix and target vector.

    Parameters
    ----------
    values:
        1-D float array of time-series values (no NaN).
    index:
        Corresponding DatetimeIndex.
    lags:
        List of lag steps to use.

    Returns
    -------
    (X, y) where X has shape (n_samples, n_features) and y has shape (n_samples,).
    The first max(lags) rows are discarded.
    """
    max_lag = max(lags)
    n = len(values)

    rows_x: list[np.ndarray] = []
    rows_y: list[float] = []

    time_feats = _make_time_features(index)

    for i in range(max_lag, n):
        lag_feats = np.array([values[i - lag] for lag in lags], dtype=float)
        time_arr = np.array([v[i] for v in time_feats.values()], dtype=float)
        rows_x.append(np.concatenate([lag_feats, time_arr]))
        rows_y.append(values[i])

    if not rows_x:
        return np.empty((0, len(lags) + len(time_feats)), dtype=float), np.empty((0,), dtype=float)

    return np.array(rows_x, dtype=float), np.array(rows_y, dtype=float)


class WalkForwardModel:
    """
    Multi-step recursive forecast using an sklearn-compatible estimator.

    Feature engineering:
    - Lag features: configurable base lags (default [1,2,3,6,12,24,48]) plus an optional
      seasonal lag added when the series is long enough.
    - Time features: hour_sin, hour_cos, dow_sin, dow_cos, is_weekend.

    Prediction uses recursive (autoregressive) strategy: each new step is
    predicted using the previous predictions as lag features.

    Parameters
    ----------
    estimator:
        Name of the estimator (see _ESTIMATOR_REGISTRY).
    params:
        Extra keyword arguments passed to the estimator constructor.
    lags:
        Base lag steps to always include. Defaults to [1, 2, 3, 6, 12, 24, 48].
    seasonal_lag:
        The seasonal lag step added when the series is long enough. Default 288.
    seasonal_lag_min_len:
        Minimum series length required to add the seasonal lag.
        Defaults to seasonal_lag + 50.
    """

    def __init__(
        self,
        estimator: str = "lightgbm",
        params: dict[str, Any] | None = None,
        lags: list[int] | None = None,
        seasonal_lag: int = _SEASONAL_LAG,
        seasonal_lag_min_len: int | None = None,
    ) -> None:
        self._estimator_name = estimator
        self._params = params or {}
        self._lags_config: list[int] = list(lags) if lags is not None else list(_BASE_LAGS)
        self._seasonal_lag = seasonal_lag
        self._seasonal_lag_min_len = seasonal_lag_min_len if seasonal_lag_min_len is not None else seasonal_lag + 50
        self._model: Any = None
        self._lags: list[int] = []
        self._history: np.ndarray = np.array([], dtype=float)
        self._last_timestamp: pd.Timestamp | None = None
        self.name: str = f"walkforward/{estimator}"

    def fit(self, series: pd.Series) -> "WalkForwardModel":
        if series.empty:
            raise ValueError(f"{self.name}.fit: series is empty.")

        series = series.dropna().sort_index()
        values = series.to_numpy(dtype=float)

        self._lags = list(self._lags_config)
        if len(values) >= self._seasonal_lag_min_len:
            if self._seasonal_lag not in self._lags:
                self._lags.append(self._seasonal_lag)
            logger.debug("%s: adding seasonal lag %d (series length %d).", self.name, self._seasonal_lag, len(values))

        X, y = _build_features(values, series.index, self._lags)

        if len(X) == 0:
            raise ValueError(
                f"{self.name}: not enough data to build lag features "
                f"(need > {max(self._lags)} points, got {len(values)})."
            )

        logger.debug("%s: fitting on %d samples, %d features.", self.name, len(X), X.shape[1])

        self._model = _import_estimator(self._estimator_name, self._params)
        self._model.fit(X, y)

        # Keep the last max_lag values for recursive prediction
        self._history = values[-max(self._lags) :].copy()
        self._last_timestamp = series.index[-1]

        logger.debug("%s: fit complete.", self.name)
        return self

    def predict(self, horizon: int, freq: str) -> pd.Series:
        if self._model is None or self._last_timestamp is None:
            raise RuntimeError("Model must be fitted before calling predict().")

        idx = pd.date_range(
            start=self._last_timestamp + pd.tseries.frequencies.to_offset(freq),  # type: ignore[operator]
            periods=horizon,
            freq=freq,
            tz="UTC",
        )
        time_feats = _make_time_features(idx)

        # Sliding window: history + predictions so far
        window = list(self._history)
        predictions: list[float] = []

        for i in range(horizon):
            lag_feats = np.array([window[-(lag)] for lag in self._lags], dtype=float)
            time_arr = np.array([v[i] for v in time_feats.values()], dtype=float)
            x_row = np.concatenate([lag_feats, time_arr]).reshape(1, -1)
            pred = float(self._model.predict(x_row)[0])
            predictions.append(pred)
            window.append(pred)

        return pd.Series(predictions, index=idx, name="forecast")
