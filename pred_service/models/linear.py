"""
Linear and polynomial trend forecast models.

Uses scikit-learn under the hood (always available as a core dependency).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

logger = logging.getLogger(__name__)


class LinearTrendModel:
    """
    Fits an OLS linear trend on a numeric time index and extrapolates.

    The model uses ordinary least squares (via Ridge with alpha=0 equivalent,
    actually just numpy lstsq) for speed and simplicity.
    """

    name: str = "linear_trend"

    def __init__(self) -> None:
        self._coef: float = 0.0
        self._intercept: float = 0.0
        self._origin_ordinal: int = 0
        self._last_timestamp: pd.Timestamp | None = None

    def fit(self, series: pd.Series) -> "LinearTrendModel":
        if series.empty:
            raise ValueError("LinearTrendModel.fit: series is empty.")
        series = series.dropna()
        if len(series) < 2:
            raise ValueError("LinearTrendModel requires at least 2 non-NaN observations.")

        # Use integer index (ordinal days from epoch) for numerical stability
        ordinals = np.array([ts.toordinal() for ts in series.index], dtype=float)
        self._origin_ordinal = int(ordinals[0])
        x = ordinals - self._origin_ordinal
        y = series.to_numpy(dtype=float)

        # Solve via least squares
        X = np.column_stack([x, np.ones_like(x)])
        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        self._coef = float(coeffs[0])
        self._intercept = float(coeffs[1])
        self._last_timestamp = series.index[-1]
        self._last_ordinal = float(ordinals[-1])

        logger.debug(
            "%s fitted: coef=%.6f, intercept=%.4f", self.name, self._coef, self._intercept
        )
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
        ordinals = np.array([ts.toordinal() - self._origin_ordinal for ts in idx], dtype=float)
        values = self._coef * ordinals + self._intercept
        return pd.Series(values, index=idx, name="forecast")


class PolynomialTrendModel:
    """
    Polynomial trend model using PolynomialFeatures + Ridge regression.

    Parameters
    ----------
    degree:
        Degree of the polynomial. Default is 2 (quadratic).
    alpha:
        Ridge regularisation strength.
    """

    name: str = "polynomial_trend"

    def __init__(self, degree: int = 2, alpha: float = 1.0) -> None:
        self._degree = degree
        self._alpha = alpha
        self._poly: PolynomialFeatures | None = None
        self._ridge: Ridge | None = None
        self._origin_ordinal: int = 0
        self._last_timestamp: pd.Timestamp | None = None

    def fit(self, series: pd.Series) -> "PolynomialTrendModel":
        if series.empty:
            raise ValueError("PolynomialTrendModel.fit: series is empty.")
        series = series.dropna()
        if len(series) < self._degree + 1:
            raise ValueError(
                f"PolynomialTrendModel(degree={self._degree}) requires at least "
                f"{self._degree + 1} non-NaN observations."
            )

        ordinals = np.array([ts.toordinal() for ts in series.index], dtype=float)
        self._origin_ordinal = int(ordinals[0])
        x = (ordinals - self._origin_ordinal).reshape(-1, 1)
        y = series.to_numpy(dtype=float)

        self._poly = PolynomialFeatures(degree=self._degree, include_bias=True)
        X_poly = self._poly.fit_transform(x)
        self._ridge = Ridge(alpha=self._alpha)
        self._ridge.fit(X_poly, y)
        self._last_timestamp = series.index[-1]

        logger.debug(
            "%s fitted: degree=%d, alpha=%.4f, n_samples=%d",
            self.name, self._degree, self._alpha, len(series),
        )
        return self

    def predict(self, horizon: int, freq: str) -> pd.Series:
        if self._last_timestamp is None or self._poly is None or self._ridge is None:
            raise RuntimeError("Model must be fitted before calling predict().")
        idx = pd.date_range(
            start=self._last_timestamp + pd.tseries.frequencies.to_offset(freq),  # type: ignore[operator]
            periods=horizon,
            freq=freq,
            tz="UTC",
        )
        ordinals = np.array([ts.toordinal() - self._origin_ordinal for ts in idx], dtype=float)
        X_poly = self._poly.transform(ordinals.reshape(-1, 1))
        values = self._ridge.predict(X_poly)
        return pd.Series(values, index=idx, name="forecast")
