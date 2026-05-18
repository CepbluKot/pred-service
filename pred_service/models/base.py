"""
Protocol definition for forecast models.

Using Protocol (structural subtyping) rather than ABC so that third-party
models can be used without inheriting from our base class.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class ForecastModel(Protocol):
    """Duck-typing interface for all forecast models."""

    name: str

    def fit(self, series: pd.Series) -> "ForecastModel":
        """
        Fit the model on the provided time series.

        Parameters
        ----------
        series:
            pd.Series with DatetimeIndex (UTC) and float values.

        Returns
        -------
        self (for chaining)
        """
        ...

    def predict(self, horizon: int, freq: str) -> pd.Series:
        """
        Generate a forecast.

        Parameters
        ----------
        horizon:
            Number of steps ahead to predict.
        freq:
            Pandas frequency string for the output DatetimeIndex (e.g. "5min", "1h").

        Returns
        -------
        pd.Series with DatetimeIndex (UTC) starting one step after the last training point.
        """
        ...
