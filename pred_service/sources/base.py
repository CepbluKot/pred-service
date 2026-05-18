"""Abstract base class for all data sources."""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

import pandas as pd


class DataSource(ABC):
    """Fetch a time series for a given query and time range."""

    @abstractmethod
    def fetch(self, query: str, start: datetime, end: datetime) -> pd.Series:
        """
        Fetch metric data.

        Parameters
        ----------
        query:
            SQL query (for ClickHouse) or PromQL expression (for Prometheus).
        start:
            Inclusive start of the time range (UTC).
        end:
            Inclusive end of the time range (UTC).

        Returns
        -------
        pd.Series
            DatetimeIndex (UTC, sorted ascending), float values.
        """
