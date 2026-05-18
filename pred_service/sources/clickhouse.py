"""ClickHouse data source implementation."""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import pandas as pd

from pred_service.config import ClickHouseSourceConfig, PreprocessConfig
from pred_service.sources.base import DataSource

logger = logging.getLogger(__name__)


class ClickHouseSource(DataSource):
    """
    Fetches time-series data from ClickHouse.

    The query must return exactly two columns:
      - Column 0: timestamp (DateTime or DateTime64)
      - Column 1: value (numeric)

    The start/end arguments are passed to the query as {start} and {end}
    named parameters if the query contains those placeholders, otherwise
    ClickHouse executes the query as-is (the query itself must handle filtering).
    """

    def __init__(
        self,
        cfg: ClickHouseSourceConfig,
        preprocess: Optional[PreprocessConfig] = None,
    ) -> None:
        self._cfg = cfg
        self._preprocess = preprocess

    def _get_client(self):  # type: ignore[return]
        """Create and return a clickhouse_connect client."""
        try:
            import clickhouse_connect  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "clickhouse-connect is required for ClickHouseSource. "
                "Install it with: pip install clickhouse-connect"
            ) from exc

        return clickhouse_connect.get_client(
            host=self._cfg.host,
            port=self._cfg.port,
            username=self._cfg.user,
            password=self._cfg.password,
            database=self._cfg.database,
        )

    def fetch(self, query: str, start: datetime, end: datetime) -> pd.Series:
        logger.debug(
            "ClickHouseSource.fetch: host=%s db=%s start=%s end=%s",
            self._cfg.host,
            self._cfg.database,
            start.isoformat(),
            end.isoformat(),
        )

        client = self._get_client()

        # Pass start/end as query parameters; clickhouse_connect supports named params
        parameters = {
            "start": start,
            "end": end,
        }

        try:
            result = client.query(query, parameters=parameters)
        except Exception as exc:
            logger.error("ClickHouse query failed: %s", exc)
            raise

        if result.column_count < 2:
            raise ValueError(
                f"ClickHouse query must return at least 2 columns (timestamp, value), "
                f"got {result.column_count}."
            )

        rows = result.result_rows
        if not rows:
            logger.warning("ClickHouse query returned 0 rows.")
            return pd.Series(dtype=float, name="value")

        timestamps = []
        values = []
        for row in rows:
            ts_raw = row[0]
            val_raw = row[1]
            # Normalise timestamp to pandas Timestamp (UTC-aware)
            ts = pd.Timestamp(ts_raw)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            else:
                ts = ts.tz_convert("UTC")
            timestamps.append(ts)
            values.append(float(val_raw) if val_raw is not None else float("nan"))

        series = pd.Series(values, index=pd.DatetimeIndex(timestamps), name="value")
        series = series.sort_index()

        # Apply preprocessing
        if self._preprocess is not None and self._preprocess.scale is not None:
            logger.debug("Applying scale factor: %s", self._preprocess.scale)
            series = series * self._preprocess.scale

        logger.info("Fetched %d data points from ClickHouse.", len(series))
        return series
