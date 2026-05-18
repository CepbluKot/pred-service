"""Prometheus data source implementation."""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import pandas as pd

from pred_service.config import PreprocessConfig, PrometheusSourceConfig
from pred_service.sources.base import DataSource

logger = logging.getLogger(__name__)


class PrometheusSource(DataSource):
    """
    Fetches time-series data from a Prometheus-compatible endpoint.

    Uses prometheus_api_client.PrometheusConnect under the hood.
    The query is a PromQL expression. The result is parsed into a pd.Series
    by extracting the first returned metric's values.
    """

    def __init__(
        self,
        cfg: PrometheusSourceConfig,
        preprocess: Optional[PreprocessConfig] = None,
    ) -> None:
        self._cfg = cfg
        self._preprocess = preprocess

    def _get_client(self):  # type: ignore[return]
        try:
            from prometheus_api_client import PrometheusConnect  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "prometheus-api-client is required for PrometheusSource. "
                "Install it with: pip install prometheus-api-client"
            ) from exc

        auth = None
        if self._cfg.username:
            auth = (self._cfg.username, self._cfg.password)

        return PrometheusConnect(
            url=self._cfg.url,
            auth=auth,
            disable_ssl=self._cfg.disable_ssl,
        )

    def fetch(self, query: str, start: datetime, end: datetime) -> pd.Series:
        logger.debug(
            "PrometheusSource.fetch: url=%s step=%s start=%s end=%s",
            self._cfg.url,
            self._cfg.step,
            start.isoformat(),
            end.isoformat(),
        )

        client = self._get_client()

        try:
            result = client.custom_query_range(
                query=query,
                start_time=start,
                end_time=end,
                step=self._cfg.step,
            )
        except Exception as exc:
            logger.error("Prometheus query failed: %s", exc)
            raise

        if not result:
            logger.warning("Prometheus query returned empty result for query: %s", query)
            return pd.Series(dtype=float, name="value")

        # Use the first metric series returned
        if len(result) > 1:
            logger.warning(
                "Prometheus query returned %d metric series; using the first one. "
                "Consider making your PromQL more specific.",
                len(result),
            )

        metric_data = result[0]
        values_raw = metric_data.get("values", [])

        if not values_raw:
            logger.warning("Prometheus metric has no values.")
            return pd.Series(dtype=float, name="value")

        timestamps = []
        values = []
        for ts_epoch, val_str in values_raw:
            ts = pd.Timestamp(float(ts_epoch), unit="s", tz="UTC")
            try:
                val = float(val_str)
            except (ValueError, TypeError):
                val = float("nan")
            timestamps.append(ts)
            values.append(val)

        series = pd.Series(values, index=pd.DatetimeIndex(timestamps), name="value")
        series = series.sort_index()

        # Apply preprocessing
        if self._preprocess is not None and self._preprocess.scale is not None:
            logger.debug("Applying scale factor: %s", self._preprocess.scale)
            series = series * self._preprocess.scale

        logger.info("Fetched %d data points from Prometheus.", len(series))
        return series
