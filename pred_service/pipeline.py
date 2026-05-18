"""
Main orchestrator for the prediction service.

run_metric() is the top-level function: given a MetricConfig and Settings,
it fetches data, trains a model, generates forecasts, writes outputs, and
returns a PredictionResult.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

import pandas as pd

from pred_service.config import MetricConfig, SourceConfig
from pred_service.evaluator import evaluate
from pred_service.models.selector import BestOfStrategy, SelectionResult, SingleStrategy, _step_to_freq
from pred_service.settings import Settings

if TYPE_CHECKING:
    from pred_service.sources.base import DataSource

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Encapsulates everything produced for a single metric."""
    service: str
    metric: str
    model_name: str
    strategy: str
    eval_scores: dict[str, float]
    eval_series: pd.Series        # predictions on holdout window
    forecast_series: pd.Series    # future predictions
    step_seconds: int
    all_candidate_scores: dict[str, dict[str, float]] = field(default_factory=dict)


def _build_source(source_cfg: SourceConfig) -> "DataSource":
    """Instantiate the appropriate DataSource from config."""
    src_type = source_cfg.type
    preprocess = source_cfg.preprocess

    if src_type == "clickhouse":
        from pred_service.sources.clickhouse import ClickHouseSource
        if source_cfg.clickhouse is None:
            raise ValueError("source.type='clickhouse' requires source.clickhouse connection config.")
        return ClickHouseSource(cfg=source_cfg.clickhouse, preprocess=preprocess)
    elif src_type == "prometheus":
        from pred_service.sources.prometheus import PrometheusSource
        if source_cfg.prometheus is None:
            raise ValueError("source.type='prometheus' requires source.prometheus connection config.")
        return PrometheusSource(cfg=source_cfg.prometheus, preprocess=preprocess)
    else:
        raise ValueError(f"Unknown source type '{src_type}'. Valid: 'clickhouse', 'prometheus'.")


def _resolve_time_range(source_cfg: SourceConfig) -> tuple[datetime, datetime]:
    """Compute start/end datetimes from the time_range config."""
    now = datetime.now(tz=timezone.utc)
    tr = source_cfg.time_range

    if tr is None:
        # Default: 90 days lookback
        return now - timedelta(days=90), now

    if tr.lookback_days is not None:
        return now - timedelta(days=tr.lookback_days), now

    if tr.start is not None and tr.end is not None:
        from dateutil import parser as dtparser
        start = dtparser.parse(tr.start)
        end = dtparser.parse(tr.end)
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)
        return start, end

    # Fallback
    return now - timedelta(days=90), now


def _step_to_seconds(step: str) -> int:
    """Convert a step string like '5m', '1h' to seconds."""
    step = step.strip().lower()
    if step.endswith("s"):
        return int(step[:-1])
    if step.endswith("min"):
        return int(step[:-3]) * 60
    if step.endswith("m"):
        return int(step[:-1]) * 60
    if step.endswith("h"):
        return int(step[:-1]) * 3600
    if step.endswith("d"):
        return int(step[:-1]) * 86400
    raise ValueError(f"Cannot convert step '{step}' to seconds.")


def run_metric(
    metric_cfg: MetricConfig,
    settings: Settings,
    run_id: str,
) -> PredictionResult:
    """
    Run the full prediction pipeline for one metric.

    Steps:
    1. Build DataSource from config.
    2. Fetch time series.
    3. Apply preprocess (already handled in the source, but scale is noted here).
    4. Run strategy (Single or BestOf) → fitted model + eval scores.
    5. Generate future forecast.
    6. Write to configured outputs.
    7. Return PredictionResult.
    """
    service = metric_cfg.service
    metric = metric_cfg.metric

    logger.info("=" * 60)
    logger.info("Processing metric: service='%s' metric='%s'", service, metric)
    logger.info("=" * 60)

    # ── 1. Build source ───────────────────────────────────────────────────────
    source = _build_source(metric_cfg.source)

    # ── 2. Fetch data ─────────────────────────────────────────────────────────
    start, end = _resolve_time_range(metric_cfg.source)
    logger.info("Fetching data from %s to %s ...", start.isoformat(), end.isoformat())
    series = source.fetch(query=metric_cfg.source.query, start=start, end=end)

    if series.empty:
        raise ValueError(f"No data returned for service='{service}' metric='{metric}'.")

    logger.info("Fetched %d data points. Range: %s → %s", len(series), series.index[0], series.index[-1])

    # ── 3. Strategy selection ─────────────────────────────────────────────────
    model_cfg = metric_cfg.model
    step = metric_cfg.forecast.step

    if model_cfg.strategy == "single":
        if not model_cfg.candidates:
            raise ValueError("SingleStrategy requires at least one candidate in model.candidates.")
        strategy = SingleStrategy(
            model_spec=model_cfg.candidates[0]
            if not hasattr(model_cfg.candidates[0], "model_dump")
            else model_cfg.candidates[0].model_dump(),  # type: ignore[union-attr]
            eval_fraction=model_cfg.eval_fraction,
            refit_on_full_data=model_cfg.refit_on_full_data,
        )
        strategy_name = "single"
    else:
        # Convert ModelSpec objects to dicts if needed
        raw_candidates = []
        for c in model_cfg.candidates:
            if isinstance(c, str):
                raw_candidates.append(c)
            else:
                raw_candidates.append(c.model_dump(exclude_none=True))
        strategy = BestOfStrategy(
            candidates=raw_candidates,
            eval_metric=model_cfg.eval_metric,
            eval_fraction=model_cfg.eval_fraction,
            refit_on_full_data=model_cfg.refit_on_full_data,
        )
        strategy_name = "best_of"

    logger.info("Running strategy: %s (eval_metric=%s)", strategy_name, model_cfg.eval_metric)
    selection: SelectionResult = strategy.select(series=series, step=step)

    # ── 4. Future forecast ────────────────────────────────────────────────────
    horizon = metric_cfg.forecast.horizon_steps
    freq = _step_to_freq(step)
    logger.info("Generating %d-step forecast (freq=%s) with model '%s' ...", horizon, freq, selection.model.name)
    forecast_series = selection.model.predict(horizon=horizon, freq=freq)
    logger.info("Forecast generated: %s → %s", forecast_series.index[0], forecast_series.index[-1])

    # ── 5. Assemble result ────────────────────────────────────────────────────
    try:
        step_seconds = _step_to_seconds(step)
    except ValueError:
        logger.warning("Could not parse step '%s' to seconds; defaulting to 300.", step)
        step_seconds = 300

    result = PredictionResult(
        service=service,
        metric=metric,
        model_name=selection.model.name,
        strategy=strategy_name,
        eval_scores=selection.eval_scores,
        eval_series=selection.eval_series,
        forecast_series=forecast_series,
        step_seconds=step_seconds,
        all_candidate_scores=selection.all_scores,
    )

    # ── 6. Write outputs ──────────────────────────────────────────────────────
    output_cfg = metric_cfg.output

    if output_cfg.console:
        from pred_service.output.console import ConsoleSink
        ConsoleSink().write(result)

    if output_cfg.clickhouse is not None:
        from pred_service.output.clickhouse import ClickHouseSink
        table = output_cfg.clickhouse.table
        sink = ClickHouseSink(settings=settings, table=table, run_id=run_id)
        sink.write(result, save_eval=output_cfg.save_eval)

    return result
