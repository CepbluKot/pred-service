"""ClickHouse output sink — writes forecast results to metrics_forecast table."""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from pred_service.output.base import OutputSink
from pred_service.pipeline import PredictionResult
from pred_service.settings import Settings

logger = logging.getLogger(__name__)

_CREATE_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS {table} (
    generated_at   DateTime64(3),
    run_id         String,
    service        String,
    metric         String,
    model          String,
    strategy       LowCardinality(String),
    kind           LowCardinality(String),
    step_seconds   UInt32,
    timestamp      DateTime64(3),
    value          Float64,
    eval_mae       Float64,
    eval_rmse      Float64,
    eval_mape      Float64,
    eval_r2        Float64
)
ENGINE = MergeTree()
ORDER BY (service, metric, generated_at, kind, timestamp)
"""

_INSERT_QUERY = """
INSERT INTO {table}
(generated_at, run_id, service, metric, model, strategy, kind,
 step_seconds, timestamp, value, eval_mae, eval_rmse, eval_mape, eval_r2)
VALUES
"""


class ClickHouseSink(OutputSink):
    """
    Writes PredictionResult rows to a ClickHouse table.

    Two kinds of rows are written:
    - kind="eval": predictions on the holdout window (for back-testing review).
    - kind="forecast": future predictions.

    The table is created (IF NOT EXISTS) on first use.
    """

    def __init__(self, settings: Settings, table: str = "metrics_forecast", run_id: str = "") -> None:
        self._settings = settings
        self._table = table
        self._run_id = run_id

    def _get_client(self):  # type: ignore[return]
        try:
            import clickhouse_connect  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "clickhouse-connect is required for ClickHouseSink. "
                "Install it with: pip install clickhouse-connect"
            ) from exc

        s = self._settings
        return clickhouse_connect.get_client(
            host=s.pred_ch_host,
            port=s.pred_ch_port,
            username=s.pred_ch_user,
            password=s.pred_ch_password,
            database=s.pred_ch_database,
        )

    def _ensure_table(self, client: object) -> None:
        ddl = _CREATE_TABLE_DDL.format(table=self._table)
        client.command(ddl)  # type: ignore[attr-defined]
        logger.debug("Ensured table '%s' exists.", self._table)

    def write(self, result: PredictionResult, save_eval: bool = True) -> None:
        client = self._get_client()
        self._ensure_table(client)

        generated_at = datetime.now(tz=timezone.utc)
        mae = result.eval_scores.get("mae", float("nan"))
        rmse = result.eval_scores.get("rmse", float("nan"))
        mape = result.eval_scores.get("mape", float("nan"))
        r2 = result.eval_scores.get("r2", float("nan"))

        rows: list[list] = []

        # Eval rows (holdout predictions) — optional
        if save_eval:
            for ts, val in result.eval_series.items():
                rows.append([
                    generated_at,
                    self._run_id,
                    result.service,
                    result.metric,
                    result.model_name,
                    result.strategy,
                    "eval",
                    result.step_seconds,
                    ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts,
                    float(val),
                    mae,
                    rmse,
                    mape,
                    r2,
                ])

        # Forecast rows (future predictions)
        for ts, val in result.forecast_series.items():
            rows.append([
                generated_at,
                self._run_id,
                result.service,
                result.metric,
                result.model_name,
                result.strategy,
                "forecast",
                result.step_seconds,
                ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts,
                float(val),
                mae,
                rmse,
                mape,
                r2,
            ])

        if not rows:
            logger.warning("ClickHouseSink: no rows to insert for %s/%s.", result.service, result.metric)
            return

        column_names = [
            "generated_at", "run_id", "service", "metric", "model", "strategy",
            "kind", "step_seconds", "timestamp", "value",
            "eval_mae", "eval_rmse", "eval_mape", "eval_r2",
        ]

        try:
            client.insert(  # type: ignore[attr-defined]
                table=self._table,
                data=rows,
                column_names=column_names,
            )
            logger.info(
                "ClickHouseSink: inserted %d rows into '%s' for %s/%s.",
                len(rows),
                self._table,
                result.service,
                result.metric,
            )
        except Exception as exc:
            logger.error("ClickHouseSink: insert failed: %s", exc)
            raise
