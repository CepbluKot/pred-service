"""Console/log output sink — prints a human-readable forecast summary."""
from __future__ import annotations

import logging

from pred_service.output.base import OutputSink
from pred_service.pipeline import PredictionResult

logger = logging.getLogger(__name__)


class ConsoleSink(OutputSink):
    """
    Writes a summary of the prediction result to the log (INFO level).

    Shows: eval scores, first/last forecast values, model name, strategy.
    """

    def write(self, result: PredictionResult) -> None:
        scores = result.eval_scores
        fc = result.forecast_series

        logger.info("─" * 60)
        logger.info(
            "RESULT  service=%-20s metric=%s",
            result.service,
            result.metric,
        )
        logger.info("  Model:    %s  (strategy=%s)", result.model_name, result.strategy)
        logger.info(
            "  Eval:     MAE=%.4f  RMSE=%.4f  MAPE=%.2f%%  R2=%.4f",
            scores.get("mae", float("nan")),
            scores.get("rmse", float("nan")),
            scores.get("mape", float("nan")),
            scores.get("r2", float("nan")),
        )
        if not fc.empty:
            logger.info(
                "  Forecast: %d steps  [%s → %s]  first=%.4f  last=%.4f",
                len(fc),
                fc.index[0].isoformat(),
                fc.index[-1].isoformat(),
                float(fc.iloc[0]),
                float(fc.iloc[-1]),
            )
        else:
            logger.info("  Forecast: (empty)")

        if result.all_candidate_scores and len(result.all_candidate_scores) > 1:
            logger.info("  All candidates:")
            for name, s in result.all_candidate_scores.items():
                marker = " <-- winner" if name == result.model_name else ""
                logger.info(
                    "    %-40s  RMSE=%.4f  MAE=%.4f%s",
                    name,
                    s.get("rmse", float("nan")),
                    s.get("mae", float("nan")),
                    marker,
                )
        logger.info("─" * 60)
