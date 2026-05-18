"""
Model selection strategies.

SingleStrategy: use exactly one model.
BestOfStrategy: train N candidates, evaluate on holdout, pick the winner.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Union

import pandas as pd

from pred_service.evaluator import evaluate
from pred_service.models.base import ForecastModel
from pred_service.models.registry import get_model

logger = logging.getLogger(__name__)


@dataclass
class SelectionResult:
    """Result of a strategy.select() call."""
    model: ForecastModel
    eval_scores: dict[str, float]
    eval_series: pd.Series  # predictions on the holdout window
    all_scores: dict[str, dict[str, float]] = field(default_factory=dict)


def _step_to_freq(step: str) -> str:
    """
    Convert a step string like "5m", "1h", "15m", "1d" to a pandas freq string.

    Handles common abbreviations used in config and Prometheus.
    """
    step = step.strip()
    # Map single-letter suffixes to pandas freq suffixes
    _MAP = {
        "s": "s",
        "m": "min",
        "min": "min",
        "h": "h",
        "d": "D",
        "w": "W",
    }
    # Try to split into number + unit
    for suffix in sorted(_MAP.keys(), key=len, reverse=True):
        if step.lower().endswith(suffix):
            number = step[: -len(suffix)]
            if number.isdigit():
                return f"{number}{_MAP[suffix]}"
    # If no known suffix, return as-is and hope pandas can parse it
    logger.warning("Could not map step '%s' to pandas freq; using as-is.", step)
    return step


class SingleStrategy:
    """
    Uses a single model specified in the config.

    Parameters
    ----------
    model_spec:
        Shorthand string or dict spec for the model.
    eval_fraction:
        Fraction of series to use as holdout for eval metrics.
    refit_on_full_data:
        Whether to refit on the full series after evaluation (default True).
    """

    def __init__(
        self,
        model_spec: Union[str, dict[str, Any]],
        eval_fraction: float = 0.2,
        refit_on_full_data: bool = True,
    ) -> None:
        self._spec = model_spec
        self._eval_fraction = eval_fraction
        self._refit_on_full_data = refit_on_full_data

    def select(self, series: pd.Series, step: str) -> SelectionResult:
        freq = _step_to_freq(step)
        n_eval = max(1, int(len(series) * self._eval_fraction))
        train = series.iloc[: len(series) - n_eval]
        holdout = series.iloc[len(series) - n_eval :]

        model = get_model(self._spec)
        logger.info("SingleStrategy: fitting model '%s' on %d points.", model.name, len(train))
        model.fit(train)

        eval_pred = model.predict(horizon=len(holdout), freq=freq)
        scores = evaluate(holdout, eval_pred)
        logger.info(
            "SingleStrategy: eval scores for '%s': MAE=%.4f RMSE=%.4f MAPE=%.2f%% R2=%.4f",
            model.name,
            scores["mae"],
            scores["rmse"],
            scores["mape"],
            scores["r2"],
        )

        if self._refit_on_full_data:
            model.fit(series)
        return SelectionResult(
            model=model,
            eval_scores=scores,
            eval_series=eval_pred,
            all_scores={model.name: scores},
        )


class BestOfStrategy:
    """
    Trains multiple candidate models on a training split, evaluates on a holdout,
    and selects the best model by eval_metric (lower is better for mae/rmse/mape).

    The winner is refit on the full series before being returned (unless
    refit_on_full_data=False).

    Parameters
    ----------
    candidates:
        List of model specs (strings or dicts).
    eval_metric:
        One of "mae", "rmse", "mape". Lower is better.
    eval_fraction:
        Fraction of the series to hold out for evaluation.
    refit_on_full_data:
        Whether to refit the winner on the full series after selection (default True).
    """

    def __init__(
        self,
        candidates: list[Union[str, dict[str, Any]]],
        eval_metric: str = "rmse",
        eval_fraction: float = 0.2,
        refit_on_full_data: bool = True,
    ) -> None:
        self._candidates = candidates
        self._eval_metric = eval_metric.lower()
        self._eval_fraction = eval_fraction
        self._refit_on_full_data = refit_on_full_data

    def select(self, series: pd.Series, step: str) -> SelectionResult:
        freq = _step_to_freq(step)
        n_eval = max(1, int(len(series) * self._eval_fraction))
        train = series.iloc[: len(series) - n_eval]
        holdout = series.iloc[len(series) - n_eval :]

        logger.info(
            "BestOfStrategy: evaluating %d candidates on %d holdout steps (metric=%s).",
            len(self._candidates),
            n_eval,
            self._eval_metric,
        )

        best_model: ForecastModel | None = None
        best_score: float = float("inf")
        best_eval_pred: pd.Series = pd.Series(dtype=float)
        best_scores: dict[str, float] = {}
        all_scores: dict[str, dict[str, float]] = {}

        for spec in self._candidates:
            try:
                model = get_model(spec)
            except (ValueError, TypeError) as exc:
                logger.warning("Skipping invalid model spec '%s': %s", spec, exc)
                continue

            try:
                model.fit(train)
                eval_pred = model.predict(horizon=len(holdout), freq=freq)
                scores = evaluate(holdout, eval_pred)
                all_scores[model.name] = scores
                metric_val = scores[self._eval_metric]
                logger.info(
                    "  [%s] MAE=%.4f RMSE=%.4f MAPE=%.2f%% R2=%.4f",
                    model.name,
                    scores["mae"],
                    scores["rmse"],
                    scores["mape"],
                    scores["r2"],
                )
                if not (metric_val != metric_val):  # skip NaN
                    if metric_val < best_score:
                        best_score = metric_val
                        best_model = model
                        best_eval_pred = eval_pred
                        best_scores = scores
            except Exception as exc:
                logger.warning("Model '%s' failed during evaluation: %s", getattr(model, "name", spec), exc)
                continue

        if best_model is None:
            raise RuntimeError("All candidate models failed. Cannot select a winner.")

        logger.info(
            "BestOfStrategy: winner is '%s' with %s=%.4f.",
            best_model.name,
            self._eval_metric,
            best_score,
        )

        if self._refit_on_full_data:
            best_model.fit(series)
        return SelectionResult(
            model=best_model,
            eval_scores=best_scores,
            eval_series=best_eval_pred,
            all_scores=all_scores,
        )
