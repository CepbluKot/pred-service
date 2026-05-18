"""
Model registry: converts shorthand strings or dict specs into model instances.

Shorthand string format: "type" or "type/estimator"
  - Uses only default parameters for the model.
  - To customise any parameter, use the dict form instead.
Examples:
  "walkforward/lightgbm"         -> WalkForwardModel(estimator="lightgbm")
  "walkforward/ridge"            -> WalkForwardModel(estimator="ridge")
  "linear_trend"                 -> LinearTrendModel()
  "polynomial_trend"             -> PolynomialTrendModel()
  "seasonal_naive"               -> SeasonalNaiveModel()
  "naive_constant"               -> NaiveConstantModel()
  "drift"                        -> DriftModel()

Dict spec format — all top-level keys outside "params" are model-level options;
  "params" contains estimator constructor kwargs (for walkforward only):
  {
    "type": "walkforward",
    "estimator": "lightgbm",
    "params": {"n_estimators": 200},
    "lags": [1, 2, 3, 6, 12, 24, 48],
    "seasonal_lag": 288,
    "seasonal_lag_min_len": 338
  }
  {"type": "seasonal_naive", "params": {"period_steps": 144}}
  {"type": "polynomial_trend", "params": {"degree": 3, "alpha": 0.5}}
  {"type": "naive_constant", "params": {"n": 20}}
  {"type": "drift", "params": {"n": 30}}
"""
from __future__ import annotations

import logging
from typing import Any, Union

from pred_service.models.base import ForecastModel
from pred_service.models.linear import LinearTrendModel, PolynomialTrendModel
from pred_service.models.naive import DriftModel, NaiveConstantModel, SeasonalNaiveModel
from pred_service.models.walkforward import WalkForwardModel

logger = logging.getLogger(__name__)


def get_model(spec: Union[str, dict[str, Any]]) -> ForecastModel:
    """
    Parse a shorthand string or dict spec and return an instantiated model.

    Parameters
    ----------
    spec:
        A string shorthand ("walkforward/lightgbm") or a dict
        {"type": ..., "estimator": ..., "params": {...}}.

    Returns
    -------
    An instance satisfying the ForecastModel protocol.

    Raises
    ------
    ValueError:
        If the spec is not recognised.
    """
    if isinstance(spec, dict):
        return _from_dict(spec)
    if isinstance(spec, str):
        return _from_string(spec)
    raise TypeError(f"Model spec must be str or dict, got {type(spec).__name__}.")


def _from_string(spec: str) -> ForecastModel:
    parts = spec.strip().split("/", 1)
    model_type = parts[0].lower()
    estimator = parts[1].lower() if len(parts) > 1 else None

    if model_type == "walkforward":
        est = estimator or "lightgbm"
        logger.debug("Registry: WalkForwardModel(estimator=%s)", est)
        return WalkForwardModel(estimator=est)
    if model_type == "linear_trend":
        logger.debug("Registry: LinearTrendModel()")
        return LinearTrendModel()
    if model_type == "polynomial_trend":
        logger.debug("Registry: PolynomialTrendModel()")
        return PolynomialTrendModel()
    if model_type == "seasonal_naive":
        logger.debug("Registry: SeasonalNaiveModel()")
        return SeasonalNaiveModel()
    if model_type == "naive_constant":
        logger.debug("Registry: NaiveConstantModel()")
        return NaiveConstantModel()
    if model_type == "drift":
        logger.debug("Registry: DriftModel()")
        return DriftModel()

    raise ValueError(
        f"Unknown model type '{model_type}'. "
        "Valid types: walkforward, linear_trend, polynomial_trend, "
        "seasonal_naive, naive_constant, drift."
    )


def _from_dict(spec: dict[str, Any]) -> ForecastModel:
    model_type = spec.get("type", "").lower()
    estimator: str = spec.get("estimator", "lightgbm")
    params: dict[str, Any] = spec.get("params", {})

    if model_type == "walkforward":
        lags = spec.get("lags")  # None → WalkForwardModel uses its default
        seasonal_lag_raw = spec.get("seasonal_lag")
        seasonal_lag_min_len_raw = spec.get("seasonal_lag_min_len")
        kwargs: dict[str, Any] = {"estimator": estimator, "params": params}
        if lags is not None:
            kwargs["lags"] = [int(x) for x in lags]
        if seasonal_lag_raw is not None:
            kwargs["seasonal_lag"] = int(seasonal_lag_raw)
        if seasonal_lag_min_len_raw is not None:
            kwargs["seasonal_lag_min_len"] = int(seasonal_lag_min_len_raw)
        logger.debug("Registry: WalkForwardModel(%s)", kwargs)
        return WalkForwardModel(**kwargs)
    if model_type == "linear_trend":
        return LinearTrendModel()
    if model_type == "polynomial_trend":
        degree = int(params.get("degree", 2))
        alpha = float(params.get("alpha", 1.0))
        return PolynomialTrendModel(degree=degree, alpha=alpha)
    if model_type == "seasonal_naive":
        period = int(params.get("period_steps", 288))
        return SeasonalNaiveModel(period_steps=period)
    if model_type == "naive_constant":
        n = int(params.get("n", 10))
        return NaiveConstantModel(n=n)
    if model_type == "drift":
        n = int(params.get("n", 60))
        return DriftModel(n=n)

    raise ValueError(
        f"Unknown model type '{model_type}' in dict spec. "
        "Valid types: walkforward, linear_trend, polynomial_trend, "
        "seasonal_naive, naive_constant, drift."
    )
