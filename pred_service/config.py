"""
Pydantic schema for the JSON configuration file / PRED_SERVICE_CONFIG env var.

The top-level structure is PredServiceConfig.
Each metric config is MetricConfig (after deep-merging with defaults).
"""
from __future__ import annotations

import copy
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ── Source sub-configs ────────────────────────────────────────────────────────

class ClickHouseSourceConfig(BaseModel):
    host: str = "localhost"
    port: int = 8123
    user: str = "default"
    password: str = ""
    database: str = "default"


class PrometheusSourceConfig(BaseModel):
    url: str = "http://localhost:9090"
    step: str = "5m"
    username: str = ""
    password: str = ""
    disable_ssl: bool = False


class TimeRangeConfig(BaseModel):
    lookback_days: Optional[int] = None
    start: Optional[str] = None  # ISO8601
    end: Optional[str] = None    # ISO8601

    @model_validator(mode="after")
    def validate_range(self) -> "TimeRangeConfig":
        has_lookback = self.lookback_days is not None
        has_explicit = self.start is not None or self.end is not None
        if has_lookback and has_explicit:
            raise ValueError("Specify either lookback_days or start/end, not both.")
        if has_explicit and (self.start is None or self.end is None):
            raise ValueError("Both 'start' and 'end' must be provided together.")
        return self


class PreprocessConfig(BaseModel):
    scale: Optional[float] = None


class SourceConfig(BaseModel):
    type: Literal["clickhouse", "prometheus"] = "clickhouse"
    clickhouse: Optional[ClickHouseSourceConfig] = None
    prometheus: Optional[PrometheusSourceConfig] = None
    query: str = ""
    time_range: Optional[TimeRangeConfig] = None
    preprocess: Optional[PreprocessConfig] = None


# ── Model / strategy sub-configs ─────────────────────────────────────────────

class ModelSpec(BaseModel):
    """
    Full model spec dict form: {"type": "walkforward", "estimator": "lightgbm", "params": {...}}.

    Extra top-level keys are forwarded to the model constructor as-is, allowing
    model-specific params (e.g. lags, seasonal_lag for WalkForwardModel) without
    enumerating them all here.
    """
    model_config = ConfigDict(extra="allow")

    type: str
    estimator: Optional[str] = None
    params: dict[str, Any] = Field(default_factory=dict)


class ModelConfig(BaseModel):
    strategy: Literal["single", "best_of"] = "best_of"
    # Each candidate is either a shorthand string ("walkforward/lightgbm") or a ModelSpec dict
    candidates: list[Union[str, ModelSpec]] = Field(
        default_factory=lambda: [
            "walkforward/lightgbm",
            "walkforward/ridge",
            "linear_trend",
            "seasonal_naive",
        ]
    )
    eval_metric: Literal["mae", "rmse", "mape"] = "rmse"
    eval_fraction: float = Field(default=0.2, ge=0.05, le=0.5)
    # Whether to refit the selected model on the full series after evaluation.
    # Set to false to keep the model trained only on the training split.
    refit_on_full_data: bool = True


# ── Forecast sub-config ───────────────────────────────────────────────────────

class ForecastConfig(BaseModel):
    horizon_steps: int = Field(default=288, gt=0)
    step: str = "5m"  # e.g. "5m", "1h", "15m"

    @field_validator("step")
    @classmethod
    def validate_step(cls, v: str) -> str:
        # Normalize: just check it's parseable later; keep original for now
        if not v:
            raise ValueError("step must not be empty")
        return v


# ── Output sub-config ─────────────────────────────────────────────────────────

class ClickHouseOutputConfig(BaseModel):
    table: str = "metrics_forecast"


class OutputConfig(BaseModel):
    clickhouse: Optional[ClickHouseOutputConfig] = None
    console: bool = True
    # Whether to write holdout eval predictions to ClickHouse (kind="eval" rows).
    save_eval: bool = True


# ── Per-metric config ─────────────────────────────────────────────────────────

class MetricConfig(BaseModel):
    service: str
    metric: str
    source: SourceConfig
    model: ModelConfig = Field(default_factory=ModelConfig)
    forecast: ForecastConfig = Field(default_factory=ForecastConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)


# ── Top-level config ──────────────────────────────────────────────────────────

class PredServiceConfig(BaseModel):
    defaults: dict[str, Any] = Field(default_factory=dict)
    metrics: list[dict[str, Any]]
    continue_on_error: bool = True

    def resolved_metrics(self) -> list[MetricConfig]:
        """Return list of MetricConfig after deep-merging defaults into each metric."""
        result: list[MetricConfig] = []
        for raw in self.metrics:
            merged = _deep_merge(copy.deepcopy(self.defaults), copy.deepcopy(raw))
            result.append(MetricConfig.model_validate(merged))
        return result


# ── Deep merge utility ────────────────────────────────────────────────────────

def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively merge override into base. Override wins on scalar conflicts.
    Dicts are merged recursively; all other types (including lists) are replaced.
    """
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result
