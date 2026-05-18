# CLAUDE.md — Prediction Generator Codebase Guide

## Codebase map

```
prediction-generator/
├── requirements.txt                  # pandas, numpy, scikit-learn, lightgbm, clickhouse-connect, prometheus-api-client, pydantic>=2, pydantic-settings, python-dateutil
├── .env.pred-service.example         # template for output-ClickHouse env vars
├── airflow/
│   └── trigger_dag.py               # standalone script to POST a DAG run via Airflow REST API
└── pred_service/
    ├── __init__.py                   # empty
    ├── main.py                       # CLI entry point; loads config, runs per-metric loop
    ├── settings.py                   # pydantic-settings: output ClickHouse creds + LOG_LEVEL
    ├── config.py                     # all Pydantic config models; _deep_merge; resolved_metrics()
    ├── pipeline.py                   # run_metric(); PredictionResult dataclass; _build_source(); _resolve_time_range()
    ├── evaluator.py                  # evaluate() -> {mae, rmse, mape, r2}; NaN-safe
    ├── sources/
    │   ├── __init__.py               # empty
    │   ├── base.py                   # DataSource ABC with abstract fetch()
    │   ├── clickhouse.py             # ClickHouseSource: 2-column SQL → pd.Series
    │   └── prometheus.py             # PrometheusSource: PromQL range query → pd.Series
    ├── models/
    │   ├── __init__.py               # empty
    │   ├── base.py                   # ForecastModel Protocol (runtime_checkable)
    │   ├── naive.py                  # NaiveConstantModel, SeasonalNaiveModel, DriftModel
    │   ├── linear.py                 # LinearTrendModel, PolynomialTrendModel
    │   ├── walkforward.py            # WalkForwardModel; _ESTIMATOR_REGISTRY; feature engineering
    │   ├── registry.py               # get_model(spec) — string/dict → model instance
    │   └── selector.py               # SingleStrategy, BestOfStrategy, SelectionResult, _step_to_freq()
    └── output/
        ├── __init__.py               # empty
        ├── base.py                   # OutputSink ABC with abstract write()
        ├── clickhouse.py             # ClickHouseSink: writes eval+forecast rows; DDL
        └── console.py                # ConsoleSink: logs summary at INFO
```

---

## Key abstractions

### DataSource (`sources/base.py`)

```python
class DataSource(ABC):
    @abstractmethod
    def fetch(self, query: str, start: datetime, end: datetime) -> pd.Series: ...
```

Contract: returns a `pd.Series` with a UTC-aware, sorted-ascending `DatetimeIndex` and float values. Empty series is valid (pipeline raises `ValueError` on empty). Source applies `preprocess.scale` multiplication if configured.

### ForecastModel (`models/base.py`)

```python
@runtime_checkable
class ForecastModel(Protocol):
    name: str

    def fit(self, series: pd.Series) -> "ForecastModel": ...
    def predict(self, horizon: int, freq: str) -> pd.Series: ...
```

Contract: `fit()` stores state and returns `self`. `predict()` returns a `pd.Series` with UTC `DatetimeIndex` starting exactly one `freq` step after the last training timestamp, length == `horizon`. Protocol is structural (no inheritance needed).

### OutputSink (`output/base.py`)

```python
class OutputSink(ABC):
    @abstractmethod
    def write(self, result: PredictionResult) -> None: ...
```

Contract: receives a completed `PredictionResult` and persists or displays it. Must not modify the result.

### Strategy (`models/selector.py`)

No formal ABC. Both strategies expose:

```python
def select(self, series: pd.Series, step: str) -> SelectionResult: ...
```

`SelectionResult` fields: `model: ForecastModel`, `eval_scores: dict[str, float]`, `eval_series: pd.Series`, `all_scores: dict[str, dict[str, float]]`.

Contract: on return, `model` is already refit on the **full** series (both `SingleStrategy` and `BestOfStrategy` call `.fit(series)` a second time on the winner after evaluation).

---

## Data flow

1. **`main.py:main()`** — reads `sys.argv[1]` (file path) or `PRED_SERVICE_CONFIG` env var as raw JSON string.
2. **`config.py:PredServiceConfig.model_validate(raw_config)`** — validates top-level structure; does not merge yet.
3. **`config.py:resolved_metrics()`** — for each raw metric dict, calls `_deep_merge(defaults, metric_dict)` then `MetricConfig.model_validate(merged)`.
4. **`pipeline.py:run_metric(metric_cfg, settings, run_id)`** called per metric:
   a. `_build_source(metric_cfg.source)` — instantiates `ClickHouseSource` or `PrometheusSource`.
   b. `_resolve_time_range(metric_cfg.source)` — computes `(start, end)` datetimes; default 90-day lookback.
   c. `source.fetch(query, start, end)` — returns `pd.Series`.
   d. Constructs `SingleStrategy` or `BestOfStrategy` from `metric_cfg.model`.
   e. `strategy.select(series, step)` → `SelectionResult` (winner already refit on full data).
   f. `selection.model.predict(horizon, freq)` → `forecast_series`.
5. **Outputs** written: `ConsoleSink().write(result)` if `output.console=true`; `ClickHouseSink(...).write(result)` if `output.clickhouse` is set.

---

## Config schema (machine-readable summary)

### `ClickHouseSourceConfig`
| field | type | default |
|---|---|---|
| `host` | `str` | `"localhost"` |
| `port` | `int` | `8123` |
| `user` | `str` | `"default"` |
| `password` | `str` | `""` |
| `database` | `str` | `"default"` |

### `PrometheusSourceConfig`
| field | type | default |
|---|---|---|
| `url` | `str` | `"http://localhost:9090"` |
| `step` | `str` | `"5m"` |
| `username` | `str` | `""` |
| `password` | `str` | `""` |
| `disable_ssl` | `bool` | `False` |

### `TimeRangeConfig`
| field | type | default |
|---|---|---|
| `lookback_days` | `Optional[int]` | `None` |
| `start` | `Optional[str]` | `None` |
| `end` | `Optional[str]` | `None` |

Validator: `lookback_days` and `start`/`end` are mutually exclusive; `start` and `end` must both be present if either is set.

### `PreprocessConfig`
| field | type | default |
|---|---|---|
| `scale` | `Optional[float]` | `None` |

### `SourceConfig`
| field | type | default |
|---|---|---|
| `type` | `Literal["clickhouse", "prometheus"]` | `"clickhouse"` |
| `clickhouse` | `Optional[ClickHouseSourceConfig]` | `None` |
| `prometheus` | `Optional[PrometheusSourceConfig]` | `None` |
| `query` | `str` | `""` |
| `time_range` | `Optional[TimeRangeConfig]` | `None` |
| `preprocess` | `Optional[PreprocessConfig]` | `None` |

### `ModelSpec`
| field | type | default |
|---|---|---|
| `type` | `str` | required |
| `estimator` | `Optional[str]` | `None` |
| `params` | `dict[str, Any]` | `{}` |

### `ModelConfig`
| field | type | default |
|---|---|---|
| `strategy` | `Literal["single", "best_of"]` | `"best_of"` |
| `candidates` | `list[Union[str, ModelSpec]]` | `["walkforward/lightgbm", "walkforward/ridge", "linear_trend", "seasonal_naive"]` |
| `eval_metric` | `Literal["mae", "rmse", "mape"]` | `"rmse"` |
| `eval_fraction` | `float` (ge=0.05, le=0.5) | `0.2` |

### `ForecastConfig`
| field | type | default |
|---|---|---|
| `horizon_steps` | `int` (gt=0) | `288` |
| `step` | `str` | `"5m"` |

### `ClickHouseOutputConfig`
| field | type | default |
|---|---|---|
| `table` | `str` | `"metrics_forecast"` |

### `OutputConfig`
| field | type | default |
|---|---|---|
| `clickhouse` | `Optional[ClickHouseOutputConfig]` | `None` |
| `console` | `bool` | `True` |

### `MetricConfig`
| field | type | default |
|---|---|---|
| `service` | `str` | required |
| `metric` | `str` | required |
| `source` | `SourceConfig` | required |
| `model` | `ModelConfig` | `ModelConfig()` |
| `forecast` | `ForecastConfig` | `ForecastConfig()` |
| `output` | `OutputConfig` | `OutputConfig()` |

### `PredServiceConfig`
| field | type | default |
|---|---|---|
| `defaults` | `dict[str, Any]` | `{}` |
| `metrics` | `list[dict[str, Any]]` | required |
| `continue_on_error` | `bool` | `True` |

### `Settings` (pydantic-settings, from env / `.env.pred-service`)
| field | env var | default |
|---|---|---|
| `pred_ch_host` | `PRED_CH_HOST` | `"localhost"` |
| `pred_ch_port` | `PRED_CH_PORT` | `8123` |
| `pred_ch_user` | `PRED_CH_USER` | `"default"` |
| `pred_ch_password` | `PRED_CH_PASSWORD` | `""` |
| `pred_ch_database` | `PRED_CH_DATABASE` | `"default"` |
| `log_level` | `LOG_LEVEL` | `"INFO"` |

---

## Model registry

Defined in `models/registry.py` — `get_model(spec: str | dict) -> ForecastModel`:

| Shorthand string | Class instantiated |
|---|---|
| `"walkforward"` or `"walkforward/<est>"` | `WalkForwardModel(estimator=<est or "lightgbm">)` |
| `"linear_trend"` | `LinearTrendModel()` |
| `"polynomial_trend"` | `PolynomialTrendModel()` |
| `"seasonal_naive"` | `SeasonalNaiveModel()` |
| `"naive_constant"` | `NaiveConstantModel()` |
| `"drift"` | `DriftModel()` |

Dict form: `{"type": ..., "estimator": ..., "params": {...}}`. Only `walkforward` uses `estimator`. `params` is passed to the constructor: `polynomial_trend` reads `degree` and `alpha`; `seasonal_naive` reads `period_steps`; `naive_constant` and `drift` read `n`.

### `_ESTIMATOR_REGISTRY` in `models/walkforward.py`

| Key (estimator name) | Module | Class |
|---|---|---|
| `"ridge"` | `sklearn.linear_model` | `Ridge` |
| `"lasso"` | `sklearn.linear_model` | `Lasso` |
| `"linear"` | `sklearn.linear_model` | `LinearRegression` |
| `"random_forest"` | `sklearn.ensemble` | `RandomForestRegressor` |
| `"extra_trees"` | `sklearn.ensemble` | `ExtraTreesRegressor` |
| `"hist_gradient_boosting"` | `sklearn.ensemble` | `HistGradientBoostingRegressor` |
| `"lightgbm"` | `lightgbm` | `LGBMRegressor` |
| `"xgboost"` | `xgboost` | `XGBRegressor` |

All estimators are imported lazily via `importlib.import_module` at fit time — missing optional packages (lightgbm, xgboost) raise `ImportError` only when that estimator is actually used.

---

## The deep-merge algorithm

Defined in `config.py:_deep_merge(base, override)`:

```
result = copy of base
for each (key, val) in override:
    if key in result AND result[key] is dict AND val is dict:
        result[key] = _deep_merge(result[key], val)  # recurse
    else:
        result[key] = val  # override wins; lists are REPLACED not merged
return result
```

Called by `resolved_metrics()`:
```python
merged = _deep_merge(copy.deepcopy(self.defaults), copy.deepcopy(raw_metric_dict))
MetricConfig.model_validate(merged)
```

Both sides are deep-copied before merging, so defaults are not mutated between metrics. Key consequence: a list in a metric entry (e.g. `model.candidates`) replaces the default list entirely — there is no list append/union behavior.

---

## WalkForward feature engineering

Defined in `models/walkforward.py`.

**Base lags** (always used): `[1, 2, 3, 6, 12, 24, 48]`

**Seasonal lag** (288 steps): added only if `len(series) >= 338` (i.e. `_SEASONAL_LAG + _SEASONAL_LAG_MIN_LEN_OFFSET = 288 + 50`). Specifically: `_SEASONAL_LAG_MIN_LEN = 338`.

**Time features** (computed from the DatetimeIndex):
- `hour_sin = sin(2π × hour / 24)`
- `hour_cos = cos(2π × hour / 24)`
- `dow_sin = sin(2π × dayofweek / 7)`
- `dow_cos = cos(2π × dayofweek / 7)`
- `is_weekend = float(dayofweek >= 5)`

**Feature matrix shape**: `(n - max_lag, len(lags) + 5)` where 5 is the number of time features. The first `max_lag` rows are discarded because not all lags are available.

**Recursive prediction step by step**:
1. After `fit()`, `self._history` stores the last `max(lags)` values from the training series.
2. In `predict()`, a sliding `window = list(self._history)` is initialized.
3. For each step `i` in `range(horizon)`:
   - Lag features: `[window[-lag] for lag in self._lags]`
   - Time features: computed from the forecast DatetimeIndex at position `i`
   - Concatenate → 1-row feature matrix → `self._model.predict(x_row)[0]`
   - Append prediction to `window` (so it becomes a lag for the next step)
4. Returns `pd.Series(predictions, index=forecast_idx)`.

---

## Output table schema

Exact DDL from `output/clickhouse.py`:

```sql
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
```

`kind="eval"` rows: model predictions over the holdout window (last `eval_fraction` of the input series). Used for back-testing and quality review. Timestamps are historical.

`kind="forecast"` rows: model predictions for future steps (`horizon_steps` beyond the last observed timestamp). These are the operationally useful rows.

Both kinds share the same `eval_mae/rmse/mape/r2` values (computed once on the holdout, repeated on every row of the batch). The table is created via `client.command(ddl)` at the start of every `ClickHouseSink.write()` call.

---

## Common extension points

| What to add | Where |
|---|---|
| New source type | Create `sources/mysource.py` extending `DataSource`; add config model to `config.py`; add branch in `pipeline._build_source()` |
| New model type | Implement `ForecastModel` protocol; register string shorthand in `registry._from_string()` and `_from_dict()` |
| New output sink | Create `output/mysink.py` extending `OutputSink`; instantiate and call `.write(result)` in `pipeline.run_metric()` |
| New WalkForward estimator | Add `"name": ("module", "Class")` entry to `_ESTIMATOR_REGISTRY` in `walkforward.py`; no other changes needed |

---

## Non-obvious design decisions

1. **`SeasonalNaiveModel` uses tiling, not shifting.** `predict()` calls `np.tile(season_values, ceil(horizon / period))[:horizon]`. This means the seasonal pattern always starts from the same phase as the last observed period, regardless of how far in the future the forecast window starts.

2. **Both strategies refit the winner on full data.** After evaluating on the train/holdout split, both `SingleStrategy` and `BestOfStrategy` call `model.fit(series)` a second time using the complete series before returning. The eval scores are from the first fit; the model that generates the forecast is from the second fit.

3. **`WalkForwardModel._history` stores only `max(lags)` values, not the full series.** After `fit()`, only the tail `values[-max(self._lags):]` is retained. This is sufficient for recursive prediction but means the model cannot be re-evaluated without refitting.

4. **The seasonal lag (288) is conditional on series length.** It is only included if `len(series) >= 338` (288 + 50). Below this threshold, lag-288 features are silently omitted and the feature matrix is narrower. This can cause shape mismatches if a model is fit on a long series then somehow asked to predict on a shorter one (though the current code does not support that path).

5. **`ClickHouseSink` uses the output ClickHouse from `Settings` (env vars), not from `source.clickhouse`.** Source credentials (for reading metrics) and output credentials (for writing forecasts) are configured entirely separately. A single run can read from one ClickHouse and write to a different one.

6. **`PrometheusSource` uses only the first metric series returned by the query.** If a PromQL expression matches multiple time series (e.g. without enough label matchers), the extras are silently ignored with a WARNING log. No error is raised.

7. **`LinearTrendModel` uses `ts.toordinal()` (integer days since year 1) as its time axis, not Unix timestamps.** This gives numerical stability but means the slope unit is "value change per day". The origin is pinned to the first training point to keep `x` values small.

8. **`_deep_merge` replaces lists entirely.** If `defaults.model.candidates = ["walkforward/lightgbm", "seasonal_naive"]` and a metric sets `model.candidates = ["drift"]`, the result is `["drift"]` — not a union. There is no append/extend behavior.

9. **`run_id` defaults to a fresh UUID but is overridden by `AIRFLOW_RUN_ID` env var.** This makes ClickHouse rows from Airflow-triggered runs traceable back to a specific DAG run without any extra plumbing.

10. **`BestOfStrategy` skips invalid model specs with a warning rather than failing.** In `select()`, a `ValueError`/`TypeError` from `get_model(spec)` or any exception from `model.fit()`/`model.predict()` logs a WARNING and continues to the next candidate. Only if every single candidate fails does it raise `RuntimeError("All candidate models failed.")`.
