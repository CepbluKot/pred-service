# Prediction Generator

## 1. What is this

`pred_service` is a time-series forecasting microservice that reads historical metric data from ClickHouse or Prometheus, trains one or more regression models, generates a future forecast, and writes the results (both the holdout back-test and the forward forecast) back to a ClickHouse table. It is designed to run as a one-shot job — locally from a config file, inside Docker, or scheduled via Airflow — producing ready-to-query forecast rows that downstream dashboards or alerting systems can consume.

---

## 2. Architecture overview

```
┌─────────────────────────────────────────────────────────┐
│                       main.py                           │
│  CLI entry point — loads config, iterates metrics,      │
│  calls run_metric() for each, handles continue_on_error │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                    pipeline.py                          │
│  run_metric(): orchestrates one metric end-to-end       │
│  _build_source()  _resolve_time_range()                 │
└──┬──────────────────────────┬──────────────────────────┘
   │                          │
   ▼                          ▼
┌──────────────────┐  ┌───────────────────────────────────┐
│   sources/       │  │          models/                  │
│  DataSource ABC  │  │  selector.py  (strategies)        │
│  ┌────────────┐  │  │  ┌─────────────┐ ┌─────────────┐ │
│  │ClickHouse  │  │  │  │SingleStrat. │ │BestOfStrat. │ │
│  │ Source     │  │  │  └─────────────┘ └─────────────┘ │
│  ├────────────┤  │  │  registry.py  (get_model)         │
│  │Prometheus  │  │  │  ┌──────────┬──────────────────┐  │
│  │ Source     │  │  │  │naive.py  │linear.py         │  │
│  └────────────┘  │  │  │walkfwd.py│                  │  │
└──────────────────┘  │  └──────────┴──────────────────┘  │
                      └───────────────────────────────────┘
                                       │
                                       ▼
                      ┌───────────────────────────────────┐
                      │   evaluator.py                    │
                      │   compute MAE/RMSE/MAPE/R2        │
                      └───────────────────────────────────┘
                                       │
                                       ▼
                      ┌───────────────────────────────────┐
                      │         output/                   │
                      │  OutputSink ABC                   │
                      │  ┌──────────────┐ ┌────────────┐ │
                      │  │ClickHouseSink│ │ConsoleSink │ │
                      │  └──────────────┘ └────────────┘ │
                      └───────────────────────────────────┘
```

| Component | File | Responsibility |
|---|---|---|
| CLI entry point | `pred_service/main.py` | Parses config from file or `PRED_SERVICE_CONFIG`, iterates over metrics, respects `continue_on_error` |
| Config schema | `pred_service/config.py` | Pydantic models for all config fields; `_deep_merge` and `resolved_metrics()` |
| Settings | `pred_service/settings.py` | pydantic-settings for output ClickHouse credentials loaded from env / `.env.pred-service` |
| Pipeline | `pred_service/pipeline.py` | `run_metric()` — end-to-end orchestration for one metric; `PredictionResult` dataclass |
| DataSource ABC | `pred_service/sources/base.py` | Abstract `fetch(query, start, end) -> pd.Series` |
| ClickHouse source | `pred_service/sources/clickhouse.py` | Reads 2-column (timestamp, value) result set via `clickhouse_connect` |
| Prometheus source | `pred_service/sources/prometheus.py` | Calls `custom_query_range` via `prometheus_api_client`, picks first returned metric |
| Evaluator | `pred_service/evaluator.py` | Computes MAE, RMSE, MAPE, R2 with NaN-safe alignment |
| ForecastModel Protocol | `pred_service/models/base.py` | Structural protocol: `fit(series)`, `predict(horizon, freq)`, `name` |
| Naive models | `pred_service/models/naive.py` | `NaiveConstantModel`, `SeasonalNaiveModel`, `DriftModel` |
| Linear/poly models | `pred_service/models/linear.py` | `LinearTrendModel` (numpy lstsq), `PolynomialTrendModel` (Ridge) |
| WalkForward model | `pred_service/models/walkforward.py` | Lag + time features + any sklearn-compatible estimator, recursive prediction |
| Model registry | `pred_service/models/registry.py` | `get_model(spec)` — parses shorthand strings and dict specs |
| Model selector | `pred_service/models/selector.py` | `SingleStrategy`, `BestOfStrategy`, `_step_to_freq()` |
| OutputSink ABC | `pred_service/output/base.py` | Abstract `write(result: PredictionResult)` |
| ClickHouse sink | `pred_service/output/clickhouse.py` | Writes eval + forecast rows to `metrics_forecast`; creates table if absent |
| Console sink | `pred_service/output/console.py` | Logs a human-readable summary at INFO level |

---

## 3. JSON config reference

The top-level JSON object is validated against `PredServiceConfig`. All `metrics` entries are deep-merged with `defaults` before validation — see section "The deep-merge behavior of defaults" below.

### Top-level fields

| Field | Type | Default | Required | Description |
|---|---|---|---|---|
| `metrics` | `list[object]` | — | **yes** | List of per-metric config objects |
| `defaults` | `object` | `{}` | no | Shared config merged into every metric entry |
| `continue_on_error` | `bool` | `true` | no | When `true`, a failed metric is logged and skipped; when `false`, the service aborts on first error |

### `source`

| Field | Type | Default | Required | Description |
|---|---|---|---|---|
| `source.type` | `"clickhouse"` \| `"prometheus"` | `"clickhouse"` | no | Which backend to query |
| `source.query` | `string` | `""` | **yes** | SQL (ClickHouse) or PromQL (Prometheus) |
| `source.time_range` | `object` | `null` (90-day lookback) | no | See below |
| `source.preprocess.scale` | `float` | `null` | no | Multiply every value by this factor after fetching (e.g. `1e-9` for bytes→GiB) |
| `source.clickhouse.host` | `string` | `"localhost"` | no | |
| `source.clickhouse.port` | `int` | `8123` | no | |
| `source.clickhouse.user` | `string` | `"default"` | no | |
| `source.clickhouse.password` | `string` | `""` | no | |
| `source.clickhouse.database` | `string` | `"default"` | no | |
| `source.prometheus.url` | `string` | `"http://localhost:9090"` | no | |
| `source.prometheus.step` | `string` | `"5m"` | no | Query resolution step |
| `source.prometheus.username` | `string` | `""` | no | |
| `source.prometheus.password` | `string` | `""` | no | |
| `source.prometheus.disable_ssl` | `bool` | `false` | no | |

**`source.time_range`** — mutually exclusive options:

| Field | Type | Description |
|---|---|---|
| `lookback_days` | `int` | Fetch the last N days from now |
| `start` | `string` (ISO 8601) | Explicit start; must be paired with `end` |
| `end` | `string` (ISO 8601) | Explicit end; must be paired with `start` |

If `time_range` is omitted entirely, the default is a 90-day lookback from now.

### `model`

| Field | Type | Default | Description |
|---|---|---|---|
| `model.strategy` | `"single"` \| `"best_of"` | `"best_of"` | Selection strategy |
| `model.candidates` | `list[string \| object]` | `["walkforward/lightgbm", "walkforward/ridge", "linear_trend", "seasonal_naive"]` | Models to try; see Model catalog |
| `model.eval_metric` | `"mae"` \| `"rmse"` \| `"mape"` | `"rmse"` | Metric used to rank candidates in `best_of` |
| `model.eval_fraction` | `float` (0.05–0.5) | `0.2` | Fraction of the series held out for evaluation |

**String shorthand form:** `"walkforward/lightgbm"`, `"linear_trend"`, `"seasonal_naive"`, etc.

**Dict form:**
```json
{
  "type": "walkforward",
  "estimator": "lightgbm",
  "params": {"n_estimators": 300, "learning_rate": 0.05}
}
```

For non-WalkForward models, `params` accepts model-specific keys: `polynomial_trend` accepts `degree` (int, default 2) and `alpha` (float, default 1.0); `seasonal_naive` accepts `period_steps` (int, default 288); `naive_constant` accepts `n` (int, default 10); `drift` accepts `n` (int, default 60).

### `forecast`

| Field | Type | Default | Description |
|---|---|---|---|
| `forecast.horizon_steps` | `int` (> 0) | `288` | Number of future steps to predict |
| `forecast.step` | `string` | `"5m"` | Time resolution: `"5m"`, `"15m"`, `"1h"`, `"1d"`, etc. |

### `output`

| Field | Type | Default | Description |
|---|---|---|---|
| `output.console` | `bool` | `true` | Print a summary to the log |
| `output.clickhouse` | `object \| null` | `null` | If present, write results to ClickHouse |
| `output.clickhouse.table` | `string` | `"metrics_forecast"` | Destination table name |

### The deep-merge behavior of defaults

`defaults` is merged recursively into each metric entry before Pydantic validation. Rules:

- Both sides have the key and both values are dicts → merge recursively.
- One side is missing the key → the other side wins.
- Both sides have the key and at least one value is not a dict (scalars, lists) → the metric-level value wins, replacing the default entirely (lists are **not** merged, they are replaced).

```json
{
  "defaults": {
    "source": {
      "type": "prometheus",
      "prometheus": { "url": "https://prom.internal" }
    },
    "forecast": { "step": "5m", "horizon_steps": 288 }
  },
  "metrics": [
    {
      "service": "api",
      "metric": "latency_p99",
      "source": { "query": "histogram_quantile(0.99, ...)" },
      "forecast": { "horizon_steps": 576 }
    }
  ]
}
```

After merge the metric gets `source.type="prometheus"`, `source.prometheus.url="https://prom.internal"`, `forecast.step="5m"`, and `forecast.horizon_steps=576` (metric overrides default).

### Complete working example

```json
{
  "continue_on_error": true,
  "defaults": {
    "source": {
      "type": "clickhouse",
      "clickhouse": {
        "host": "ch.internal",
        "port": 8123,
        "user": "reader",
        "password": "secret",
        "database": "metrics"
      },
      "time_range": { "lookback_days": 90 }
    },
    "model": {
      "strategy": "best_of",
      "candidates": [
        "walkforward/lightgbm",
        "walkforward/ridge",
        "linear_trend",
        "seasonal_naive"
      ],
      "eval_metric": "rmse",
      "eval_fraction": 0.2
    },
    "forecast": { "horizon_steps": 288, "step": "5m" },
    "output": {
      "console": true,
      "clickhouse": { "table": "metrics_forecast" }
    }
  },
  "metrics": [
    {
      "service": "my-service",
      "metric": "cpu_usage",
      "source": {
        "query": "SELECT ts, cpu FROM cpu_metrics WHERE ts BETWEEN {start} AND {end}"
      }
    },
    {
      "service": "my-service",
      "metric": "memory_bytes",
      "source": {
        "query": "SELECT ts, mem FROM mem_metrics WHERE ts BETWEEN {start} AND {end}",
        "preprocess": { "scale": 1e-9 }
      },
      "model": {
        "strategy": "single",
        "candidates": [
          { "type": "walkforward", "estimator": "lightgbm", "params": { "n_estimators": 300 } }
        ]
      }
    }
  ]
}
```

---

## 4. Model catalog

| Name | Shorthand | Description | Best for | Configurable params (dict form) |
|---|---|---|---|---|
| `NaiveConstantModel` | `naive_constant` | Predicts the mean of the last N observed values (flat line) | Stable, non-trending metrics | `n` (int, default 10) |
| `SeasonalNaiveModel` | `seasonal_naive` | Repeats the last complete seasonal period (default 288 steps = 1 day at 5-min resolution) as the forecast | Strongly periodic daily patterns | `period_steps` (int, default 288) |
| `DriftModel` | `drift` | Extrapolates the linear trend between the first and last of the N most recent observations | Slowly trending metrics | `n` (int, default 60) |
| `LinearTrendModel` | `linear_trend` | Fits an OLS line over the full series (integer ordinal time index) via `numpy.linalg.lstsq` | Long-term linear growth/decline | none |
| `PolynomialTrendModel` | `polynomial_trend` | Polynomial regression via `PolynomialFeatures + Ridge` | Non-linear growth curves | `degree` (int, default 2), `alpha` (float, default 1.0) |
| `WalkForwardModel` | `walkforward/lightgbm` | Lag + time features + LightGBM; recursive multi-step prediction | Complex seasonality and non-linear patterns | any `LGBMRegressor` kwargs in `params` |
| `WalkForwardModel` | `walkforward/ridge` | Same lag/time features + Ridge regression | Quick baseline for ML approach | any `Ridge` kwargs in `params` |
| `WalkForwardModel` | `walkforward/random_forest` | Same feature set + `RandomForestRegressor` | Non-linear, robust to outliers | any `RandomForestRegressor` kwargs |
| `WalkForwardModel` | `walkforward/extra_trees` | Same feature set + `ExtraTreesRegressor` | Faster ensemble alternative to random forest | any `ExtraTreesRegressor` kwargs |
| `WalkForwardModel` | `walkforward/hist_gradient_boosting` | Same feature set + `HistGradientBoostingRegressor` | Handles NaN natively, fast on large datasets | any `HistGradientBoostingRegressor` kwargs |
| `WalkForwardModel` | `walkforward/lasso` | Same feature set + `Lasso` | Sparse feature selection | any `Lasso` kwargs |
| `WalkForwardModel` | `walkforward/linear` | Same feature set + `LinearRegression` | Interpretable linear baseline | any `LinearRegression` kwargs |
| `WalkForwardModel` | `walkforward/xgboost` | Same feature set + `XGBRegressor` | Alternative boosted tree implementation | any `XGBRegressor` kwargs |

---

## 5. Model strategies

### `single`

Trains exactly one model (the first entry in `candidates`) on the training split, evaluates it on the holdout, then refits it on the full series. Use when you already know which model works best for this metric and want to skip comparison overhead.

```json
"model": {
  "strategy": "single",
  "candidates": ["walkforward/lightgbm"]
}
```

### `best_of`

Trains every candidate on the training split (the first `1 - eval_fraction` of the series), evaluates each on the holdout window, picks the winner by the lowest `eval_metric` value, then **refits the winner on the full series** before forecasting. NaN scores are skipped. If all candidates fail, the pipeline raises an error.

```json
"model": {
  "strategy": "best_of",
  "candidates": [
    "walkforward/lightgbm",
    "walkforward/ridge",
    "linear_trend",
    "seasonal_naive"
  ],
  "eval_metric": "rmse",
  "eval_fraction": 0.2
}
```

**When to use `best_of`:** When you don't know which model family will win on a given metric. The overhead is one extra train+predict pass per candidate over the training split. For 4 candidates and a 90-day series this typically takes seconds.

**When to use `single`:** When a specific model is known to perform well (e.g. a well-tuned LightGBM for a metric you've profiled before), or in production where you want deterministic runtime.

---

## 6. ClickHouse output schema

### DDL

```sql
CREATE TABLE IF NOT EXISTS metrics_forecast (
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

The table is created with `CREATE TABLE IF NOT EXISTS` on every run, so no manual DDL is needed.

### Column reference

| Column | Description |
|---|---|
| `generated_at` | Wall-clock time when the service wrote this batch of rows (UTC) |
| `run_id` | UUID for this run (or `AIRFLOW_RUN_ID` env var when run from Airflow) |
| `service` | Service name from config |
| `metric` | Metric name from config |
| `model` | Name of the winning model, e.g. `walkforward/lightgbm` |
| `strategy` | `single` or `best_of` |
| `kind` | `eval` (back-test predictions on holdout) or `forecast` (future predictions) |
| `step_seconds` | Time resolution in seconds (e.g. 300 for 5-minute data) |
| `timestamp` | The timestamp this prediction refers to |
| `value` | Predicted metric value |
| `eval_mae` | Mean Absolute Error on the holdout window |
| `eval_rmse` | Root Mean Squared Error on the holdout window |
| `eval_mape` | Mean Absolute Percentage Error on the holdout window (%) |
| `eval_r2` | R² coefficient on the holdout window |

### Get the latest forecast

```sql
SELECT
    timestamp,
    value,
    model,
    eval_rmse
FROM metrics_forecast
WHERE
    service = 'my-service'
    AND metric = 'cpu_usage'
    AND kind = 'forecast'
    AND generated_at = (
        SELECT max(generated_at)
        FROM metrics_forecast
        WHERE service = 'my-service' AND metric = 'cpu_usage'
    )
ORDER BY timestamp;
```

---

## 7. Running locally

### Prerequisites

```bash
pip install -r requirements.txt
```

### From a config file

```bash
python -m pred_service my_config.json
```

### From an environment variable

```bash
export PRED_SERVICE_CONFIG='{"metrics": [...], "defaults": {...}}'
python -m pred_service
```

### Setting up `.env.pred-service`

Create `.env.pred-service` in the working directory (copy from `.env.pred-service.example`):

```ini
# ClickHouse where forecast results are stored
PRED_CH_HOST=localhost
PRED_CH_PORT=8123
PRED_CH_USER=default
PRED_CH_PASSWORD=
PRED_CH_DATABASE=default

LOG_LEVEL=INFO
```

Note: credentials for the *source* ClickHouse or Prometheus (where you read metrics from) go in the JSON config, not here. The `.env.pred-service` file is only for the *output* ClickHouse where forecast rows are written.

### Minimal config example

```json
{
  "metrics": [
    {
      "service": "my-service",
      "metric": "requests_per_second",
      "source": {
        "type": "clickhouse",
        "clickhouse": {
          "host": "localhost",
          "database": "metrics"
        },
        "query": "SELECT ts, rps FROM rps_table WHERE ts BETWEEN {start} AND {end}",
        "time_range": { "lookback_days": 30 }
      },
      "output": {
        "console": true,
        "clickhouse": { "table": "metrics_forecast" }
      }
    }
  ]
}
```

---

## 8. Running via Airflow

The service is designed to run as a Kubernetes pod DAG. The Airflow DAG reads the following **Airflow Variables**:

| Variable | Description |
|---|---|
| `PRED_SERVICE_IMAGE` | Docker image to run (e.g. `registry.internal/pred-service:latest`) |
| `PRED_SERVICE_NAMESPACE` | Kubernetes namespace for the pod |
| `PRED_SERVICE_DATA_PVC` | PVC name mounted at `/data` inside the pod |
| `PRED_CH_HOST` | Output ClickHouse host |
| `PRED_CH_PORT` | Output ClickHouse port |
| `PRED_CH_USER` | Output ClickHouse user |
| `PRED_CH_PASSWORD` | Output ClickHouse password |
| `PRED_CH_DATABASE` | Output ClickHouse database |

### Triggering manually

Edit the `CONFIG` dict in `airflow/trigger_dag.py` to set your Airflow URL, credentials, and the desired pred-service config JSON, then run:

```bash
python airflow/trigger_dag.py
```

The script calls `POST /api/v1/dags/pred_service/dagRuns` with `conf.config_json` set to the serialised config JSON, and prints the `dag_run_id` and initial state.

---

## 9. Adding a new data source

**Step 1.** Create `pred_service/sources/mysource.py` extending `DataSource`:

```python
from pred_service.sources.base import DataSource
import pandas as pd
from datetime import datetime

class MySource(DataSource):
    def __init__(self, cfg, preprocess=None):
        self._cfg = cfg
        self._preprocess = preprocess

    def fetch(self, query: str, start: datetime, end: datetime) -> pd.Series:
        # ... fetch data, return pd.Series with UTC DatetimeIndex
        pass
```

**Step 2.** Add a Pydantic config model in `pred_service/config.py` (e.g. `MySourceConfig`) and add a field `mysource: Optional[MySourceConfig] = None` to `SourceConfig`. Update the `type` literal: `Literal["clickhouse", "prometheus", "mysource"]`.

**Step 3.** Register the source in `pipeline._build_source()` in `pred_service/pipeline.py`:

```python
elif src_type == "mysource":
    from pred_service.sources.mysource import MySource
    if source_cfg.mysource is None:
        raise ValueError("source.type='mysource' requires source.mysource config.")
    return MySource(cfg=source_cfg.mysource, preprocess=preprocess)
```

**Step 4.** The `fetch()` method must return a `pd.Series` with a UTC-aware `DatetimeIndex`, float values, sorted ascending. Apply `preprocess.scale` if set.

---

## 10. Adding a new model

**Step 1.** Implement the `ForecastModel` protocol in a new or existing file under `pred_service/models/`:

```python
class MyModel:
    name: str = "my_model"

    def fit(self, series: pd.Series) -> "MyModel":
        # train on series (UTC DatetimeIndex, float values)
        return self

    def predict(self, horizon: int, freq: str) -> pd.Series:
        # return pd.Series with UTC DatetimeIndex, horizon points,
        # starting one step after the last training point
        ...
```

**Step 2.** Register the shorthand string in `pred_service/models/registry.py`. In `_from_string()`, add:

```python
if model_type == "my_model":
    from pred_service.models.mymodel import MyModel
    return MyModel()
```

In `_from_dict()`, add the same pattern, reading any relevant keys from `params`.

**Step 3.** Add the shorthand to the docstring at the top of `registry.py` and to the `raise ValueError` messages so error output stays accurate.

**Step 4.** To add a new `WalkForward` estimator (without creating a new model class), just add an entry to `_ESTIMATOR_REGISTRY` in `pred_service/models/walkforward.py`:

```python
"my_estimator": ("my_package", "MyEstimatorClass"),
```

It will then be usable as `"walkforward/my_estimator"` in configs.
