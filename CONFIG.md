# pred-service — JSON Config Guide

Сервис принимает конфиг двумя способами:
- переменная окружения `PRED_SERVICE_CONFIG` (используется в Kubernetes / Airflow)
- напрямую: `python airflow/trigger_dag.py airflow/config_example.json`

---

## Структура верхнего уровня

```json
{
  "defaults":          { },     // общие настройки — наследуются всеми метриками
  "metrics":           [ ],     // список метрик (обязателен, минимум одна)
  "continue_on_error": true     // не падать при ошибке одной метрики (default: true)
}
```

`defaults` и каждая запись `metrics` имеют одну и ту же схему.  
При обработке метрики сервис делает **deep merge**: `defaults` — база, поля метрики её перекрывают.  
Это значит, что в `defaults` можно один раз описать источник данных и модель, а в метриках писать только отличия.

---

## Схема метрики

```
{
  "service":  "my-service",   // обязательно — имя сервиса
  "metric":   "memory_gb",    // обязательно — имя метрики

  "source":   { ... },        // откуда брать данные
  "model":    { ... },        // как выбирать и обучать модель
  "forecast": { ... },        // горизонт и шаг прогноза
  "output":   { ... }         // куда писать результат
}
```

---

## source — источник данных

```json
"source": {
  "type": "prometheus",         // "prometheus" | "clickhouse"

  "prometheus": {
    "url":         "http://prometheus:9090",
    "step":        "5m",        // шаг дискретизации (должен совпадать с forecast.step)
    "username":    "",
    "password":    "",
    "disable_ssl": false
  },

  "clickhouse": {               // используется если type = "clickhouse"
    "host":     "localhost",
    "port":     8123,
    "user":     "default",
    "password": "",
    "database": "default"
  },

  "query": "sum(container_memory_working_set_bytes{container='svc'})",

  "time_range": {
    "lookback_days": 90         // ИЛИ явные границы:
    // "start": "2025-01-01T00:00:00Z",
    // "end":   "2025-04-01T00:00:00Z"
  },

  "preprocess": {
    "scale": 1e-9               // умножить значения на коэффициент (например bytes → GB)
  }
}
```

`time_range` принимает либо `lookback_days`, либо пару `start`/`end` — не оба вместе.  
Если `time_range` не задан, по умолчанию берётся 90 дней назад.

---

## model — выбор и обучение модели

```json
"model": {
  "strategy":           "best_of",  // "best_of" | "single"
  "candidates":         [ ],        // список моделей
  "eval_metric":        "rmse",     // метрика отбора: "rmse" | "mae" | "mape"
  "eval_fraction":      0.2,        // доля ряда под holdout (0.05 – 0.5)
  "refit_on_full_data": true        // переобучить победителя на полном ряду
}
```

### strategy

**`best_of`** — обучает все `candidates` на тренировочной части, оценивает на holdout,
выбирает лучшую по `eval_metric`. Победитель переобучается на полном ряду (если `refit_on_full_data: true`).

**`single`** — использует первый элемент `candidates`, остальные игнорирует.

### candidates — форматы моделей

**Строка** (только тип и estimator, все параметры по умолчанию):

```json
"candidates": [
  "walkforward/lightgbm",
  "walkforward/ridge",
  "linear_trend",
  "seasonal_naive"
]
```

**Словарь** (полный контроль над параметрами):

```json
"candidates": [
  {
    "type":      "walkforward",
    "estimator": "lightgbm",
    "params": {                        // гиперпараметры estimator
      "n_estimators":  300,
      "num_leaves":    63,
      "learning_rate": 0.05
    },
    "lags":                [1, 2, 3, 6, 12, 24, 48],  // базовые лаги (в шагах)
    "seasonal_lag":        288,                         // сезонный лаг
    "seasonal_lag_min_len": 338                         // мин. длина ряда для сезонного лага
  },
  {
    "type": "seasonal_naive",
    "params": { "period_steps": 288 }   // длина одного сезонного периода в шагах
  },
  {
    "type": "polynomial_trend",
    "params": { "degree": 2, "alpha": 1.0 }
  },
  {
    "type": "naive_constant",
    "params": { "n": 20 }               // среднее последних N точек
  },
  {
    "type": "drift",
    "params": { "n": 60 }               // линейная экстраполяция по последним N точкам
  }
]
```

### Доступные типы моделей

| Тип | Estimator | Что делает |
|-----|-----------|------------|
| `walkforward` | `lightgbm`, `ridge`, `lasso`, `linear`, `random_forest`, `extra_trees`, `hist_gradient_boosting`, `xgboost` | Рекурсивный мультишаговый прогноз с лаг-фичами и циклическим временем |
| `linear_trend` | — | Линейный тренд (OLS) |
| `polynomial_trend` | — | Полиномиальный тренд (степень задаётся `degree`) |
| `seasonal_naive` | — | Повторяет последний сезонный период |
| `naive_constant` | — | Среднее последних N точек (горизонтальная линия) |
| `drift` | — | Линейная экстраполяция тренда последних N точек |

### Параметры WalkForward

| Параметр | Где задаётся | Default | Описание |
|----------|-------------|---------|----------|
| `estimator` | верхний уровень dict | `lightgbm` | Имя ML-библиотеки |
| `params` | `params: {}` | `{}` | Kwargs конструктора estimator |
| `lags` | `lags: [...]` | `[1,2,3,6,12,24,48]` | Базовые лаги в шагах |
| `seasonal_lag` | `seasonal_lag: N` | `288` | Шаг сезонного лага |
| `seasonal_lag_min_len` | `seasonal_lag_min_len: N` | `seasonal_lag + 50` | Мин. длина ряда для включения сезонного лага |

Сезонный лаг добавляется автоматически, если длина обучающего ряда ≥ `seasonal_lag_min_len`.

---

## forecast — горизонт прогноза

```json
"forecast": {
  "horizon_steps": 288,   // количество шагов вперёд (288 × 5m = 24 часа)
  "step":          "5m"   // шаг: "1m", "5m", "15m", "1h", "1d" и т.д.
}
```

`step` должен совпадать с шагом данных из источника.

---

## output — куда писать результат

```json
"output": {
  "clickhouse": {
    "table": "metrics_forecast"   // таблица в ClickHouse (задаётся env PRED_CH_*)
  },
  "console":    true,             // печатать сводку в stdout
  "save_eval":  true              // писать holdout-предсказания (kind="eval") в ClickHouse
}
```

Если `clickhouse` не задан (или не указаны env-переменные `PRED_CH_*`), запись в CH пропускается.

Строки в таблице `metrics_forecast` делятся по полю `kind`:
- `"forecast"` — будущие предсказания (основной результат)
- `"eval"` — предсказания на holdout-окне (для оценки качества модели в ретроспективе)

---

## Полный пример с defaults

```json
{
  "defaults": {
    "source": {
      "type": "prometheus",
      "prometheus": {
        "url":         "https://prometheus.internal",
        "step":        "5m",
        "disable_ssl": true
      },
      "time_range": { "lookback_days": 90 }
    },
    "model": {
      "strategy":      "best_of",
      "candidates":    ["walkforward/lightgbm", "walkforward/ridge", "linear_trend", "seasonal_naive"],
      "eval_metric":   "rmse",
      "eval_fraction": 0.2,
      "refit_on_full_data": true
    },
    "forecast": {
      "horizon_steps": 288,
      "step":          "5m"
    },
    "output": {
      "clickhouse": { "table": "metrics_forecast" },
      "console":    true,
      "save_eval":  true
    }
  },
  "metrics": [
    {
      "service": "api-gateway",
      "metric":  "memory_gb",
      "source": {
        "query": "sum(container_memory_working_set_bytes{container='api-gateway'})",
        "preprocess": { "scale": 1e-9 }
      }
    },
    {
      "service": "api-gateway",
      "metric":  "rps",
      "source": {
        "query": "sum(rate(http_requests_total{service='api-gateway'}[1m]))"
      },
      "forecast": {
        "horizon_steps": 576,
        "step": "5m"
      }
    },
    {
      "service": "worker",
      "metric":  "queue_depth",
      "source": {
        "query": "rabbitmq_queue_messages{queue='main'}"
      },
      "model": {
        "strategy": "single",
        "candidates": [
          {
            "type":      "walkforward",
            "estimator": "lightgbm",
            "params":    { "n_estimators": 500 },
            "lags":      [1, 2, 3, 6, 12, 24, 48],
            "seasonal_lag": 288
          }
        ]
      }
    }
  ],
  "continue_on_error": true
}
```

---

## Запуск

```bash
# С файлом конфига
python airflow/trigger_dag.py airflow/config_example.json

# Или запустить сервис напрямую (без Airflow)
PRED_SERVICE_CONFIG=$(cat airflow/config_example.json) \
PRED_CH_HOST=localhost \
PRED_CH_PORT=8123 \
python -m pred_service
```

Готовый пример для старта — [airflow/config_example.json](airflow/config_example.json).
