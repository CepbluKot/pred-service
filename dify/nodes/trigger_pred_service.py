"""Dify Code Node: Trigger pred_service Airflow DAG

Запускает pred_service DAG через Airflow REST API.
Конфиг передаётся как JSON-строка (PRED_SERVICE_CONFIG).

Inputs:
  airflow_url      (str) — базовый URL Airflow, например "http://airflow:8080"
  airflow_username (str) — логин Airflow (Basic Auth)
  airflow_password (str) — пароль Airflow (Basic Auth)
  config_json      (str) — полный JSON-конфиг pred_service (defaults + metrics[])

Outputs:
  dag_run_id   (str) — ID запущенного dag run
  state        (str) — начальное состояние ("queued")
  logical_date (str) — логическая дата запуска
  error        (str) — текст ошибки (пустая строка если успех)

Пример config_json:
  {
    "defaults": {
      "source": {
        "type": "prometheus",
        "prometheus": {"url": "http://prometheus:9090", "step": "5m"},
        "time_range": {"lookback_days": 90}
      },
      "model": {
        "strategy": "best_of",
        "candidates": ["walkforward/lightgbm", "walkforward/ridge", "linear_trend", "seasonal_naive"],
        "eval_metric": "rmse",
        "eval_fraction": 0.2
      },
      "forecast": {"horizon_steps": 288, "step": "5m"},
      "output": {"clickhouse": {"table": "metrics_forecast"}, "console": true}
    },
    "metrics": [
      {
        "service": "my-service",
        "metric": "memory_gb",
        "source": {
          "query": "sum(container_memory_working_set_bytes{container='my-service'})",
          "preprocess": {"scale": 1e-9}
        }
      }
    ],
    "continue_on_error": true
  }
"""

import json
import urllib.request
import urllib.error
from base64 import b64encode


def main(
    airflow_url: str,
    airflow_username: str,
    airflow_password: str,
    config_json: str,
) -> dict:
    # Валидируем что config_json — корректный JSON
    try:
        config = json.loads(config_json)
    except json.JSONDecodeError as e:
        return {
            "dag_run_id":   "",
            "state":        "error",
            "logical_date": "",
            "error":        f"Invalid config_json: {e}",
        }

    url = f"{airflow_url.rstrip('/')}/api/v1/dags/pred_service/dagRuns"

    payload = {
        "conf": {
            "config_json": json.dumps(config),
        }
    }

    token = b64encode(f"{airflow_username}:{airflow_password}".encode()).decode()
    body  = json.dumps(payload).encode()

    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type":  "application/json",
            "Authorization": f"Basic {token}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req) as resp:
            result = json.loads(resp.read())
        return {
            "dag_run_id":   result.get("dag_run_id", ""),
            "state":        result.get("state", ""),
            "logical_date": result.get("logical_date", ""),
            "error":        "",
        }
    except urllib.error.HTTPError as e:
        return {
            "dag_run_id":   "",
            "state":        "error",
            "logical_date": "",
            "error":        f"HTTP {e.code}: {e.read().decode()}",
        }
    except Exception as e:
        return {
            "dag_run_id":   "",
            "state":        "error",
            "logical_date": "",
            "error":        str(e),
        }
