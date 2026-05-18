"""Тригерит pred_service DAG через Airflow REST API.

Запуск с вшитым конфигом:
    python airflow/trigger_dag.py

Запуск с конфигом из файла:
    python airflow/trigger_dag.py airflow/config_example.json
"""
import json
import sys
import urllib.request
import urllib.error
from base64 import b64encode
from pathlib import Path


# ── CONFIG ────────────────────────────────────────────────────────────────────

AIRFLOW_URL      = "http://localhost:8080"
AIRFLOW_USERNAME = "airflow"
AIRFLOW_PASSWORD = "airflow"

# ── PRED-SERVICE CONFIG ───────────────────────────────────────────────────────

CONFIG = {
    "defaults": {
        "source": {
            "type": "prometheus",
            "prometheus": {
                "url": "https://prometheus.your-company.com",
                "step": "5m",
                "disable_ssl": True,
            },
            "time_range": {"lookback_days": 90},
        },
        "model": {
            "strategy": "best_of",
            # Use string shorthand for default params, or dict form for full control.
            # Dict form example (all keys optional — only override what you need):
            #   {"type": "walkforward", "estimator": "lightgbm",
            #    "params": {"n_estimators": 300},
            #    "lags": [1, 2, 3, 6, 12, 24, 48], "seasonal_lag": 288}
            #   {"type": "seasonal_naive", "params": {"period_steps": 144}}
            #   {"type": "polynomial_trend", "params": {"degree": 3}}
            #   {"type": "naive_constant", "params": {"n": 20}}
            "candidates": [
                "walkforward/lightgbm",
                "walkforward/ridge",
                "linear_trend",
                "seasonal_naive",
            ],
            "eval_metric": "rmse",
            "eval_fraction": 0.2,
            "refit_on_full_data": True,   # refit winner on full series after evaluation
        },
        "forecast": {
            "horizon_steps": 288,
            "step": "5m",
        },
        "output": {
            "clickhouse": {"table": "metrics_forecast"},
            "console": True,
            "save_eval": True,  # write holdout eval rows to ClickHouse (kind="eval")
        },
    },
    "metrics": [
        {
            "service": "airflow-worker",
            "metric": "memory_gb",
            "source": {
                "query": (
                    "sum(container_memory_working_set_bytes"
                    "{container='airflow-worker', node='ndp-v01wnl-n19'})"
                ),
                "preprocess": {"scale": 1e-9},
            },
        },
    ],
    "continue_on_error": True,
}

# ── RUN ───────────────────────────────────────────────────────────────────────

def _load_config() -> dict:
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
        print(f"Loading config from: {path}")
        return json.loads(path.read_text())
    return CONFIG


def main() -> None:
    url = f"{AIRFLOW_URL.rstrip('/')}/api/v1/dags/pred_service/dagRuns"

    payload = {
        "conf": {
            "config_json": json.dumps(_load_config()),
        }
    }

    token = b64encode(f"{AIRFLOW_USERNAME}:{AIRFLOW_PASSWORD}".encode()).decode()
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
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code}: {e.read().decode()}", file=sys.stderr)
        sys.exit(1)

    print(f"dag_run_id:   {result['dag_run_id']}")
    print(f"state:        {result['state']}")
    print(f"logical_date: {result.get('logical_date', '—')}")


if __name__ == "__main__":
    main()
