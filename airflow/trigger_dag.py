"""Тригерит pred_service DAG через Airflow REST API.

Настрой CONFIG и запусти:
    python airflow/trigger_dag.py
"""
import json
import sys
import urllib.request
import urllib.error
from base64 import b64encode


# ── CONFIG ────────────────────────────────────────────────────────────────────

AIRFLOW_URL      = "http://localhost:8080"
AIRFLOW_USERNAME = "airflow"
AIRFLOW_PASSWORD = "airflow"

# ── PRED-SERVICE CONFIG ───────────────────────────────────────────────────────

CONFIG = {
    "time_range": {
        "lookback_days": 300,
    },
    "defaults": {
        "source": {
            "type": "prometheus",
            "prometheus": {
                "url": "https://prometheus.your-company.com",
                "username": "",
                "password": "",
                "disable_ssl": True,
            },
        },
        "step": "5m",
        "predict_model": {
            "name": "walkfwd_lightgbm",
            "kwargs": {"auto_lags": True},
        },
        "forecast": {
            "horizon_days": 1,
            "overflow_condition": "gte",
            "overflow_model": {"name": "linear_trend"},
            "save_eval": True,
        },
    },
    "metrics": [
        {
            "service": "airflow-worker",
            "metric_name": "mem",
            "source": {
                "query": (
                    "sum(container_memory_working_set_bytes"
                    "{container='airflow-worker', node='ndp-v01wnl-n19'})"
                ),
            },
            "forecast": {
                "resource_limit": 200,   # GiB (если value_mul=1e-9)
            },
            "preprocess": {"value_mul": 1e-9},
        },
    ],
    "plots_dir": "/data/prediction_plots",
    "continue_on_error": True,
}

# ── RUN ───────────────────────────────────────────────────────────────────────

def main() -> None:
    url = f"{AIRFLOW_URL.rstrip('/')}/api/v1/dags/pred_service/dagRuns"

    payload = {
        "conf": {
            "config_json": json.dumps(CONFIG),
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
