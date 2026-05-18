"""
Airflow DAG: Prediction Service (metrics forecasting)

Запускает pred-service.py как Pod в Kubernetes.
Принимает JSON-конфиг метрик, строит прогнозы, сохраняет в ClickHouse.

Airflow Variables (Admin → Variables):
  PRED_SERVICE_IMAGE     — Docker-образ (default: registry.your-company.com/pred-service:latest)
  PRED_SERVICE_NAMESPACE — Kubernetes namespace (default: airflow)
  PRED_SERVICE_DATA_PVC  — PVC для графиков/артефактов (default: pred-service-data)

Params (задаются при ручном триггере):
  config_json       — полный JSON-конфиг pred-service (metrics, time_range, defaults, ...)
  continue_on_error — продолжать при ошибке одной метрики (default true)
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta

from airflow import DAG
from airflow.models import Param, Variable
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from kubernetes.client import models as k8s

# ── CONFIG ────────────────────────────────────────────────────────────────────

IMAGE     = Variable.get("PRED_SERVICE_IMAGE",     default_var="registry.your-company.com/pred-service:latest")
NAMESPACE = Variable.get("PRED_SERVICE_NAMESPACE", default_var="airflow")
DATA_PVC  = Variable.get("PRED_SERVICE_DATA_PVC",  default_var="pred-service-data")

# ── DAG ───────────────────────────────────────────────────────────────────────

default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

_CONFIG_EXAMPLE = json.dumps({
    "time_range": {"lookback_days": 300},
    "defaults": {
        "source": {
            "type": "prometheus",
            "prometheus": {
                "url": "http://prometheus:9090",
                "disable_ssl": True,
            },
        },
        "step": "5m",
    },
    "metrics": [
        {
            "service": "svc-a",
            "metric_name": "mem",
            "source": {"query": "sum(container_memory_working_set_bytes{container='airflow-worker'})"},
            "predict_model": {"name": "walkfwd_lightgbm", "kwargs": {"auto_lags": True}},
            "forecast": {
                "horizon_days": 1,
                "resource_limit": 200,
                "overflow_condition": "gte",
                "overflow_model": {"name": "linear_trend"},
            },
        }
    ],
    "continue_on_error": True,
}, indent=2)

with DAG(
    dag_id="pred_service",
    default_args=default_args,
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["pred-service", "forecasting", "ml", "k8s"],
    params={
        "config_json": Param(
            _CONFIG_EXAMPLE,
            type="string",
            description=(
                "JSON-конфиг pred-service: metrics[], time_range, defaults, continue_on_error. "
                "Полная схема — см. документацию pred-service."
            ),
        ),
    },
) as dag:

    run_pred_service = KubernetesPodOperator(
        task_id="run_pred_service",
        name="pred-service",
        namespace=NAMESPACE,
        image=IMAGE,
        image_pull_policy="Always",

        env_vars={
            # ClickHouse для сохранения прогнозов
            "PRED_CH_HOST":     "{{ var.value.PRED_CH_HOST }}",
            "PRED_CH_PORT":     "{{ var.value.PRED_CH_PORT }}",
            "PRED_CH_USER":     "{{ var.value.PRED_CH_USER }}",
            "PRED_CH_PASSWORD": "{{ var.value.PRED_CH_PASSWORD }}",
            "PRED_CH_DATABASE": "{{ var.value.PRED_CH_DATABASE }}",
            # Конфиг передаётся через env — сервис читает PRED_SERVICE_CONFIG
            "PRED_SERVICE_CONFIG": "{{ params.config_json }}",
            "AIRFLOW_RUN_ID":      "{{ run_id }}",
        },

        security_context=k8s.V1PodSecurityContext(
            run_as_non_root=True,
            run_as_user=1000,
        ),
        container_security_context=k8s.V1SecurityContext(
            read_only_root_filesystem=False,  # сервис пишет PNG-графики
            run_as_non_root=True,
            run_as_user=1000,
        ),

        volumes=[
            k8s.V1Volume(
                name="data",
                persistent_volume_claim=k8s.V1PersistentVolumeClaimVolumeSource(claim_name=DATA_PVC),
            ),
            k8s.V1Volume(
                name="tmp",
                empty_dir=k8s.V1EmptyDirVolumeSource(),
            ),
        ],

        volume_mounts=[
            k8s.V1VolumeMount(name="data", mount_path="/data"),
            k8s.V1VolumeMount(name="tmp",  mount_path="/tmp"),
        ],

        container_resources=k8s.V1ResourceRequirements(
            requests={"cpu": "500m", "memory": "1Gi"},
            limits={"cpu": "4",     "memory": "8Gi"},
        ),

        get_logs=True,
        is_delete_operator_pod=True,
        in_cluster=True,
    )
