"""
CLI entry point for the prediction service.

Usage:
    python -m pred_service [config.json]

If no argument is provided, reads JSON config from the PRED_SERVICE_CONFIG env var
(useful for Docker/Airflow where config is passed as an environment variable).
"""
from __future__ import annotations

import json
import logging
import os
import sys
import uuid
from typing import Any

from pred_service.config import PredServiceConfig
from pred_service.settings import Settings


def _setup_logging(level: str) -> None:
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stdout,
    )


def _load_config_dict() -> dict[str, Any]:
    """Load raw JSON config from CLI arg or PRED_SERVICE_CONFIG env var."""
    if len(sys.argv) >= 2:
        config_path = sys.argv[1]
        logging.getLogger(__name__).info("Loading config from file: %s", config_path)
        with open(config_path, encoding="utf-8") as fh:
            return json.load(fh)  # type: ignore[no-any-return]

    env_config = os.environ.get("PRED_SERVICE_CONFIG")
    if env_config:
        logging.getLogger(__name__).info("Loading config from PRED_SERVICE_CONFIG env var.")
        return json.loads(env_config)  # type: ignore[no-any-return]

    print(
        "ERROR: No config provided.\n"
        "Usage: python -m pred_service [config.json]\n"
        "       OR set PRED_SERVICE_CONFIG env var with JSON string.",
        file=sys.stderr,
    )
    sys.exit(1)


def main() -> None:
    # Load settings first so we can set log level before anything else
    settings = Settings()
    _setup_logging(settings.log_level)
    logger = logging.getLogger(__name__)

    run_id = os.environ.get("AIRFLOW_RUN_ID") or str(uuid.uuid4())
    logger.info("Prediction service starting. run_id=%s", run_id)

    # ── Load and parse config ─────────────────────────────────────────────────
    raw_config = _load_config_dict()

    try:
        config = PredServiceConfig.model_validate(raw_config)
    except Exception as exc:
        logger.error("Config validation failed: %s", exc)
        sys.exit(1)

    metrics = config.resolved_metrics()
    logger.info("Loaded config: %d metric(s) to process. continue_on_error=%s", len(metrics), config.continue_on_error)

    # ── Run pipeline for each metric ──────────────────────────────────────────
    from pred_service.pipeline import run_metric

    success_count = 0
    error_count = 0

    for metric_cfg in metrics:
        label = f"{metric_cfg.service}/{metric_cfg.metric}"
        try:
            run_metric(metric_cfg=metric_cfg, settings=settings, run_id=run_id)
            success_count += 1
        except Exception as exc:
            error_count += 1
            if config.continue_on_error:
                logger.error("FAILED [%s]: %s — continuing.", label, exc, exc_info=True)
            else:
                logger.error("FAILED [%s]: %s — aborting (continue_on_error=false).", label, exc)
                sys.exit(1)

    logger.info(
        "Prediction service finished. success=%d error=%d total=%d",
        success_count,
        error_count,
        len(metrics),
    )

    if error_count > 0 and not config.continue_on_error:
        sys.exit(1)


if __name__ == "__main__":
    main()
