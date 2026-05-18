"""
Environment-based settings for the prediction service.

Source credentials (for reading metrics) come from the JSON config.
Env vars here are ONLY for the output ClickHouse where forecasts are stored.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env.pred-service",
        extra="ignore",
    )

    # ClickHouse for storing forecast results
    pred_ch_host: str = "localhost"
    pred_ch_port: int = 8123
    pred_ch_user: str = "default"
    pred_ch_password: str = ""
    pred_ch_database: str = "default"

    log_level: str = "INFO"
