"""
Configuración centralizada del backend RiskLab USTA.

Toda la configuración se carga vía `pydantic-settings` desde:
  1. Variables de entorno del proceso (mayor prioridad).
  2. Archivo `.env` en la raíz del proyecto.
  3. Defaults declarados en la clase `Settings`.

Uso típico en una ruta FastAPI:

    from fastapi import Depends
    from api.config import Settings, get_settings

    @app.get("/foo")
    def foo(settings: Settings = Depends(get_settings)):
        return {"benchmark": settings.benchmark}
"""
from __future__ import annotations

import secrets
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DB_FILE = (_PROJECT_ROOT / "data" / "risklab.db").as_posix()


class Settings(BaseSettings):
    """Configuración del sistema, cargada desde .env y variables de entorno."""

    model_config = SettingsConfigDict(
        env_file=str(_PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
    )

    # ── API ───────────────────────────────────────────────────
    api_title: str = "RiskLab USTA API"
    api_version: str = "2.1.0"

    # ── Auth JWT ──────────────────────────────────────────────
    jwt_secret: str = Field(default_factory=lambda: secrets.token_hex(32))
    jwt_ttl_minutes: int = Field(60, ge=1, le=10080)

    # ── Persistencia (SQLAlchemy / SQLite) ────────────────────
    database_url: str = f"sqlite:///{_DEFAULT_DB_FILE}"

    # ── Portafolio del análisis ───────────────────────────────
    tickers: list[str] = Field(
        default_factory=lambda: ["NU", "AMZN", "SONY", "XOM", "WPM"],
        validation_alias="PORTFOLIO_TICKERS",
    )
    benchmark: str = Field("^GSPC", validation_alias="BENCHMARK_TICKER")
    default_rf: float = Field(
        0.04, ge=0.0, le=1.0, validation_alias="RISK_FREE_RATE_DEFAULT"
    )

    # ── APIs externas ─────────────────────────────────────────
    fred_api_key: Optional[str] = None
    yfinance_cache_ttl_hours: int = Field(24, ge=1, le=720)

    # ── Machine Learning ──────────────────────────────────────
    ml_model_path: str = "api/ml/model.joblib"

    # ── Rango de fechas disponibles ───────────────────────────
    data_min_date: str = "2020-01-01"

    @field_validator("tickers", mode="before")
    @classmethod
    def _split_csv(cls, v):
        """Permite escribir la lista como CSV en .env (PORTFOLIO_TICKERS=NU,AMZN,...)."""
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Acceso cacheado a la configuración (instancia única por proceso)."""
    return Settings()
