"""
Configuración central del backend con Pydantic BaseSettings.

Carga variables desde .env (nunca hardcodeadas) y las expone vía Depends().
Documentación de variables: ver `.env.example` en la raíz del proyecto.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import Field
try:
    # Pydantic v2 con pydantic-settings (recomendado)
    from pydantic_settings import BaseSettings, SettingsConfigDict
    _USING_PYDANTIC_SETTINGS = True
except ImportError:
    # Fallback para Pydantic v2 sin pydantic-settings: usar BaseModel + os.getenv
    from pydantic import BaseModel as BaseSettings  # type: ignore
    SettingsConfigDict = dict  # type: ignore
    _USING_PYDANTIC_SETTINGS = False

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_ENV_FILE = _PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    """Configuración global. Se accede vía Depends(get_settings)."""

    # ── Autenticación ──────────────────────────────────────────────────────
    JWT_SECRET:      str = Field(default="changeme_for_production",
                                 description="Clave HS256 para firmar JWT.")
    JWT_TTL_MINUTES: int = Field(default=60, ge=1, le=1440,
                                 description="Tiempo de vida del token JWT (minutos).")

    # ── APIs externas ──────────────────────────────────────────────────────
    FRED_API_KEY:    str = Field(default="",
                                 description="Clave gratuita de FRED. Si vacía, usa fallback DEMO.")

    # ── Persistencia ───────────────────────────────────────────────────────
    DATABASE_URL:    str = Field(
        default=f"sqlite:///{(_PROJECT_ROOT / 'data' / 'risklab_users.db').as_posix()}",
        description="URL de conexión SQLAlchemy.",
    )

    # ── Configuración del portafolio ───────────────────────────────────────
    PORTFOLIO_TICKERS: str = Field(default="NU,AMZN,SONY,XOM,WPM",
                                   description="Tickers separados por coma.")
    BENCHMARK_TICKER:  str = Field(default="^GSPC",
                                   description="Benchmark para CAPM y M8.")
    RISK_FREE_RATE:    float = Field(default=0.04, ge=0, le=1,
                                     description="Tasa libre de riesgo fallback (decimal).")

    # ── Modelo ML ──────────────────────────────────────────────────────────
    ML_MODEL_PATH: str = Field(
        default=str(_PROJECT_ROOT / "backend" / "app" / "ml" / "model.joblib"),
        description="Ruta al joblib del modelo ML servido en /predict.",
    )

    # ── Entorno (desarrollo/producción) ────────────────────────────────────
    ENVIRONMENT: str = Field(default="development",
                             description="development | production")

    if _USING_PYDANTIC_SETTINGS:
        model_config = SettingsConfigDict(
            env_file=str(_ENV_FILE),
            env_file_encoding="utf-8",
            case_sensitive=True,
            extra="ignore",
        )

    # ── Helpers ────────────────────────────────────────────────────────────
    @property
    def tickers_list(self) -> List[str]:
        return [t.strip() for t in self.PORTFOLIO_TICKERS.split(",") if t.strip()]

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT.lower() == "production"

    @property
    def fred_enabled(self) -> bool:
        key = self.FRED_API_KEY.strip().lower()
        placeholders = {"", "your_fred_key_here", "your_fred_api_key", "changeme",
                        "tu_api_key_aqui", "xxx", "todo"}
        return key not in placeholders


# Si pydantic-settings no está disponible, parchear desde os.getenv
if not _USING_PYDANTIC_SETTINGS:
    import os
    from dotenv import load_dotenv
    load_dotenv(_ENV_FILE)
    _orig_init = Settings.__init__
    def _patched_init(self, **kwargs):  # type: ignore
        defaults = {
            "JWT_SECRET":        os.getenv("JWT_SECRET", "changeme_for_production"),
            "JWT_TTL_MINUTES":   int(os.getenv("JWT_TTL_MINUTES", "60")),
            "FRED_API_KEY":      os.getenv("FRED_API_KEY", ""),
            "DATABASE_URL":      os.getenv("DATABASE_URL", f"sqlite:///{(_PROJECT_ROOT / 'data' / 'risklab_users.db').as_posix()}"),
            "PORTFOLIO_TICKERS": os.getenv("PORTFOLIO_TICKERS", "NU,AMZN,SONY,XOM,WPM"),
            "BENCHMARK_TICKER":  os.getenv("BENCHMARK_TICKER", "^GSPC"),
            "RISK_FREE_RATE":    float(os.getenv("RISK_FREE_RATE", "0.04")),
            "ML_MODEL_PATH":     os.getenv("ML_MODEL_PATH", str(_PROJECT_ROOT / "backend" / "app" / "ml" / "model.joblib")),
            "ENVIRONMENT":       os.getenv("ENVIRONMENT", "development"),
        }
        defaults.update(kwargs)
        _orig_init(self, **defaults)
    Settings.__init__ = _patched_init  # type: ignore


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Singleton inyectable vía Depends(get_settings).

    Cachea con lru_cache para que la lectura de .env ocurra UNA sola vez.
    Cualquier endpoint puede recibir `settings: Settings = Depends(get_settings)`.
    """
    return Settings()
