"""
Dependencias inyectables centrales — patrón Depends() de FastAPI.

Estas funciones desacoplan los handlers de los servicios concretos. Cada router
recibe sus dependencias por inyección, lo que facilita los tests con override.

Cubre los 4 puntos exigidos por la guía del profesor:
  1. Sesión SQLAlchemy (BD inyectada por request).
  2. Servicios de datos (FRED, precios).
  3. Configuración (BaseSettings).
  4. Carga del modelo ML (Singleton).
"""
from __future__ import annotations

from typing import Annotated

from fastapi import Depends
from sqlalchemy.orm import Session

from backend.app.config import Settings, get_settings
from backend.app.database import get_db
from backend.app.ml.predictor import ModelPredictor, get_predictor
from backend.app.services import fred_service
from backend.app.services.price_service import get_prices

# ── Aliases tipados para que los routers los importen limpiamente ─────────────
SettingsDep = Annotated[Settings,        Depends(get_settings)]
DBSession   = Annotated[Session,         Depends(get_db)]
Predictor   = Annotated[ModelPredictor,  Depends(get_predictor)]


# ── Servicios de datos (delegan a módulos puros para no romper la firma) ─────

def get_fred_service():
    """Devuelve el módulo fred_service. Útil para sobreescribir en tests."""
    return fred_service


def get_price_service():
    """Devuelve la función pública del servicio de precios."""
    return get_prices


FredService  = Annotated[object, Depends(get_fred_service)]
PriceService = Annotated[object, Depends(get_price_service)]


__all__ = [
    "SettingsDep", "DBSession", "Predictor",
    "FredService", "PriceService",
    "get_settings", "get_db", "get_predictor",
    "get_fred_service", "get_price_service",
]
