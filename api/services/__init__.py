"""
Servicios reutilizables del backend RiskLab USTA.

  - data_service  : cache transparente de datos externos en SQLite.
  - decorators    : decoradores propios (logging de tiempo, etc.).
  - fixed_income  : FRED, curva Nelson-Siegel y bono sintetico (M9).
  - options       : Black-Scholes, Greeks y volatilidad implicita (M10).
  - stress        : escenarios de stress testing sobre el portafolio (M11).
"""
from api.services.data_service import DataService, get_data_service
from api.services.decorators import log_execution_time
from api.services.fixed_income import (
    Bond,
    FredClient,
    NelsonSiegelParams,
    YieldCurve,
    get_fred_client,
)
from api.services.options import OptionPricer
from api.services.stress import StressTester

__all__ = [
    "DataService",
    "get_data_service",
    "log_execution_time",
    "Bond",
    "FredClient",
    "NelsonSiegelParams",
    "YieldCurve",
    "get_fred_client",
    "OptionPricer",
    "StressTester",
]
