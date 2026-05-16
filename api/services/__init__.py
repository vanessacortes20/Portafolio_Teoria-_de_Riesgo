"""
Servicios reutilizables del backend RiskLab USTA.

  - data_service : cache transparente de datos externos en SQLite.
  - decorators   : decoradores propios (logging de tiempo, etc.).
"""
from api.services.data_service import DataService, get_data_service
from api.services.decorators import log_execution_time

__all__ = ["DataService", "get_data_service", "log_execution_time"]
