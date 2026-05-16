"""
Decoradores reutilizables del backend.

`log_execution_time` cumple el requisito de la rubrica del Proyecto
Integrador (al menos un decorador personalizado en el codigo).
"""
from __future__ import annotations

import logging
import time
from functools import wraps
from typing import Any, Callable

logger = logging.getLogger(__name__)

_DEFAULT_LEVEL = logging.INFO


def _wrap_with_level(func: Callable, level: int) -> Callable:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            logger.log(
                level,
                "%s.%s ejecutado en %.1f ms",
                func.__module__,
                func.__qualname__,
                elapsed_ms,
            )

    return wrapper


def log_execution_time(func_or_level: Any = None) -> Callable:
    """
    Decorador que loguea el tiempo de ejecucion de la funcion decorada.

    Uso:
        @log_execution_time
        def fetch_prices(...): ...

        @log_execution_time(logging.DEBUG)
        def deep_calc(...): ...
    """
    # Uso sin parentesis: @log_execution_time
    if callable(func_or_level):
        return _wrap_with_level(func_or_level, _DEFAULT_LEVEL)

    # Uso con parentesis: @log_execution_time() o @log_execution_time(DEBUG)
    level = func_or_level if func_or_level is not None else _DEFAULT_LEVEL

    def decorator(func: Callable) -> Callable:
        return _wrap_with_level(func, level)

    return decorator
