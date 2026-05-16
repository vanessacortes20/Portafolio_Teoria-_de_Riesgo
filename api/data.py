"""
Shim de compatibilidad. La logica real con cache transparente vive
en api.services.data_service.DataService.

Las firmas de las funciones publicas se preservan para que el codigo
existente (api/main.py, generate_data.py) siga funcionando sin cambios.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from api.services.data_service import DataService


def get_historical_data(
    ticker: str,
    period: str = "2y",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """
    Descarga OHLCV de un activo con cache transparente en SQLite.

    En la primera llamada hace fetch a Yahoo Finance y persiste el resultado
    en la tabla `prices`. Las llamadas posteriores dentro del TTL configurado
    (default 24 h) se sirven directo desde la BD sin tocar la red.

    Mantiene la firma original: si start_date y end_date estan ambos
    presentes se usan; de lo contrario se cae al `period`.
    """
    with DataService() as svc:
        return svc.get_prices_df(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            period=period,
        )


def get_portfolio_data(
    tickers: list,
    period: str = "2y",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> dict:
    """Descarga OHLCV para una lista de tickers, reusando una sola sesion."""
    out: dict = {}
    with DataService() as svc:
        for t in tickers:
            df = svc.get_prices_df(
                ticker=t,
                start_date=start_date,
                end_date=end_date,
                period=period,
            )
            if df is not None:
                out[t] = df
    return out
