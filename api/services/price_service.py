"""
Cache transparente de precios OHLCV en SQLite vía SQLAlchemy.

Implementa la estrategia recomendada por el instructivo:
"si el dato existe en BD y la fecha es reciente, leer de BD; si no, llamar
a la API externa y persistir el resultado antes de retornarlo."

Reduce los rate limits de Yahoo y acelera tiempos de respuesta del frontend.
La capa antigua `api.data.get_historical_data` sigue funcionando como
fallback directo (sin cache) para no romper código que ya la usa.
"""
from __future__ import annotations

import time
from datetime import date, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf
from sqlalchemy import and_, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from api.db_models import Asset, Price

# TTL de cache: si la fecha más reciente almacenada es >= hoy - REFRESH_DAYS
# se considera fresca. Default 1 día (precios de hoy quedan disponibles tras cierre).
REFRESH_DAYS = 1
DEFAULT_LOOKBACK_DAYS = 365 * 2  # 2 años (mínimo del instructivo)
MAX_RETRIES = 3
RETRY_BACKOFF_SEC = 1.5


def _ensure_asset(db: Session, ticker: str) -> Asset:
    asset = db.query(Asset).filter(Asset.ticker == ticker).first()
    if asset is None:
        asset = Asset(ticker=ticker, name=ticker, sector=None)
        db.add(asset)
        db.commit()
        db.refresh(asset)
    return asset


def _query_cached(db: Session, asset: Asset,
                  start_date: Optional[date], end_date: Optional[date]) -> pd.DataFrame:
    """Devuelve DataFrame con los precios cacheados en el rango pedido."""
    q = select(Price).where(Price.asset_id == asset.id)
    if start_date:
        q = q.where(Price.date >= start_date)
    if end_date:
        q = q.where(Price.date <= end_date)
    q = q.order_by(Price.date.asc())
    rows = db.execute(q).scalars().all()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([{
        "Date":   r.date,
        "Open":   r.open,
        "High":   r.high,
        "Low":    r.low,
        "Close":  r.close,
        "Volume": r.volume,
    } for r in rows])


def _is_cache_fresh(db: Session, asset: Asset, end_date: Optional[date]) -> bool:
    """True si el último precio cacheado está dentro del TTL respecto a la fecha pedida."""
    last = (db.query(Price.date)
              .filter(Price.asset_id == asset.id)
              .order_by(Price.date.desc()).first())
    if last is None or last[0] is None:
        return False
    target = end_date or date.today()
    return (target - last[0]).days <= REFRESH_DAYS


def _download_with_retries(ticker: str,
                           start: Optional[str], end: Optional[str],
                           period: str = "2y") -> pd.DataFrame:
    """Descarga de yfinance con reintentos exponenciales simples."""
    last_exc: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if start and end:
                df = yf.download(ticker, start=start, end=end, interval="1d", progress=False)
            else:
                df = yf.download(ticker, period=period, interval="1d", progress=False)
            if df is not None and not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.reset_index(inplace=True)
                if "Date" not in df.columns and "index" in df.columns:
                    df.rename(columns={"index": "Date"}, inplace=True)
                return df
            return pd.DataFrame()
        except Exception as exc:
            last_exc = exc
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SEC * attempt)
    if last_exc:
        print(f"[price_service] yfinance falló para {ticker} tras {MAX_RETRIES} intentos: {last_exc}")
    return pd.DataFrame()


def _persist_prices(db: Session, asset: Asset, df: pd.DataFrame) -> int:
    """Guarda precios nuevos evitando duplicados (asset_id, date) gracias al unique constraint."""
    if df.empty:
        return 0
    inserted = 0
    for _, row in df.iterrows():
        d = row["Date"]
        d_obj = d.date() if hasattr(d, "date") else d
        exists = db.query(Price).filter(
            and_(Price.asset_id == asset.id, Price.date == d_obj)
        ).first()
        if exists:
            continue
        try:
            db.add(Price(
                asset_id=asset.id,
                date=d_obj,
                open=float(row.get("Open"))    if pd.notna(row.get("Open"))   else None,
                high=float(row.get("High"))    if pd.notna(row.get("High"))   else None,
                low=float(row.get("Low"))      if pd.notna(row.get("Low"))    else None,
                close=float(row.get("Close"))  if pd.notna(row.get("Close"))  else None,
                volume=float(row.get("Volume")) if pd.notna(row.get("Volume")) else None,
            ))
            inserted += 1
        except SQLAlchemyError:
            db.rollback()
            continue
    if inserted:
        db.commit()
    return inserted


def get_prices(
    db: Session,
    ticker: str,
    period: str = "2y",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    fresh: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """API pública del servicio.

    Devuelve (DataFrame, meta) donde meta incluye `cache_status` (hit/miss/refresh),
    `n_cached`, `n_downloaded`, `source`.

    `fresh=True` fuerza descarga incluso si hay cache fresco.
    """
    meta = {"cache_status": "miss", "n_cached": 0, "n_downloaded": 0, "source": "yfinance"}
    asset = _ensure_asset(db, ticker)

    sd = pd.to_datetime(start_date).date() if start_date else None
    ed = pd.to_datetime(end_date).date() if end_date else None

    # Si hay cache fresco y no piden refresh, lee de BD
    if not fresh and _is_cache_fresh(db, asset, ed):
        cached = _query_cached(db, asset, sd, ed)
        if not cached.empty:
            meta.update({"cache_status": "hit", "n_cached": len(cached), "source": "sqlite_cache"})
            return cached, meta

    # Cache vacío o vencido → descarga + persiste
    df = _download_with_retries(ticker, start_date, end_date, period=period)
    if df.empty:
        # Si el download falla pero hay cache (aunque viejo), úsalo como fallback
        cached = _query_cached(db, asset, sd, ed)
        if not cached.empty:
            meta.update({"cache_status": "stale_used", "n_cached": len(cached), "source": "sqlite_cache_stale"})
            return cached, meta
        meta["cache_status"] = "unavailable"
        return pd.DataFrame(), meta

    n_inserted = _persist_prices(db, asset, df)
    meta.update({
        "cache_status": "refresh" if fresh else "miss",
        "n_cached":     0,
        "n_downloaded": n_inserted,
    })
    return df, meta


def cache_summary(db: Session) -> dict:
    """Reporta métricas del cache para fines de auditoría/dashboard."""
    n_assets = db.query(Asset).count()
    n_prices = db.query(Price).count()
    last = (db.query(Price.date).order_by(Price.date.desc()).first())
    return {
        "n_assets":       n_assets,
        "n_prices_cached": n_prices,
        "last_date":      last[0].isoformat() if last and last[0] else None,
    }
