"""
Servicio FRED (Federal Reserve Economic Data) con cache transparente en SQLite.

Devuelve la última observación de cada serie y mantiene un cache con TTL
configurable (24h por defecto) para evitar consumir el rate limit gratuito de
FRED en cada request del frontend.

Si FRED_API_KEY no está configurada o la API falla, los métodos retornan None
y el endpoint que lo consume debe activar su fallback (yfinance).
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Optional

import requests
from sqlalchemy.orm import Session

from api.db_models import FredCache

# Series FRED soportadas — el resto se puede pedir por id directo
SERIES = {
    "rf_3m":         "DGS3MO",   # Treasury 3-Month Constant Maturity Rate
    "treasury_10y":  "DGS10",    # Treasury 10-Year Constant Maturity Rate
    "inflation_cpi": "CPIAUCSL", # CPI All Urban Consumers
    "yield_1y":      "DGS1",
    "yield_2y":      "DGS2",
    "yield_5y":      "DGS5",
    "yield_30y":     "DGS30",
}

CACHE_TTL_HOURS = 24
FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"
MAX_RETRIES = 3
RETRY_BACKOFF_SEC = 1.5
HTTP_TIMEOUT_SEC = 15


_PLACEHOLDER_VALUES = {
    "your_fred_key_here", "your_fred_api_key", "changeme",
    "tu_api_key_aqui", "xxx", "todo",
}


def _api_key() -> Optional[str]:
    key = os.getenv("FRED_API_KEY", "").strip()
    if not key or key.lower() in _PLACEHOLDER_VALUES:
        return None
    return key


def is_available() -> bool:
    """Indica si el servicio FRED está configurado con una API key válida."""
    return _api_key() is not None


def _fetch_remote(series_id: str) -> Optional[dict]:
    """Llama al endpoint de FRED con reintentos exponenciales.

    Reintenta hasta MAX_RETRIES veces ante fallos de red o respuestas 5xx.
    No reintenta ante 4xx (key inválida, serie inexistente). Retorna None si
    no hay key configurada o si todos los intentos fallan.
    """
    import time
    key = _api_key()
    if not key:
        return None
    params = {
        "series_id":  series_id,
        "api_key":    key,
        "file_type":  "json",
        "sort_order": "desc",
        "limit":      100,
    }
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(FRED_BASE, params=params, timeout=HTTP_TIMEOUT_SEC)
            if r.status_code == 200:
                return r.json()
            if 400 <= r.status_code < 500:
                # 4xx: error del cliente — no tiene sentido reintentar
                return None
            last_error = f"HTTP {r.status_code}"
        except requests.RequestException as exc:
            last_error = str(exc)
        if attempt < MAX_RETRIES:
            time.sleep(RETRY_BACKOFF_SEC * attempt)
    if last_error:
        # Solo log; el caller maneja el None devuelto
        print(f"[fred_service] fallo descargando {series_id} tras {MAX_RETRIES} intentos: {last_error}")
    return None


def _extract_last_value(payload: dict) -> tuple[Optional[float], Optional[str]]:
    """De una respuesta de FRED extrae el último valor numérico válido."""
    obs = payload.get("observations") or []
    for o in obs:
        v = o.get("value")
        if v in (None, "", "."):
            continue
        try:
            return float(v), o.get("date")
        except (TypeError, ValueError):
            continue
    return None, None


def _cache_fresh(entry: FredCache, ttl_hours: int) -> bool:
    if entry is None or entry.fetched_at is None:
        return False
    age = datetime.utcnow() - entry.fetched_at
    return age < timedelta(hours=ttl_hours)


def get_series(
    db: Session,
    series_id: str,
    ttl_hours: int = CACHE_TTL_HOURS,
    force_refresh: bool = False,
) -> Optional[dict]:
    """Devuelve un dict con `value`, `date`, `series_id`, `cache_status`.

    `cache_status` puede ser: 'hit', 'miss', 'stale_used' (cache vencido pero
    se reutilizó por fallo de FRED), o 'unavailable' (sin valor).
    """
    entry = db.query(FredCache).filter(FredCache.series_id == series_id).first()

    if entry and _cache_fresh(entry, ttl_hours) and not force_refresh:
        return {
            "series_id":    series_id,
            "value":        entry.last_value,
            "date":         entry.last_date,
            "cache_status": "hit",
        }

    payload = _fetch_remote(series_id)
    if payload is None:
        # Fallo de FRED: si hay cache (aunque vencido) lo reutilizamos
        if entry is not None:
            return {
                "series_id":    series_id,
                "value":        entry.last_value,
                "date":         entry.last_date,
                "cache_status": "stale_used",
            }
        return None

    value, date = _extract_last_value(payload)
    if value is None:
        return None

    if entry is None:
        entry = FredCache(series_id=series_id, payload=payload,
                          last_value=value, last_date=date,
                          fetched_at=datetime.utcnow())
        db.add(entry)
    else:
        entry.payload = payload
        entry.last_value = value
        entry.last_date = date
        entry.fetched_at = datetime.utcnow()
    db.commit()

    return {
        "series_id":    series_id,
        "value":        value,
        "date":         date,
        "cache_status": "miss",
    }


def get_rf_rate_3m(db: Session) -> Optional[dict]:
    """Tasa libre de riesgo (T-Bill 3 meses). FRED publica el rendimiento en %.

    Retorna el dict con `value` ya en decimal (por ej. 0.052 para 5.2%).
    """
    out = get_series(db, SERIES["rf_3m"])
    if out is None or out.get("value") is None:
        return None
    out["value_decimal"] = round(out["value"] / 100.0, 6)
    return out


def get_treasury_10y(db: Session) -> Optional[dict]:
    """Tasa del Tesoro a 10 años. Retorna decimal en `value_decimal`."""
    out = get_series(db, SERIES["treasury_10y"])
    if out is None or out.get("value") is None:
        return None
    out["value_decimal"] = round(out["value"] / 100.0, 6)
    return out


def get_inflation_yoy(db: Session) -> Optional[dict]:
    """Inflación interanual aproximada a partir del CPI (CPIAUCSL).

    Compara el último valor disponible contra el de hace ~12 meses dentro del
    payload completo cacheado.
    """
    out = get_series(db, SERIES["inflation_cpi"])
    if out is None:
        return None
    entry = db.query(FredCache).filter(FredCache.series_id == SERIES["inflation_cpi"]).first()
    if not entry or not entry.payload:
        return None
    obs = [o for o in entry.payload.get("observations", []) if o.get("value") not in ("", ".", None)]
    if len(obs) < 13:
        return out
    try:
        last  = float(obs[0]["value"])
        prev  = float(obs[12]["value"])
        yoy   = round((last / prev - 1.0), 5)
        out["yoy"] = yoy
    except Exception:
        pass
    return out
