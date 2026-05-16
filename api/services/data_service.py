"""
DataService: cache transparente de datos externos sobre SQLite.

Cuando el sistema pide precios de un ticker:
  1. Resuelve (o crea) el Asset correspondiente.
  2. Si la tabla `prices` tiene filas recientes (fetched_at < TTL) que
     cubren el rango solicitado, devuelve la query directa.
  3. Si falta o esta viejo, llama a yfinance, hace upsert en `prices`
     y devuelve el resultado.

El shape del DataFrame retornado es identico al que devolvia la
funcion `get_historical_data` original, para que el resto del backend
no requiera ningun cambio.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Iterator, Optional

import pandas as pd
import yfinance as yf
from sqlalchemy import func, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from api.config import Settings, get_settings
from api.db.base import SessionLocal
from api.db.models import Asset, Price
from api.services.decorators import log_execution_time

logger = logging.getLogger(__name__)


class DataService:
    """Servicio de datos con cache transparente en SQLite."""

    def __init__(
        self,
        db: Optional[Session] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self._db = db
        self._owns_session = db is None
        self.settings = settings or get_settings()

    # ── Manejo de sesion ─────────────────────────────────────────────────────

    def _session(self) -> Session:
        if self._db is None:
            self._db = SessionLocal()
        return self._db

    def close(self) -> None:
        if self._owns_session and self._db is not None:
            self._db.close()
            self._db = None

    def __enter__(self) -> "DataService":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ── API publica ──────────────────────────────────────────────────────────

    def get_or_create_asset(self, ticker: str) -> Asset:
        """Devuelve el Asset del ticker, creandolo si no existe."""
        db = self._session()
        asset = db.scalar(select(Asset).where(Asset.ticker == ticker))
        if asset is not None:
            return asset
        asset = Asset(ticker=ticker)
        db.add(asset)
        db.commit()
        db.refresh(asset)
        return asset

    @log_execution_time
    def get_prices_df(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: str = "2y",
        fresh: bool = False,
    ) -> Optional[pd.DataFrame]:
        """
        Retorna OHLCV del ticker con columnas Date, Open, High, Low, Close, Volume.

        Si fresh=True ignora el cache y fuerza una llamada a yfinance.
        """
        # ── Resolver rango efectivo ─────────────────────────────────────────
        if start_date and end_date:
            start = (
                date.fromisoformat(start_date)
                if isinstance(start_date, str)
                else start_date
            )
            end = (
                date.fromisoformat(end_date)
                if isinstance(end_date, str)
                else end_date
            )
        else:
            end = date.today()
            years = self._parse_period_years(period)
            start = end - timedelta(days=int(years * 365))

        asset = self.get_or_create_asset(ticker)

        # ── Cache hit ───────────────────────────────────────────────────────
        if not fresh and self._cache_fresh_enough(asset, start, end):
            df_cached = self._read_from_db(asset, start, end)
            if df_cached is not None and not df_cached.empty:
                logger.debug("cache HIT %s [%s, %s] -> %d filas",
                             ticker, start, end, len(df_cached))
                return df_cached

        # ── Cache miss: llamada a yfinance + upsert ─────────────────────────
        logger.debug("cache MISS %s [%s, %s] -> fetch yfinance", ticker, start, end)
        df_yf = self._fetch_yfinance(ticker, start, end)
        if df_yf is None or df_yf.empty:
            # Fallback: devolver lo que haya en BD (puede estar incompleto)
            df_fallback = self._read_from_db(asset, start, end)
            return df_fallback if (df_fallback is not None and not df_fallback.empty) else None

        self._upsert_prices(asset, df_yf)
        return df_yf

    def cache_stats(self) -> dict:
        """Estadisticas agregadas del cache (uso diagnostico)."""
        db = self._session()
        total_assets = db.scalar(select(func.count()).select_from(Asset)) or 0
        total_prices = db.scalar(select(func.count()).select_from(Price)) or 0
        last_fetch = db.scalar(select(func.max(Price.fetched_at)))
        return {
            "assets": int(total_assets),
            "price_rows": int(total_prices),
            "last_fetched_at": last_fetch.isoformat() if last_fetch else None,
            "ttl_hours": self.settings.yfinance_cache_ttl_hours,
        }

    # ── Internos ─────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_period_years(period: str) -> float:
        p = (period or "").strip().lower()
        if p.endswith("y"):
            try:
                return float(p[:-1])
            except ValueError:
                pass
        if p.endswith("mo"):
            try:
                return float(p[:-2]) / 12.0
            except ValueError:
                pass
        if p.endswith("d"):
            try:
                return float(p[:-1]) / 365.0
            except ValueError:
                pass
        return 2.0  # default razonable

    def _cache_fresh_enough(self, asset: Asset, start: date, end: date) -> bool:
        """True si en BD hay datos recientes y suficientes para el rango."""
        db = self._session()
        ttl = timedelta(hours=self.settings.yfinance_cache_ttl_hours)
        threshold = datetime.utcnow() - ttl

        last = db.scalar(
            select(func.max(Price.fetched_at)).where(Price.asset_id == asset.id)
        )
        if last is None or last < threshold:
            return False

        rows_in_range = db.scalar(
            select(func.count(Price.id))
            .where(Price.asset_id == asset.id)
            .where(Price.date >= start)
            .where(Price.date <= end)
        ) or 0

        # ~252 dias habiles por anio. Umbral conservador: 80% de 200/anio.
        expected = max(1, int(((end - start).days / 365.0) * 200))
        return rows_in_range >= int(expected * 0.8)

    def _read_from_db(
        self, asset: Asset, start: date, end: date
    ) -> Optional[pd.DataFrame]:
        db = self._session()
        rows = db.scalars(
            select(Price)
            .where(Price.asset_id == asset.id)
            .where(Price.date >= start)
            .where(Price.date <= end)
            .order_by(Price.date)
        ).all()
        if not rows:
            return None
        return pd.DataFrame(
            [
                {
                    "Date": pd.Timestamp(r.date),
                    "Open": r.open,
                    "High": r.high,
                    "Low": r.low,
                    "Close": r.close,
                    "Volume": r.volume,
                }
                for r in rows
            ]
        )

    @staticmethod
    def _fetch_yfinance(
        ticker: str, start: date, end: date
    ) -> Optional[pd.DataFrame]:
        try:
            df = yf.download(
                ticker,
                start=start.isoformat(),
                end=(end + timedelta(days=1)).isoformat(),  # yf end es exclusivo
                interval="1d",
                progress=False,
                auto_adjust=False,
            )
            if df is None or df.empty:
                return None
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if df.index.name != "Date":
                df.index.name = "Date"
            df = df.reset_index()
            if "Date" not in df.columns and "index" in df.columns:
                df = df.rename(columns={"index": "Date"})
            return df
        except Exception as exc:
            logger.warning("Error al descargar %s de yfinance: %s", ticker, exc)
            return None

    def _upsert_prices(self, asset: Asset, df: pd.DataFrame) -> None:
        if df.empty:
            return
        db = self._session()
        now = datetime.utcnow()
        rows = []
        for _, r in df.iterrows():
            d = r["Date"]
            if isinstance(d, pd.Timestamp):
                d = d.date()
            rows.append(
                {
                    "asset_id": asset.id,
                    "date": d,
                    "open": float(r.get("Open")) if pd.notna(r.get("Open")) else None,
                    "high": float(r.get("High")) if pd.notna(r.get("High")) else None,
                    "low": float(r.get("Low")) if pd.notna(r.get("Low")) else None,
                    "close": float(r.get("Close")) if pd.notna(r.get("Close")) else None,
                    "volume": float(r.get("Volume")) if pd.notna(r.get("Volume")) else None,
                    "fetched_at": now,
                }
            )
        if not rows:
            return
        stmt = sqlite_insert(Price).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=["asset_id", "date"],
            set_={
                "open": stmt.excluded.open,
                "high": stmt.excluded.high,
                "low": stmt.excluded.low,
                "close": stmt.excluded.close,
                "volume": stmt.excluded.volume,
                "fetched_at": stmt.excluded.fetched_at,
            },
        )
        db.execute(stmt)
        db.commit()


# ── Dependency injection ────────────────────────────────────────────────────


def get_data_service() -> Iterator[DataService]:
    """
    Dependencia FastAPI. Crea un DataService que gestiona su propia sesion.

    Uso:
        @app.get('/foo')
        def foo(svc: DataService = Depends(get_data_service)):
            ...
    """
    service = DataService()
    try:
        yield service
    finally:
        service.close()
