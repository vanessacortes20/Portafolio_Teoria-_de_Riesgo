"""
Modulo de renta fija (M9 del Proyecto Integrador).

Contenido:
  - FredClient   : wrapper sobre fredapi con cache en SQLite (MacroSeries).
  - YieldCurve   : ajuste de Nelson-Siegel sobre puntos de la curva.
  - Bond         : bono sintetico con duracion Macaulay/modificada y
                   convexidad, mas sensibilidad ante shocks de tasa.

Las clases estan disenadas para inyeccion sencilla en endpoints FastAPI
y para uso aislado en scripts o notebooks.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Iterator, Optional

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from api.config import Settings, get_settings
from api.db.base import SessionLocal
from api.db.models import MacroSeries
from api.services.decorators import log_execution_time

logger = logging.getLogger(__name__)


# =============================================================================
# 1. FredClient: cache transparente para FRED + fallback a yfinance
# =============================================================================


class FredClient:
    """
    Cliente con cache para FRED (Federal Reserve Economic Data).

    Si la API key esta configurada llama a fredapi y persiste los valores en
    `macro_series`. Si no, usa yfinance como fallback para las series
    equivalentes.

    Mapeo curva de tesoros US:
        0.25 anos -> DGS3MO    (yfinance: ^IRX)
        1.00 anos -> DGS1
        2.00 anos -> DGS2
        5.00 anos -> DGS5      (yfinance: ^FVX)
       10.00 anos -> DGS10     (yfinance: ^TNX)
       30.00 anos -> DGS30     (yfinance: ^TYX)
    """

    YIELD_CURVE_SERIES: dict[float, str] = {
        0.25: "DGS3MO",
        1.00: "DGS1",
        2.00: "DGS2",
        5.00: "DGS5",
        10.00: "DGS10",
        30.00: "DGS30",
    }

    # Fallback yfinance (parcial: FRED tiene mas plazos)
    YF_FALLBACK: dict[float, str] = {
        0.25: "^IRX",
        5.00: "^FVX",
        10.00: "^TNX",
        30.00: "^TYX",
    }

    def __init__(
        self,
        db: Optional[Session] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._db = db
        self._owns_session = db is None
        self._fred = None  # lazy

    # ── Session ──────────────────────────────────────────────────────────────

    def _session(self) -> Session:
        if self._db is None:
            self._db = SessionLocal()
        return self._db

    def close(self) -> None:
        if self._owns_session and self._db is not None:
            self._db.close()
            self._db = None

    def __enter__(self) -> "FredClient":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ── Cliente FRED lazy ────────────────────────────────────────────────────

    def _client(self):
        if self._fred is not None:
            return self._fred
        if not self.settings.fred_api_key:
            return None
        try:
            from fredapi import Fred  # type: ignore
            self._fred = Fred(api_key=self.settings.fred_api_key)
            return self._fred
        except Exception as exc:
            logger.warning("No se pudo inicializar fredapi: %s", exc)
            return None

    # ── API publica ──────────────────────────────────────────────────────────

    @log_execution_time
    def get_yield_curve(self) -> dict:
        """
        Construye la curva de rendimiento con los puntos definidos en
        YIELD_CURVE_SERIES. Retorna un dict por madurez:

            {0.25: {"yield": 0.0525, "date": "2025-04-15", "source": "FRED", "series_id": "DGS3MO"},
             1.00: {...}, ...}

        Tambien retorna metadata "source" agregada y "as_of".
        """
        out: dict[float, dict] = {}
        source_used = None
        as_of_max: Optional[date] = None

        client = self._client()
        if client is not None:
            for mat, series_id in self.YIELD_CURVE_SERIES.items():
                res = self.get_series_latest(series_id)
                if res is None:
                    continue
                d, v = res
                out[mat] = {
                    "yield":     v / 100.0,
                    "date":      d.isoformat(),
                    "source":    "FRED",
                    "series_id": series_id,
                }
                if as_of_max is None or d > as_of_max:
                    as_of_max = d
            if out:
                source_used = "FRED"

        # Fallback con yfinance para los plazos faltantes
        if len(out) < len(self.YIELD_CURVE_SERIES):
            yf_points = self._yfinance_curve_points()
            for mat, info in yf_points.items():
                if mat not in out:
                    out[mat] = info
                    if as_of_max is None or info["_date_obj"] > as_of_max:
                        as_of_max = info["_date_obj"]
            for v in out.values():
                v.pop("_date_obj", None)
            if not source_used and yf_points:
                source_used = "yfinance"
            elif source_used == "FRED" and yf_points:
                source_used = "mixed"

        return {
            "as_of":  as_of_max.isoformat() if as_of_max else None,
            "source": source_used or "none",
            "points": dict(sorted(out.items())),
        }

    def get_series_latest(
        self, series_id: str, ttl_hours: int = 24
    ) -> Optional[tuple[date, float]]:
        """Devuelve (date, value_in_percent) del ultimo dato disponible."""
        db = self._session()
        ttl = timedelta(hours=ttl_hours)
        threshold = datetime.utcnow() - ttl

        last_cached = db.execute(
            select(MacroSeries.date, MacroSeries.value, MacroSeries.fetched_at)
            .where(MacroSeries.series_id == series_id)
            .order_by(MacroSeries.date.desc())
            .limit(1)
        ).first()

        if (
            last_cached is not None
            and last_cached.fetched_at is not None
            and last_cached.fetched_at >= threshold
        ):
            return (last_cached.date, float(last_cached.value))

        # Fetch fresco
        client = self._client()
        if client is None:
            return None

        try:
            series = client.get_series(series_id)
            if series is None or len(series) == 0:
                return None
            series = series.dropna()
            if series.empty:
                return None
            # Persistir las ultimas ~60 observaciones para historico ligero
            self._upsert_series(series_id, series.tail(60))
            last_idx = series.index[-1]
            last_date = (
                last_idx.date() if hasattr(last_idx, "date") else last_idx
            )
            return (last_date, float(series.iloc[-1]))
        except Exception as exc:
            logger.warning("FRED fetch %s fallo: %s", series_id, exc)
            return None

    # ── Internos ─────────────────────────────────────────────────────────────

    def _upsert_series(self, series_id: str, series: pd.Series) -> None:
        if series.empty:
            return
        db = self._session()
        now = datetime.utcnow()
        rows = []
        for idx, val in series.items():
            if pd.isna(val):
                continue
            d = idx.date() if hasattr(idx, "date") else idx
            rows.append(
                {
                    "series_id":  series_id,
                    "date":       d,
                    "value":      float(val),
                    "fetched_at": now,
                }
            )
        if not rows:
            return
        stmt = sqlite_insert(MacroSeries).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=["series_id", "date"],
            set_={
                "value":      stmt.excluded.value,
                "fetched_at": stmt.excluded.fetched_at,
            },
        )
        db.execute(stmt)
        db.commit()

    def _yfinance_curve_points(self) -> dict[float, dict]:
        """Fallback: obtiene plazos via yfinance cuando FRED no esta disponible."""
        try:
            import yfinance as yf
        except ImportError:
            return {}

        out: dict[float, dict] = {}
        for mat, ticker in self.YF_FALLBACK.items():
            try:
                hist = yf.Ticker(ticker).history(period="5d", auto_adjust=False)
                if hist is None or hist.empty:
                    continue
                last_idx = hist.index[-1]
                last_date = (
                    last_idx.date() if hasattr(last_idx, "date") else last_idx
                )
                # yfinance reporta el rendimiento ya en porcentaje
                yld_pct = float(hist["Close"].iloc[-1])
                out[mat] = {
                    "yield":     yld_pct / 100.0,
                    "date":      last_date.isoformat(),
                    "source":    "yfinance",
                    "series_id": ticker,
                    "_date_obj": last_date,
                }
            except Exception as exc:
                logger.debug("yfinance fallback %s fallo: %s", ticker, exc)
        return out


def get_fred_client() -> Iterator[FredClient]:
    """Dependencia FastAPI: crea un FredClient con sesion gestionada."""
    client = FredClient()
    try:
        yield client
    finally:
        client.close()


# =============================================================================
# 2. YieldCurve: ajuste Nelson-Siegel y consultas spot
# =============================================================================


@dataclass
class NelsonSiegelParams:
    """Parametros estimados del modelo Nelson-Siegel."""
    beta0: float
    beta1: float
    beta2: float
    lambda_: float
    rmse: float
    success: bool

    def to_dict(self) -> dict:
        return {
            "beta0":   self.beta0,
            "beta1":   self.beta1,
            "beta2":   self.beta2,
            "lambda":  self.lambda_,
            "rmse":    self.rmse,
            "success": self.success,
        }


class YieldCurve:
    """
    Curva de rendimiento. Permite ajustar Nelson-Siegel y consultar la
    tasa spot a un plazo arbitrario.

    Convencion: maturities en anos, yields en decimal (0.045 = 4.5%).
    """

    def __init__(self, maturities: list[float], yields: list[float]) -> None:
        if len(maturities) != len(yields):
            raise ValueError("maturities y yields deben tener el mismo largo.")
        if len(maturities) < 3:
            raise ValueError("Se requieren al menos 3 puntos para ajustar NS.")
        order = np.argsort(maturities)
        self.maturities = np.array(maturities, dtype=float)[order]
        self.yields = np.array(yields, dtype=float)[order]
        self.ns_params: Optional[NelsonSiegelParams] = None

    # ── Modelo Nelson-Siegel ─────────────────────────────────────────────────

    @staticmethod
    def _ns(tau: np.ndarray, b0: float, b1: float, b2: float, lam: float) -> np.ndarray:
        """y(tau) = b0 + b1 * f1(tau) + b2 * (f1(tau) - exp(-tau/lam))."""
        tau = np.where(tau == 0, 1e-9, np.asarray(tau, dtype=float))
        decay = np.exp(-tau / lam)
        factor = (1.0 - decay) / (tau / lam)
        return b0 + b1 * factor + b2 * (factor - decay)

    def fit_nelson_siegel(self) -> NelsonSiegelParams:
        """Ajusta los 4 parametros por minimos cuadrados no lineales."""
        from scipy.optimize import least_squares

        def residuals(params: np.ndarray) -> np.ndarray:
            return YieldCurve._ns(self.maturities, *params) - self.yields

        b0_init = float(np.mean(self.yields))
        b1_init = float(self.yields[0] - self.yields[-1])
        b2_init = 0.0
        lam_init = 1.5

        res = least_squares(
            residuals,
            x0=[b0_init, b1_init, b2_init, lam_init],
            bounds=([-np.inf, -np.inf, -np.inf, 0.01], [np.inf, np.inf, np.inf, 30.0]),
            max_nfev=2000,
        )
        rmse = float(np.sqrt(np.mean(res.fun ** 2)))
        self.ns_params = NelsonSiegelParams(
            beta0=float(res.x[0]),
            beta1=float(res.x[1]),
            beta2=float(res.x[2]),
            lambda_=float(res.x[3]),
            rmse=rmse,
            success=bool(res.success),
        )
        return self.ns_params

    def spot_rate(self, tau: float) -> float:
        """Tasa spot al plazo tau (anos), interpolada via Nelson-Siegel."""
        if self.ns_params is None:
            self.fit_nelson_siegel()
        p = self.ns_params
        return float(
            YieldCurve._ns(
                np.array([float(tau)]), p.beta0, p.beta1, p.beta2, p.lambda_
            )[0]
        )

    def shape(self) -> str:
        """Clasifica la curva por la diferencia (10Y - 3M)."""
        short_idx = int(np.argmin(np.abs(self.maturities - 0.25)))
        long_idx = int(np.argmin(np.abs(self.maturities - 10.0)))
        diff = float(self.yields[long_idx] - self.yields[short_idx])
        if diff > 0.005:
            return "normal"
        if diff < -0.005:
            return "inverted"
        return "flat"

    def to_dict(self) -> dict:
        return {
            "maturities":    self.maturities.tolist(),
            "yields":        self.yields.tolist(),
            "nelson_siegel": self.ns_params.to_dict() if self.ns_params else None,
            "shape":         self.shape(),
        }


# =============================================================================
# 3. Bond: duracion Macaulay/modificada, convexidad y sensibilidad
# =============================================================================


class Bond:
    """
    Bono sintetico con cupon fijo, vencimiento conocido y frecuencia de
    pago discreta (anual, semestral o trimestral).

    Convencion: face en moneda, coupon_rate en decimal anual (0.05 = 5%),
    maturity_years en anos, freq en pagos por anio.
    """

    def __init__(
        self,
        face: float = 1000.0,
        coupon_rate: float = 0.05,
        maturity_years: float = 10.0,
        freq: int = 2,
    ) -> None:
        if face <= 0:
            raise ValueError("face debe ser positivo.")
        if not 0.0 <= coupon_rate <= 1.0:
            raise ValueError("coupon_rate debe estar en [0, 1].")
        if maturity_years <= 0:
            raise ValueError("maturity_years debe ser positivo.")
        if freq not in (1, 2, 4):
            raise ValueError("freq debe ser 1, 2 o 4.")

        self.face = float(face)
        self.coupon_rate = float(coupon_rate)
        self.maturity_years = float(maturity_years)
        self.freq = int(freq)
        self.coupon = self.face * self.coupon_rate / self.freq
        self.n_periods = max(1, int(round(self.maturity_years * self.freq)))

    # ── Flujos ───────────────────────────────────────────────────────────────

    def cash_flows(self) -> list[tuple[float, float]]:
        """Lista [(t_anios, CF)]; el ultimo flujo incluye el principal."""
        flows: list[tuple[float, float]] = []
        for k in range(1, self.n_periods + 1):
            t = k / self.freq
            cf = self.coupon + (self.face if k == self.n_periods else 0.0)
            flows.append((t, cf))
        return flows

    # ── Precio ───────────────────────────────────────────────────────────────

    def price(self, ytm: float) -> float:
        """Precio del bono descontando flujos a YTM (anual nominal)."""
        y_per = ytm / self.freq
        total = 0.0
        for k, (_, cf) in enumerate(self.cash_flows(), start=1):
            total += cf / ((1.0 + y_per) ** k)
        return total

    # ── Sensibilidades ───────────────────────────────────────────────────────

    def macaulay_duration(self, ytm: float) -> float:
        y_per = ytm / self.freq
        p = self.price(ytm)
        if p <= 0:
            return float("nan")
        weighted = 0.0
        for k, (t, cf) in enumerate(self.cash_flows(), start=1):
            weighted += t * cf / ((1.0 + y_per) ** k)
        return weighted / p

    def modified_duration(self, ytm: float) -> float:
        return self.macaulay_duration(ytm) / (1.0 + ytm / self.freq)

    def convexity(self, ytm: float) -> float:
        """Convexidad anualizada (segunda derivada / precio)."""
        y_per = ytm / self.freq
        p = self.price(ytm)
        if p <= 0:
            return float("nan")
        total = 0.0
        for k, (_, cf) in enumerate(self.cash_flows(), start=1):
            total += k * (k + 1) * cf / ((1.0 + y_per) ** (k + 2))
        # Convertir convexidad por periodo a anual: dividir por freq^2
        return total / (p * (self.freq ** 2))

    # ── Sensibilidad ante shocks de tasa ─────────────────────────────────────

    def price_change(self, ytm: float, delta_y_bp: float) -> dict:
        """
        Compara tres aproximaciones del cambio de precio ante un shock
        de delta_y_bp puntos basicos en la YTM:
          1. Lineal (solo duracion modificada).
          2. Lineal + termino de convexidad.
          3. Reprice exacto descontando los flujos a la nueva YTM.
        """
        delta_y = float(delta_y_bp) / 10_000.0
        p0 = self.price(ytm)
        mod_dur = self.modified_duration(ytm)
        conv = self.convexity(ytm)

        linear_pct = -mod_dur * delta_y
        full_pct = -mod_dur * delta_y + 0.5 * conv * (delta_y ** 2)
        p_new = self.price(ytm + delta_y)
        reprice_pct = (p_new - p0) / p0 if p0 > 0 else float("nan")

        return {
            "delta_y_bp":        float(delta_y_bp),
            "price_base":        float(p0),
            "price_repriced":    float(p_new),
            "approx_linear_pct": float(linear_pct),
            "approx_full_pct":   float(full_pct),
            "reprice_pct":       float(reprice_pct),
        }

    def sensitivity_table(
        self, ytm: float, shocks_bp: list[float] = None
    ) -> list[dict]:
        """Tabla de sensibilidad para una grilla de shocks (default +-50/100/200)."""
        if shocks_bp is None:
            shocks_bp = [-200, -100, -50, 50, 100, 200]
        return [self.price_change(ytm, s) for s in shocks_bp]

    def summary(self, ytm: float) -> dict:
        return {
            "face":              self.face,
            "coupon_rate":       self.coupon_rate,
            "maturity_years":    self.maturity_years,
            "freq":              self.freq,
            "ytm":               float(ytm),
            "price":             float(self.price(ytm)),
            "macaulay_duration": float(self.macaulay_duration(ytm)),
            "modified_duration": float(self.modified_duration(ytm)),
            "convexity":         float(self.convexity(ytm)),
        }
