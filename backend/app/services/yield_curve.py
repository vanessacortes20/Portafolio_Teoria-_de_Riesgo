"""
Curva de rendimiento del Tesoro EE.UU. + ajuste Nelson-Siegel.

Las series base se obtienen de FRED a través de api.services.fred_service.
Si FRED_API_KEY no está configurada, se devuelve una curva DEMO claramente
marcada como `source: fallback_demo` para que el frontend pueda renderizar
algo sin engañar al usuario sobre la procedencia del dato.
"""
from __future__ import annotations

from datetime import datetime
from typing import Iterable, Optional

import numpy as np
from scipy.optimize import least_squares
from sqlalchemy.orm import Session

from backend.app.services import fred_service as fred

# Mapping FRED series → maturity en años
SERIES_BY_MATURITY = [
    ("DGS3MO", 0.25),
    ("DGS1",   1.0),
    ("DGS2",   2.0),
    ("DGS5",   5.0),
    ("DGS10", 10.0),
    ("DGS30", 30.0),
]

# Curva DEMO (valores ilustrativos típicos — NO son datos reales actuales)
_DEMO_CURVE = {
    0.25: 5.30,
    1.0:  4.85,
    2.0:  4.52,
    5.0:  4.30,
    10.0: 4.40,
    30.0: 4.62,
}


# ── Núcleo Nelson-Siegel ─────────────────────────────────────────────────────

def nelson_siegel(tau: np.ndarray, beta0: float, beta1: float,
                  beta2: float, lam: float) -> np.ndarray:
    """
    y(τ) = β₀ + β₁ · ((1 - e^(-τ/λ)) / (τ/λ))
              + β₂ · ((1 - e^(-τ/λ)) / (τ/λ) - e^(-τ/λ))

    β₀  = nivel de largo plazo
    β₁  = pendiente (corto - largo)
    β₂  = curvatura
    λ   = velocidad de decay
    """
    tau = np.asarray(tau, dtype=float)
    x = tau / lam
    decay = np.exp(-x)
    factor = (1.0 - decay) / np.where(x == 0, 1e-12, x)
    return beta0 + beta1 * factor + beta2 * (factor - decay)


def _interpret_shape(beta0: float, beta1: float, beta2: float) -> str:
    """Diagnóstico cualitativo de la forma de la curva ajustada."""
    long_term = beta0
    short_term = beta0 + beta1
    spread = long_term - short_term
    curvature_msg = ""
    if abs(beta2) > 1.0:
        curvature_msg = " La curvatura intermedia es marcada (β₂ alto en magnitud)."
    if spread > 0.4:
        return f"Curva con pendiente positiva (largo {long_term:.2f}% > corto {short_term:.2f}%) — expectativa de crecimiento.{curvature_msg}"
    if spread < -0.2:
        return f"Curva invertida (corto {short_term:.2f}% > largo {long_term:.2f}%) — señal clásica de recesión esperada.{curvature_msg}"
    return f"Curva relativamente plana (spread {spread:.2f}%) — incertidumbre macro o fin de ciclo.{curvature_msg}"


class YieldCurve:
    """Encapsula descarga de curva FRED + ajuste Nelson-Siegel."""

    def __init__(self, db: Session):
        self.db = db
        self.points: list[dict] = []
        self.source: str = "unknown"
        self.cache_status: dict[str, str] = {}
        self.as_of: Optional[str] = None

    # ── Descarga de puntos ──────────────────────────────────────────────────

    def fetch_curve(self) -> list[dict]:
        """Descarga los puntos de la curva. Retorna lista de {maturity, yield, source}."""
        if not fred.is_available():
            self.source = "fallback_demo"
            self.as_of = datetime.utcnow().date().isoformat()
            self.points = [
                {"maturity": m, "yield_pct": y, "series_id": None}
                for m, y in _DEMO_CURVE.items()
            ]
            return self.points

        pts: list[dict] = []
        latest_date: Optional[str] = None
        for series_id, maturity in SERIES_BY_MATURITY:
            data = fred.get_series(self.db, series_id)
            if data is None or data.get("value") is None:
                continue
            pts.append({
                "maturity":  maturity,
                "yield_pct": float(data["value"]),  # FRED ya da en %
                "series_id": series_id,
                "date":      data.get("date"),
            })
            self.cache_status[series_id] = data.get("cache_status", "miss")
            if data.get("date") and (latest_date is None or data["date"] > latest_date):
                latest_date = data["date"]

        if pts:
            self.source = "FRED"
            self.as_of = latest_date or datetime.utcnow().date().isoformat()
        else:
            self.source = "fallback_demo"
            self.as_of = datetime.utcnow().date().isoformat()
            pts = [{"maturity": m, "yield_pct": y, "series_id": None}
                   for m, y in _DEMO_CURVE.items()]

        self.points = pts
        return pts

    # ── Ajuste Nelson-Siegel ────────────────────────────────────────────────

    def fit_nelson_siegel(
        self,
        maturities: Optional[Iterable[float]] = None,
        yields: Optional[Iterable[float]] = None,
    ) -> dict:
        """Ajusta NS por mínimos cuadrados no lineales y devuelve parámetros + RMSE.

        Si no se pasan maturities/yields, usa los puntos cargados con fetch_curve.
        """
        if maturities is None or yields is None:
            if not self.points:
                self.fetch_curve()
            mats   = np.array([p["maturity"]  for p in self.points], dtype=float)
            ys     = np.array([p["yield_pct"] for p in self.points], dtype=float)
        else:
            mats = np.array(list(maturities), dtype=float)
            ys   = np.array(list(yields),     dtype=float)

        if len(mats) < 4:
            raise ValueError(f"Se requieren al menos 4 puntos para Nelson-Siegel (recibidos: {len(mats)}).")
        if (mats <= 0).any():
            raise ValueError("Las madureces deben ser estrictamente positivas.")
        if not np.isfinite(ys).all():
            raise ValueError("Los rendimientos deben ser numéricos finitos.")

        # Estimación inicial razonable: nivel = promedio, pendiente = corto-promedio
        b0_init = float(np.mean(ys))
        b1_init = float(ys[mats.argmin()] - b0_init)
        b2_init = 0.0
        lam_init = 2.0

        def residuals(params):
            b0, b1, b2, lam = params
            return nelson_siegel(mats, b0, b1, b2, lam) - ys

        try:
            res = least_squares(
                residuals,
                x0=[b0_init, b1_init, b2_init, lam_init],
                bounds=([-30, -30, -30, 0.05], [30, 30, 30, 30]),
                max_nfev=2000,
            )
            b0, b1, b2, lam = [float(x) for x in res.x]
            fitted = nelson_siegel(mats, b0, b1, b2, lam)
            rmse = float(np.sqrt(np.mean((fitted - ys) ** 2)))
            converged = bool(res.success)
        except Exception as exc:  # pragma: no cover
            return {
                "fitted": False,
                "error":  str(exc),
            }

        return {
            "fitted":          True,
            "converged":       converged,
            "beta0":           round(b0, 6),
            "beta1":           round(b1, 6),
            "beta2":           round(b2, 6),
            "tau":             round(lam, 6),
            "rmse":            round(rmse, 6),
            "fitted_yields":   [round(float(v), 6) for v in fitted.tolist()],
            "maturities":      mats.tolist(),
            "interpretation":  _interpret_shape(b0, b1, b2),
        }

    # ── Empaquetado para endpoint ───────────────────────────────────────────

    def to_response(self) -> dict:
        """Empaqueta puntos + ajuste NS + metadata para JSON de endpoint."""
        if not self.points:
            self.fetch_curve()
        try:
            ns = self.fit_nelson_siegel()
        except Exception as exc:
            ns = {"fitted": False, "error": str(exc)}

        return {
            "as_of":           self.as_of,
            "source":          self.source,
            "cache_status":    self.cache_status,
            "points":          self.points,
            "nelson_siegel":   ns,
        }
