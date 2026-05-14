"""
Bono sintético: precio, duración Macaulay, duración modificada, convexidad
y sensibilidad ante shocks de tasa.

Convención:
- yield_rate y coupon_rate son tasas anuales nominales en decimal (ej: 0.05 = 5%).
- frequency = pagos por año (1, 2, 4, 12).
- maturity_years puede ser fraccional.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np


class Bond:
    """Bono cupón fijo con cálculos analíticos."""

    def __init__(
        self,
        face_value: float,
        coupon_rate: float,
        maturity_years: float,
        yield_rate: float,
        frequency: int = 2,
    ):
        if face_value <= 0:
            raise ValueError("face_value debe ser > 0")
        if coupon_rate < 0:
            raise ValueError("coupon_rate debe ser >= 0")
        if maturity_years <= 0:
            raise ValueError("maturity_years debe ser > 0")
        if yield_rate < 0:
            raise ValueError("yield_rate debe ser >= 0")
        if frequency not in (1, 2, 4, 12):
            raise ValueError("frequency debe estar en {1, 2, 4, 12}")

        self.face_value     = float(face_value)
        self.coupon_rate    = float(coupon_rate)
        self.maturity_years = float(maturity_years)
        self.yield_rate     = float(yield_rate)
        self.frequency      = int(frequency)

    # ── Estructura de flujos ────────────────────────────────────────────────

    def _cash_flows(self) -> tuple[np.ndarray, np.ndarray]:
        """Devuelve (tiempos en años, flujos)."""
        n = max(int(round(self.maturity_years * self.frequency)), 1)
        times  = np.array([(i + 1) / self.frequency for i in range(n)])
        coupon = self.face_value * self.coupon_rate / self.frequency
        flows  = np.full(n, coupon)
        flows[-1] += self.face_value
        return times, flows

    # ── Métricas ────────────────────────────────────────────────────────────

    def price(self, yield_rate: float | None = None) -> float:
        y = self.yield_rate if yield_rate is None else float(yield_rate)
        times, flows = self._cash_flows()
        discount = (1.0 + y / self.frequency) ** (times * self.frequency)
        return float(np.sum(flows / discount))

    def macaulay_duration(self) -> float:
        times, flows = self._cash_flows()
        discount = (1.0 + self.yield_rate / self.frequency) ** (times * self.frequency)
        pv = flows / discount
        p  = pv.sum()
        if p <= 0:
            return 0.0
        return float(np.sum(times * pv) / p)

    def modified_duration(self) -> float:
        d_mac = self.macaulay_duration()
        return float(d_mac / (1.0 + self.yield_rate / self.frequency))

    def convexity(self) -> float:
        times, flows = self._cash_flows()
        m = self.frequency
        discount = (1.0 + self.yield_rate / m) ** (times * m)
        pv = flows / discount
        p  = pv.sum()
        if p <= 0:
            return 0.0
        # C = (1 / P) · Σ (t · (t + 1/m) · CF_t / (1 + y/m)^(t·m + 2))
        weight = times * (times + 1.0 / m)
        adj    = (1.0 + self.yield_rate / m) ** 2
        return float(np.sum(weight * pv) / (p * adj))

    # ── Sensibilidad ────────────────────────────────────────────────────────

    def shock_sensitivity(
        self,
        shocks_bp: Iterable[int] = (-200, -100, -50, 50, 100, 200),
    ) -> list[dict]:
        """Para cada shock en puntos básicos compara tres aproximaciones:

        - lineal con duración: ΔP/P ≈ −D* · Δy
        - duración + convexidad: ΔP/P ≈ −D* · Δy + ½ · C · (Δy)²
        - reprice exacto descontando flujos a la nueva yield

        Retorna lista con los tres porcentajes y el precio resultante.
        """
        p0   = self.price()
        d_mod = self.modified_duration()
        c     = self.convexity()
        out: list[dict] = []
        for bp in shocks_bp:
            dy = bp / 10000.0
            duration_pct        = -d_mod * dy
            duration_convex_pct = -d_mod * dy + 0.5 * c * dy ** 2
            new_price = self.price(yield_rate=self.yield_rate + dy)
            exact_pct = (new_price / p0 - 1.0) if p0 > 0 else 0.0
            out.append({
                "shock_bp":             int(bp),
                "delta_y":              round(dy, 6),
                "duration_only_pct":    round(duration_pct, 6),
                "duration_convex_pct":  round(duration_convex_pct, 6),
                "exact_pct":            round(exact_pct, 6),
                "new_price":            round(new_price, 4),
            })
        return out

    # ── Empaquetado ─────────────────────────────────────────────────────────

    def summary(self) -> dict:
        p     = self.price()
        d_mac = self.macaulay_duration()
        d_mod = self.modified_duration()
        cnv   = self.convexity()
        shocks = self.shock_sensitivity()

        # Interpretación cualitativa
        if d_mac >= 10:
            interp = (f"Bono de larga duración ({d_mac:.2f} años) — alta sensibilidad "
                      f"a cambios de tasa. La convexidad {cnv:.2f} suaviza grandes movimientos.")
        elif d_mac >= 5:
            interp = (f"Duración intermedia ({d_mac:.2f} años) — sensibilidad moderada. "
                      f"Un shock de +100bp implica ~{abs(d_mod):.2f}% de pérdida aproximada.")
        else:
            interp = (f"Bono corto ({d_mac:.2f} años) — baja sensibilidad a cambios de tasa.")

        return {
            "inputs": {
                "face_value":     self.face_value,
                "coupon_rate":    self.coupon_rate,
                "maturity_years": self.maturity_years,
                "yield_rate":     self.yield_rate,
                "frequency":      self.frequency,
            },
            "bond_price":         round(p, 6),
            "macaulay_duration":  round(d_mac, 6),
            "modified_duration":  round(d_mod, 6),
            "convexity":          round(cnv, 6),
            "shock_sensitivity":  shocks,
            "interpretation":     interp,
        }
