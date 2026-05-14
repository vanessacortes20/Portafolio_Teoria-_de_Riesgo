"""
Stress testing de portafolio bajo escenarios extremos.

Toma un portafolio (lista de tickers + pesos) y aplica shocks predefinidos
o personalizados:

- shock de mercado: caída/alza del benchmark, propagada por Beta de cada activo
- shock de tasa: ΔRf en puntos básicos, afecta CAPM
- shock de volatilidad: σ → σ × multiplicador, afecta VaR/Greeks
- combinado: superpone los tres simultáneamente

El cálculo es una aproximación de primer/segundo orden — útil para análisis
forward-looking y reportes de Basilea III, no para reprice exacto.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


class StressTester:
    """Aplica escenarios de stress sobre un portafolio."""

    def __init__(
        self,
        weights: dict[str, float],
        prices: dict[str, float],
        betas: Optional[dict[str, float]] = None,
        sigmas: Optional[dict[str, float]] = None,
    ):
        if not weights:
            raise ValueError("weights no puede estar vacío")
        total = sum(weights.values())
        if abs(total - 1.0) > 1e-3:
            raise ValueError(f"Los pesos deben sumar 1; suman {total:.4f}")
        for w in weights.values():
            if not isinstance(w, (int, float)):
                raise ValueError("Todos los pesos deben ser numéricos")
        if not prices:
            raise ValueError("prices no puede estar vacío")
        for t in weights:
            if t not in prices:
                raise ValueError(f"Falta el precio del ticker '{t}'")

        self.weights = {t: float(w) for t, w in weights.items()}
        self.prices  = {t: float(p) for t, p in prices.items()}
        # Por default, cada activo tiene Beta=1 (neutral) y σ=20% si no se aporta
        self.betas  = {t: float(betas.get(t, 1.0)) if betas else 1.0 for t in weights}
        self.sigmas = {t: float(sigmas.get(t, 0.20)) if sigmas else 0.20 for t in weights}

    @property
    def base_value(self) -> float:
        """Valor base del portafolio (suma de precio · peso)."""
        return float(sum(self.prices[t] * self.weights[t] for t in self.weights))

    # ── Escenarios ──────────────────────────────────────────────────────────

    def apply(self, scenario: dict) -> dict:
        """Aplica un escenario y retorna el impacto.

        scenario: {
            "name": str,
            "market_drop_pct": float | None,    # ej: -0.20 = caída 20% del mercado
            "rate_shock_bp":   int | None,      # ej: 200 = +200bp en Rf
            "vol_multiplier":  float | None,    # ej: 2.0 = σ se duplica
        }
        """
        name = scenario.get("name", "scenario")
        market_drop = scenario.get("market_drop_pct")
        rate_shock  = scenario.get("rate_shock_bp")
        vol_mult    = scenario.get("vol_multiplier")

        # Shocks por activo
        impact_by_asset: list[dict] = []
        new_value = 0.0
        for t, w in self.weights.items():
            p0 = self.prices[t]
            beta  = self.betas[t]
            sigma = self.sigmas[t]
            delta_pct = 0.0

            # Shock de mercado vía Beta
            if market_drop is not None:
                delta_pct += float(beta) * float(market_drop)

            # Shock de tasa: aproximación lineal — equivale a -beta · Δrf
            # (sensibilidad estilo CAPM, no de bono)
            if rate_shock is not None:
                dy = float(rate_shock) / 10000.0
                delta_pct += -float(beta) * dy

            # Shock de volatilidad: aprox por una desviación negativa de 1σ adicional
            if vol_mult is not None and vol_mult > 1:
                extra_sigma = sigma * (vol_mult - 1.0)
                # Usamos un factor de impacto de ~50% del extra σ (proxy conservador)
                delta_pct += -0.5 * extra_sigma

            new_price = p0 * (1.0 + delta_pct)
            asset_value = new_price * w
            new_value += asset_value
            impact_by_asset.append({
                "ticker":           t,
                "weight":           w,
                "base_price":       round(p0, 6),
                "stressed_price":   round(new_price, 6),
                "delta_pct":        round(delta_pct, 6),
                "asset_value":      round(asset_value, 6),
            })

        loss = new_value - self.base_value
        loss_pct = loss / self.base_value if self.base_value > 0 else 0.0

        return {
            "scenario_name":       name,
            "shocks":              {
                "market_drop_pct": market_drop,
                "rate_shock_bp":   rate_shock,
                "vol_multiplier":  vol_mult,
            },
            "stressed_value":      round(new_value, 6),
            "estimated_loss":      round(loss, 6),
            "percentage_loss":     round(loss_pct, 6),
            "impact_by_asset":     impact_by_asset,
        }

    # ── Batería completa ────────────────────────────────────────────────────

    def run_all(self, scenarios: Optional[list[dict]] = None) -> dict:
        """Corre la batería de escenarios. Si no se pasan, usa los obligatorios."""
        if not scenarios:
            scenarios = [
                {"name": "Caída de mercado -20%",   "market_drop_pct": -0.20},
                {"name": "Caída de mercado -30%",   "market_drop_pct": -0.30},
                {"name": "Subida de tasa +200bp",   "rate_shock_bp":   200},
                {"name": "Bajada de tasa -200bp",   "rate_shock_bp":  -200},
                {"name": "Shock de volatilidad x2", "vol_multiplier":  2.0},
                {"name": "Tormenta perfecta",
                 "market_drop_pct": -0.20, "rate_shock_bp": 200, "vol_multiplier": 2.0},
            ]

        results = [self.apply(s) for s in scenarios]
        worst = min(results, key=lambda r: r["percentage_loss"]) if results else None

        # Resumen interpretativo
        risk_msg: str
        if worst:
            wp = worst["percentage_loss"]
            if wp <= -0.30:
                risk_msg = (f"El escenario peor ('{worst['scenario_name']}') implica una pérdida del "
                            f"{abs(wp)*100:.1f}% — exposición severa, considerar coberturas.")
            elif wp <= -0.15:
                risk_msg = (f"El escenario peor ('{worst['scenario_name']}') implica una pérdida del "
                            f"{abs(wp)*100:.1f}% — riesgo significativo bajo estrés.")
            else:
                risk_msg = (f"El portafolio resiste razonablemente: peor caso "
                            f"{abs(wp)*100:.1f}% en '{worst['scenario_name']}'.")
        else:
            risk_msg = "Sin escenarios evaluados."

        return {
            "base_portfolio_value":  round(self.base_value, 6),
            "scenarios_run":         len(results),
            "stressed_values":       [{"name": r["scenario_name"], "value": r["stressed_value"]} for r in results],
            "scenario_summary":      results,
            "worst_scenario":        worst["scenario_name"] if worst else None,
            "worst_loss_pct":        worst["percentage_loss"] if worst else None,
            "risk_interpretation":   risk_msg,
        }
