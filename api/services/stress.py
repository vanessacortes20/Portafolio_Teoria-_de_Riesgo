"""
Modulo de stress testing (M11 del Proyecto Integrador).

A diferencia del backtesting (Kupiec, M5) que valida el VaR contra historia
observada, el stress testing aplica escenarios hipoteticos extremos para
estimar la perdida potencial bajo condiciones adversas. Es obligatorio en
el marco regulatorio de Basilea III.

Escenarios soportados:
  - Shock de tasa libre de riesgo en puntos basicos.
  - Caida porcentual del benchmark de mercado.
  - Multiplicacion de la volatilidad (sigma -> sigma * mult).
  - Combinacion arbitraria de los tres (tormenta perfecta).

Cada activo del portafolio absorbe el shock de mercado segun su Beta
respecto al benchmark:  dR_i = Beta_i * shock_mkt. La perdida del
portafolio agregada es la suma ponderada por los pesos.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)


# =============================================================================
# StressTester
# =============================================================================


class StressTester:
    """
    Aplica escenarios de stress sobre un portafolio dado.

    Argumentos:
      returns_df          : DataFrame de retornos diarios (columnas = tickers).
      weights             : dict ticker -> peso. Se renormaliza si no suma 1.
      benchmark_returns   : Serie de retornos del benchmark (para Betas).
      rf_rate             : tasa libre de riesgo anual base.
      portfolio_value     : valor monetario base del portafolio.
      equity_rate_duration: sensibilidad implicita de equities a la tasa,
                            usada para el componente directo del shock de
                            tasa. Default 0 (no impacto directo). Para una
                            cartera de equities el rango tipico es 4-7.
    """

    def __init__(
        self,
        returns_df: pd.DataFrame,
        weights: dict[str, float],
        benchmark_returns: pd.Series,
        rf_rate: float = 0.04,
        portfolio_value: float = 100_000.0,
        equity_rate_duration: float = 0.0,
    ) -> None:
        common = [t for t in weights if t in returns_df.columns]
        if len(common) < 2:
            raise ValueError(
                "Se necesitan al menos 2 activos comunes entre weights y returns_df."
            )
        self.returns_df = returns_df[common].dropna().copy()
        # Normalizar pesos
        w_raw = {t: float(weights[t]) for t in common}
        s = sum(w_raw.values())
        if s <= 0:
            raise ValueError("La suma de pesos debe ser positiva.")
        self.weights = {t: w_raw[t] / s for t in common}
        self.benchmark = benchmark_returns
        self.rf_rate = float(rf_rate)
        self.portfolio_value = float(portfolio_value)
        self.equity_rate_duration = float(equity_rate_duration)

        # Caches
        self._betas: Optional[dict[str, float]] = None

    # ── Caracterizacion del portafolio base ──────────────────────────────────

    def betas(self) -> dict[str, float]:
        """Beta de cada activo respecto al benchmark via regresion lineal."""
        if self._betas is not None:
            return self._betas
        out: dict[str, float] = {}
        for t in self.returns_df.columns:
            r = self.returns_df[t].dropna()
            common_idx = r.index.intersection(self.benchmark.index)
            if len(common_idx) >= 10:
                slope, _, _, _, _ = scipy_stats.linregress(
                    self.benchmark.loc[common_idx], r.loc[common_idx]
                )
                out[t] = float(slope)
            else:
                out[t] = 1.0
        self._betas = out
        return out

    def portfolio_daily_returns(self) -> np.ndarray:
        """Retornos diarios del portafolio (combinacion lineal con weights)."""
        w_vec = np.array(
            [self.weights[t] for t in self.returns_df.columns], dtype=float
        )
        return self.returns_df.values @ w_vec

    def base_metrics(self, confidence: float = 0.95) -> dict:
        """Metricas del portafolio antes de aplicar cualquier shock."""
        port_ret = self.portfolio_daily_returns()
        mu = float(np.mean(port_ret))
        sigma = float(np.std(port_ret))
        z_a = float(scipy_stats.norm.ppf(1.0 - confidence))
        z_99 = float(scipy_stats.norm.ppf(0.01))
        var_a = abs(mu + sigma * z_a)
        var_99 = abs(mu + sigma * z_99)
        return {
            "portfolio_value":     self.portfolio_value,
            "mu_daily":            mu,
            "sigma_daily":         sigma,
            "var_alpha_daily":     var_a,
            "var_99_daily":        var_99,
            "var_alpha_amount":    var_a * self.portfolio_value,
            "var_99_amount":       var_99 * self.portfolio_value,
            "confidence":          confidence,
            "rf_rate":             self.rf_rate,
        }

    # ── Aplicacion de un escenario ───────────────────────────────────────────

    def apply_scenario(self, scenario: dict, confidence: float = 0.95) -> dict:
        """
        scenario = {
            "name":             str,
            "rate_shock_bp":    float,   (+200 = subida de 200 pb)
            "market_drop_pct":  float,   (-0.20 = caida 20% del benchmark)
            "vol_multiplier":   float,   (1.0 = sin cambio; 2.0 = sigma *2)
        }
        """
        name = str(scenario.get("name", "scenario"))
        rate_bp = float(scenario.get("rate_shock_bp", 0.0))
        market_drop = float(scenario.get("market_drop_pct", 0.0))
        vol_mult = float(scenario.get("vol_multiplier", 1.0))

        betas = self.betas()
        port_ret = self.portfolio_daily_returns()
        mu_base = float(np.mean(port_ret))
        sigma_base = float(np.std(port_ret))

        # ── Cambio por shock de mercado: dR_i = Beta_i * market_drop ─────────
        asset_market_pct = {t: betas[t] * market_drop for t in self.weights}

        # ── Cambio por shock de tasa: dR_i = -equity_duration * dRf ──────────
        delta_rf = rate_bp / 10_000.0
        asset_rate_pct = {
            t: -self.equity_rate_duration * delta_rf for t in self.weights
        }

        # ── Cambio total por activo ──────────────────────────────────────────
        asset_total_pct = {
            t: asset_market_pct[t] + asset_rate_pct[t] for t in self.weights
        }
        portfolio_delta_return = sum(
            self.weights[t] * asset_total_pct[t] for t in self.weights
        )

        # ── Volatilidad post-shock ───────────────────────────────────────────
        sigma_stressed = sigma_base * vol_mult

        # ── VaR base y estresado (parametrico) ───────────────────────────────
        z_a = float(scipy_stats.norm.ppf(1.0 - confidence))
        z_99 = float(scipy_stats.norm.ppf(0.01))
        var_base_alpha = abs(mu_base + sigma_base * z_a)
        var_base_99 = abs(mu_base + sigma_base * z_99)
        var_stress_alpha = abs(mu_base + sigma_stressed * z_a)
        var_stress_99 = abs(mu_base + sigma_stressed * z_99)

        # ── Perdida puntual del portafolio ───────────────────────────────────
        loss_pct = -portfolio_delta_return
        loss_amount = loss_pct * self.portfolio_value

        # ── Rf estresada (para Sharpe/CAPM) ──────────────────────────────────
        rf_stressed = self.rf_rate + delta_rf

        return {
            "name": name,
            "shocks": {
                "rate_shock_bp":   rate_bp,
                "market_drop_pct": market_drop,
                "vol_multiplier":  vol_mult,
            },
            "rf_base":             self.rf_rate,
            "rf_stressed":         float(rf_stressed),
            "portfolio_loss_pct":  float(loss_pct),
            "portfolio_loss_amount": float(loss_amount),
            "portfolio_value_after": float(self.portfolio_value * (1.0 + portfolio_delta_return)),
            "sigma_base_daily":    sigma_base,
            "sigma_stressed_daily": float(sigma_stressed),
            "var_base_alpha":      float(var_base_alpha),
            "var_stressed_alpha":  float(var_stress_alpha),
            "var_base_99":         float(var_base_99),
            "var_stressed_99":     float(var_stress_99),
            "var_stressed_alpha_amount": float(var_stress_alpha * self.portfolio_value),
            "var_stressed_99_amount":    float(var_stress_99 * self.portfolio_value),
            "asset_impact_pct":    {
                t: {
                    "market_pct": float(asset_market_pct[t]),
                    "rate_pct":   float(asset_rate_pct[t]),
                    "total_pct":  float(asset_total_pct[t]),
                    "beta":       float(betas[t]),
                    "weight":     float(self.weights[t]),
                }
                for t in self.weights
            },
        }

    # ── Escenarios por defecto (los 4 obligatorios de la spec) ───────────────

    @staticmethod
    def default_scenarios() -> list[dict]:
        return [
            {"name": "Tasa +200 pb",      "rate_shock_bp":  200, "market_drop_pct":  0.00, "vol_multiplier": 1.0},
            {"name": "Tasa -200 pb",      "rate_shock_bp": -200, "market_drop_pct":  0.00, "vol_multiplier": 1.0},
            {"name": "Mercado -20%",      "rate_shock_bp":    0, "market_drop_pct": -0.20, "vol_multiplier": 1.0},
            {"name": "Mercado -30%",      "rate_shock_bp":    0, "market_drop_pct": -0.30, "vol_multiplier": 1.0},
            {"name": "Volatilidad x2",    "rate_shock_bp":    0, "market_drop_pct":  0.00, "vol_multiplier": 2.0},
            {"name": "Tormenta perfecta", "rate_shock_bp":  200, "market_drop_pct": -0.20, "vol_multiplier": 2.0},
        ]

    # ── Orquestacion completa ────────────────────────────────────────────────

    def run(
        self,
        scenarios: Optional[list[dict]] = None,
        confidence: float = 0.95,
    ) -> dict:
        """
        Ejecuta los escenarios indicados (o los default) y devuelve el
        diccionario completo listo para serializar a JSON.
        """
        if scenarios is None:
            scenarios = self.default_scenarios()

        base = self.base_metrics(confidence)
        results = [self.apply_scenario(sc, confidence) for sc in scenarios]

        # Heatmap activo x escenario (estructura amigable para el dashboard)
        heatmap = {
            t: {res["name"]: res["asset_impact_pct"][t]["total_pct"] for res in results}
            for t in self.weights
        }

        # Resumen tipo bar: nombre vs perdida
        loss_bar = [
            {
                "name":   r["name"],
                "loss_pct":     r["portfolio_loss_pct"],
                "loss_amount":  r["portfolio_loss_amount"],
            }
            for r in results
        ]

        return {
            "base":      base,
            "betas":     self.betas(),
            "weights":   self.weights,
            "scenarios": results,
            "heatmap":   heatmap,
            "loss_bar":  loss_bar,
        }
