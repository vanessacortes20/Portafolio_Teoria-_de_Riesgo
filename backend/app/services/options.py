"""
Valoración de opciones europeas con Black-Scholes y las cinco Greeks.

Convenciones:
- S, K: precios en la misma moneda.
- T: tiempo a vencimiento en años (fraccional).
- r: tasa libre de riesgo anual decimal.
- sigma: volatilidad anual decimal.
- option_type: "call" o "put".
"""
from __future__ import annotations

import math
from typing import Literal

from scipy.stats import norm

OptionType = Literal["call", "put"]


class OptionPricer:
    """Black-Scholes + Greeks para opciones europeas."""

    def __init__(self, S: float, K: float, T: float, r: float, sigma: float):
        if S <= 0:
            raise ValueError("S (spot) debe ser > 0")
        if K <= 0:
            raise ValueError("K (strike) debe ser > 0")
        if T <= 0:
            raise ValueError("T (tiempo a vencimiento) debe ser > 0")
        if sigma <= 0:
            raise ValueError("sigma (volatilidad) debe ser > 0")

        self.S = float(S)
        self.K = float(K)
        self.T = float(T)
        self.r = float(r)
        self.sigma = float(sigma)

    # ── d1 / d2 ──────────────────────────────────────────────────────────────

    @property
    def d1(self) -> float:
        return (math.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * math.sqrt(self.T))

    @property
    def d2(self) -> float:
        return self.d1 - self.sigma * math.sqrt(self.T)

    # ── Precio ───────────────────────────────────────────────────────────────

    def black_scholes(self, option_type: OptionType) -> float:
        d1, d2 = self.d1, self.d2
        if option_type == "call":
            return self.S * norm.cdf(d1) - self.K * math.exp(-self.r * self.T) * norm.cdf(d2)
        if option_type == "put":
            return self.K * math.exp(-self.r * self.T) * norm.cdf(-d2) - self.S * norm.cdf(-d1)
        raise ValueError(f"option_type inválido: {option_type}")

    # ── Greeks ───────────────────────────────────────────────────────────────

    def greeks(self, option_type: OptionType) -> dict:
        d1, d2 = self.d1, self.d2
        sqrtT = math.sqrt(self.T)
        n_d1  = norm.pdf(d1)

        gamma = n_d1 / (self.S * self.sigma * sqrtT)
        vega  = self.S * sqrtT * n_d1   # por unidad de σ (no por 1%)

        if option_type == "call":
            delta = norm.cdf(d1)
            theta = (-self.S * n_d1 * self.sigma / (2 * sqrtT)
                     - self.r * self.K * math.exp(-self.r * self.T) * norm.cdf(d2))
            rho   =  self.K * self.T * math.exp(-self.r * self.T) * norm.cdf(d2)
        elif option_type == "put":
            delta = norm.cdf(d1) - 1.0
            theta = (-self.S * n_d1 * self.sigma / (2 * sqrtT)
                     + self.r * self.K * math.exp(-self.r * self.T) * norm.cdf(-d2))
            rho   = -self.K * self.T * math.exp(-self.r * self.T) * norm.cdf(-d2)
        else:
            raise ValueError(f"option_type inválido: {option_type}")

        return {
            "delta": round(delta, 6),
            "gamma": round(gamma, 6),
            "vega":  round(vega,  6),
            "theta": round(theta, 6),
            "rho":   round(rho,   6),
        }

    # ── Paridad put-call ─────────────────────────────────────────────────────

    def put_call_parity_check(self) -> dict:
        """Verifica numéricamente que C - P = S - K·e^(-rT)."""
        c = self.black_scholes("call")
        p = self.black_scholes("put")
        lhs = c - p
        rhs = self.S - self.K * math.exp(-self.r * self.T)
        diff = abs(lhs - rhs)
        return {
            "call":     round(c, 6),
            "put":      round(p, 6),
            "lhs":      round(lhs, 6),
            "rhs":      round(rhs, 6),
            "abs_diff": round(diff, 9),
            "satisfied": bool(diff < 1e-6),
        }

    # ── Volatilidad implícita ────────────────────────────────────────────────

    def implied_volatility(
        self,
        market_price: float,
        option_type: OptionType,
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> float | None:
        """Newton-Raphson para σ implícita. Retorna None si no converge."""
        sigma = max(self.sigma, 0.2)
        for _ in range(max_iter):
            self.sigma = sigma
            price = self.black_scholes(option_type)
            v = self.greeks(option_type)["vega"]
            if v < 1e-10:
                return None
            diff = price - market_price
            if abs(diff) < tol:
                return round(sigma, 6)
            sigma = sigma - diff / v
            if sigma <= 0 or sigma > 5:
                return None
        return None

    # ── Empaquetado ─────────────────────────────────────────────────────────

    def summary(self, option_type: OptionType) -> dict:
        if option_type not in ("call", "put"):
            raise ValueError("option_type debe ser 'call' o 'put'")
        price = self.black_scholes(option_type)
        gks   = self.greeks(option_type)
        parity = self.put_call_parity_check()

        moneyness = self.S / self.K
        if abs(moneyness - 1) < 0.02:
            mny_label = "ATM (cerca del strike)"
        elif (option_type == "call" and moneyness > 1) or (option_type == "put" and moneyness < 1):
            mny_label = "ITM (in-the-money)"
        else:
            mny_label = "OTM (out-of-the-money)"

        interp = (
            f"Opción {option_type.upper()} {mny_label}: precio teórico "
            f"{price:.4f}. Delta {gks['delta']:.3f} indica sensibilidad al spot; "
            f"Vega {gks['vega']:.3f} mide la exposición a la volatilidad."
        )

        return {
            "inputs": {
                "S": self.S, "K": self.K, "T": self.T, "r": self.r, "sigma": self.sigma,
            },
            "option_type":            option_type,
            "price":                  round(price, 6),
            "greeks":                 gks,
            "put_call_parity_check":  parity,
            "moneyness":              round(moneyness, 6),
            "interpretation":         interp,
        }
