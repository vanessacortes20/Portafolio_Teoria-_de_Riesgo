"""
Modulo de opciones europeas (M10 del Proyecto Integrador).

Implementa el modelo de Black-Scholes:
  d1 = [ln(S/K) + (r + sigma^2/2) * T] / (sigma * sqrt(T))
  d2 = d1 - sigma * sqrt(T)
  Call = S * N(d1) - K * exp(-rT) * N(d2)
  Put  = K * exp(-rT) * N(-d2) - S * N(-d1)

Calcula las cinco Greeks (Delta, Gamma, Vega, Theta, Rho), verifica la
paridad put-call y resuelve la volatilidad implicita por Newton-Raphson
usando vega como derivada de BS respecto a sigma.

Las funciones de calculo son static helpers; la clase OptionPricer es un
wrapper OO comodo para inyectar en endpoints.
"""
from __future__ import annotations

import math
from typing import Optional

from scipy.stats import norm


# =============================================================================
# 1. Funciones puras (static helpers) — usables sin instanciar la clase
# =============================================================================


def _d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    return (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))


def _d2_from_d1(d1: float, sigma: float, T: float) -> float:
    return d1 - sigma * math.sqrt(T)


def bs_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str,
) -> float:
    """Precio Black-Scholes de una opcion europea (call o put)."""
    if option_type not in ("call", "put"):
        raise ValueError("option_type debe ser 'call' o 'put'.")
    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2_from_d1(d1, sigma, T)
    disc = math.exp(-r * T)
    if option_type == "call":
        return float(S * norm.cdf(d1) - K * disc * norm.cdf(d2))
    return float(K * disc * norm.cdf(-d2) - S * norm.cdf(-d1))


def bs_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Vega = dV / d sigma (igual para call y put)."""
    d1 = _d1(S, K, T, r, sigma)
    return float(S * math.sqrt(T) * norm.pdf(d1))


def bs_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str,
) -> dict:
    """
    Las cinco Greeks de una opcion europea.

    Convencion: Greeks "raw" por unidad de cambio (no escaladas a 1% o
    a un dia). El frontend puede multiplicar/dividir si quiere mostrar
    Vega por 1% (vega/100) o Theta por dia (theta/365).
    """
    if option_type not in ("call", "put"):
        raise ValueError("option_type debe ser 'call' o 'put'.")
    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2_from_d1(d1, sigma, T)
    nd1_pdf = norm.pdf(d1)
    sqrt_t = math.sqrt(T)
    disc = math.exp(-r * T)

    gamma = nd1_pdf / (S * sigma * sqrt_t)
    vega = S * sqrt_t * nd1_pdf

    if option_type == "call":
        delta = norm.cdf(d1)
        theta = (
            -S * nd1_pdf * sigma / (2.0 * sqrt_t)
            - r * K * disc * norm.cdf(d2)
        )
        rho = K * T * disc * norm.cdf(d2)
    else:
        delta = norm.cdf(d1) - 1.0
        theta = (
            -S * nd1_pdf * sigma / (2.0 * sqrt_t)
            + r * K * disc * norm.cdf(-d2)
        )
        rho = -K * T * disc * norm.cdf(-d2)

    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "vega":  float(vega),
        "theta": float(theta),
        "rho":   float(rho),
    }


def put_call_parity(
    S: float, K: float, T: float, r: float, sigma: float
) -> dict:
    """
    Verifica numericamente C - P = S - K * exp(-rT).

    Calcula ambos precios con BS, evalua la identidad y reporta el error
    absoluto. Es la verificacion que la rubrica exige.
    """
    c = bs_price(S, K, T, r, sigma, "call")
    p = bs_price(S, K, T, r, sigma, "put")
    lhs = c - p
    rhs = S - K * math.exp(-r * T)
    err = lhs - rhs
    return {
        "call":             c,
        "put":              p,
        "lhs_c_minus_p":    float(lhs),
        "rhs_s_minus_pv_k": float(rhs),
        "error":            float(err),
        "valid":            bool(abs(err) < 1e-6),
    }


def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str,
    initial_sigma: float = 0.20,
    tolerance: float = 1e-7,
    max_iter: int = 100,
) -> dict:
    """
    Resuelve sigma tal que BS(sigma) = market_price usando Newton-Raphson:

        sigma_{n+1} = sigma_n - [BS(sigma_n) - market_price] / vega(sigma_n)

    Retorna metadata de convergencia.
    """
    if market_price <= 0:
        raise ValueError("market_price debe ser positivo.")
    if option_type not in ("call", "put"):
        raise ValueError("option_type debe ser 'call' o 'put'.")

    sigma = float(initial_sigma)
    for i in range(1, max_iter + 1):
        try:
            price_now = bs_price(S, K, T, r, sigma, option_type)
            v = bs_vega(S, K, T, r, sigma)
        except (ValueError, ZeroDivisionError):
            return {
                "sigma":      None,
                "iterations": i,
                "converged":  False,
                "reason":     "evaluacion fallo",
            }
        diff = price_now - market_price
        if abs(diff) < tolerance:
            return {
                "sigma":      float(sigma),
                "iterations": i,
                "converged":  True,
                "price_at_sigma": float(price_now),
            }
        if v < 1e-12:
            return {
                "sigma":      float(sigma),
                "iterations": i,
                "converged":  False,
                "reason":     "vega cerca de cero",
            }
        sigma = sigma - diff / v
        # Forzar sigma positiva
        if sigma <= 0:
            sigma = 1e-3

    return {
        "sigma":      float(sigma),
        "iterations": max_iter,
        "converged":  False,
        "reason":     "max_iter alcanzado",
    }


def payoff_at_expiry(strikes: list[float], K: float, option_type: str) -> list[float]:
    """Payoff intrinseco a vencimiento (call: max(S-K, 0); put: max(K-S, 0))."""
    if option_type == "call":
        return [max(s - K, 0.0) for s in strikes]
    if option_type == "put":
        return [max(K - s, 0.0) for s in strikes]
    raise ValueError("option_type debe ser 'call' o 'put'.")


# =============================================================================
# 2. OptionPricer: wrapper OO comodo
# =============================================================================


class OptionPricer:
    """
    Wrapper orientado a objetos sobre las funciones puras de Black-Scholes.

    Pensado para uso en endpoints (recibe un BSRequest tipado) y para
    notebooks (instanciar una vez y pedir varios resultados).
    """

    def __init__(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
    ) -> None:
        if S <= 0:
            raise ValueError("S (precio del subyacente) debe ser positivo.")
        if K <= 0:
            raise ValueError("K (strike) debe ser positivo.")
        if T <= 0:
            raise ValueError("T (tiempo a vencimiento) debe ser positivo.")
        if sigma <= 0:
            raise ValueError("sigma (volatilidad) debe ser positiva.")
        if r < 0 or r > 1:
            raise ValueError("r (tasa libre de riesgo) debe estar en [0, 1].")

        self.S = float(S)
        self.K = float(K)
        self.T = float(T)
        self.r = float(r)
        self.sigma = float(sigma)

    # ── Math ─────────────────────────────────────────────────────────────────

    @property
    def d1(self) -> float:
        return _d1(self.S, self.K, self.T, self.r, self.sigma)

    @property
    def d2(self) -> float:
        return _d2_from_d1(self.d1, self.sigma, self.T)

    def price(self, option_type: str) -> float:
        return bs_price(self.S, self.K, self.T, self.r, self.sigma, option_type)

    def greeks(self, option_type: str) -> dict:
        return bs_greeks(self.S, self.K, self.T, self.r, self.sigma, option_type)

    def vega(self) -> float:
        return bs_vega(self.S, self.K, self.T, self.r, self.sigma)

    def put_call_parity(self) -> dict:
        return put_call_parity(self.S, self.K, self.T, self.r, self.sigma)

    def implied_volatility(
        self,
        market_price: float,
        option_type: str,
        initial_sigma: float = 0.20,
        tolerance: float = 1e-7,
        max_iter: int = 100,
    ) -> dict:
        return implied_volatility(
            market_price=market_price,
            S=self.S,
            K=self.K,
            T=self.T,
            r=self.r,
            option_type=option_type,
            initial_sigma=initial_sigma,
            tolerance=tolerance,
            max_iter=max_iter,
        )

    # ── Resumen completo (para endpoint) ─────────────────────────────────────

    def summary(self, option_type: str, market_price: Optional[float] = None) -> dict:
        out = {
            "S":           self.S,
            "K":           self.K,
            "T":           self.T,
            "r":           self.r,
            "sigma":       self.sigma,
            "option_type": option_type,
            "price":       self.price(option_type),
            "greeks":      self.greeks(option_type),
            "put_call_parity": self.put_call_parity(),
        }
        if market_price is not None:
            out["implied_volatility"] = self.implied_volatility(
                market_price, option_type
            )
        return out
