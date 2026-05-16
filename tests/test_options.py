"""Tests unitarios de Black-Scholes (modulo M10)."""
from __future__ import annotations

import math

import pytest

from api.services.options import (
    OptionPricer,
    bs_greeks,
    bs_price,
    bs_vega,
    implied_volatility,
    put_call_parity,
)


# ── Paridad put-call C - P = S - K * exp(-rT) ────────────────────────────────


def test_put_call_parity_basic():
    """Sobre un par (call, put) BS la paridad debe cumplirse hasta 1e-6."""
    parity = put_call_parity(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.20)
    assert parity["valid"] is True
    assert abs(parity["error"]) < 1e-6


def test_put_call_parity_otm_call():
    """Mismo test fuera del dinero (K > S)."""
    parity = put_call_parity(S=100.0, K=120.0, T=0.5, r=0.05, sigma=0.30)
    assert parity["valid"] is True
    assert abs(parity["error"]) < 1e-6


# ── Precios y Greeks ─────────────────────────────────────────────────────────


def test_atm_call_price_positive():
    p = bs_price(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.20, option_type="call")
    assert p > 0


def test_deep_itm_call_approx_intrinsic():
    """Una call muy ITM con T pequeno se aproxima al payoff intrinseco S - K e^(-rT)."""
    S, K, T, r, sigma = 200.0, 100.0, 0.01, 0.05, 0.20
    p = bs_price(S, K, T, r, sigma, "call")
    intrinsic = S - K * math.exp(-r * T)
    # margen amplio porque T no es exactamente 0
    assert abs(p - intrinsic) < 0.5


def test_call_delta_in_zero_one():
    g = bs_greeks(S=100, K=100, T=1.0, r=0.05, sigma=0.20, option_type="call")
    assert 0.0 <= g["delta"] <= 1.0


def test_put_delta_in_minus_one_zero():
    g = bs_greeks(S=100, K=100, T=1.0, r=0.05, sigma=0.20, option_type="put")
    assert -1.0 <= g["delta"] <= 0.0


def test_vega_is_positive():
    v = bs_vega(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
    assert v > 0


# ── Volatilidad implicita ────────────────────────────────────────────────────


def test_implied_volatility_recovers_input_sigma():
    """Si tomamos el precio BS con sigma=0.25 y lo invertimos, devuelve 0.25."""
    S, K, T, r, sigma_true = 100.0, 100.0, 1.0, 0.05, 0.25
    market_price = bs_price(S, K, T, r, sigma_true, "call")
    res = implied_volatility(
        market_price=market_price,
        S=S, K=K, T=T, r=r,
        option_type="call",
        initial_sigma=0.20,
        tolerance=1e-8,
    )
    assert res["converged"] is True
    assert abs(res["sigma"] - sigma_true) < 1e-5


# ── Validacion de constructor ────────────────────────────────────────────────


def test_option_pricer_rejects_negative_sigma():
    with pytest.raises(ValueError):
        OptionPricer(S=100, K=100, T=1.0, r=0.05, sigma=-0.20)


def test_option_pricer_rejects_zero_time():
    with pytest.raises(ValueError):
        OptionPricer(S=100, K=100, T=0.0, r=0.05, sigma=0.20)
