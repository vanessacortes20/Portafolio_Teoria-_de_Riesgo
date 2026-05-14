"""
Tests unitarios sobre las funciones puras de api/logic.py y api/services/.

NO dependen de yfinance, FRED ni internet. Usan series sintéticas.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest


# ── 1. RSI sobre serie conocida ─────────────────────────────────────────────

def test_rsi_constante_es_neutral():
    """RSI sobre una serie con cambios uniformes debe estar bien acotado."""
    from backend.app.services.logic import calculate_rsi

    # Serie con tendencia alcista pura (cada día sube 1)
    df = pd.DataFrame({"Close": [100 + i for i in range(50)]})
    rsi = calculate_rsi(df, window=14).dropna()

    assert len(rsi) > 0
    # En tendencia alcista pura el RSI tiende a 100
    assert rsi.iloc[-1] >= 70, f"RSI alcista esperado >=70, obtenido {rsi.iloc[-1]}"
    assert (rsi.between(0, 100)).all(), "RSI fuera de [0, 100]"


def test_rsi_bajista_pura():
    """RSI sobre una serie bajista pura tiende a 0."""
    from backend.app.services.logic import calculate_rsi

    df = pd.DataFrame({"Close": [100 - i for i in range(50)]})
    rsi = calculate_rsi(df, window=14).dropna()
    assert rsi.iloc[-1] <= 30, f"RSI bajista esperado <=30, obtenido {rsi.iloc[-1]}"


# ── 2. VaR paramétrico contra valor analítico ───────────────────────────────

def test_var_parametrico_normal_analitico():
    """
    Para retornos N(μ, σ²) el VaR paramétrico al 95% debe ser ≈ |μ + σ·z_0.05|.
    Con μ=0, σ=0.02, z_0.05=-1.6449 → VaR ≈ 0.0329.
    """
    from scipy.stats import norm
    from backend.app.services.logic import calculate_var_cvar

    np.random.seed(42)
    rng_returns = pd.Series(np.random.normal(0.0, 0.02, 10_000))
    result = calculate_var_cvar(rng_returns, confidence=0.95, n_simulations=5_000)

    var_param = result["Parametrico"]["VaR"]
    expected  = abs(0.0 + 0.02 * norm.ppf(0.05))  # ≈ 0.03290
    # Tolerancia ±5% (estimación muestral de μ y σ varía un poco con la semilla)
    assert abs(var_param - expected) < 0.005, (
        f"VaR paramétrico {var_param:.4f} muy distante del valor analítico {expected:.4f}"
    )
    # El VaR siempre se reporta como pérdida positiva
    assert var_param > 0


# ── 3. Paridad put-call ─────────────────────────────────────────────────────

def test_put_call_parity_clasica():
    """C - P debe igualar S - K·e^(-rT) hasta error numérico."""
    from backend.app.services.options import OptionPricer

    op = OptionPricer(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
    parity = op.put_call_parity_check()

    assert parity["satisfied"] is True
    assert parity["abs_diff"] < 1e-6, f"Paridad rota: |diff| = {parity['abs_diff']}"
    # Valor de referencia conocido
    assert abs(parity["call"] - 10.4506) < 0.001
    assert abs(parity["put"]  -  5.5735) < 0.001


def test_put_call_parity_strike_distinto():
    """Paridad debe cumplirse para cualquier strike."""
    from backend.app.services.options import OptionPricer

    op = OptionPricer(S=100, K=110, T=0.5, r=0.04, sigma=0.25)
    parity = op.put_call_parity_check()
    assert parity["satisfied"] is True
    expected_diff = 100 - 110 * math.exp(-0.04 * 0.5)
    assert abs(parity["lhs"] - expected_diff) < 1e-6
