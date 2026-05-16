"""
Tests unitarios de VaR / CVaR (modulo M5).

El VaR parametrico debe coincidir con la expresion analitica
|mu + sigma * Phi^-1(alpha)| sobre una serie normal i.i.d. conocida.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from api.logic import calculate_var_cvar, kupiec_test


def test_parametric_var_matches_analytical():
    """VaR parametrico vs analitico sobre N(0, 0.01) con N=10_000."""
    rng = np.random.default_rng(seed=42)
    returns = pd.Series(rng.normal(0.0, 0.01, size=10_000))

    result = calculate_var_cvar(returns, confidence=0.95, n_simulations=100)

    # Esperado: |mu + sigma * z_{0.05}|, con z_{0.05} ~ -1.6449
    mu = float(returns.mean())
    sigma = float(returns.std())
    z = float(stats.norm.ppf(0.05))
    expected_var = abs(mu + sigma * z)

    assert abs(result["Parametrico"]["VaR"] - expected_var) < 1e-9


def test_parametric_var_higher_at_99_than_95():
    """VaR_99 > VaR_95 para la misma serie."""
    rng = np.random.default_rng(seed=7)
    returns = pd.Series(rng.normal(0.0, 0.015, size=2_000))

    r95 = calculate_var_cvar(returns, confidence=0.95, n_simulations=100)
    r99 = calculate_var_cvar(returns, confidence=0.99, n_simulations=100)

    assert r99["Parametrico"]["VaR"] > r95["Parametrico"]["VaR"]


def test_var_returns_positive_loss():
    """El VaR siempre se reporta como magnitud positiva."""
    rng = np.random.default_rng(seed=1)
    returns = pd.Series(rng.normal(0.0, 0.02, size=500))
    result = calculate_var_cvar(returns, confidence=0.95, n_simulations=100)
    for method in ("Historico", "Parametrico", "Montecarlo"):
        assert result[method]["VaR"] >= 0


# ── Kupiec POF ───────────────────────────────────────────────────────────────


def test_kupiec_no_exceptions_returns_p_value_none():
    """Si no hay excedencias el LR no esta definido; el helper devuelve None."""
    rng = np.random.default_rng(seed=11)
    returns = pd.Series(rng.normal(0.0, 0.01, size=200))
    res = kupiec_test(returns, var_value=1.0, confidence=0.95)  # VaR enorme -> 0 excedencias
    assert res["N_exceptions"] == 0
    assert res["LR_stat"] is None
    assert res["p_value"] is None
    assert res["passed"] is None
