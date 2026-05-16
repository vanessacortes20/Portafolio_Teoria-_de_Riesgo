"""Tests unitarios de indicadores tecnicos (modulo M1)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from api.logic import calculate_rsi, calculate_sma


# ── RSI sobre series con resultado conocido ──────────────────────────────────


def test_rsi_monotone_up_is_100():
    """
    Si todos los retornos son positivos, no hay perdidas en la ventana.
    rolling(loss).mean() = 0, RS = inf, RSI = 100.
    """
    prices = pd.DataFrame({"Close": list(range(100, 130))})
    rsi = calculate_rsi(prices, window=14)
    assert rsi.iloc[-1] == 100.0


def test_rsi_monotone_down_is_0():
    """
    Si todos los retornos son negativos, no hay ganancias.
    rolling(gain).mean() = 0, RS = 0, RSI = 0.
    """
    prices = pd.DataFrame({"Close": list(range(130, 100, -1))})
    rsi = calculate_rsi(prices, window=14)
    assert rsi.iloc[-1] == 0.0


def test_rsi_series_length_matches_input():
    prices = pd.DataFrame({"Close": np.linspace(100, 110, 50)})
    rsi = calculate_rsi(prices, window=14)
    assert len(rsi) == 50


# ── SMA ──────────────────────────────────────────────────────────────────────


def test_sma_constant_series_equals_value():
    """SMA de una serie constante c en cualquier ventana es c."""
    prices = pd.DataFrame({"Close": [10.0] * 30})
    sma = calculate_sma(prices, window=14)
    # Despues de la ventana inicial, el valor debe ser exactamente 10.0
    assert sma.iloc[-1] == pytest.approx(10.0, abs=1e-9)


def test_sma_known_window():
    """SMA(window=3) sobre [1,2,3,4,5,6] en t=2 debe ser (1+2+3)/3 = 2."""
    prices = pd.DataFrame({"Close": [1, 2, 3, 4, 5, 6]})
    sma = calculate_sma(prices, window=3)
    assert sma.iloc[2] == pytest.approx(2.0, abs=1e-9)
    assert sma.iloc[3] == pytest.approx(3.0, abs=1e-9)
    assert sma.iloc[5] == pytest.approx(5.0, abs=1e-9)
