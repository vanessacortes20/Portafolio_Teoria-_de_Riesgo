"""
Tests unitarios de la clase SignalGenerator (Plan III, M7).

Cubre las cinco reglas obligatorias del Módulo 7:
  1. MACD cross
  2. RSI extreme
  3. Bollinger touch
  4. Moving average cross (Golden/Death)
  5. Stochastic cross

Cada test construye dos registros técnicos sintéticos (último y penúltimo)
que disparan la regla, sin tocar la red ni la BD.
"""
from __future__ import annotations

from api.services.signals import SignalGenerator


def _empty_record() -> dict:
    """Registro neutro: todos los indicadores en valores que no disparan reglas."""
    return {
        "RSI":         50.0,
        "MACD_Line":   0.0,
        "MACD_Signal": 0.0,
        "MACD_Hist":   0.0,
        "Close":       100.0,
        "BB_Upper":    110.0,
        "BB_Lower":    90.0,
        "SMA_20":      100.0,
        "SMA_50":      100.0,
        "Stoch_K":     50.0,
        "Stoch_D":     50.0,
    }


def test_macd_bullish_cross_se_detecta():
    """MACD cruza por encima de la señal -> señal de compra."""
    prev = _empty_record()
    prev["MACD_Line"]   = -0.5
    prev["MACD_Signal"] = -0.2
    last = _empty_record()
    last["MACD_Line"]   = 0.1
    last["MACD_Signal"] = -0.1
    gen = SignalGenerator(last=last, prev=prev)
    sigs = gen.macd_cross()
    assert len(sigs) == 1
    assert sigs[0]["id"]   == "MACD_BULL_CROSS"
    assert sigs[0]["type"] == "buy"


def test_rsi_overbought_dispara_sell():
    """RSI > umbral configurable -> señal de venta."""
    last = _empty_record()
    last["RSI"] = 85.0
    gen = SignalGenerator(last=last, prev=_empty_record(), rsi_overbought=70)
    sigs = gen.rsi_extreme()
    assert len(sigs) == 1
    assert sigs[0]["id"]   == "RSI_OVERBOUGHT"
    assert sigs[0]["type"] == "sell"
    assert sigs[0]["value"] == 85.0


def test_rsi_oversold_dispara_buy_con_umbral_personalizado():
    """RSI <= umbral configurable -> señal de compra."""
    last = _empty_record()
    last["RSI"] = 25.0
    gen = SignalGenerator(last=last, prev=_empty_record(), rsi_oversold=30)
    sigs = gen.rsi_extreme()
    assert len(sigs) == 1
    assert sigs[0]["id"]   == "RSI_OVERSOLD"
    assert sigs[0]["type"] == "buy"


def test_golden_cross_se_detecta():
    """SMA20 cruza por encima de SMA50 -> Golden cross (buy)."""
    prev = _empty_record()
    prev["SMA_20"] = 99.0
    prev["SMA_50"] = 100.0
    last = _empty_record()
    last["SMA_20"] = 101.0
    last["SMA_50"] = 100.0
    gen = SignalGenerator(last=last, prev=prev)
    sigs = gen.moving_average_cross()
    assert len(sigs) == 1
    assert sigs[0]["id"]   == "GOLDEN_CROSS"
    assert sigs[0]["type"] == "buy"


def test_death_cross_se_detecta():
    """SMA20 cruza por debajo de SMA50 -> Death cross (sell)."""
    prev = _empty_record()
    prev["SMA_20"] = 101.0
    prev["SMA_50"] = 100.0
    last = _empty_record()
    last["SMA_20"] = 99.0
    last["SMA_50"] = 100.0
    gen = SignalGenerator(last=last, prev=prev)
    sigs = gen.moving_average_cross()
    assert len(sigs) == 1
    assert sigs[0]["id"]   == "DEATH_CROSS"
    assert sigs[0]["type"] == "sell"


def test_bollinger_upper_touch_dispara_sell():
    """Precio cierra por encima de la banda superior -> sell."""
    last = _empty_record()
    last["Close"]    = 112.0
    last["BB_Upper"] = 110.0
    gen = SignalGenerator(last=last, prev=_empty_record())
    sigs = gen.bollinger_touch()
    assert any(s["id"] == "BB_UPPER_TOUCH" and s["type"] == "sell" for s in sigs)


def test_stochastic_bullish_en_sobreventa_se_detecta():
    """%K cruza %D al alza estando ambos en sobreventa -> buy."""
    prev = _empty_record()
    prev["Stoch_K"] = 15.0
    prev["Stoch_D"] = 18.0
    last = _empty_record()
    last["Stoch_K"] = 19.0
    last["Stoch_D"] = 17.0
    gen = SignalGenerator(last=last, prev=prev, stoch_oversold=20)
    sigs = gen.stochastic_signal()
    assert len(sigs) == 1
    assert sigs[0]["id"]   == "STOCH_BULL"
    assert sigs[0]["type"] == "buy"


def test_evaluate_all_acumula_multiples_senales():
    """evaluate_all() debe acumular señales de varias reglas activas."""
    prev = _empty_record()
    prev["SMA_20"] = 99.0
    prev["SMA_50"] = 100.0
    last = _empty_record()
    last["RSI"]    = 85.0
    last["SMA_20"] = 101.0
    last["SMA_50"] = 100.0
    gen = SignalGenerator(last=last, prev=prev)
    all_signals = gen.evaluate_all()
    ids = {s["id"] for s in all_signals}
    # Al menos RSI_OVERBOUGHT y GOLDEN_CROSS deben dispararse simultáneamente
    assert "RSI_OVERBOUGHT" in ids
    assert "GOLDEN_CROSS"   in ids


def test_signal_no_recomienda_absolutamente():
    """Las interpretaciones no deben usar lenguaje de recomendación absoluta."""
    last = _empty_record()
    last["RSI"] = 85.0
    gen = SignalGenerator(last=last, prev=_empty_record())
    sigs = gen.rsi_extreme()
    assert sigs
    text = sigs[0]["interpretation"].lower()
    # No usar lenguaje de recomendación absoluta
    forbidden = ["compre seguro", "venda seguro", "ganancia garantizada", "compra garantizada"]
    for word in forbidden:
        assert word not in text
