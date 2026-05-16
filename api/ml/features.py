"""
Ingenieria de features y etiquetado para el clasificador buy/hold/sell.

Features (todas derivadas de OHLCV historico):
  - ret_1d, ret_5d, ret_20d  : retornos retrospectivos (no leakage).
  - rsi_14                    : RSI de 14 sesiones.
  - macd_hist                 : histograma MACD.
  - vol_20d                   : volatilidad rolling 20 sesiones.
  - pos_in_bb                 : posicion relativa del precio en bandas
                                de Bollinger (0=banda inferior, 1=superior).
  - volume_ratio              : volumen actual / promedio rolling 20d.

Etiqueta (forward 5 dias):
  - Buy   si ret_futuro >= +2%
  - Sell  si ret_futuro <= -2%
  - Hold  si esta entre los dos umbrales

Asi la etiqueta usa el FUTURO y las features el PASADO -> no hay leakage.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

FEATURE_NAMES: list[str] = [
    "ret_1d",
    "ret_5d",
    "ret_20d",
    "rsi_14",
    "macd_hist",
    "vol_20d",
    "pos_in_bb",
    "volume_ratio",
]


# ── Indicadores tecnicos privados ────────────────────────────────────────────


def _rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=window).mean()
    rs = gain / loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _macd_histogram(prices: pd.Series) -> pd.Series:
    ema_12 = prices.ewm(span=12, adjust=False).mean()
    ema_26 = prices.ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    signal = macd_line.ewm(span=9, adjust=False).mean()
    return macd_line - signal


def _bb_position(prices: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.Series:
    sma = prices.rolling(window=window).mean()
    std = prices.rolling(window=window).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    denom = (upper - lower).replace(0.0, np.nan)
    return (prices - lower) / denom


# ── API publica ──────────────────────────────────────────────────────────────


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construye el DataFrame de features a partir de OHLCV."""
    if df is None or df.empty:
        return pd.DataFrame(columns=FEATURE_NAMES)

    close = df["Close"]
    if "Volume" in df.columns:
        vol = df["Volume"]
    else:
        vol = pd.Series(np.nan, index=df.index)

    out = pd.DataFrame(index=df.index)
    out["ret_1d"] = close.pct_change(1)
    out["ret_5d"] = close.pct_change(5)
    out["ret_20d"] = close.pct_change(20)
    out["rsi_14"] = _rsi(close, window=14)
    out["macd_hist"] = _macd_histogram(close)
    out["vol_20d"] = close.pct_change().rolling(window=20).std()
    out["pos_in_bb"] = _bb_position(close)
    vol_ma_20 = vol.rolling(window=20).mean().replace(0.0, np.nan)
    out["volume_ratio"] = vol / vol_ma_20

    out = out.replace([np.inf, -np.inf], np.nan)
    return out[FEATURE_NAMES]


def build_labels(
    df: pd.DataFrame,
    horizon_days: int = 5,
    threshold: float = 0.02,
) -> pd.Series:
    """
    Etiqueta cada barra con Buy/Hold/Sell segun el retorno acumulado
    en los proximos horizon_days. threshold se interpreta en decimal
    (0.02 = 2%).
    """
    if df is None or df.empty:
        return pd.Series(dtype=object)

    close = df["Close"]
    future_ret = close.shift(-horizon_days) / close - 1.0

    labels = pd.Series(index=df.index, dtype=object)
    labels[future_ret >= threshold] = "Buy"
    labels[future_ret <= -threshold] = "Sell"
    labels[(future_ret > -threshold) & (future_ret < threshold)] = "Hold"
    return labels


def build_features_and_labels(
    df: pd.DataFrame,
    horizon_days: int = 5,
    threshold: float = 0.02,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Devuelve (X, y) alineados y sin NaN, listos para fit."""
    X = build_features(df)
    y = build_labels(df, horizon_days=horizon_days, threshold=threshold)
    mask = X.notna().all(axis=1) & y.notna()
    return (
        X.loc[mask].reset_index(drop=True),
        y.loc[mask].reset_index(drop=True),
    )


def latest_features_for_ticker(df: pd.DataFrame) -> dict:
    """
    Devuelve el ultimo vector de features valido del DataFrame.

    Si la ultima barra contiene NaN, retrocede hasta encontrar una
    completa. Retorna {} si no hay ninguna fila valida.
    """
    X = build_features(df)
    if X.empty:
        return {}
    valid = X.dropna()
    if valid.empty:
        return {}
    last = valid.iloc[-1]
    return {k: float(last[k]) for k in FEATURE_NAMES}
