"""
Entrenamiento del modelo de clasificación direccional buy/hold/sell.

Uso:
    python -m api.ml.train [--ticker AMZN] [--period 5y] [--output api/ml/model.joblib]

El modelo predice la dirección del retorno del día siguiente sobre la base
de features técnicas (retornos rezagados, RSI, MACD, volatilidad rodante).

⚠️  ADVERTENCIA: este modelo es una herramienta analítica de demostración
académica. NO constituye recomendación financiera ni garantía de rentabilidad.
Las predicciones son probabilísticas y no consideran costos de transacción,
liquidez ni cambios de régimen.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from backend.app.data_yf import get_historical_data
from backend.app.services.logic import calculate_macd, calculate_rsi, calculate_returns

MODEL_VERSION   = "v1.0.0"
DEFAULT_OUTPUT  = Path(__file__).parent / "model.joblib"
DEFAULT_TICKER  = "AMZN"

# Umbral en decimal para clasificar el retorno del día siguiente
UP_THRESHOLD   = 0.005   # > +0.5% → buy
DOWN_THRESHOLD = -0.005  # < -0.5% → sell


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    """Construye las features de entrada del modelo a partir de OHLCV."""
    df = data.copy().reset_index(drop=True)

    # Retornos
    simple_ret, log_ret = calculate_returns(data)
    # Re-alinear con el dataframe (calculate_returns elimina la primera fila)
    log_ret = log_ret.reset_index(drop=True)

    # Indicadores técnicos
    rsi  = calculate_rsi(df, window=14).reset_index(drop=True)
    macd_line, macd_signal, macd_hist = calculate_macd(df)
    macd_hist = macd_hist.reset_index(drop=True)

    # Construcción de features sobre el mismo índice (alineado con log_ret)
    n = min(len(log_ret), len(rsi), len(macd_hist))
    feats = pd.DataFrame({
        "ret_lag_1": log_ret.shift(1).iloc[:n].values,
        "ret_lag_2": log_ret.shift(2).iloc[:n].values,
        "ret_lag_3": log_ret.shift(3).iloc[:n].values,
        "rsi":       rsi.iloc[-n:].values,
        "macd_hist": macd_hist.iloc[-n:].values,
        "vol_5":     log_ret.shift(1).rolling(5).std().iloc[:n].values,
        "vol_20":    log_ret.shift(1).rolling(20).std().iloc[:n].values,
    })
    feats["target_ret"] = log_ret.iloc[:n].values
    return feats.dropna().reset_index(drop=True)


def label_target(df: pd.DataFrame) -> pd.Series:
    """0 = sell, 1 = hold, 2 = buy. Basado en el retorno realizado del día."""
    y = pd.Series(1, index=df.index, dtype=int)
    y[df["target_ret"] > UP_THRESHOLD]  = 2
    y[df["target_ret"] < DOWN_THRESHOLD] = 0
    return y


def train_model(ticker: str = DEFAULT_TICKER, period: str = "5y") -> dict:
    """Entrena el modelo y devuelve métricas + lo persiste con joblib."""
    print(f"[train] Descargando {ticker} ({period})...")
    data = get_historical_data(ticker, period=period)
    if data is None or len(data) < 200:
        raise RuntimeError(f"Datos insuficientes para {ticker}")

    feats = build_features(data)
    if len(feats) < 100:
        raise RuntimeError(f"Features insuficientes (n={len(feats)})")

    feature_cols = ["ret_lag_1", "ret_lag_2", "ret_lag_3",
                    "rsi", "macd_hist", "vol_5", "vol_20"]
    X = feats[feature_cols].values
    y = label_target(feats).values

    # shuffle=False respeta el orden temporal — crítico en series financieras
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False,
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_tr, y_tr)

    train_acc = float(model.score(X_tr, y_tr))
    test_acc  = float(model.score(X_te, y_te))
    report    = classification_report(y_te, model.predict(X_te), output_dict=True, zero_division=0)

    metadata = {
        "model_version":  MODEL_VERSION,
        "ticker_trained": ticker,
        "period":         period,
        "n_samples":      int(len(feats)),
        "n_train":        int(len(y_tr)),
        "n_test":         int(len(y_te)),
        "feature_cols":   feature_cols,
        "labels":         {"0": "sell", "1": "hold", "2": "buy"},
        "thresholds":     {"up": UP_THRESHOLD, "down": DOWN_THRESHOLD},
        "train_accuracy": round(train_acc, 4),
        "test_accuracy":  round(test_acc, 4),
        "trained_at":     datetime.utcnow().isoformat(timespec="seconds"),
        "warning":        ("Modelo académico — NO constituye recomendación de inversión "
                           "ni garantiza rentabilidad. Considerar siempre costos de transacción."),
    }

    bundle = {"model": model, "metadata": metadata}
    return bundle, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default=DEFAULT_TICKER)
    parser.add_argument("--period", default="5y")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    bundle, report = train_model(args.ticker, args.period)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out_path)

    print(f"[train] Modelo guardado en: {out_path}")
    print(f"[train] Train accuracy: {bundle['metadata']['train_accuracy']}")
    print(f"[train] Test accuracy:  {bundle['metadata']['test_accuracy']}")
    print(f"[train] Classification report:")
    print(json.dumps(report, indent=2, default=str)[:600])


if __name__ == "__main__":
    main()
