"""
Script de entrenamiento del clasificador buy/hold/sell.

Ejecucion:
    python -m api.ml.train

Pipeline:
  1. Descarga ~2 anos de historico para cada ticker de Settings.tickers.
  2. Construye features y etiquetas (forward 5 dias, threshold +-2%).
  3. Particion train/test temporal (shuffle=False) -> sin leakage.
  4. Fit de Pipeline(StandardScaler + RandomForestClassifier).
  5. Reporte classification_report + accuracy + F1.
  6. Guarda artefacto en Settings.ml_model_path (joblib).
  7. Guarda metadata .meta.json al lado.

Despues de entrenar, reinicia el backend para que el Singleton recargue
el modelo nuevo (el Singleton solo lee al primer __new__).
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from api.config import get_settings
from api.data import get_historical_data
from api.db.base import init_db
from api.ml.features import FEATURE_NAMES, build_features_and_labels

logger = logging.getLogger(__name__)

MODEL_VERSION = "v1.0.0"


def _gather_dataset() -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Junta features y etiquetas de todos los tickers configurados."""
    settings = get_settings()
    X_parts: list[pd.DataFrame] = []
    y_parts: list[pd.Series] = []
    tickers_used: list[str] = []

    for ticker in settings.tickers:
        df = get_historical_data(ticker)
        if df is None or df.empty:
            print(f"[skip] {ticker}: sin datos.")
            continue
        X_t, y_t = build_features_and_labels(df)
        if X_t.empty:
            print(f"[skip] {ticker}: features vacias.")
            continue
        X_parts.append(X_t)
        y_parts.append(y_t)
        tickers_used.append(ticker)
        print(f"[ok]   {ticker}: {len(X_t)} filas.")

    if not X_parts:
        raise RuntimeError("Sin datos para entrenar.")

    X = pd.concat(X_parts, ignore_index=True)
    y = pd.concat(y_parts, ignore_index=True)
    return X, y, tickers_used


def train(verbose: bool = True) -> dict:
    """Entrena el modelo, persiste el artefacto y devuelve la metadata."""
    settings = get_settings()
    # Asegura que las tablas existan (DataService persiste precios via ORM).
    init_db()
    print(f"Entrenando clasificador buy/hold/sell para: {settings.tickers}")

    X, y, tickers_used = _gather_dataset()
    print(f"\nDataset total: {len(X)} filas, {X.shape[1]} features.")
    print(f"Distribucion de clases:\n{y.value_counts(normalize=True).round(3)}\n")

    if len(X) < 100:
        raise RuntimeError(
            f"Datos insuficientes para entrenar ({len(X)} filas). "
            "Verifica que los tickers tengan al menos ~6 meses de historia."
        )

    # Particion temporal: shuffle=False evita data leakage en series financieras.
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, random_state=42, shuffle=False
    )

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=10,
                    min_samples_leaf=5,
                    random_state=42,
                    n_jobs=-1,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    pipeline.fit(X_tr, y_tr)
    y_pred = pipeline.predict(X_te)
    acc = float(accuracy_score(y_te, y_pred))
    f1 = float(f1_score(y_te, y_pred, average="weighted"))

    if verbose:
        print("Classification report (test):")
        print(classification_report(y_te, y_pred, digits=4))
        print(f"Accuracy:        {acc:.4f}")
        print(f"F1 (weighted):   {f1:.4f}")

    # Persistir
    model_path = Path(settings.ml_model_path)
    if not model_path.is_absolute():
        project_root = Path(__file__).resolve().parent.parent.parent
        model_path = project_root / model_path
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)

    meta = {
        "model_version": MODEL_VERSION,
        "trained_at":    datetime.utcnow().isoformat(timespec="seconds"),
        "model_type":    "RandomForestClassifier (sklearn Pipeline + StandardScaler)",
        "n_train":       int(len(X_tr)),
        "n_test":        int(len(X_te)),
        "accuracy":      acc,
        "f1_weighted":   f1,
        "features":      list(FEATURE_NAMES),
        "labels":        ["Buy", "Hold", "Sell"],
        "tickers_used":  tickers_used,
        "horizon_days":  5,
        "label_threshold": 0.02,
    }
    meta_path = model_path.with_suffix(".meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\nModelo guardado:   {model_path}")
    print(f"Metadata guardada: {meta_path}")
    print("Reinicia el backend para que el Singleton recargue el modelo nuevo.")
    return meta


if __name__ == "__main__":
    try:
        train()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
