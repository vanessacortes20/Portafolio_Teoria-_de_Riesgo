"""
Pipeline de Machine Learning del Proyecto Integrador.

Estructura:
  - features.py   : ingenieria de features y etiquetado buy/hold/sell.
  - predictor.py  : ModelPredictor (patron Singleton verificable) y la
                    dependencia FastAPI get_predictor.
  - train.py      : script entrenable con `python -m api.ml.train`.

El artefacto serializado vive en Settings.ml_model_path (default
api/ml/model.joblib). Si no existe, ModelPredictor cae a una heuristica
basada en RSI + MACD para garantizar que /predict siempre responda.
"""
from api.ml.features import (
    FEATURE_NAMES,
    build_features,
    build_features_and_labels,
    build_labels,
    latest_features_for_ticker,
)
from api.ml.predictor import ModelPredictor, get_predictor

__all__ = [
    "FEATURE_NAMES",
    "build_features",
    "build_labels",
    "build_features_and_labels",
    "latest_features_for_ticker",
    "ModelPredictor",
    "get_predictor",
]
