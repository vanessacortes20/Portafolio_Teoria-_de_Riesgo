"""
ModelPredictor: Singleton verificable que sirve el modelo de ML al
endpoint /api/v1/predict.

Por que Singleton:
  Sin este patron, cada request a /predict recargaria el .joblib desde
  disco (~10-50 MB para un Random Forest serializado), pagando el costo
  de I/O en cada llamada. Con el Singleton el modelo se carga UNA sola
  vez al primer acceso y todas las llamadas siguientes reutilizan la
  misma instancia.

Como verificarlo:
  En el primer __new__ se imprime un mensaje "[ModelPredictor] modelo
  cargado". Si haces dos llamadas a /predict consecutivas y el mensaje
  aparece UNA SOLA VEZ en los logs/stdout del servidor, el Singleton
  funciona correctamente.

Fallback heuristico:
  Si Settings.ml_model_path no existe (todavia no se entreno el modelo),
  ModelPredictor cae a una regla simple basada en RSI y MACD. Esto
  garantiza que /predict siempre devuelva una respuesta valida y que el
  proyecto pueda demostrar el flujo end-to-end sin entrenar primero.

Reentrenar con:
  python -m api.ml.train
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np

from api.config import get_settings
from api.ml.features import FEATURE_NAMES
from api.services.decorators import log_execution_time

logger = logging.getLogger(__name__)


def _resolve_model_path() -> Path:
    """Resuelve la ruta del modelo a absoluta para no depender del cwd."""
    settings = get_settings()
    p = Path(settings.ml_model_path)
    if p.is_absolute():
        return p
    project_root = Path(__file__).resolve().parent.parent.parent
    return project_root / p


class ModelPredictor:
    """Singleton del modelo de prediccion."""

    _instance: Optional["ModelPredictor"] = None
    _model = None
    _meta: dict = {}

    def __new__(cls) -> "ModelPredictor":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._load()
        return cls._instance

    @classmethod
    def _load(cls) -> None:
        """Carga el modelo y la metadata. Se ejecuta UNA sola vez."""
        model_path = _resolve_model_path()

        if not model_path.exists():
            logger.warning(
                "Modelo ML no encontrado en %s. Usando fallback heuristico.",
                model_path,
            )
            cls._model = None
            cls._meta = {
                "model_version": "fallback-heuristic-v0",
                "model_type":    "heuristic",
                "features":      FEATURE_NAMES,
                "labels":        ["Buy", "Hold", "Sell"],
                "note":          "Entrene el modelo real con: python -m api.ml.train",
            }
            msg = (
                f"[ModelPredictor] modelo cargado (carga UNICA, Singleton): "
                f"{cls._meta['model_version']}"
            )
            print(msg)
            logger.info(msg)
            return

        try:
            cls._model = joblib.load(model_path)
        except Exception as exc:
            logger.error("Fallo al cargar %s: %s. Cayendo a heuristica.", model_path, exc)
            cls._model = None

        meta_path = model_path.with_suffix(".meta.json")
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    cls._meta = json.load(f)
            except Exception as exc:
                logger.warning("Fallo al leer metadata %s: %s", meta_path, exc)
                cls._meta = {"model_version": "unknown"}
        else:
            cls._meta = {"model_version": "unknown"}

        msg = (
            f"[ModelPredictor] modelo cargado (carga UNICA, Singleton): "
            f"{cls._meta.get('model_version', 'unknown')}"
        )
        # Imprimimos en stdout para que sea visible en la demo (uvicorn loguea stdout)
        print(msg)
        logger.info(msg)

    # ── Metadatos ────────────────────────────────────────────────────────────

    @property
    def model_version(self) -> str:
        return str(self._meta.get("model_version", "unknown"))

    @property
    def model_type(self) -> str:
        return str(self._meta.get("model_type", "RandomForest"))

    @property
    def meta(self) -> dict:
        return dict(self._meta)

    @property
    def has_real_model(self) -> bool:
        return self._model is not None

    @property
    def expected_features(self) -> list[str]:
        return list(FEATURE_NAMES)

    # ── Inferencia ───────────────────────────────────────────────────────────

    @log_execution_time
    def predict(self, features: dict) -> dict:
        """
        Predice una accion (Buy/Hold/Sell) sobre un vector de features.

        Args:
            features: dict con todas las claves de FEATURE_NAMES como
                      valores numericos. NaN/None no admitidos.

        Returns:
            dict con keys: action, confidence (puede ser None), model_version,
                            model_type.
        """
        # Validar y armar el vector
        missing = [k for k in FEATURE_NAMES if k not in features]
        if missing:
            raise ValueError(f"Features faltantes: {missing}")
        invalid = [k for k in FEATURE_NAMES if features[k] is None]
        if invalid:
            raise ValueError(f"Features con valor None: {invalid}")
        try:
            x = np.array([[float(features[k]) for k in FEATURE_NAMES]], dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Features no convertibles a float: {exc}")
        if not np.isfinite(x).all():
            raise ValueError("Features contienen NaN o Inf.")

        # Inferencia real
        if self._model is not None:
            pred = str(self._model.predict(x)[0])
            try:
                proba = self._model.predict_proba(x)[0]
                confidence = float(np.max(proba))
            except Exception:
                confidence = None
            return {
                "action":         pred,
                "confidence":     confidence,
                "model_version":  self.model_version,
                "model_type":     self.model_type,
            }

        # Fallback: regla heuristica RSI + MACD
        rsi = float(features.get("rsi_14", 50.0))
        macd_h = float(features.get("macd_hist", 0.0))
        if rsi < 30.0 and macd_h > 0.0:
            action = "Buy"
        elif rsi > 70.0 and macd_h < 0.0:
            action = "Sell"
        else:
            action = "Hold"
        return {
            "action":         action,
            "confidence":     None,
            "model_version":  self.model_version,
            "model_type":     "heuristic",
        }


# ── Dependency injection ────────────────────────────────────────────────────


def get_predictor() -> ModelPredictor:
    """
    Dependencia FastAPI: devuelve la instancia Singleton del predictor.

    Importante: NO devuelve un nuevo objeto cada vez. __new__ garantiza
    que la misma instancia se reutiliza para todos los requests.
    """
    return ModelPredictor()
