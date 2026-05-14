"""
Predictor ML servido vía /predict con patrón Singleton.

El modelo entrenado se carga UNA sola vez en memoria al primer request o al
arranque del servidor. Cada llamada subsecuente reutiliza la misma instancia,
evitando el costo de joblib.load por petición.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np

MODEL_PATH = Path(__file__).parent / "model.joblib"


class ModelPredictor:
    """Singleton que mantiene el modelo en memoria.

    `__new__` garantiza que solo exista una instancia y que la carga del
    modelo ocurra una única vez. Si el archivo no existe, `model` queda en
    None y el endpoint debe responder con HTTP 503.
    """

    _instance: Optional["ModelPredictor"] = None
    _model:    Any = None
    _metadata: Optional[dict] = None
    _loaded:   bool = False

    def __new__(cls) -> "ModelPredictor":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        """Intenta cargar el modelo desde disco. Solo se ejecuta una vez."""
        if self._loaded:
            return
        try:
            import joblib
            if MODEL_PATH.exists():
                bundle = joblib.load(MODEL_PATH)
                if isinstance(bundle, dict) and "model" in bundle:
                    type(self)._model    = bundle["model"]
                    type(self)._metadata = bundle.get("metadata", {})
                else:
                    # Bundle legacy (solo el modelo, sin metadata)
                    type(self)._model    = bundle
                    type(self)._metadata = {"model_version": "unknown"}
                print(f"[ModelPredictor] modelo cargado desde {MODEL_PATH.name} "
                      f"(version {type(self)._metadata.get('model_version', '?')})")
            else:
                print(f"[ModelPredictor] {MODEL_PATH} no existe — /predict responderá 503")
        except Exception as exc:
            print(f"[ModelPredictor] error cargando modelo: {exc}")
        finally:
            type(self)._loaded = True

    # ── API pública ─────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    @property
    def model_version(self) -> str:
        return (self._metadata or {}).get("model_version", "unknown")

    @property
    def metadata(self) -> dict:
        return dict(self._metadata or {})

    def feature_columns(self) -> list[str]:
        return list((self._metadata or {}).get("feature_cols", []))

    def labels(self) -> dict:
        return dict((self._metadata or {}).get("labels", {}))

    def predict(self, features: np.ndarray) -> tuple[int, Optional[list[float]]]:
        """Predice la clase. Retorna (label_int, probabilidades_o_None)."""
        if not self.is_ready:
            raise RuntimeError("Modelo no cargado")
        X = np.asarray(features, dtype=float).reshape(1, -1)
        pred = int(self._model.predict(X)[0])
        probs: Optional[list[float]] = None
        if hasattr(self._model, "predict_proba"):
            try:
                probs = [round(float(p), 6) for p in self._model.predict_proba(X)[0]]
            except Exception:
                probs = None
        return pred, probs


def get_predictor() -> ModelPredictor:
    """Dependencia inyectable para FastAPI: retorna la instancia compartida."""
    return ModelPredictor()
