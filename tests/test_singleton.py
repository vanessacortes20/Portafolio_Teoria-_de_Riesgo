"""
Tests del patron Singleton del ModelPredictor (criterio 11 de la rubrica).

La rubrica exige Singleton verificable: el modelo se carga UNA SOLA VEZ.
"""
from __future__ import annotations

from api.ml import ModelPredictor, get_predictor


def test_two_instantiations_return_same_object():
    """ModelPredictor() llamado dos veces devuelve la misma instancia."""
    a = ModelPredictor()
    b = ModelPredictor()
    assert a is b


def test_get_predictor_returns_singleton_instance():
    """La dependencia get_predictor reutiliza la misma instancia."""
    a = ModelPredictor()
    c = get_predictor()
    assert c is a


def test_predictor_has_expected_features():
    p = ModelPredictor()
    assert isinstance(p.expected_features, list)
    assert len(p.expected_features) == 8
