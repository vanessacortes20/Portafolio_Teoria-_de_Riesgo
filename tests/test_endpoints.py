"""
Tests de integracion sobre la API. Usan TestClient + BD en memoria
inyectada via conftest.py. No realizan llamadas a Internet.
"""
from __future__ import annotations


def test_root_returns_200(client):
    """GET / debe devolver 200 con el shape esperado."""
    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert body.get("status") == "ok"
    assert "data_range" in body
    assert "min" in body["data_range"]
    assert "max" in body["data_range"]


def test_docs_endpoint_available(client):
    """FastAPI sirve Swagger UI en /docs."""
    r = client.get("/docs")
    assert r.status_code == 200
    assert "swagger" in r.text.lower() or "openapi" in r.text.lower()


def test_option_price_happy_path(client):
    """POST /api/v1/option/price con parametros validos."""
    payload = {
        "S": 100.0, "K": 100.0, "T": 1.0, "r": 0.05, "sigma": 0.20,
        "option_type": "call",
    }
    r = client.post("/api/v1/option/price", json=payload)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["price"] > 0
    assert "greeks" in data
    for g in ("delta", "gamma", "vega", "theta", "rho"):
        assert g in data["greeks"]
    parity = data["put_call_parity"]
    assert parity["valid"] is True


def test_option_price_invalid_sigma_returns_422(client):
    """sigma negativa debe disparar HTTP 422."""
    payload = {
        "S": 100.0, "K": 100.0, "T": 1.0, "r": 0.05, "sigma": -0.20,
        "option_type": "call",
    }
    r = client.post("/api/v1/option/price", json=payload)
    assert r.status_code == 422


def test_option_price_invalid_type_returns_422(client):
    """option_type fuera del Literal debe devolver 422."""
    payload = {
        "S": 100.0, "K": 100.0, "T": 1.0, "r": 0.05, "sigma": 0.20,
        "option_type": "swap",
    }
    r = client.post("/api/v1/option/price", json=payload)
    assert r.status_code == 422


def test_stress_invalid_weights_returns_422(client):
    """
    POST /api/v1/stress con pesos que NO suman 1 debe devolver 422.

    Este test cumple el item obligatorio de la rubrica: validacion en
    boundary con field_validator personalizado.
    """
    payload = {
        "tickers":         ["AAPL", "MSFT"],
        "weights":         {"AAPL": 0.7, "MSFT": 0.5},  # suma 1.2
        "portfolio_value": 100_000.0,
        "confidence":      0.95,
    }
    r = client.post("/api/v1/stress", json=payload)
    assert r.status_code == 422


def test_predict_with_manual_features(client):
    """POST /api/v1/predict con features explicitas (no toca yfinance)."""
    features = {
        "ret_1d":       0.01,
        "ret_5d":       0.02,
        "ret_20d":      0.05,
        "rsi_14":       55.0,
        "macd_hist":    0.10,
        "vol_20d":      0.015,
        "pos_in_bb":    0.50,
        "volume_ratio": 1.1,
    }
    r = client.post(
        "/api/v1/predict",
        json={"ticker": "AAPL", "features": features},
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["ticker"] == "AAPL"
    assert data["prediction"]["action"] in ("Buy", "Hold", "Sell")
    assert data["features_source"] == "user_provided"


def test_predict_info_returns_metadata(client):
    """GET /api/v1/predict/info expone metadata del modelo activo."""
    r = client.get("/api/v1/predict/info")
    assert r.status_code == 200
    body = r.json()
    assert "model_version" in body
    assert "expected_features" in body
    assert isinstance(body["expected_features"], list)
    assert len(body["expected_features"]) == 8
