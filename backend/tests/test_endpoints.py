"""
Tests de integración con TestClient — validan contratos de los endpoints
sin tocar yfinance ni FRED. La BD usada es SQLite en memoria (ver conftest).
"""
from __future__ import annotations


# ── 4. /docs responde 200 ───────────────────────────────────────────────────

def test_docs_disponible(client):
    """La documentación de FastAPI debe estar accesible."""
    res = client.get("/docs")
    assert res.status_code == 200
    assert "swagger" in res.text.lower() or "openapi" in res.text.lower()


def test_openapi_json(client):
    """El esquema OpenAPI debe servirse correctamente."""
    res = client.get("/openapi.json")
    assert res.status_code == 200
    data = res.json()
    assert "paths" in data
    # Verifica que los endpoints clave estén registrados
    expected = [
        "/api/v1/curva-rendimiento",
        "/api/v1/bono/duracion",
        "/api/v1/opcion/precio",
        "/api/v1/stress",
        "/api/v1/predict",
    ]
    for p in expected:
        assert p in data["paths"], f"Falta endpoint en OpenAPI: {p}"


# ── 5. /opcion/precio con T=0 retorna 422 ──────────────────────────────────

def test_opcion_T_cero_retorna_422(client):
    res = client.post("/api/v1/opcion/precio", json={
        "S": 100, "K": 100, "T": 0, "r": 0.05, "sigma": 0.20,
        "option_type": "call",
    })
    assert res.status_code == 422


def test_opcion_call_atm_clasica(client):
    """Caso de referencia BS: S=K=100, T=1, r=5%, σ=20% → call ≈ 10.4506."""
    res = client.post("/api/v1/opcion/precio", json={
        "S": 100, "K": 100, "T": 1, "r": 0.05, "sigma": 0.20,
        "option_type": "call",
    })
    assert res.status_code == 200
    data = res.json()
    assert abs(data["price"] - 10.4506) < 0.001
    # Las 5 Greeks deben estar presentes
    for g in ("delta", "gamma", "vega", "theta", "rho"):
        assert g in data["greeks"]
    # La paridad put-call debe cumplirse
    assert data["put_call_parity_check"]["satisfied"] is True


# ── 6. /predict retorna esquema válido si el modelo existe ──────────────────

def test_predict_schema_si_modelo_existe(client):
    """Si el modelo no está disponible debe devolver 503; si está, schema válido."""
    info = client.get("/api/v1/predict/info")
    assert info.status_code == 200
    info_data = info.json()

    if not info_data.get("is_ready"):
        # Sin modelo: /predict debe responder 503 con detail
        res = client.post("/api/v1/predict", json={
            "ticker": "AMZN",
            "features": [0.005, -0.002, 0.001, 55.0, 0.0001, 0.012, 0.018],
        })
        assert res.status_code == 503
        return

    # Con modelo: schema esperado
    n_features = len(info_data.get("feature_cols", []))
    res = client.post("/api/v1/predict", json={
        "ticker":   "AMZN",
        "features": [0.005, -0.002, 0.001, 55.0, 0.0001, 0.012, 0.018][:n_features],
    })
    assert res.status_code == 200
    data = res.json()
    for k in ("ticker", "prediction", "prediction_label",
              "model_version", "features_used", "interpretation", "warning"):
        assert k in data, f"Falta campo en /predict response: {k}"


def test_predict_features_invalidas_retorna_422(client):
    """Features con NaN deben rechazarse en validación Pydantic."""
    res = client.post("/api/v1/predict", json={
        "ticker": "AMZN",
        "features": [None, 0, 0, 50, 0, 0.01, 0.02],
    })
    assert res.status_code == 422


# ── 7. /stress retorna estructura válida ────────────────────────────────────

def test_stress_estructura_valida(client):
    res = client.post("/api/v1/stress", json={
        "weights": {"A": 0.6, "B": 0.4},
        "prices":  {"A": 100, "B": 50},
    })
    assert res.status_code == 200
    data = res.json()
    for k in ("base_portfolio_value", "scenarios_run", "scenario_summary",
              "worst_scenario", "risk_interpretation"):
        assert k in data
    assert data["scenarios_run"] >= 4
    # Cada escenario debe tener loss y nombre
    for s in data["scenario_summary"]:
        assert "scenario_name"  in s
        assert "percentage_loss" in s
        assert "stressed_value" in s


def test_stress_pesos_invalidos_retorna_422(client):
    """Pesos que no suman 1 deben rechazarse."""
    res = client.post("/api/v1/stress", json={
        "weights": {"A": 0.5, "B": 0.3},  # suma = 0.8
        "prices":  {"A": 100, "B": 50},
    })
    assert res.status_code == 422


# ── Bono adicional ──────────────────────────────────────────────────────────

def test_bono_al_par(client):
    """Bono cupón = yield → precio debe = face_value."""
    res = client.post("/api/v1/bono/duracion", json={
        "face_value": 1000, "coupon_rate": 0.05, "maturity_years": 10,
        "yield_rate": 0.05, "frequency": 2,
    })
    assert res.status_code == 200
    data = res.json()
    assert abs(data["bond_price"] - 1000.0) < 0.01
    assert data["macaulay_duration"] > 0
    assert data["convexity"] > 0


def test_bono_frequency_invalida_retorna_422(client):
    res = client.post("/api/v1/bono/duracion", json={
        "face_value": 1000, "coupon_rate": 0.05, "maturity_years": 10,
        "yield_rate": 0.05, "frequency": 3,
    })
    assert res.status_code == 422


# ── Curva de rendimiento (usa fallback DEMO sin FRED key) ───────────────────

def test_curva_rendimiento_responde(client):
    """Sin FRED_API_KEY el endpoint debe responder con curva DEMO."""
    res = client.get("/api/v1/curva-rendimiento")
    assert res.status_code == 200
    data = res.json()
    assert data["source"] in ("FRED", "fallback_demo")
    assert len(data["points"]) >= 4
    assert "nelson_siegel" in data
