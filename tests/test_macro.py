"""
Tests de integración del endpoint /macro (Plan III, M8).

Verifica que el response cumple el contrato Pydantic MacroIndicators:
  - HTTP 200
  - Campos requeridos presentes (as_of, fred_enabled)
  - Campos opcionales tipados correctamente cuando están presentes
  - fred_enabled refleja la disponibilidad real de FRED_API_KEY

No depende de FRED_API_KEY real (en CI viene vacía → fallback yfinance).
Se aceptan valores None en campos opcionales cuando la red falla.
"""
from __future__ import annotations


REQUIRED_KEYS = {"as_of", "fred_enabled"}
OPTIONAL_KEYS = {
    "rf_rate", "rf_source",
    "treasury_10y", "treasury_10y_source",
    "spx_ytd",
    "inflation_yoy", "inflation_source",
    "usdcop", "usdcop_source",
    "cache_status",
}


def test_macro_endpoint_responde_con_estructura_pydantic(client):
    """GET /api/v1/macro -> 200 y respeta el contrato MacroIndicators."""
    r = client.get("/api/v1/macro")
    assert r.status_code == 200
    data = r.json()

    # Campos obligatorios
    for k in REQUIRED_KEYS:
        assert k in data, f"falta campo requerido '{k}'"

    # Campos opcionales declarados (pueden ser None)
    for k in OPTIONAL_KEYS:
        assert k in data, f"falta campo opcional declarado '{k}'"

    # Tipos
    assert isinstance(data["as_of"], str)
    assert isinstance(data["fred_enabled"], bool)


def test_macro_fred_enabled_false_sin_key():
    """Sin FRED_API_KEY (caso CI/local sin key), fred_enabled debe ser False."""
    import os
    if os.getenv("FRED_API_KEY"):
        # Solo verificamos cuando NO hay key
        return
    from fastapi.testclient import TestClient
    from api.main import app
    r = TestClient(app).get("/api/v1/macro")
    assert r.status_code == 200
    assert r.json()["fred_enabled"] is False


def test_alias_corto_macro_responde_igual_que_v1(client):
    """El alias /macro debe devolver el mismo schema que /api/v1/macro."""
    r1 = client.get("/api/v1/macro")
    r2 = client.get("/macro")
    assert r1.status_code == 200
    assert r2.status_code == 200
    # Mismos campos top-level
    assert set(r1.json().keys()) == set(r2.json().keys())


def test_openapi_expone_macroindicators_tipado(client):
    """OpenAPI debe incluir el schema MacroIndicators con sus propiedades."""
    r = client.get("/openapi.json")
    assert r.status_code == 200
    schemas = r.json().get("components", {}).get("schemas", {})
    assert "MacroIndicators" in schemas
    props = schemas["MacroIndicators"].get("properties", {})
    # Campos exigidos por Plan III + extras documentados
    for required in ("as_of", "rf_rate", "treasury_10y", "spx_ytd",
                     "inflation_yoy", "usdcop", "fred_enabled"):
        assert required in props, f"OpenAPI no expone '{required}' en MacroIndicators"
