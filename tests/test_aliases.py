"""
Tests de integración de los aliases cortos en español (Plan III).

El Plan III exige 16 endpoints específicos con paths cortos en español
(Arquitectura del proyecto, sección "Endpoints mínimos requeridos"). Este
test verifica que TODAS las rutas existen en la app y NO retornan 404 /
405 (Method Not Allowed). Acepta códigos de validación (422) o de error
de datos (500) — lo importante es que la ruta esté registrada.
"""
from __future__ import annotations


# Plan III, sección "Endpoints mínimos requeridos"
ALIASES_GET_SIN_PATH_PARAM = [
    "/activos",
    "/portafolios",   # GET listar (requiere auth → 401, no 404)
    "/capm",
    "/alertas",
    "/macro",
    "/curva-rendimiento",
]
ALIASES_GET_CON_TICKER = [
    "/precios/{ticker}",
    "/rendimientos/{ticker}",
    "/indicadores/{ticker}",
    "/volatilidad/{ticker}",
]
ALIASES_POST = [
    "/portafolios",
    "/var",
    "/frontera-eficiente",
    "/bono/duracion",
    "/opcion/precio",
    "/stress",
    "/predict",
]


def _accept_codes(code: int) -> bool:
    """Una ruta existe si NO devuelve 404 ni 405."""
    return code not in (404, 405)


def test_los_seis_aliases_get_existen(client):
    """Las rutas GET sin params del Plan III deben existir."""
    for path in ALIASES_GET_SIN_PATH_PARAM:
        r = client.get(path)
        assert _accept_codes(r.status_code), \
            f"{path} -> HTTP {r.status_code} (debería existir)"


def test_los_cuatro_aliases_con_ticker_existen(client):
    """Las rutas GET con ticker del Plan III deben existir."""
    for path_tpl in ALIASES_GET_CON_TICKER:
        path = path_tpl.format(ticker="AMZN")
        r = client.get(path)
        assert _accept_codes(r.status_code), \
            f"{path} -> HTTP {r.status_code} (debería existir)"


def test_los_siete_aliases_post_existen(client):
    """Las rutas POST del Plan III deben existir (aceptan 401/422 con body vacío)."""
    for path in ALIASES_POST:
        r = client.post(path, json={})
        assert _accept_codes(r.status_code), \
            f"POST {path} -> HTTP {r.status_code} (debería existir)"


def test_alias_alertas_history_existe(client):
    """Endpoint adicional de historial Plan III."""
    r = client.get("/alertas/AMZN/history")
    assert _accept_codes(r.status_code)


def test_endpoints_legacy_api_v1_siguen_funcionando(client):
    """Los endpoints /api/v1/... originales no deben haberse borrado."""
    for path in (
        "/api/v1/macro",
        "/api/v1/yield-curve",
        "/api/v1/all",
    ):
        r = client.get(path)
        assert _accept_codes(r.status_code), \
            f"{path} -> HTTP {r.status_code} (legacy no debe estar muerto)"


def test_total_de_rutas_supera_los_30(client):
    """El backend debe exponer >=30 rutas (aliases + legacy + auth)."""
    r = client.get("/openapi.json")
    assert r.status_code == 200
    paths = r.json().get("paths", {})
    assert len(paths) >= 30, f"solo {len(paths)} rutas registradas"
