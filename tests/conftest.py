"""
Fixtures de pytest para la suite del proyecto.

Estrategia:
  - Base de datos: SQLite en memoria con StaticPool (todas las
    conexiones comparten la misma BD).
  - Las tablas se crean una vez al inicio de la suite via
    Base.metadata.create_all().
  - La dependencia get_db de FastAPI se sobreescribe para que apunte
    al engine de tests.
  - El TestClient se construye una sola vez por test (fixture function).

Las pruebas de endpoints que requieren llamadas externas (yfinance,
FRED) se acotan al camino feliz que no toca la red, validando solo
el contrato de entrada/salida y los errores 4xx.
"""
from __future__ import annotations

import os

# Forzar variables de entorno antes de importar la app (algunas se leen
# en el import-time via Settings).
os.environ.setdefault("JWT_SECRET", "test-secret-key-do-not-use-in-prod")
os.environ.setdefault("FRED_API_KEY", "")

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from api.db import get_db
from api.db.base import Base
from api.main import app

# ── Engine y session de pruebas (in-memory, compartido) ──────────────────────
_test_engine = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
    future=True,
)
_TestSessionLocal = sessionmaker(
    bind=_test_engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
    future=True,
)

# Crear todas las tablas en el engine de tests
Base.metadata.create_all(bind=_test_engine)


def _override_get_db():
    db = _TestSessionLocal()
    try:
        yield db
    finally:
        db.close()


# Override permanente para toda la suite
app.dependency_overrides[get_db] = _override_get_db


# ── Fixtures expuestos a los tests ──────────────────────────────────────────


@pytest.fixture(scope="function")
def client() -> TestClient:
    """TestClient apuntando a la app con BD en memoria."""
    return TestClient(app)


@pytest.fixture(scope="function")
def db_session():
    """Sesion SQLAlchemy directa a la BD de pruebas."""
    db = _TestSessionLocal()
    try:
        yield db
    finally:
        db.close()
