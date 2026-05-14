"""
Configuración pytest compartida.

Crea un cliente FastAPI con BD en memoria y override de Depends(get_db),
para que los tests no toquen la BD real ni APIs externas (yfinance, FRED).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Asegura que la raíz del proyecto esté en sys.path para `from backend.app.main import app`
_PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# Variables de entorno predecibles para los tests (no dependen de .env real)
os.environ.setdefault("JWT_SECRET", "test_secret_for_pytest_only")
os.environ.setdefault("JWT_TTL_MINUTES", "60")
# FRED_API_KEY queda vacía → fred_service usa fallback DEMO


@pytest.fixture(scope="session")
def app():
    """Importa la app FastAPI ya configurada."""
    from backend.app.main import app as fastapi_app
    return fastapi_app


@pytest.fixture(scope="session")
def client(app):
    """TestClient compartido para toda la sesión de tests."""
    from fastapi.testclient import TestClient
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from backend.app.models.db_models import Base
    from backend.app.database import get_db

    # BD SQLite en memoria — aislada y rápida
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    TestSessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    Base.metadata.create_all(bind=engine)

    def _override_get_db():
        db = TestSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = _override_get_db
    return TestClient(app)
