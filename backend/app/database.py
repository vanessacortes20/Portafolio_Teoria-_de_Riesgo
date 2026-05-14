"""
Sesión SQLAlchemy compartida con la BD existente de RiskLab USTA.

Usa el mismo archivo SQLite que api/database.py (data/risklab_users.db) para
no duplicar la persistencia. Las tablas nuevas convivien con las existentes
(users, reset_tokens) sin tocarlas.
"""
from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from backend.app.models.db_models import Base

# BD en la raíz del proyecto (compartida con backend/app/auth_db.py).
# __file__ = backend/app/database.py → parent.parent.parent = raíz del proyecto
_PROJECT_ROOT = Path(__file__).parent.parent.parent
_DB_PATH = _PROJECT_ROOT / "data" / "risklab_users.db"
_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

DATABASE_URL = f"sqlite:///{_DB_PATH.as_posix()}"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    future=True,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, future=True)


def init_orm_tables() -> None:
    """Crea las tablas ORM si no existen. Idempotente.

    No toca tablas existentes (users, reset_tokens) creadas por sqlite3 directo
    en api/database.py.
    """
    Base.metadata.create_all(bind=engine)


def get_db() -> Session:
    """Generador de sesión para inyección con Depends(get_db)."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
