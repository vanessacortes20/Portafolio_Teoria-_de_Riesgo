"""
Motor de SQLAlchemy y dependencia get_db() para inyeccion en FastAPI.

El engine y SessionLocal se construyen una sola vez al importar el modulo.
get_db() es un generador que abre una sesion por request y la cierra al
finalizar, evitando fugas de conexiones.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from api.config import get_settings


class Base(DeclarativeBase):
    """Clase base para todos los modelos ORM del proyecto."""

    pass


_settings = get_settings()


def _resolve_sqlite_path(url: str) -> str:
    """
    Convierte rutas relativas de SQLite a absolutas para evitar problemas
    cuando uvicorn se inicia desde una carpeta distinta a la raiz del repo.
    """
    if not url.startswith("sqlite:///") or url.startswith("sqlite:////"):
        return url
    rel = url.removeprefix("sqlite:///")
    if rel.startswith("./") or not Path(rel).is_absolute():
        rel = rel.lstrip("./")
        abs_path = (Path(__file__).resolve().parent.parent.parent / rel).as_posix()
        return f"sqlite:///{abs_path}"
    return url


DATABASE_URL = _resolve_sqlite_path(_settings.database_url)

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
    future=True,
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
    future=True,
)


def get_db() -> Iterator[Session]:
    """
    Dependencia FastAPI: abre una sesion por request y la cierra al final.

    Uso:
        @app.get('/foo')
        def foo(db: Session = Depends(get_db)):
            ...
    """
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """
    Crea las tablas que no existen. Idempotente.

    Importa api.db.models para registrar todas las clases en Base.metadata
    antes de llamar a create_all().
    """
    from api.db import models  # noqa: F401  -- side effect: registrar modelos

    Base.metadata.create_all(bind=engine)
