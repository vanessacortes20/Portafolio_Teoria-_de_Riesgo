"""
Capa de repositorio sobre SQLAlchemy.

Las funciones de usuario y tokens mantienen las firmas que usaba la version
sqlite3 cruda (api/database.py), por lo que api/main.py no necesita cambiar
sus imports. Internamente cada funcion abre su propia sesion via
SessionLocal, igual que el contextmanager _conn() previo.
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterator, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from api.db.base import SessionLocal, init_db as _init_db_tables
from api.db.models import Portfolio, ResetToken, User

# Exporte del path de users.json (mantengo la misma ruta que la version antigua)
USERS_JSON = Path(__file__).resolve().parent.parent.parent / "data" / "users.json"


@contextmanager
def _session_scope() -> Iterator[Session]:
    """Abre una sesion, hace commit al final o rollback si hubo excepcion."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def _user_to_dict(u: User, *, public: bool = False) -> dict:
    """Serializa un objeto User a dict (compatible con la API antigua)."""
    data = {
        "id": u.id,
        "username": u.username,
        "email": u.email,
        "hashed_password": u.hashed_password,
        "full_name": u.full_name,
        "last_name": u.last_name,
        "phone": u.phone,
        "cedula": u.cedula,
        "is_active": u.is_active,
        "role": u.role,
        "created_at": u.created_at,
        "last_login": u.last_login,
    }
    if public:
        data.pop("hashed_password", None)
    return data


def _export_users_json() -> None:
    """Regenera users.json (sin contrasenas) cada vez que el registro cambia."""
    users = get_all_users()
    USERS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(USERS_JSON, "w", encoding="utf-8") as f:
        json.dump(
            {
                "exported_at": datetime.utcnow().isoformat(timespec="seconds"),
                "total": len(users),
                "users": users,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


# ── Inicializacion ───────────────────────────────────────────────────────────


def init_db() -> None:
    """Crea tablas si no existen. Idempotente."""
    _init_db_tables()


def seed_demo_users(hash_fn: Callable[[str], str]) -> None:
    """Inserta admin/demo solo si la tabla de usuarios esta vacia."""
    with _session_scope() as db:
        existing = db.scalar(select(User).limit(1))
        if existing is not None:
            return
        now = datetime.utcnow().isoformat(timespec="seconds")
        db.add_all(
            [
                User(
                    username="admin",
                    email="admin@risklab.usta",
                    hashed_password=hash_fn("Admin2025!"),
                    full_name="Administrador",
                    last_name="Sistema",
                    phone="0000000000",
                    cedula=None,
                    role="admin",
                    created_at=now,
                ),
                User(
                    username="demo",
                    email="demo@risklab.usta",
                    hashed_password=hash_fn("Demo2025!"),
                    full_name="Usuario",
                    last_name="Demo",
                    phone="1111111111",
                    cedula=None,
                    role="user",
                    created_at=now,
                ),
            ]
        )
    _export_users_json()


# ── Consultas de usuario ─────────────────────────────────────────────────────


def get_user_by_username(username: str) -> Optional[dict]:
    with _session_scope() as db:
        u = db.scalar(select(User).where(User.username == username))
        return _user_to_dict(u) if u else None


def get_user_by_email(email: str) -> Optional[dict]:
    with _session_scope() as db:
        u = db.scalar(select(User).where(User.email == email))
        return _user_to_dict(u) if u else None


def get_user_by_cedula(cedula: str) -> Optional[dict]:
    with _session_scope() as db:
        u = db.scalar(select(User).where(User.cedula == cedula))
        return _user_to_dict(u) if u else None


def get_user_by_id(user_id: int) -> Optional[dict]:
    with _session_scope() as db:
        u = db.get(User, user_id)
        return _user_to_dict(u) if u else None


def get_all_users() -> list[dict]:
    """Devuelve todos los usuarios SIN la contrasena hasheada."""
    with _session_scope() as db:
        rows = db.scalars(select(User).order_by(User.created_at.desc())).all()
        return [_user_to_dict(u, public=True) for u in rows]


def create_user(
    username: str,
    email: str,
    hashed_password: str,
    full_name: str = "",
    last_name: str = "",
    phone: str = "",
    cedula: Optional[str] = None,
    role: str = "user",
) -> dict:
    now = datetime.utcnow().isoformat(timespec="seconds")
    with _session_scope() as db:
        u = User(
            username=username,
            email=email,
            hashed_password=hashed_password,
            full_name=full_name,
            last_name=last_name,
            phone=phone,
            cedula=cedula,
            role=role,
            created_at=now,
        )
        db.add(u)
        db.flush()
        result = _user_to_dict(u)
    _export_users_json()
    return result


def update_user_password(user_id: int, hashed_password: str) -> None:
    with _session_scope() as db:
        u = db.get(User, user_id)
        if u is not None:
            u.hashed_password = hashed_password


def update_last_login(user_id: int) -> None:
    now = datetime.utcnow().isoformat(timespec="seconds")
    with _session_scope() as db:
        u = db.get(User, user_id)
        if u is not None:
            u.last_login = now
    _export_users_json()


# ── Tokens de restablecimiento ───────────────────────────────────────────────


def save_reset_token(user_id: int, token: str, expires_at: str) -> None:
    """Invalida tokens previos del usuario y guarda uno nuevo."""
    with _session_scope() as db:
        previous = db.scalars(
            select(ResetToken).where(ResetToken.user_id == user_id)
        ).all()
        for t in previous:
            t.used = 1
        db.add(ResetToken(user_id=user_id, token=token, expires_at=expires_at))


def get_reset_token(token: str) -> Optional[dict]:
    """
    Retorna el token con datos del usuario asociado (manteniendo el shape
    que esperaba api/main.py: claves token, user_id, expires_at, email,
    username, used).
    """
    with _session_scope() as db:
        row = db.execute(
            select(
                ResetToken.id,
                ResetToken.user_id,
                ResetToken.token,
                ResetToken.expires_at,
                ResetToken.used,
                User.email,
                User.username,
            )
            .join(User, User.id == ResetToken.user_id)
            .where(ResetToken.token == token, ResetToken.used == 0)
        ).first()
        if row is None:
            return None
        return {
            "id": row.id,
            "user_id": row.user_id,
            "token": row.token,
            "expires_at": row.expires_at,
            "used": row.used,
            "email": row.email,
            "username": row.username,
        }


def mark_token_used(token: str) -> None:
    with _session_scope() as db:
        t = db.scalar(select(ResetToken).where(ResetToken.token == token))
        if t is not None:
            t.used = 1


# ── Portafolios ──────────────────────────────────────────────────────────────
#
# A diferencia de las funciones de usuario (que abren su propia sesion para
# preservar compatibilidad con la API antigua), las de portafolios reciben
# la Session por parametro. Esto encaja con el patron moderno de FastAPI:
#   db: Session = Depends(get_db)
#   pf = create_portfolio(db, name=..., weights=...)


def _portfolio_to_dict(p: Portfolio) -> dict:
    return {
        "id":          p.id,
        "user_id":     p.user_id,
        "name":        p.name,
        "weights":     p.weights,
        "description": p.description,
        "created_at":  p.created_at.isoformat() if p.created_at else None,
    }


def create_portfolio(
    db: Session,
    name: str,
    weights: dict,
    user_id: Optional[int] = None,
    description: Optional[str] = None,
) -> dict:
    """Crea un portafolio persistido y devuelve su representacion serializable."""
    p = Portfolio(
        user_id=user_id,
        name=name,
        weights=weights,
        description=description,
    )
    db.add(p)
    db.commit()
    db.refresh(p)
    return _portfolio_to_dict(p)


def get_portfolio_by_id(db: Session, portfolio_id: int) -> Optional[dict]:
    p = db.get(Portfolio, portfolio_id)
    return _portfolio_to_dict(p) if p else None


def list_portfolios(db: Session, user_id: Optional[int] = None) -> list[dict]:
    """Lista portafolios. Si user_id se especifica, filtra por propietario."""
    stmt = select(Portfolio).order_by(Portfolio.created_at.desc())
    if user_id is not None:
        stmt = stmt.where(Portfolio.user_id == user_id)
    rows = db.scalars(stmt).all()
    return [_portfolio_to_dict(p) for p in rows]


def update_portfolio(
    db: Session,
    portfolio_id: int,
    name: Optional[str] = None,
    weights: Optional[dict] = None,
    description: Optional[str] = None,
) -> Optional[dict]:
    """Actualiza un portafolio. Devuelve None si no existe."""
    p = db.get(Portfolio, portfolio_id)
    if p is None:
        return None
    if name is not None:
        p.name = name
    if weights is not None:
        p.weights = weights
    if description is not None:
        p.description = description
    db.commit()
    db.refresh(p)
    return _portfolio_to_dict(p)


def delete_portfolio(db: Session, portfolio_id: int) -> bool:
    """Borra un portafolio. Devuelve True si existia, False si no."""
    p = db.get(Portfolio, portfolio_id)
    if p is None:
        return False
    db.delete(p)
    db.commit()
    return True
