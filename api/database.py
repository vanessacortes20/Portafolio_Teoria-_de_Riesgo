"""
RiskLab USTA — Capa de persistencia SQLite.
Esquema extendido: nombres, apellidos, teléfono, cédula, último acceso.
Exporta users.json automáticamente en cada modificación del registro.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

DB_PATH      = Path(__file__).parent.parent / "data" / "risklab_users.db"
USERS_JSON   = Path(__file__).parent.parent / "data" / "users.json"


# ── Conexión ─────────────────────────────────────────────────────────────────

@contextmanager
def _conn():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# ── Inicialización y migración ────────────────────────────────────────────────

def init_db() -> None:
    """Crea tablas y aplica migraciones si la BD ya existía. Idempotente."""
    with _conn() as db:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                username         TEXT    UNIQUE NOT NULL,
                email            TEXT    UNIQUE NOT NULL,
                hashed_password  TEXT    NOT NULL,
                full_name        TEXT    NOT NULL DEFAULT '',
                last_name        TEXT    NOT NULL DEFAULT '',
                phone            TEXT    NOT NULL DEFAULT '',
                cedula           TEXT    DEFAULT NULL,
                is_active        INTEGER DEFAULT 1,
                role             TEXT    DEFAULT 'user',
                created_at       TEXT    NOT NULL,
                last_login       TEXT    DEFAULT NULL
            );

            CREATE TABLE IF NOT EXISTS reset_tokens (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id    INTEGER NOT NULL,
                token      TEXT    UNIQUE NOT NULL,
                expires_at TEXT    NOT NULL,
                used       INTEGER DEFAULT 0,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );
        """)

        # Migraciones: agrega columnas nuevas a BD existente sin romper nada
        existing = {row[1] for row in db.execute("PRAGMA table_info(users)")}
        migrations = {
            "full_name":  "ALTER TABLE users ADD COLUMN full_name  TEXT NOT NULL DEFAULT ''",
            "last_name":  "ALTER TABLE users ADD COLUMN last_name  TEXT NOT NULL DEFAULT ''",
            "phone":      "ALTER TABLE users ADD COLUMN phone      TEXT NOT NULL DEFAULT ''",
            "cedula":     "ALTER TABLE users ADD COLUMN cedula     TEXT DEFAULT NULL",
            "last_login": "ALTER TABLE users ADD COLUMN last_login TEXT DEFAULT NULL",
        }
        for col, sql in migrations.items():
            if col not in existing:
                db.execute(sql)

        # Índice único para cédula (seguro de re-ejecutar)
        db.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_users_cedula ON users(cedula) WHERE cedula IS NOT NULL"
        )


def seed_demo_users(hash_fn) -> None:
    """Inserta usuarios demo si la tabla está vacía."""
    with _conn() as db:
        count = db.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        if count > 0:
            return
        now = datetime.utcnow().isoformat(timespec="seconds")
        db.execute(
            """INSERT INTO users
               (username, email, hashed_password, full_name, last_name, phone, cedula, role, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ("admin", "admin@risklab.usta", hash_fn("Admin2025!"),
             "Administrador", "Sistema", "0000000000", None, "admin", now),
        )
        db.execute(
            """INSERT INTO users
               (username, email, hashed_password, full_name, last_name, phone, cedula, role, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ("demo", "demo@risklab.usta", hash_fn("Demo2025!"),
             "Usuario", "Demo", "1111111111", None, "user", now),
        )
    _export_users_json()


# ── Exportación JSON ──────────────────────────────────────────────────────────

def _export_users_json() -> None:
    """Regenera users.json en la raíz del proyecto (sin contraseñas)."""
    users = get_all_users()
    with open(USERS_JSON, "w", encoding="utf-8") as f:
        json.dump(
            {
                "exported_at": datetime.utcnow().isoformat(timespec="seconds"),
                "total":       len(users),
                "users":       users,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


# ── Consultas de usuario ──────────────────────────────────────────────────────

def get_user_by_username(username: str) -> Optional[dict]:
    with _conn() as db:
        row = db.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()
    return dict(row) if row else None


def get_user_by_email(email: str) -> Optional[dict]:
    with _conn() as db:
        row = db.execute(
            "SELECT * FROM users WHERE email = ?", (email,)
        ).fetchone()
    return dict(row) if row else None


def get_user_by_cedula(cedula: str) -> Optional[dict]:
    with _conn() as db:
        row = db.execute(
            "SELECT * FROM users WHERE cedula = ?", (cedula,)
        ).fetchone()
    return dict(row) if row else None


def get_user_by_id(user_id: int) -> Optional[dict]:
    with _conn() as db:
        row = db.execute(
            "SELECT * FROM users WHERE id = ?", (user_id,)
        ).fetchone()
    return dict(row) if row else None


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
    with _conn() as db:
        db.execute(
            """INSERT INTO users
               (username, email, hashed_password, full_name, last_name, phone, cedula, role, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (username, email, hashed_password, full_name, last_name, phone, cedula, role, now),
        )
        row = db.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()
    _export_users_json()
    return dict(row)


def get_all_users() -> list[dict]:
    """Devuelve todos los usuarios sin la contraseña hasheada."""
    with _conn() as db:
        rows = db.execute(
            """SELECT id, username, email, full_name, last_name, phone,
                      cedula, is_active, role, created_at, last_login
               FROM users ORDER BY created_at DESC"""
        ).fetchall()
    return [dict(r) for r in rows]


def update_user_password(user_id: int, hashed_password: str) -> None:
    with _conn() as db:
        db.execute(
            "UPDATE users SET hashed_password = ? WHERE id = ?",
            (hashed_password, user_id),
        )


def update_last_login(user_id: int) -> None:
    now = datetime.utcnow().isoformat(timespec="seconds")
    with _conn() as db:
        db.execute(
            "UPDATE users SET last_login = ? WHERE id = ?",
            (now, user_id),
        )
    _export_users_json()


# ── Tokens de restablecimiento ────────────────────────────────────────────────

def save_reset_token(user_id: int, token: str, expires_at: str) -> None:
    with _conn() as db:
        db.execute(
            "UPDATE reset_tokens SET used = 1 WHERE user_id = ?", (user_id,)
        )
        db.execute(
            """INSERT INTO reset_tokens (user_id, token, expires_at)
               VALUES (?, ?, ?)""",
            (user_id, token, expires_at),
        )


def get_reset_token(token: str) -> Optional[dict]:
    with _conn() as db:
        row = db.execute(
            """SELECT rt.*, u.email, u.username
               FROM reset_tokens rt
               JOIN users u ON u.id = rt.user_id
               WHERE rt.token = ? AND rt.used = 0""",
            (token,),
        ).fetchone()
    return dict(row) if row else None


def mark_token_used(token: str) -> None:
    with _conn() as db:
        db.execute(
            "UPDATE reset_tokens SET used = 1 WHERE token = ?", (token,)
        )
