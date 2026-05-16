"""
Shim de compatibilidad.

La implementacion real de la capa de persistencia vive ahora en api/db/.
Este modulo se conserva como un re-export para que el codigo existente
(api/main.py, generate_data.py, scripts) siga funcionando sin cambios.

Para codigo nuevo se recomienda importar directamente desde api.db.
"""
from api.db.repository import (
    USERS_JSON,
    create_user,
    get_all_users,
    get_reset_token,
    get_user_by_cedula,
    get_user_by_email,
    get_user_by_id,
    get_user_by_username,
    init_db,
    mark_token_used,
    save_reset_token,
    seed_demo_users,
    update_last_login,
    update_user_password,
)

# Ruta heredada que algunos scripts antiguos podrian leer directamente.
# Apunta al archivo SQLite real configurado en api.config.Settings.
from pathlib import Path as _Path  # noqa: E402
from api.config import get_settings as _get_settings  # noqa: E402


def _legacy_db_path() -> _Path:
    url = _get_settings().database_url
    if url.startswith("sqlite:///"):
        return _Path(url.removeprefix("sqlite:///"))
    return _Path("data/risklab_users.db")


DB_PATH = _legacy_db_path()

# Garantiza que la carpeta data/ exista (la BD y users.json no se versionan)
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


__all__ = [
    "DB_PATH",
    "USERS_JSON",
    "init_db",
    "seed_demo_users",
    "get_user_by_username",
    "get_user_by_email",
    "get_user_by_cedula",
    "get_user_by_id",
    "get_all_users",
    "create_user",
    "update_user_password",
    "update_last_login",
    "save_reset_token",
    "get_reset_token",
    "mark_token_used",
]
