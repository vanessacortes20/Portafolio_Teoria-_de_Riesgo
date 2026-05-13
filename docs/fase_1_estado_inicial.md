# Fase 1 — Estado Inicial del Proyecto

**Proyecto:** RiskLab USTA — Plataforma de Análisis Cuantitativo de Riesgo Financiero
**Fase:** 1 — Auditoría inicial, validación de arquitectura y protección de lo existente
**Fecha de ejecución:** 2026-05-12 (martes, noche)
**Rama de trabajo:** `feature/instructivo-iii`
**Rama base:** `main` (sincronizada con `origin/main`)
**Ejecutor:** Asistente Claude (modo ejecución directa)

---

## 1. Resumen del estado actual

El proyecto está **funcionalmente operativo**. Tras corregir incompatibilidades del entorno Python (pandas/numpy), el backend FastAPI arranca sin errores, los 14 endpoints de negocio responden con HTTP 200 y datos válidos, la autenticación JWT funciona con las credenciales reales documentadas, el dashboard HTML existe con su snapshot estático válido, y los scripts auxiliares importan correctamente.

**Veredicto:** la línea base es **defensible y reproducible**. El proyecto está listo para iniciar Fase 2.

---

## 2. Arquitectura confirmada

```
FastAPI backend (api/) ⟶ API REST en :8001 ⟶ frontend HTML/JS/Plotly (dashboard/)
                  │
                  ├── api/main.py        — endpoints + auth + Pydantic
                  ├── api/logic.py       — cálculos puros M1–M8
                  ├── api/data.py        — descarga yfinance
                  ├── api/database.py    — SQLite directo (sqlite3)
                  └── data/              — risklab_users.db + users.json (no versionados)
```

**Decisión arquitectónica obligatoria respetada:** el frontend sigue siendo HTML + JS + Plotly. La lógica financiera vive en FastAPI.

---

## 3. Archivos principales detectados

### Raíz del proyecto
- `README.md` (28 KB)
- `CLAUDE.md` (memoria local, no versionada)
- `requirements.txt`
- `Dockerfile` (single-stage)
- `Procfile` (Heroku/Railway)
- `render.yaml`
- `.env.example`
- `.gitignore`
- `generate_data.py` (script principal)

### Carpetas
| Carpeta | Contenido | Estado |
|---|---|---|
| `api/` | `main.py`, `logic.py`, `data.py`, `database.py`, `__init__.py` | ✅ OK |
| `dashboard/` | `dashboard.html` (217 KB), `data.js` (1.63 MB) | ✅ OK |
| `data/` | `risklab_users.db`, `users.json` (locales, gitignored) | ✅ OK |
| `docs/` | 4 HTML: instructivo viejo, instructivo nuevo (Python III), 2 auditorías | ✅ OK |
| `tests/` | `test_yf.py` — solo smoke test | ⚠️ Falta pytest formal |

---

## 4. Endpoints detectados (22 totales)

### Documentación (4)
- `GET /` · `GET /docs` · `GET /redoc` · `GET /openapi.json`

### Autenticación (7)
- `POST /auth/register`
- `POST /auth/login`
- `GET /auth/me`
- `POST /auth/change-password`
- `POST /auth/reset-password`
- `POST /auth/reset-password/confirm`
- `GET /auth/users` (admin)

### Análisis financiero M1–M8 (10)
- `GET /api/v1/technical/{ticker}` (M1)
- `GET /api/v1/returns/{ticker}` (M2)
- `GET /api/v1/volatility/{ticker}` (M3)
- `GET /api/v1/risk/{ticker}` (M4 + M5)
- `GET /api/v1/risk/{ticker}/backtest` (M5 Kupiec)
- `GET /api/v1/portfolio/optimize` (M6 simulación)
- `GET /api/v1/portfolio/target` (M6 SLSQP)
- `GET /api/v1/macro` (M8)
- `GET /api/v1/signals/{ticker}` (M7)
- `GET /api/v1/all` (bulk)

**Detalle completo (auth, payloads, response keys, status codes):** [`baseline_endpoints.md`](baseline_endpoints.md)

---

## 5. Dashboard detectado

| Archivo | Tamaño | Estado |
|---|---|---|
| `dashboard/dashboard.html` | 217.6 KB · 3,556 líneas | ✅ OK — Plotly CDN, auth-overlay, 9 tabs M1–M9 |
| `dashboard/data.js` | 1.63 MB | ✅ JSON válido bajo `window.RISKLAB_DATA = {...}` |

- 5 tickers en el snapshot: NU, AMZN, SONY, XOM, WPM
- Generated_at: 2026-04-22 18:39:32
- rf_rate vigente: 3.60% (^IRX)
- API_BASE hardcoded a `http://localhost:8001`
- TOKEN_KEY: `rl_jwt` en localStorage
- Login overlay obligatorio al cargar

> ⚠️ data.js usa `window.RISKLAB_DATA`, no `const DATA` como mencionaba documentación previa.

---

## 6. Tests detectados

| Script | Tipo | Resultado |
|---|---|---|
| `tests/test_yf.py` | Smoke test ad-hoc (no pytest) | ✅ Corre OK como script. Descarga NU desde yfinance. |
| `pytest -v` | — | ❌ pytest no instalado en el entorno |

**Conclusión:** no hay suite de tests formal. Esta brecha se cubrirá en Fase 5 (criterio 12 de la rúbrica).

---

## 7. Dependencias detectadas

### En `requirements.txt`
fastapi · uvicorn · yfinance · pandas · streamlit · python-dotenv · plotly · scipy · statsmodels · arch · numpy · requests · bcrypt · python-jose[cryptography] · email-validator

> ⚠️ `streamlit` aparece en requirements pero **ya no se usa** (se eliminó `dashboard/app.py` en una limpieza anterior). Se puede sacar en Fase 5 sin afectar nada.

### Versiones efectivas tras corrección de Fase 1
| Paquete | Versión |
|---|---|
| Python | 3.11.3 |
| pandas | 3.0.3 (era 2.2.1) |
| numpy | 1.26.4 (downgrade automático por statsmodels) |
| scipy | 1.12.0 |
| arch | 8.0.0 |
| statsmodels | 0.14.6 |

---

## 8. Problemas encontrados

| # | Problema | Severidad | Resuelto en Fase 1 |
|---|---|---|---|
| 1 | pandas 2.2.1 incompatible con numpy 2 — bloqueaba arranque del backend | 🔴 Alta | ✅ Sí (upgrade a 3.0.3) |
| 2 | Login fallaba con `admin123` y `demo1234` (README) — credenciales reales son `Admin2025!` y `Demo2025!` | 🟡 Media | ❌ No (documentado en baseline, fix en Fase 5) |
| 3 | Endpoints `/api/v1/*` no exigen autenticación a pesar de tener `OAuth2PasswordBearer` configurado | 🟡 Media | ❌ No (documentado, decisión arquitectónica para Fase 2/5) |
| 4 | data.js usa `window.RISKLAB_DATA` no `const DATA` — documentación interna desactualizada | 🟢 Baja | ❌ No (documentado, fix en Fase 5) |
| 5 | pytest no instalado | 🟡 Media | ❌ No (planificado para Fase 5) |
| 6 | `streamlit` en requirements.txt pero ya no se usa | 🟢 Baja | ❌ No (limpieza en Fase 5) |
| 7 | Dockerfile single-stage (~600 MB potencial vs 150 MB esperado) | 🟡 Media | ❌ No (Fase 5 lo migra) |

---

## 9. Riesgos para Fase 2

| Riesgo | Probabilidad | Mitigación recomendada |
|---|---|---|
| Migrar a SQLAlchemy y romper login | 🟡 Media | Cohabitación de capas: mantener `api/database.py` (sqlite3) activo durante la migración. Solo eliminar al validar la nueva. |
| Migrar `users` table y perder datos de prueba | 🟡 Media | Hacer backup previo de `data/risklab_users.db` antes de tocar cualquier query. |
| FRED requiere API key gratuita | 🟢 Baja | El servicio debe tener fallback a yfinance. Documentar `FRED_API_KEY` en `.env.example`. |
| FRED rate limit o caída | 🟢 Baja | Cache obligatorio en SQLite con TTL de 24h. |
| Romper la respuesta actual de `/api/v1/macro` al cambiar fuente | 🟡 Media | Mantener exactamente las mismas keys en el response (`as_of, rf_rate, rf_source, treasury_10y, spx_ytd`). Solo cambiar el valor de `rf_source`. |
| sqlalchemy + pandas 3.0.3 incompatibilidad | 🟢 Baja | Probar import en cuanto se agregue. |
| Conflicto de versiones en numpy al añadir más paquetes | 🟡 Media | Pinear versiones en requirements.txt al cerrar Fase 5. |

---

## 10. Recomendación exacta para iniciar Fase 2

### Orden recomendado de tareas

1. **Hacer backup de `data/risklab_users.db`** antes de cualquier cambio (`cp risklab_users.db risklab_users.db.bak`)
2. **Agregar a `requirements.txt`:** `sqlalchemy>=2.0`, `fredapi` (opcional — si se usa requests directo, no hace falta)
3. **Crear `api/db_models.py`** con clases ORM:
   - `Base = declarative_base()`
   - `Asset` (id, ticker, name, sector)
   - `Price` (id, asset_id FK, date, open/high/low/close, volume)
   - `Portfolio` (id, name, weights JSON, created_at)
   - `PredictionLog` (id, model_version, timestamp, ticker, input_features JSON, prediction, actual)
   - `SignalLog` (id, ticker, regla, valor, timestamp)
   - `FredCache` (id, series_id, value, fetched_at)
   - `UserORM` (espejo del User actual — migración separada)
4. **Crear `api/database_session.py`** con `engine`, `SessionLocal`, `get_db()` generator y `Base.metadata.create_all()` en startup event de FastAPI
5. **Inyectar `db: Session = Depends(get_db)`** en endpoints relevantes — empezar por uno de prueba (ej. `/auth/me`) sin migrar la lógica todavía
6. **Crear `api/services/fred_service.py`** con clase `FredService` y cache transparente vía `FredCache`
7. **Modificar `api/main.py`** en el endpoint `/api/v1/macro` para usar FRED con fallback a yfinance — mantener exactamente las mismas keys del response
8. **Migrar funciones de `api/database.py`** a SQLAlchemy en paralelo — cohabitar las dos capas
9. **Validar**: ejecutar las pruebas de baseline (login admin, /auth/me, /api/v1/macro, dashboard carga)
10. **Eliminar el código viejo de sqlite3** SOLO cuando todo lo nuevo funcione idénticamente

### Archivos esperados al terminar Fase 2

- `api/db_models.py` (nuevo)
- `api/database_session.py` (nuevo)
- `api/services/__init__.py` (nuevo)
- `api/services/fred_service.py` (nuevo)
- `api/database.py` (mantenido durante transición, eliminado al final)
- `api/main.py` (modificado — inyección de `get_db`)
- `requirements.txt` (modificado — sqlalchemy)
- `.env.example` (modificado — FRED_API_KEY opcional)

### Bloqueos que NO existen para iniciar Fase 2 ahora

- ✅ Backend arranca limpio (entorno corregido)
- ✅ Endpoints baseline documentados con sus keys exactas
- ✅ Credenciales reales conocidas para validar regresión de login
- ✅ Rama `feature/instructivo-iii` creada
- ✅ Backup de auth implícito (la BD existe en local)
- ✅ Tabla baseline lista para detectar regresiones

**No hay bloqueos. Fase 2 puede iniciar inmediatamente después de aprobar este reporte.**

---

## 11. Preparación para Fase 2

| Pregunta | Respuesta |
|---|---|
| ¿El backend está listo para migrar a SQLAlchemy? | **Sí.** Las tablas actuales (`users`, `reset_tokens`) son simples y se mapean directamente a modelos ORM. La migración no requiere cambios estructurales. |
| ¿Qué archivos deben tocarse primero? | `requirements.txt` → `api/db_models.py` (nuevo) → `api/database_session.py` (nuevo). Después se itera sobre `api/main.py` para inyectar `get_db`. |
| ¿Qué riesgos hay con login y BD? | El riesgo principal es romper la verificación BCrypt al migrar la tabla `users`. **Mitigación:** mantener `api/database.py` original activo durante la transición; solo eliminar cuando la nueva capa pase la validación de baseline. |
| ¿FRED puede integrarse directamente? | **Sí, sin bloqueos.** Requiere obtener API key gratuita en `https://fred.stlouisfed.org/`. El servicio debe tener fallback a yfinance para no romper si falta la key. La key se agrega a `.env` (no al repo). |
| ¿Qué dependencias se deberán agregar? | `sqlalchemy>=2.0` (obligatorio). Para FRED se puede usar `requests` directo (ya está) o `fredapi` (opcional, simplifica). |
| ¿Hay algo que bloquea iniciar Fase 2 hoy? | **No.** Todos los pre-requisitos están cumplidos: backend arranca, baseline documentada, rama lista, credenciales válidas. |

---

## 12. Confirmación: lógica funcional NO modificada

Durante la Fase 1 se ejecutaron únicamente:

✅ Comandos de diagnóstico (curl, pip show, python -c)
✅ Lectura de archivos
✅ Creación de archivos nuevos en `docs/` (este archivo y `baseline_endpoints.md`)
✅ Operaciones git: creación de rama `feature/instructivo-iii`
✅ Correcciones del entorno Python (pip upgrade pandas, statsmodels, arch) — **no afectan archivos del proyecto, solo el entorno local**

❌ NO se modificó `api/main.py`
❌ NO se modificó `api/logic.py`
❌ NO se modificó `api/database.py`
❌ NO se modificó `api/data.py`
❌ NO se modificó `dashboard/dashboard.html`
❌ NO se modificó `dashboard/data.js`
❌ NO se modificó `generate_data.py`
❌ NO se modificó `tests/test_yf.py`
❌ NO se modificó `requirements.txt`, `Dockerfile`, `Procfile`, `render.yaml`
❌ NO se modificó `README.md`
❌ NO se modificó `CLAUDE.md`

**El código funcional del proyecto está intacto.** Solo se agregaron documentos de baseline en `docs/`.

---

*Reporte generado durante la ejecución de Fase 1 del plan de acción del Proyecto Integrador de Teoría del Riesgo Financiero (Python III) — RiskLab USTA · 2026-05-12*
