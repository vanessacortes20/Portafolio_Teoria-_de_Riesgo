# Baseline de endpoints — RiskLab USTA

**Fecha de captura:** 2026-05-12
**Rama:** `feature/instructivo-iii`
**Backend:** `python -m uvicorn api.main:app --port 8001`
**API base:** `http://localhost:8001`

> Este archivo es la **línea base contractual** de los endpoints existentes antes de las modificaciones de Fase 2 en adelante. Cualquier cambio futuro debe preservar el comportamiento aquí documentado o agregar campos sin remover los existentes.

---

## Credenciales reales detectadas

| Usuario | Contraseña real (en código) | Rol |
|---|---|---|
| `admin` | `Admin2025!` | admin |
| `demo`  | `Demo2025!`  | user |

> ⚠️ El README menciona `admin123` y `demo1234`. Las contraseñas reales en `api/database.py:97-105` son `Admin2025!` y `Demo2025!`. **Inconsistencia documental detectada — no corregida en Fase 1.**

---

## Tabla maestra de endpoints (22 totales)

### Documentación y health

| Método | Ruta | Auth | Descripción | Estado |
|---|---|---|---|---|
| GET | `/` | No | Estado de la API y rango de fechas disponible | ✅ OK (HTTP 200, 115 bytes) |
| GET | `/docs` | No | Swagger UI generado por FastAPI | ✅ OK (HTTP 200, 948 bytes) |
| GET | `/redoc` | No | ReDoc UI generado por FastAPI | ✅ OK (no probado, ruta registrada) |
| GET | `/openapi.json` | No | Esquema OpenAPI 3 | ✅ OK (HTTP 200, 18,710 bytes) |

**Response `/`:**
```json
{
  "message": "Bienvenido a RiskLab USTA API v2.0",
  "status": "ok",
  "data_range": {"min": "2020-01-01", "max": "2026-05-12"}
}
```

---

### Autenticación

| Método | Ruta | Auth | Descripción | Estado |
|---|---|---|---|---|
| POST | `/auth/register` | No | Registro de nuevo usuario con validación Pydantic | ✅ OK (no re-probado para no contaminar BD) |
| POST | `/auth/login` | No | Login con username o email → JWT | ✅ OK (HTTP 200 con creds correctas, HTTP 401 con incorrectas) |
| GET | `/auth/me` | Bearer | Perfil del usuario autenticado | ✅ OK (HTTP 200, 210 bytes) |
| POST | `/auth/change-password` | Bearer | Cambio de contraseña | ✅ Ruta registrada (no probado) |
| POST | `/auth/reset-password` | No | Solicita token de reset | ✅ Ruta registrada (no probado) |
| POST | `/auth/reset-password/confirm` | No | Confirma reset con token | ✅ Ruta registrada (no probado) |
| GET | `/auth/users` | Bearer + admin | Lista todos los usuarios (solo rol admin) | ✅ OK (HTTP 200, 2,900 bytes — 13 usuarios) |

**Login payload:**
```
POST /auth/login
Content-Type: application/x-www-form-urlencoded

username=admin&password=Admin2025!
```

**Login response:**
```json
{"access_token": "<JWT 139 chars>", "token_type": "bearer"}
```

**`/auth/me` response keys:**
`id, username, email, full_name, last_name, phone, cedula, role, is_active, created_at, last_login`

---

### Módulos de análisis (M1–M8)

| Método | Ruta | Auth | Descripción | Estado | Bytes |
|---|---|---|---|---|---|
| GET | `/api/v1/technical/{ticker}` | No real | M1 — Análisis técnico (SMA, EMA, RSI, MACD, BB, Stoch) | ✅ HTTP 200 | 213,280 |
| GET | `/api/v1/returns/{ticker}` | No real | M2 — Estadísticas de retornos + normalidad | ✅ HTTP 200 | 51,735 |
| GET | `/api/v1/volatility/{ticker}` | No real | M3 — ARCH/GARCH/EGARCH + best_model + Forecast | ✅ HTTP 200 | 21,831 |
| GET | `/api/v1/risk/{ticker}` | No real | M4+M5 — CAPM + VaR/CVaR + dispersión | ✅ HTTP 200 | 23,741 |
| GET | `/api/v1/risk/{ticker}/backtest?confidence=0.95` | No real | M5 — Backtesting Kupiec POF | ✅ HTTP 200 | 604 |
| GET | `/api/v1/portfolio/optimize` | No real | M6 — Markowitz por simulación (10k portafolios) | ✅ HTTP 200 | 1,214 |
| GET | `/api/v1/portfolio/target?target_return=X` | No real | M6 — Optimización determinista por retorno objetivo (SLSQP) | ✅ HTTP 200 (responde feasible/error) | 150 |
| GET | `/api/v1/macro` | No real | M8 — Indicadores macro (Rf, T10Y, S&P YTD) | ✅ HTTP 200 | 160 |
| GET | `/api/v1/signals/{ticker}` | No real | M7 — Señales técnicas automáticas | ✅ HTTP 200 | 30 |

> ⚠️ **Hallazgo crítico:** los endpoints de `/api/v1/*` **NO requieren autenticación** en el código actual. La validación con `OAuth2PasswordBearer(auto_error=False)` permite que respondan sin token. Esto debe documentarse en CLAUDE.md y considerarse en Fase 2/5 si se quiere endurecer el modelo.

---

## Esquema de respuestas por endpoint (response keys)

### `GET /api/v1/technical/{ticker}`
```
array(n=501) — sample_keys: [Date, Close, High, Low, Open, Volume]
                            (también SMA_20, SMA_50, EMA_20, RSI, MACD_*, BB_*, Stoch_*)
```

### `GET /api/v1/returns/{ticker}`
```
object keys: [ticker, stats, normality, plot_data, qq_data, stylized_facts]
```

### `GET /api/v1/volatility/{ticker}`
```
object keys: [ARCH(1), GARCH(1,1), EGARCH(1,1), best_model, Residuals, Forecast_10d]
```

### `GET /api/v1/risk/{ticker}`
```
object keys: [capm, scatter, var, var_99]
```

### `GET /api/v1/risk/{ticker}/backtest?confidence=0.95`
```
object keys: [ticker, confidence, backtesting]
backtesting: dict { "Historico": {...}, "Parametrico": {...}, "Montecarlo": {...} }
```

### `GET /api/v1/portfolio/optimize`
```
object keys: [Max_Sharpe, Min_Volatility, Correlation]
```

### `GET /api/v1/portfolio/target?target_return=0.15`
```
object keys: [feasible, error]
(Con target_return=0.15 retorna feasible=false — el target es muy alto para los activos)
```

### `GET /api/v1/macro`
```
object keys: [as_of, rf_rate, rf_source, treasury_10y, spx_ytd]
```

### `GET /api/v1/signals/{ticker}`
```
object keys: [ticker, signals]
```

---

## Validación de dependencias del entorno

| Paquete | Versión instalada en el entorno | Notas |
|---|---|---|
| Python | 3.11.3 | OK |
| FastAPI | (requirements no fija versión) | OK |
| Pydantic | v2 | OK |
| pandas | 3.0.3 | Actualizado en Fase 1 (era 2.2.1, incompatible con numpy 2) |
| numpy | 1.26.4 | Downgrade automático por statsmodels — funciona con pandas 3.0.3 |
| scipy | 1.12.0 | OK |
| arch | 8.0.0 | Actualizado en Fase 1 |
| statsmodels | 0.14.6 | Actualizado en Fase 1 |
| yfinance | (no verificado, smoke test OK) | OK |
| python-jose | OK | JWT funciona |
| bcrypt | OK | Login funciona |
| pytest | **NO INSTALADO** | Pendiente para Fase 5 |

---

## Validación del frontend

| Componente | Estado | Notas |
|---|---|---|
| `dashboard/dashboard.html` | ✅ Existe (217.6 KB, 3,556 líneas) | Plotly CDN cargado, auth-overlay presente, 9 tabs M1–M9 |
| `dashboard/data.js` | ✅ Existe (1.63 MB) | JSON válido bajo `window.RISKLAB_DATA = {...}` |
| 5 tickers en data.js | ✅ NU, AMZN, SONY, XOM, WPM | Correcto |
| rf_rate en data.js | ✅ 3.60% | Fuente: `^IRX (3.60%)` (yfinance) |
| Generated_at | 2026-04-22 18:39:32 | Snapshot vigente |
| API_BASE en HTML | ✅ Detectado | Hardcoded a localhost:8001 |
| TOKEN_KEY en HTML | ✅ Detectado | `rl_jwt` en localStorage |

> ⚠️ El `data.js` usa la variable global `window.RISKLAB_DATA`, no `const DATA` como mencionaba documentación previa. Esto debe reflejarse en las herramientas de análisis y en el CLAUDE.md.

---

## Validación de scripts

| Script | Estado | Comando ejecutado |
|---|---|---|
| `generate_data.py` | ✅ Importa sin error | `python -c "import generate_data"` (no se corrió completo para no sobreescribir el snapshot) |
| `tests/test_yf.py` | ✅ Corre como script | `python tests/test_yf.py` — descarga NU OK con yfinance |
| `pytest -v` | ❌ No disponible | pytest no instalado en el entorno |

---

## Hallazgos consolidados (no corregidos en Fase 1)

| # | Hallazgo | Severidad | Acción recomendada |
|---|---|---|---|
| 1 | `/api/v1/*` no exigen auth — `OAuth2PasswordBearer(auto_error=False)` | Media | Endurecer en Fase 2/5 si se quiere proteger |
| 2 | README documenta `admin123/demo1234` pero código usa `Admin2025!/Demo2025!` | Baja | Actualizar README en Fase 5 |
| 3 | data.js usa `window.RISKLAB_DATA` no `const DATA` | Baja | Actualizar CLAUDE.md y guion |
| 4 | pytest no instalado — solo smoke test ad-hoc | Media | Fase 5 lo agrega como capa formal |
| 5 | `/api/v1/portfolio/target` con target=0.15 devuelve `feasible=false` | Informativo | Verificar si es esperado (rango histórico de retornos del portafolio) |
| 6 | data.js generado 2026-04-22 (no es de hoy) | Baja | Regenerar con `python generate_data.py` cuando convenga |
| 7 | Dockerfile single-stage | Media | Fase 5 lo migra a multi-stage |

---

## Cambios técnicos hechos en Fase 1 (correcciones mínimas de entorno)

> Estos cambios afectan **solo el entorno Python local**, NO el código del proyecto.

| Acción | Comando | Razón |
|---|---|---|
| Upgrade pandas | `pip install --upgrade --user pandas` (2.2.1 → 3.0.3) | Resolver `ValueError: numpy.dtype size changed` que impedía importar pandas |
| Upgrade statsmodels + arch | `pip install --upgrade --user statsmodels arch` | Compatibilidad con pandas 3.0.3 — downgradeó numpy a 1.26.4 |

**Archivos del proyecto modificados en Fase 1:** ninguno aparte de los nuevos archivos de documentación (`docs/baseline_endpoints.md` y `docs/fase_1_estado_inicial.md`).

---

## Comandos de validación reproducibles

Para verificar la baseline en cualquier momento:

```bash
# Arrancar el backend
cd <proyecto>
python -m uvicorn api.main:app --port 8001

# Probar endpoints públicos
curl http://localhost:8001/
curl http://localhost:8001/openapi.json | head -c 200

# Login y endpoint con auth
TOK=$(curl -s -X POST http://localhost:8001/auth/login \
  -d "username=admin&password=Admin2025%21" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  | python -c "import sys,json; print(json.load(sys.stdin)['access_token'])")
curl -H "Authorization: Bearer $TOK" http://localhost:8001/auth/me

# Probar endpoint de análisis
curl -s http://localhost:8001/api/v1/macro
curl -s http://localhost:8001/api/v1/risk/AMZN/backtest?confidence=0.95
```

---

*Generado durante la ejecución de Fase 1 del plan de acción del proyecto RiskLab USTA · 2026-05-12*
