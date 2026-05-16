# RiskLab USTA — Plataforma de Análisis Cuantitativo de Riesgo Financiero

**Proyecto Integrador — Teoría del Riesgo · Python para APIs e IA**
Universidad Santo Tomás · Facultad de Estadística · 2024–2026

---

## Resumen ejecutivo

RiskLab USTA es una plataforma web de análisis cuantitativo de riesgo financiero construida íntegramente en Python. Integra **once módulos analíticos** que cubren el ciclo completo de evaluación de un portafolio: desde el diagnóstico técnico de precios hasta el stress testing, pasando por la modelación de volatilidad, optimización por programación cuadrática, valoración de renta fija con curva Nelson-Siegel y de opciones europeas con Black-Scholes, más un **componente de machine learning** que clasifica señales buy/hold/sell sobre features técnicas.

La arquitectura se organiza en **cinco capas independientes** (datos, análisis clásico, renta fija + derivados, ML, infraestructura) expuestas como una **API REST FastAPI** con autenticación JWT, persistencia ORM en SQLite vía SQLAlchemy, cache transparente de datos externos, pipeline de ML con patrón Singleton, suite de tests con pytest + TestClient, y despliegue en contenedor Docker multi-stage validado por CI en GitHub Actions.

---

## Contexto y propósito

La gestión del riesgo financiero es una competencia central para cualquier profesional de la estadística aplicada. Las instituciones financieras necesitan herramientas que traduzcan datos de mercado en información accionable: cuánto puede perder un activo en el peor caso, qué tan eficiente es un portafolio respecto al mercado, cómo cambia el precio de un bono ante un shock de tasa, qué pasa con la cartera bajo un escenario de estrés.

Este proyecto demuestra que los marcos teóricos estudiados —CAPM, Markowitz, VaR/CVaR, GARCH, EWMA, Nelson-Siegel, Black-Scholes— pueden implementarse de forma rigurosa, reproducible y completamente desplegable usando herramientas de código abierto. Cierra la brecha entre la teoría y la práctica en tres dimensiones:

1. **Implementación real**: cada modelo ejecuta sobre datos descargados en tiempo real desde Yahoo Finance y FRED, no sobre series predefinidas.
2. **Validación estadística**: los modelos se contrastan con tests formales (Jarque-Bera, Shapiro-Wilk, Kupiec POF, Ljung-Box).
3. **Ingeniería productiva**: no es un notebook ni un script aislado, sino un sistema con tests, contenedor, CI y deploy continuo.

---

## Arquitectura en cinco capas

```
┌────────────────────────────────────────────────────────────────┐
│  Frontend interactivo (dashboard.html + Plotly.js)             │
│      ─ Consume la API REST via fetch                           │
└──────────────────────────────┬─────────────────────────────────┘
                               │ HTTP
┌──────────────────────────────▼─────────────────────────────────┐
│  Backend FastAPI (api/main.py)                                 │
│      ─ Routers, Pydantic v2, Depends(), JWT, OpenAPI/docs      │
└──────────────────────────────┬─────────────────────────────────┘
                               │
   ┌───────────────────────────┼───────────────────────────┐
   ▼                           ▼                           ▼
┌──────────┐         ┌──────────────────┐         ┌──────────────┐
│  Capa 1  │         │  Capas 2 + 3     │         │   Capa 4     │
│  Datos   │         │  Análisis        │         │   ML         │
│  +       │         │  + Renta fija    │         │   Singleton  │
│  Cache   │         │  + Opciones      │         │   /predict   │
│ (ORM)    │         │  + Stress        │         │              │
└──────────┘         └──────────────────┘         └──────────────┘
   │                           │                           │
   └───────────────────────────┴───────────────────────────┘
                               │
                               ▼
                  ┌──────────────────────────┐
                  │   Capa 5: infraestructura │
                  │   ─ pytest + TestClient   │
                  │   ─ Docker multi-stage    │
                  │   ─ GitHub Actions CI     │
                  │   ─ Render PaaS           │
                  └──────────────────────────┘
```

- **Capa 1 — Datos y persistencia**: descarga de precios de Yahoo Finance y series macro de FRED con cache transparente en SQLite vía SQLAlchemy ORM (tabla `prices`, TTL configurable).
- **Capa 2 — Análisis clásico**: indicadores técnicos, rendimientos, EWMA + GARCH, CAPM, VaR/CVaR + Kupiec, Markowitz QP con/sin no-negatividad, señales.
- **Capa 3 — Renta fija + derivados + stress**: curva Nelson-Siegel, duración y convexidad, Black-Scholes con Greeks, stress testing por escenarios.
- **Capa 4 — Machine Learning**: pipeline train → joblib → load (Singleton) → predict, con logging persistente en `predictions_log`.
- **Capa 5 — Infraestructura**: tests pytest, Dockerfile multi-stage, workflow CI, deploy en Render.

---

## Estructura del repositorio

```
Portafolio_Teoria-_de_Riesgo/
├── api/
│   ├── main.py              # FastAPI app + routers + endpoints
│   ├── config.py            # Settings(BaseSettings) — .env centralizado
│   ├── data.py              # shim de get_historical_data hacia DataService
│   ├── database.py          # shim de compatibilidad sobre api/db/
│   ├── logic.py             # cálculos analíticos M1-M6 (indicadores, VaR, QP…)
│   ├── db/
│   │   ├── base.py          # engine, SessionLocal, Base, get_db, init_db
│   │   ├── models.py        # 7 modelos ORM
│   │   └── repository.py    # CRUD: usuarios, portafolios, predicciones
│   ├── services/
│   │   ├── data_service.py  # cache transparente OHLCV
│   │   ├── decorators.py    # @log_execution_time
│   │   ├── fixed_income.py  # FredClient, YieldCurve, Bond (M9)
│   │   ├── options.py       # OptionPricer, Black-Scholes, Greeks (M10)
│   │   └── stress.py        # StressTester (M11)
│   └── ml/
│       ├── features.py      # ingeniería de 8 features técnicas
│       ├── predictor.py     # ModelPredictor (Singleton verificable)
│       └── train.py         # `python -m api.ml.train`
├── dashboard/
│   ├── dashboard.html       # SPA con Plotly.js — 14 módulos
│   ├── data.js              # snapshot estático (fallback offline)
│   └── app.py               # variante Streamlit (legacy)
├── tests/
│   ├── conftest.py          # fixtures: BD en memoria + TestClient
│   ├── test_indicators.py
│   ├── test_var.py
│   ├── test_options.py
│   ├── test_qp.py
│   ├── test_endpoints.py
│   ├── test_singleton.py
│   └── test_bond.py
├── data/                    # SQLite (gitignored para .db) + users.json
├── docs/                    # HTML del instructivo del Proyecto Integrador
├── generate_data.py         # genera data.js (snapshot estático)
├── .github/workflows/ci.yml # workflow de tests en cada push
├── Dockerfile               # multi-stage python:3.11.9-slim-bookworm
├── docker-compose.yml       # dev local con hot-reload
├── .dockerignore
├── render.yaml              # config declarativa para Render
├── Procfile                 # comando de inicio para PaaS
├── pytest.ini
├── requirements.txt         # versiones fijas
├── .env.example             # plantilla de variables de entorno
└── README.md
```

---

## Los once módulos analíticos

| Módulo | Tema | Endpoint principal |
|---|---|---|
| **M1** | Análisis técnico (SMA, EMA, RSI, MACD, Bollinger, Estocástico) | `GET /api/v1/technical/{ticker}` |
| **M2** | Rendimientos + tests de normalidad + hechos estilizados | `GET /api/v1/returns/{ticker}` |
| **M3** | Volatilidad EWMA (RiskMetrics) + ARCH / GARCH / EGARCH | `GET /api/v1/volatility/{ticker}` |
| **M4** | CAPM, β, α de Jensen, R² | `GET /api/v1/risk/{ticker}` |
| **M5** | VaR paramétrico / histórico / Montecarlo + CVaR + Kupiec POF | `GET /api/v1/risk/{ticker}/backtest` |
| **M6** | Markowitz como QP con cvxpy, con/sin no-negatividad | `POST /api/v1/frontier`, `GET /api/v1/frontier/compare` |
| **M7** | Señales técnicas automáticas (RSI, MACD, Bollinger) | `GET /api/v1/signals/{ticker}` |
| **M8** | Benchmark, Tracking Error, Information Ratio, contexto macro | `GET /api/v1/macro` |
| **M9** | Renta fija: curva de tesoros (FRED) + Nelson-Siegel + duración + convexidad | `GET /api/v1/yield-curve`, `POST /api/v1/bond/duration` |
| **M10** | Opciones europeas: Black-Scholes + 5 Greeks + σ implícita Newton-Raphson | `POST /api/v1/option/price`, `GET /api/v1/option/scenarios/{ticker}` |
| **M11** | Stress testing con 6 escenarios + heatmap activo×escenario | `POST /api/v1/stress` |

Además del componente de **Machine Learning** (capa 4):

| Endpoint | Descripción |
|---|---|
| `POST /api/v1/predict` | Clasifica buy/hold/sell sobre features técnicas. Singleton + logging en BD. |
| `GET /api/v1/predict/info` | Metadata del modelo activo (versión, accuracy, F1, features esperadas). |
| `GET /api/v1/predict/log` | Últimas predicciones persistidas en `predictions_log`. |

Y los endpoints transversales:

| Endpoint | Descripción |
|---|---|
| `POST /auth/register`, `POST /auth/login`, `GET /auth/me` | Registro, login y perfil JWT. |
| `POST /api/v1/portfolios`, `GET /api/v1/portfolios`, ... | CRUD de portafolios persistidos por usuario. |
| `GET /api/v1/cache/stats` | Estado del cache OHLCV (cantidad de assets, filas, último fetch). |
| `GET /api/v1/all` | Bundle de todos los módulos en una sola llamada. |
| `GET /docs`, `GET /redoc` | Documentación automática de la API (Swagger / Redoc). |

---

## Persistencia: SQLAlchemy + SQLite

La capa de persistencia usa **SQLAlchemy 2.0 ORM** sobre SQLite con 7 modelos:

| Modelo | Tabla | Función |
|---|---|---|
| `User` | `users` | Usuarios del sistema, autenticados por JWT. |
| `ResetToken` | `reset_tokens` | Tokens de restablecimiento de contraseña con expiración. |
| `Asset` | `assets` | Universo de tickers conocidos por el sistema. |
| `Price` | `prices` | Cache OHLCV con índice único `(asset_id, date)`. |
| `Portfolio` | `portfolios` | Portafolios persistidos por usuario (CRUD). |
| `PredictionLog` | `predictions_log` | Registro de cada llamada a `/predict` para monitoreo de drift. |
| `SignalLog` | `signals_log` | Señales técnicas disparadas históricamente. |
| `MacroSeries` | `macro_series` | Cache de series macro de FRED (curva, inflación, etc.). |

La sesión se inyecta en cada endpoint vía `Depends(get_db)` (patrón estándar de FastAPI), y el cache transparente reduce los hits a Yahoo Finance / FRED a ~24 h por ticker (TTL configurable).

---

## Componente de Machine Learning

**Propósito analítico**: clasificación de señal de trading **buy / hold / sell** sobre features técnicas, etiquetada según el retorno acumulado de los próximos 5 días con umbral ±2%.

**Pipeline**:
1. `python -m api.ml.train` descarga 2 años de historia para los tickers del portafolio, construye 8 features (`ret_1d`, `ret_5d`, `ret_20d`, `rsi_14`, `macd_hist`, `vol_20d`, `pos_in_bb`, `volume_ratio`) y la etiqueta forward.
2. Particiona **temporalmente** (`shuffle=False`) para evitar leakage en series financieras.
3. Ajusta `Pipeline(StandardScaler + RandomForestClassifier(n_estimators=200, class_weight="balanced"))` y persiste el artefacto en `api/ml/model.joblib` + `.meta.json`.

**Servicio en producción**:
- La clase `ModelPredictor` implementa el **patrón Singleton verificable**: `__new__` carga el modelo UNA sola vez al primer acceso. En logs aparece `[ModelPredictor] modelo cargado (carga UNICA, Singleton): v1.0.0`. Si se realizan dos llamadas consecutivas a `/predict`, ese mensaje **aparece una sola vez** — eso demuestra el Singleton.
- El endpoint `/predict` puede recibir las features en el body (`features: {...}`) o calcularlas automáticamente desde los precios cacheados del ticker.
- Cada llamada se persiste en `predictions_log` (timestamp, ticker, features, predicción, confianza) para monitoreo de drift futuro.
- Si `model.joblib` no existe (todavía no se entrenó), el predictor cae a una **heurística RSI+MACD** para que `/predict` siempre responda.

---

## Stack técnico

| Categoría | Componentes |
|---|---|
| Lenguaje | Python 3.11.9 |
| API | FastAPI 0.135, uvicorn, Pydantic v2.12 + pydantic-settings |
| Persistencia | SQLAlchemy 2.0, SQLite |
| Autenticación | bcrypt, python-jose, OAuth2PasswordBearer |
| Datos financieros | yfinance, fredapi |
| Análisis | pandas, numpy, scipy, statsmodels, arch (GARCH) |
| Optimización | cvxpy (QP de Markowitz) |
| Machine Learning | scikit-learn, joblib |
| Visualización | Plotly.js (frontend), Streamlit (variante alternativa) |
| Tests | pytest, pytest-asyncio, httpx |
| Contenedor | Docker multi-stage sobre `python:3.11.9-slim-bookworm` |
| CI | GitHub Actions |
| Deploy | Render free-tier |

---

## Instalación y ejecución local

### Requisitos previos

- Python 3.11.9 (recomendado) o 3.12+
- pip
- Conexión a internet (para Yahoo Finance y FRED)
- **API key de FRED** (gratis): https://fredaccount.stlouisfed.org/apikeys

### Pasos

```bash
# 1. Clonar e instalar
git clone https://github.com/vanessacortes20/Portafolio_Teoria-_de_Riesgo.git
cd Portafolio_Teoria-_de_Riesgo

python -m venv venv
source venv/bin/activate            # Linux / Mac
# .\venv\Scripts\Activate.ps1       # Windows PowerShell

pip install -r requirements.txt

# 2. Configurar variables de entorno
cp .env.example .env
# Editar .env y poner valores reales:
#   - JWT_SECRET (clave aleatoria larga)
#   - FRED_API_KEY (la que obtuviste arriba)

# 3. (Opcional) Entrenar el modelo ML
python -m api.ml.train

# 4. Iniciar el backend
uvicorn api.main:app --host 0.0.0.0 --port 8001 --reload

# 5. Abrir el dashboard
# Opción A: abrir directamente dashboard/dashboard.html en el navegador
# Opción B: servirlo con un servidor estático cualquiera
```

### Documentación interactiva

Con el backend corriendo: <http://localhost:8001/docs> (Swagger UI) y <http://localhost:8001/redoc>.

### Usuarios demo

| Usuario | Contraseña | Rol |
|---|---|---|
| `admin` | `Admin2025!` | admin |
| `demo` | `Demo2025!` | user |

Se crean automáticamente la primera vez que se arranca el backend si la tabla `users` está vacía.

---

## Variables de entorno

Documentadas íntegramente en `.env.example`. Resumen:

| Variable | Default | Descripción |
|---|---|---|
| `JWT_SECRET` | aleatoria (1 sesión) | Clave para firmar tokens JWT. **Cambiar en producción.** |
| `JWT_TTL_MINUTES` | `60` | Tiempo de vida del token JWT. |
| `DATABASE_URL` | `sqlite:///./data/risklab_users.db` | URL de la BD (SQLAlchemy). |
| `PORTFOLIO_TICKERS` | `NU,AMZN,SONY,XOM,WPM` | Tickers del análisis (CSV). |
| `BENCHMARK_TICKER` | `^GSPC` | Benchmark para CAPM y stress. |
| `RISK_FREE_RATE_DEFAULT` | `0.04` | Rf anual de fallback. |
| `FRED_API_KEY` | vacía | Key de FRED (curva de tesoros, inflación). Sin esta, `/yield-curve` cae a yfinance. |
| `YFINANCE_CACHE_TTL_HOURS` | `24` | TTL del cache de precios en BD. |
| `ML_MODEL_PATH` | `api/ml/model.joblib` | Ruta del artefacto ML serializado. |

---

## Tests

```bash
pytest tests/ -v --tb=short
```

La suite cubre los **5 tests obligatorios** del Proyecto Integrador más 20 adicionales:

- `test_indicators.py`: RSI monótono → 0 / 100, SMA con valores conocidos.
- `test_var.py`: VaR paramétrico contra `|μ + σ · Φ⁻¹(α)|`, monotonía en confianza, Kupiec con cero excedencias.
- `test_options.py`: paridad put-call `C − P = S − K·e^(−rT)` en ATM y OTM, Δ ∈ [0,1] (call) y [−1,0] (put), ν > 0, σ implícita recuperada vía Newton-Raphson.
- `test_qp.py`: pesos suman 1, sin negativos cuando `allow_short=False`, frontera eficiente con múltiples puntos, mínima varianza es el punto más bajo.
- `test_bond.py`: bono a la par cuando YTM=cupón, modificada < Macaulay, convexidad > 0.
- `test_singleton.py`: dos instancias de `ModelPredictor` son la misma; `get_predictor()` también.
- `test_endpoints.py`: GET `/` → 200; GET `/docs` → 200; POST `/api/v1/option/price` happy-path y validación de errores (sigma negativa → 422, option_type inválido → 422); POST `/api/v1/stress` con pesos que no suman 1 → 422; POST `/api/v1/predict` con features manuales.

Los tests usan una BD SQLite en memoria con `StaticPool` y un override de `Depends(get_db)`, sin tocar la base productiva ni hacer llamadas a internet.

---

## Docker

### Build local

```bash
docker build -t risklab-usta .
docker run -p 8001:8001 -e JWT_SECRET=$(openssl rand -hex 32) risklab-usta
```

### Desarrollo con compose (hot-reload)

```bash
docker compose up --build
# Monta api/ y dashboard/ desde el host; cambios en código → reload automático
```

### Características de la imagen

- **Multi-stage**: stage 1 instala wheels nativos (numpy, scipy, cvxpy) con `build-essential`; stage 2 copia sólo el venv sin compiladores. La imagen final pesa ~250 MB (vs ~600 MB single-stage).
- Base **`python:3.11.9-slim-bookworm`** (versión exacta exigida por la spec).
- Usuario no-root (`app`) para correr uvicorn.
- `PYTHONUNBUFFERED=1` para que los logs salgan en tiempo real.
- `CMD` con `sh -c` para que respete `${PORT}` cuando el PaaS lo asigne.

---

## Integración continua

`.github/workflows/ci.yml` corre en cada push a `main` / `develop` y en cada PR:

1. Checkout del repo.
2. Setup de Python 3.11.9 con cache de pip basado en `requirements.txt`.
3. `pip install -r requirements.txt`.
4. `pytest tests/ -v --tb=short`.

Concurrency configurado para cancelar ejecuciones previas de la misma rama cuando llega un push nuevo. Tiempo aproximado: 3-5 min la primera ejecución, 1-2 min con cache.

---

## Despliegue en la nube

### Render (free tier)

1. Crear cuenta gratuita en <https://render.com>.
2. Conectar el repositorio de GitHub.
3. **New → Web Service** → Render detecta el `Dockerfile` automáticamente.
4. En la configuración del servicio:
   - Build: usar Dockerfile.
   - Environment: agregar `FRED_API_KEY`, `JWT_SECRET` (Render puede generarla con `generateValue: true`, ya configurado en `render.yaml`).
   - Disk persistente de 1 GB en `/app/data` (también en `render.yaml`) para que el SQLite sobreviva redeploys.
5. Cada push a `main` dispara redeploy automático.

**Limitación del free-tier**: el servicio se duerme tras 15 min sin tráfico (cold start ~30 s al despertar). Antes de la sustentación, hacer una llamada de calentamiento: `curl <url>/docs`.

### URL pública

> **TODO**: actualizar después del primer deploy con la URL real de Render.
>
> - API: `https://risklab-usta.onrender.com` *(placeholder)*
> - Swagger UI: `https://risklab-usta.onrender.com/docs`
> - Redoc: `https://risklab-usta.onrender.com/redoc`

---

## Política de uso de inteligencia artificial

Este proyecto se desarrolló con apoyo de herramientas de IA (asistentes de código, motores de búsqueda especializados) como complemento al estudio. La política seguida:

1. **Comprensión total**: cada línea de código y cada decisión de arquitectura ha sido revisada y entendida por los autores. Las herramientas se usaron como asistentes, no como sustitutos del aprendizaje.
2. **Defensa oral**: durante la sustentación, los autores pueden explicar cualquier sección del proyecto sin recurrir a la herramienta.
3. **Sin copia ciega**: ningún fragmento se copió sin revisar y adaptar al contexto. Las decisiones metodológicas (qué modelo GARCH, qué features para ML, qué umbrales de stress) se tomaron deliberadamente.

---

## Limitaciones conocidas

- **Yahoo Finance**: depende de la disponibilidad y calidad del proveedor. Activos con poca liquidez pueden producir estimaciones inestables.
- **VaR paramétrico**: asume normalidad de retornos. M2 frecuentemente la rechaza para activos individuales, por lo que el VaR histórico y Montecarlo suelen ser más conservadores.
- **Estacionariedad en GARCH**: en períodos de crisis los parámetros pueden no ser estables.
- **CAPM como modelo de un factor**: no incorpora tamaño, valor ni momentum.
- **Markowitz estático**: la frontera se calcula sobre el histórico completo. No predice pesos óptimos futuros.
- **Sin costos de transacción**: rebalanceos se asumen gratuitos.
- **Render free-tier**: cold start de 15-30 s tras inactividad.
- **Modelo ML**: heurística forward 5 días con threshold ±2% — no es una recomendación de inversión real.

---

## Conclusión

RiskLab USTA demuestra que la teoría del riesgo financiero no es solo un conjunto de fórmulas: es un marco de decisión que puede implementarse de forma completa, rigurosa y accesible con herramientas de código abierto, y desplegarse en producción con el mismo rigor que un sistema comercial.

El proyecto no solo implementa los modelos del Proyecto Integrador, sino que los conecta entre sí en una arquitectura coherente: los indicadores de M1 alimentan las señales de M7 y las features de ML; las distribuciones de M2 justifican la elección del VaR; M3 cuantifica la volatilidad que M5 usa; M4 mide la exposición que M8 contrasta contra el benchmark; M6 sintetiza todo en un portafolio óptimo; M9 y M10 valoran instrumentos derivados sobre los mismos activos; M11 estresa el portafolio bajo escenarios extremos. **No son módulos independientes: son etapas de un pipeline de análisis de riesgo cuantitativo.**

---

*Desarrollado con Python 3.11.9 · FastAPI · SQLAlchemy · Plotly.js · scikit-learn · Yahoo Finance · FRED*
*Universidad Santo Tomás — Proyecto Integrador Teoría del Riesgo · 2024–2026*
