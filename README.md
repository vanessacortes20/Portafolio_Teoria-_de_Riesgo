# RiskLab USTA — Plataforma de Análisis Cuantitativo de Riesgo Financiero

**Proyecto Integrador — Gestión de Riesgo Financiero**  
Universidad Santo Tomás · Facultad de Economía · 2024–2026

---

## Resumen Ejecutivo

RiskLab USTA es una plataforma web de análisis cuantitativo de riesgo financiero, desarrollada íntegramente en Python y desplegable desde cualquier navegador. Integra ocho módulos de análisis activos (M1–M8) que cubren el ciclo completo de evaluación de un portafolio: desde el diagnóstico técnico de precio hasta la optimización bajo la teoría de Markowitz, pasando por el modelado de volatilidad con modelos ARCH/GARCH, la estimación de riesgo extremo con VaR/CVaR, y la comparación de desempeño frente al mercado.

La plataforma no es un ejercicio académico cerrado: consume datos reales en tiempo real desde Yahoo Finance, expone una API REST documentada, implementa autenticación con tokens JWT, y puede desplegarse en la nube con un solo comando. Todo el análisis estadístico es reproducible, los parámetros son configurables por el usuario, y las visualizaciones son interactivas.

---

## Contexto y Propósito

La gestión del riesgo financiero es una de las competencias más exigentes en el análisis cuantitativo moderno. Las instituciones financieras, los fondos de inversión y los departamentos de riesgo corporativo necesitan herramientas que traduzcan datos de mercado en información accionable: cuánto puede perder un activo en el peor caso, qué tan eficiente es un portafolio respecto al mercado, o cuándo los modelos de riesgo vigentes dejan de ser estadísticamente válidos.

RiskLab USTA nace de esa necesidad aplicada al entorno académico. Su propósito es demostrar que los marcos teóricos estudiados en un programa de economía o finanzas —CAPM, Markowitz, VaR, GARCH— pueden implementarse de forma rigurosa, reproducible y completamente funcional usando herramientas de código abierto. El proyecto conecta la teoría con la práctica a través de una interfaz que cualquier analista o estudiante puede operar, interpretar y extender.

---

## El Problema que Aborda

Existe una brecha frecuente entre la enseñanza de la teoría del riesgo financiero y su aplicación práctica. Los modelos se estudian en términos de fórmulas y supuestos, pero rara vez se implementan sobre datos reales, con parámetros ajustables y resultados verificables. Esta brecha genera analistas que conocen el VaR en papel pero no saben cómo calcularlo, interpretarlo ni ponerlo en perspectiva frente a un test de validez estadística.

RiskLab USTA cierra esa brecha en tres dimensiones:

1. **Implementación real**: cada módulo ejecuta los cálculos sobre datos históricos descargados en tiempo real, no sobre series simuladas o predefinidas.
2. **Validación estadística**: los modelos no solo producen números, sino que los contrastan con tests formales (Jarque-Bera, Shapiro-Wilk, Kupiec POF, Ljung-Box).
3. **Interpretación integrada**: cada resultado viene acompañado de una explicación en lenguaje natural que conecta el valor numérico con su significado económico.

---

## Objetivos del Proyecto

- Implementar un sistema de análisis cuantitativo de riesgo que cubra el ciclo completo M1–M8 definido en el instructivo del Proyecto Integrador.
- Desarrollar un backend REST con FastAPI que exponga todos los cálculos como servicios reutilizables y documentados.
- Construir un frontend interactivo con Plotly.js que permita explorar los resultados sin necesidad de programar.
- Implementar autenticación segura con JWT y persistencia de usuarios en SQLite.
- Demostrar que Python, correctamente estructurado, puede ser el núcleo de una plataforma de riesgo financiero completa y desplegable en producción.

---

## Relación entre Python y la Teoría del Riesgo

El diseño de RiskLab no es solo técnico: cada biblioteca y cada patrón de código tiene un correlato directo con un concepto financiero o estadístico.

| Concepto teórico | Implementación en Python |
|-----------------|--------------------------|
| Retornos logarítmicos y estadísticos | `pandas`, `numpy` sobre datos de Yahoo Finance |
| Test de normalidad (Jarque-Bera, Shapiro-Wilk) | `scipy.stats` |
| Modelos ARCH/GARCH/EGARCH | `arch` (Kevin Sheppard) |
| CAPM y regresión Beta | `scipy.stats.linregress` |
| VaR histórico, paramétrico y Monte Carlo | `numpy.percentile`, `scipy.norm`, simulación aleatoria |
| Backtesting de VaR (Test de Kupiec) | Estadístico LR sobre `scipy.stats.chi2` |
| Optimización de Markowitz | Simulación aleatoria + `scipy.optimize.minimize` (SLSQP) |
| Gestión de sesiones y autenticación | `python-jose`, `bcrypt`, `FastAPI` |
| Validación de datos de entrada | `Pydantic v2` con `field_validator` y `model_validator` |

Esta correspondencia directa —teoría → código → resultado visible— es lo que convierte a RiskLab en una herramienta pedagógica y operativa al mismo tiempo. No hay cajas negras: cada cálculo está explícito en `api/logic.py` y puede ser auditado, modificado o extendido.

---

## Arquitectura del Sistema

RiskLab sigue una arquitectura de dos capas con un mecanismo de datos híbrido:

```
┌─────────────────────────────────────────────────────────┐
│                     dashboard.html                       │
│  (HTML + CSS + Plotly.js + JS — sin framework frontend)  │
│                                                          │
│  ┌─────────────────┐    ┌──────────────────────────┐    │
│  │   datos estáticos│    │  API REST (tiempo real)   │    │
│  │   data.js        │ ── │  FastAPI en :8001         │    │
│  │  (snapshot local)│    │  (Yahoo Finance en vivo)  │    │
│  └─────────────────┘    └──────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
                                │
                    ┌───────────┴──────────┐
                    │   api/               │
                    │   ├── main.py        │  FastAPI + JWT + endpoints
                    │   ├── logic.py       │  Cálculos estadísticos
                    │   ├── data.py        │  Descarga yfinance
                    │   └── database.py    │  SQLite — usuarios
                    └──────────────────────┘
```

**Patrón de datos híbrido**: el dashboard carga por defecto el snapshot estático `data.js` (generado por `generate_data.py`) para garantizar disponibilidad sin backend activo. Al presionar **"Recalcular (API)"**, el frontend consume los endpoints REST y actualiza todos los módulos con datos en tiempo real. Esto permite que la herramienta funcione tanto offline como conectada a la API.

---

## Estructura del Proyecto

```
proyecto_2/
├── api/
│   ├── __init__.py
│   ├── main.py            # FastAPI — endpoints REST, autenticación JWT, validación
│   ├── logic.py           # Funciones de cómputo estadístico (M1–M7)
│   ├── data.py            # Descarga de datos históricos (yfinance)
│   └── database.py        # SQLite — usuarios, tokens de reset, audit log
├── dashboard/
│   ├── dashboard.html     # Frontend completo (HTML + CSS + JS + Plotly.js)
│   └── data.js            # Snapshot estático generado por generate_data.py
├── docs/
│   └── instructivo_proyecto_integrador.html   # Instructivo del Proyecto Integrador
├── tests/
│   └── test_yf.py         # Smoke test de yfinance
├── data/                  # (carpeta local — la BD SQLite y users.json se generan al arrancar)
├── generate_data.py       # Generador del snapshot estático data.js
├── requirements.txt       # Dependencias Python
├── Dockerfile             # Imagen Docker lista para producción
├── Procfile               # Comando de inicio para Render / Railway / Heroku
├── render.yaml            # Configuración declarativa para Render.com
├── .env.example           # Plantilla de variables de entorno requeridas
└── .gitignore             # Reglas de exclusión del repositorio
```

> **Nota sobre `data/`:** la carpeta existe localmente para alojar `risklab_users.db` y `users.json`, pero ambos están excluidos del repositorio en `.gitignore` porque contienen datos de usuarios. Se generan automáticamente al iniciar el backend (`init_db()` y `seed_demo_users()` crean los usuarios demo).

---

## Módulos de Análisis

### M1 — Análisis Técnico

Calcula y visualiza los indicadores clásicos de análisis técnico sobre la serie de precios de cierre: media móvil simple (SMA 20 y 50 períodos), media móvil exponencial (EMA 20), RSI con zona de sobrecompra/sobreventa, MACD con histograma de momentum, Bandas de Bollinger (2σ) y Oscilador Estocástico (%K, %D). Todos los indicadores se grafican de forma interactiva con Plotly y se actualizan al cambiar el rango de fechas o el activo.

### M2 — Distribución de Rendimientos

Calcula retornos simples y logarítmicos diarios. Presenta estadísticas descriptivas completas (media, desviación estándar, asimetría, exceso de curtosis, mínimo, máximo) y aplica dos tests formales de normalidad: Jarque-Bera y Shapiro-Wilk. Incluye gráfico Q-Q para evaluar visualmente las colas de la distribución, y un panel de **hechos estilizados** (fat tails, clustering de volatilidad via Ljung-Box, asimetría negativa) que conecta la distribución empírica con los supuestos del modelo.

### M3 — Modelos de Volatilidad (ARCH/GARCH/EGARCH)

Ajusta tres modelos de volatilidad condicional: ARCH(1), GARCH(1,1) y EGARCH(1,1). Compara los tres modelos por AIC y BIC para identificar el mejor ajuste. Con el GARCH(1,1) seleccionado, calcula los residuos estandarizados (con su test de normalidad Jarque-Bera) y genera un **pronóstico de volatilidad a 10 días** que cuantifica el riesgo esperado en el horizonte de corto plazo. Los retornos se escalan por 100 para mejorar la convergencia numérica del optimizador.

### M4 — Riesgo Sistemático (CAPM)

Estima el modelo CAPM por regresión OLS entre los retornos del activo y los del benchmark (S&P 500). Produce Beta (sensibilidad al mercado), Alpha de Jensen (retorno diferencial ajustado por riesgo sistemático), R² (proporción de varianza explicada por el mercado) y retorno esperado anualizado. Clasifica el activo como agresivo (β > 1.2), neutro o defensivo (β < 0.8). El scatter plot retorno-activo vs. retorno-benchmark incluye la recta de regresión.

### M5 — Valor en Riesgo (VaR) y CVaR

Calcula VaR y CVaR bajo tres metodologías:
- **Histórico**: percentil empírico de los retornos observados.
- **Paramétrico (Normal)**: `VaR = μ + σ·z_α`, `CVaR = μ − σ·φ(z_α)/(1−α)`.
- **Monte Carlo**: simulación de 10,000 retornos normales con los parámetros empíricos.

El VaR se reporta al nivel de confianza configurado por el usuario (80%–99%) y siempre también al 99% como referencia regulatoria fija. El módulo incluye **backtesting con el Test de Kupiec (POF)**: calcula el estadístico de razón de verosimilitud LR sobre las excepciones históricas y determina si el modelo es estadísticamente válido (p-value > 0.05).

### M6 — Optimización de Portafolio (Markowitz)

Genera 10,000 portafolios aleatorios con pesos ≥ 0 y Σwᵢ = 1 para trazar la frontera eficiente. Identifica el portafolio de **máximo Sharpe Ratio** (mejor relación retorno/riesgo) y el de **mínima volatilidad**. El Sharpe se calcula correctamente deduciendo la tasa libre de riesgo: SR = (Rₚ − Rƒ)/σₚ.

Incluye una segunda optimización por **rendimiento objetivo**: dada una tasa anual deseada, encuentra la composición de mínima volatilidad que la alcanza exactamente, resolviendo un problema de optimización cuadrática con SLSQP. La matriz de correlación entre activos se visualiza como heatmap interactivo (escala RdBu de −1 a +1).

### M7 — Señales y Alertas Técnicas

Evalúa el estado actual de cada indicador (RSI, MACD, Bollinger) sobre la última sesión disponible y genera señales de compra/venta con **explicaciones en lenguaje natural**. Cada señal explica por qué se activa, qué implica económicamente y qué limitaciones tiene. No es un sistema de trading automatizado, sino un panel de diagnóstico que traduce indicadores técnicos en texto interpretable.

### M8 — Portafolio vs. Benchmark S&P 500

Compara el portafolio óptimo (pesos del máximo Sharpe) frente al S&P 500 con retorno acumulado base 100, curvas de drawdown máximo, Alpha de Jensen anualizado, Beta del portafolio, Tracking Error, Information Ratio y R². Incorpora un panel de **contexto macroeconómico** con datos en tiempo real: tasa libre de riesgo (^IRX), rendimiento del Tesoro EE.UU. a 10 años (^TNX) y retorno YTD del S&P 500. Estas tres métricas macro contextualizan los resultados del portafolio dentro del entorno de tasas y mercado vigente.

---

## Flujo de Datos Dinámicos

Cuando el usuario presiona **"Recalcular (API)"**, el frontend ejecuta el siguiente flujo:

```
1. Construye query string con fechas, confianza VaR y n_simulaciones
2. Intenta GET /api/v1/all  (respuesta completa en una sola llamada)
   └── Si falla → llama endpoints individuales en paralelo:
       /api/v1/technical/{ticker}
       /api/v1/returns/{ticker}
       /api/v1/volatility/{ticker}
       /api/v1/risk/{ticker}
       /api/v1/portfolio/optimize
3. Valida que al menos un ticker tenga datos técnicos
4. Sobreescribe D (estado global) con freshData
5. Vuelve a renderizar el módulo activo
```

El backend descarga los datos de Yahoo Finance con `yfinance`, calcula todos los indicadores en `logic.py` y serializa el resultado como JSON con manejo seguro de NaN/Inf. El rango mínimo configurable es 2020-01-01 y el máximo es el día de hoy.

---

## Fundamentos Técnicos y Matemáticos

### Retornos

- Simple: `rₜ = (Pₜ − Pₜ₋₁) / Pₜ₋₁`
- Logarítmico: `rₜ = ln(Pₜ / Pₜ₋₁)`

### Normalidad (M2)

- **Jarque-Bera**: `JB = (n/6)·[S² + (K²/4)]` ~ χ²(2) bajo H₀ de normalidad.
- **Shapiro-Wilk**: contrasta los cuantiles empíricos contra los esperados bajo normalidad.
- **Ljung-Box sobre r²**: detecta clustering de volatilidad; p-value < 0.05 → evidencia de heterocedasticidad condicional.

### ARCH/GARCH (M3)

- **ARCH(1)**: `σₜ² = ω + α·εₜ₋₁²`
- **GARCH(1,1)**: `σₜ² = ω + α·εₜ₋₁² + β·σₜ₋₁²`
- **EGARCH(1,1)**: `ln σₜ² = ω + α·(|zₜ₋₁| − E|zₜ₋₁|) + γ·zₜ₋₁ + β·ln σₜ₋₁²`

El término γ del EGARCH captura el efecto leverage (caídas generan más volatilidad que subidas equivalentes).

### CAPM (M4)

```
E[Rᵢ] = Rƒ_diaria + βᵢ · (E[Rₘ_diaria] − Rƒ_diaria)
βᵢ = Cov(Rᵢ, Rₘ) / Var(Rₘ)
Rƒ_diaria = (1 + Rƒ_anual)^(1/252) − 1
```

### VaR y CVaR (M5)

- **Paramétrico**: `VaR_α = −(μ + σ·z_α)` , `CVaR_α = −(μ − σ·φ(z_α)/(1−α))`
- Las pérdidas se reportan como valores positivos (convención de riesgo).
- VaR anualizado: `VaR_anual = VaR_diario · √252` (raíz del tiempo, supuesto de i.i.d.).

### Test de Kupiec — POF (M5)

```
LR = −2 [ N·ln(p/p̂) + (T−N)·ln((1−p)/(1−p̂)) ] ~ χ²(1) bajo H₀
```

donde `T` = observaciones, `N` = excepciones (días en que la pérdida superó el VaR), `p = 1−α` tasa esperada, `p̂ = N/T` tasa observada. Si `p-value > 0.05`, el modelo no se rechaza: la frecuencia de excepciones es estadísticamente consistente con el nivel de confianza declarado.

### Sharpe Ratio y Markowitz (M6)

```
SR = (Rₚ_anual − Rƒ_anual) / σₚ_anual
```

Optimización por rendimiento objetivo (SLSQP):
```
min   σₚ = √(wᵀ Σ w)
s.t.  Σwᵢ = 1,   wᵀμ_anual = r_objetivo,   wᵢ ≥ 0
```

### Alpha de Jensen y métricas M8

```
α_Jensen_anual = Rₚ_anual − [Rƒ + β·(Rₘ_anual − Rƒ)]
TE = std(Rₚ_diario − Rₘ_diario) · √252
IR = (Rₚ_anual − Rₘ_anual) / TE
```

---

## Autenticación y Seguridad

El sistema implementa un ciclo de autenticación completo, productivo y seguro:

- **Registro**: valida nombre, apellido, teléfono (regex), cédula, email (EmailStr), usuario (alfanumérico) y contraseña (mínimo 8 caracteres con confirmación cruzada mediante `@model_validator`).
- **Login**: acepta usuario o correo electrónico. Devuelve un `access_token` JWT firmado con HS256.
- **Autorización**: todos los endpoints de análisis requieren token válido (inyectado vía `OAuth2PasswordBearer`). El rol `admin` habilita la vista de gestión de usuarios.
- **Contraseñas**: hasheadas con BCrypt (sin passlib, usando el paquete `bcrypt` directamente).
- **Restablecimiento**: token de 32 bytes URL-safe con expiración de 1 hora, almacenado en SQLite y marcado como usado tras el consumo.
- **TTL configurable**: el tiempo de vida del token se controla con la variable `JWT_TTL_MINUTES`.

**Usuarios demo** (creados automáticamente al iniciar):

| Usuario | Contraseña | Rol |
|---------|-----------|-----|
| `admin` | `Admin2025!` | admin |
| `demo` | `Demo2025!` | user |

---

## Persistencia (SQLAlchemy ORM + sqlite3 directo)

A partir de la Fase 2 del plan de instrucciones III, la persistencia tiene **dos capas conviviendo en la misma base SQLite** (`data/risklab_users.db`):

- **`api/database.py` (sqlite3 directo)**: gestiona usuarios y tokens de reset (tablas `users`, `reset_tokens`). Soporta el flujo de autenticación completo y se conserva intacto.
- **`api/db_models.py` + `api/database_session.py` (SQLAlchemy 2.0 ORM)**: introduce 6 modelos nuevos sin tocar las tablas anteriores:
  - `Asset` — catálogo de activos
  - `Price` — cache de precios OHLCV
  - `Portfolio` — portafolios definidos por el usuario
  - `PredictionLog` — registro de predicciones del modelo ML (Fase 4)
  - `SignalLog` — persistencia de señales técnicas del M7 (Fase 3)
  - `FredCache` — cache transparente de series FRED con TTL de 24h

La sesión ORM se inyecta vía `Depends(get_db)` en endpoints que la requieran. La función `init_orm_tables()` se llama en el evento `startup` de FastAPI y es idempotente.

---

## Datos macroeconómicos (FRED + fallback yfinance)

El endpoint `/api/v1/macro` consulta **FRED (Federal Reserve Economic Data)** como fuente primaria para:

| Indicador | Serie FRED | Default fallback |
|-----------|-----------|------------------|
| Tasa libre de riesgo (3 meses) | `DGS3MO` | yfinance `^IRX` |
| Tesoro EE.UU. 10 años          | `DGS10`  | yfinance `^TNX` |
| Inflación CPI YoY (informativo) | `CPIAUCSL` | _(no fallback — campo opcional)_ |
| S&P 500 YTD                    | _(no soportado en FRED)_ | yfinance `^GSPC` |

El servicio `api/services/fred_service.py` cachea las respuestas en la tabla `fred_cache` con TTL de 24 horas. Si FRED falla (sin key, key inválida, error de red), el endpoint hace fallback a yfinance automáticamente sin romper el dashboard. El response preserva las keys originales (`as_of`, `rf_rate`, `rf_source`, `treasury_10y`, `spx_ytd`) y agrega campos opcionales (`treasury_10y_source`, `fred_enabled`, `cache_status`, `inflation_yoy`).

---

## Instalación y Ejecución Local

### Requisitos

- Python 3.10 o superior
- pip
- Conexión a internet (para Yahoo Finance)

### Pasos

```bash
# 1. Clonar el repositorio
git clone <url-del-repositorio>
cd proyecto_2

# 2. Crear entorno virtual e instalar dependencias
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Configurar variables de entorno
cp .env.example .env
# Editar .env — cambiar JWT_SECRET por un valor aleatorio seguro

# 4. Iniciar el backend
uvicorn api.main:app --port 8001 --reload

# 5. Abrir dashboard/dashboard.html en el navegador
# (apertura directa como archivo local o mediante servidor estático)
```

### Generar snapshot estático (opcional)

```bash
python generate_data.py
```

Descarga datos actuales de Yahoo Finance y escribe `data.js`. El dashboard lo carga automáticamente, lo que permite usar la herramienta sin tener el backend activo.

### Documentación interactiva de la API

Con el backend corriendo: `http://localhost:8001/docs`

---

## Endpoints REST

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | `/` | Estado de la API y rango de fechas disponible |
| POST | `/auth/register` | Registro de nuevo usuario |
| POST | `/auth/login` | Login → JWT |
| GET | `/auth/me` | Perfil del usuario autenticado |
| POST | `/auth/change-password` | Cambio de contraseña |
| POST | `/auth/reset-password` | Solicitud de restablecimiento |
| POST | `/auth/reset-password/confirm` | Confirmación con token |
| GET | `/auth/users` | Lista de usuarios (solo admin) |
| GET | `/api/v1/technical/{ticker}` | Indicadores técnicos — M1/M7 |
| GET | `/api/v1/returns/{ticker}` | Estadísticas de retornos — M2 |
| GET | `/api/v1/volatility/{ticker}` | Modelos ARCH/GARCH — M3 |
| GET | `/api/v1/risk/{ticker}` | CAPM + VaR/CVaR — M4/M5 |
| GET | `/api/v1/risk/{ticker}/backtest` | Backtesting Kupiec — M5 |
| GET | `/api/v1/portfolio/optimize` | Frontera eficiente Markowitz — M6 |
| GET | `/api/v1/portfolio/target` | Optimización por rendimiento objetivo — M6 |
| GET | `/api/v1/signals/{ticker}` | Señales técnicas automáticas — M7 |
| GET | `/api/v1/macro` | Contexto macro (Rf, T10Y, S&P YTD) — M8 |
| GET | `/api/v1/all` | Todos los módulos en una sola llamada |

**Parámetros comunes:**

| Parámetro | Rango | Default | Descripción |
|-----------|-------|---------|-------------|
| `start_date` | desde 2020-01-01 | — | Fecha de inicio (YYYY-MM-DD) |
| `end_date` | hasta hoy | — | Fecha de cierre (YYYY-MM-DD) |
| `confidence` | 0.80 – 0.99 | 0.95 | Nivel de confianza para VaR |
| `n_simulations` | 1,000 – 100,000 | 10,000 | Iteraciones Monte Carlo |
| `target_return` | −0.50 – 2.00 | — | Rendimiento anual objetivo para M6 |

---

## Variables de Entorno

| Variable | Descripción | Default |
|----------|-------------|---------|
| `JWT_SECRET` | Clave secreta para firmar tokens JWT | Aleatoria (no persistente) |
| `JWT_TTL_MINUTES` | Tiempo de vida del token JWT | `60` |
| `PORTFOLIO_TICKERS` | Tickers del portafolio (separados por coma) | `NU,AMZN,SONY,XOM,WPM` |
| `BENCHMARK_TICKER` | Ticker del benchmark | `^GSPC` |
| `RISK_FREE_RATE` | Tasa libre de riesgo anual (fallback) | `0.04` |
| `FRED_API_KEY` | Clave gratuita de FRED para macro (opcional). Solicitar en https://fred.stlouisfed.org/docs/api/api_key.html. Si está ausente o es placeholder, `/api/v1/macro` usa yfinance como fallback. | _(vacío → usa fallback)_ |

---

## Despliegue en la Nube

### Docker

```bash
docker build -t risklab-usta .
docker run -p 8001:8001 -e JWT_SECRET=tu_clave_segura risklab-usta
```

### Render.com

El archivo `render.yaml` configura automáticamente el runtime Python, el build con `pip install`, el comando de inicio con `uvicorn`, la generación automática de `JWT_SECRET` y un disco persistente de 1 GB para la base de datos SQLite.

```bash
# Despliegue directo desde el repositorio — sin configuración manual adicional
```

### Heroku / Railway

```bash
# El Procfile incluido define el comando de inicio:
# web: uvicorn api.main:app --host 0.0.0.0 --port $PORT
```

---

## Dependencias Principales

| Paquete | Versión mínima | Rol en el proyecto |
|---------|---------------|-------------------|
| `fastapi` | 0.110+ | Framework REST API con validación automática |
| `uvicorn` | 0.29+ | Servidor ASGI para FastAPI |
| `yfinance` | 0.2+ | Descarga de precios históricos de Yahoo Finance |
| `pandas` | 2.0+ | Series temporales, DataFrames, estadísticas |
| `numpy` | 1.26+ | Álgebra lineal, simulación, percentiles |
| `scipy` | 1.12+ | Tests estadísticos, optimización SLSQP |
| `arch` | 6.0+ | Estimación de modelos ARCH/GARCH/EGARCH |
| `statsmodels` | 0.14+ | Test de Ljung-Box |
| `bcrypt` | 4.0+ | Hashing seguro de contraseñas |
| `python-jose[cryptography]` | 3.3+ | Generación y verificación de tokens JWT |
| `pydantic[email]` | 2.0+ | Validación de esquemas y emails |
| `plotly` | CDN 2.35+ | Visualizaciones interactivas en el frontend |

---

## Valor Práctico de la Herramienta

RiskLab USTA está diseñada para ser útil en tres contextos distintos:

**Para el analista financiero**: la plataforma centraliza en un solo dashboard lo que normalmente requeriría múltiples herramientas: Bloomberg para datos, Excel para VaR, Python para GARCH y un informe separado para el benchmark. Aquí todo está integrado, es dinámico y se actualiza con un clic.

**Para el estudiante o investigador**: cada módulo es transparente. El código de `logic.py` es directo, sin capas de abstracción innecesarias. Cada función tiene una correspondencia uno a uno con la fórmula teórica, lo que facilita entender cómo los modelos funcionan en la práctica, no solo en papel.

---

## Limitaciones

Toda plataforma tiene supuestos y restricciones que deben conocerse para interpretar correctamente sus resultados.

- **Datos de Yahoo Finance**: la calidad y disponibilidad de los datos depende de Yahoo Finance. Activos con poca liquidez o histórico corto pueden producir estimaciones inestables.
- **Supuesto de normalidad en VaR paramétrico**: el VaR paramétrico asume retornos normales. Los tests de M2 frecuentemente rechazarán esta hipótesis para activos individuales, lo que hace que el VaR histórico y Monte Carlo sean más conservadores y recomendables.
- **Estacionariedad en GARCH**: los modelos ARCH/GARCH suponen que los parámetros de volatilidad son estables en el tiempo. En períodos de crisis o cambios estructurales de mercado, esta estacionariedad puede no mantenerse.
- **CAPM como modelo de un factor**: el CAPM captura solo el riesgo sistemático de mercado. No incorpora factores como tamaño, valor o momentum, que modelos multifactor (Fama-French) sí consideran.
- **Frontera eficiente estática**: la optimización de Markowitz en M6 produce pesos óptimos sobre el período histórico completo. No predice pesos futuros óptimos ni incorpora cambios de correlación entre activos.
- **Sin costos de transacción**: todos los análisis de portafolio asumen que los rebalanceos no tienen costo. En la práctica, comisiones y spreads bid-ask afectan la rentabilidad real.

---

## Oportunidades de Mejora

Tres extensiones naturales sobre la arquitectura actual, ordenadas por horizonte de implementación:

**Corto plazo — Versión móvil y acceso más fácil**: Adaptar el dashboard para que pueda abrirse y usarse con más comodidad desde celular o tablet, manteniendo la lectura clara de KPIs, gráficos y módulos sin saturación visual.

**Mediano plazo — Fortalecer autenticación y gestión de usuarios**: Mejorar la capa de usuario y contraseña para que el acceso sea más sólido, con mejor administración de cuentas, recuperación de credenciales y control más claro de permisos según el tipo de usuario.

**Largo plazo — Sistema de alertas más accionable**: Llevar las señales del módulo 7 a un siguiente nivel, de modo que no solo muestren estados técnicos dentro del dashboard, sino que también permitan generar avisos más útiles y oportunos para seguimiento del portafolio.

---

## Conclusión

RiskLab USTA demuestra que la teoría del riesgo financiero no es solo un conjunto de fórmulas: es un marco de decisión que puede implementarse de forma completa, rigurosa y accesible con herramientas de código abierto.

El proyecto no solo implementa los modelos exigidos por el instructivo del Proyecto Integrador, sino que los conecta entre sí con una lógica de flujo coherente. El análisis técnico de M1 diagnostica el comportamiento de precio. M2 revela si los retornos se comportan como lo asumen los modelos paramétricos. M3 cuantifica la volatilidad condicional que M5 usa para el VaR paramétrico. M4 mide la exposición al mercado que M8 contrasta contra el benchmark. M6 sintetiza toda esa información en un portafolio óptimo y M7 genera señales operacionales sobre él. No son módulos independientes: son etapas de un pipeline de análisis de riesgo.

La decisión de construirlo como una API REST con un frontend HTML puro —en lugar de un notebook o un script aislado— fue deliberada. Una API es reutilizable, testeable, versionable y desplegable. Un dashboard interactivo es explicable a un cliente o evaluador sin necesidad de que sepa programar. Esta combinación es precisamente lo que distingue a un sistema de análisis cuantitativo producible de un ejercicio académico.

El resultado es una plataforma que un analista puede usar hoy, que un estudiante puede leer y entender mañana, y que un equipo puede extender el próximo semestre. Eso, en el contexto de un proyecto académico, es exactamente lo que se busca.

---

*Desarrollado con Python 3.11 · FastAPI · Plotly.js · Yahoo Finance*  
*Universidad Santo Tomás — Proyecto Integrador Riesgo Financiero · 2024–2026*

---

## Nota sobre `CLAUDE.md` (contexto operativo local)

El proyecto incluye un archivo local `CLAUDE.md` (excluido del repositorio vía `.gitignore`) que sirve como **memoria técnica operativa** para asistencia con IA durante el desarrollo. Contiene:

- Estado actual y partes delicadas del proyecto
- Decisiones de diseño no triviales y reglas para futuras modificaciones
- Mapa exacto de validadores, dependencias inyectadas y decoradores
- Justificación de elecciones técnicas (benchmark, Rf, portafolio, stack)

**Sincronización:** `README.md` y `CLAUDE.md` deben mantenerse consistentes. Cuando se realice una modificación importante (nuevo módulo, cambio de arquitectura, nuevas dependencias), ambos archivos deben actualizarse en paralelo. Las reglas vinculantes para futuras modificaciones del proyecto residen en `CLAUDE.md`, sección **"Reglas para futuras actualizaciones"**.
