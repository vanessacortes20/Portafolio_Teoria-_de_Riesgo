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
├── backend/
│   ├── __init__.py
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py              # FastAPI — endpoints REST, alias cortos
│   │   ├── config.py            # Settings(BaseSettings) cargando .env
│   │   ├── dependencies.py      # Depends() para BD, ML, FRED, config
│   │   ├── database.py          # SQLAlchemy engine + SessionLocal + get_db
│   │   ├── auth_db.py           # sqlite3 directo (auth — capa legacy intacta)
│   │   ├── data_yf.py           # Descarga directa de yfinance (legacy)
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   └── db_models.py     # 6 modelos ORM: Asset, Price, Portfolio, etc.
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── logic.py         # Funciones puras M1–M8 (riesgo + portafolio)
│   │   │   ├── fred_service.py  # FRED con cache y reintentos
│   │   │   ├── price_service.py # Cache transparente Yahoo → SQLite
│   │   │   ├── yield_curve.py   # Curva FRED + Nelson-Siegel
│   │   │   ├── bond.py          # Bono sintético (duración, convexidad)
│   │   │   ├── options.py       # Black-Scholes + 5 Greeks
│   │   │   └── stress.py        # Stress testing
│   │   ├── ml/
│   │   │   ├── __init__.py
│   │   │   ├── train.py         # Entrenamiento offline RandomForest
│   │   │   ├── predictor.py     # Singleton + ModelPredictor
│   │   │   └── model.joblib     # Modelo serializado
│   │   └── routers/
│   │       ├── __init__.py
│   │       └── aliases.py       # /activos, /portafolios (CRUD)
│   └── tests/
│       ├── conftest.py          # Fixtures con BD SQLite en memoria
│       ├── test_logic.py        # Tests unitarios (RSI, VaR, paridad)
│       ├── test_endpoints.py    # Tests integración con TestClient
│       └── test_yf.py           # Smoke test de yfinance (no en CI)
├── frontend/
│   ├── dashboard.html           # Frontend HTML + JS + Plotly (consume API)
│   └── data.js                  # Snapshot estático generado por generate_data.py
├── docs/                        # Instructivos, auditorías, guion de sustentación
├── data/                        # (local) BD SQLite, users.json, backups
├── .github/workflows/ci.yml     # GitHub Actions con pytest
├── generate_data.py             # Generador del snapshot estático data.js
├── requirements.txt             # Dependencias con versiones fijas
├── Dockerfile                   # Imagen multi-stage
├── docker-compose.yml           # Hot-reload para desarrollo local
├── Procfile                     # Comando de inicio Heroku/Railway
├── render.yaml                  # Configuración declarativa para Render
├── pytest.ini                   # Config pytest
├── .env.example                 # Plantilla de variables de entorno
├── .dockerignore                # Excluye .env, BD, tests del contexto Docker
└── .gitignore
```

> **Comando para arrancar el backend tras la reorganización:**
> ```bash
> python -m uvicorn backend.app.main:app --port 8001 --reload
> ```

> **Nota sobre `data/`:** la carpeta existe localmente para alojar `risklab_users.db` y `users.json`, pero ambos están excluidos del repositorio en `.gitignore` porque contienen datos de usuarios. Se generan automáticamente al iniciar el backend (`init_db()` y `seed_demo_users()` crean los usuarios demo).

---

## Marco conceptual de riesgo financiero (M-I-1)

> Antes de los indicadores, el proyecto se apoya en estas definiciones canónicas. Sirven como hilo conductor de toda la plataforma.

**Riesgo financiero** es la probabilidad de pérdida económica derivada de la incertidumbre sobre el comportamiento futuro de variables del mercado, contrapartes o procesos internos. Se descompone en cuatro categorías principales:

- **Riesgo de mercado** — pérdidas por movimientos adversos en precios, tasas, tipos de cambio o índices. Es el riesgo central que cuantifica este proyecto (VaR/CVaR en M5, GARCH en M3, Beta en M4, Stress en M11).
- **Riesgo de crédito** — pérdidas por incumplimiento de pago de una contraparte o emisor. En este proyecto aparece de forma indirecta: la curva de tesoros del M9 asume riesgo soberano EE.UU. ≈ libre de crédito.
- **Riesgo de liquidez** — incapacidad de cerrar una posición a precio razonable. Se reconoce como limitación reconocida del modelo: VaR/CVaR no incorpora costos de bid-ask ni profundidad de mercado.
- **Riesgo operativo** — pérdidas por fallos en personas, procesos o sistemas. La autenticación JWT, validación Pydantic y persistencia ORM mitigan esta categoría a nivel de plataforma.

Cada módulo del dashboard (M1–M13) implementa una respuesta cuantitativa a uno o más de estos riesgos.

---

## Módulos de Análisis

### M1 — Análisis Técnico

Calcula y visualiza los indicadores clásicos de análisis técnico sobre la serie de precios de cierre: media móvil simple (SMA 20 y 50 períodos), media móvil exponencial (EMA 20), RSI con zona de sobrecompra/sobreventa, MACD con histograma de momentum, Bandas de Bollinger (2σ) y Oscilador Estocástico (%K, %D). Todos los indicadores se grafican de forma interactiva con Plotly y se actualizan al cambiar el rango de fechas o el activo.

### M2 — Distribución de Rendimientos

Calcula retornos simples y logarítmicos diarios. Presenta estadísticas descriptivas completas (media, desviación estándar, asimetría, exceso de curtosis, mínimo, máximo) **tanto para rendimientos simples como logarítmicos**, y aplica dos tests formales de normalidad: Jarque-Bera y Shapiro-Wilk. Cada prueba devuelve `stat`, `p_value`, `rejects_normal` (boolean) e `interpretation` textual con la regla de decisión: si p < 0.05 se rechaza la normalidad; si p ≥ 0.05 no se rechaza (lo que **no implica** que sí lo sea).

**Datos disponibles para visualización:**
- Histograma de rendimientos + **curva normal superpuesta** (200 puntos precomputados con μ y σ empíricos)
- Gráfico Q-Q empírico vs cuantiles teóricos N(0,1)
- **Boxplot stats**: cuartiles Q1/median/Q3, IQR, whiskers (regla 1.5·IQR), número de outliers
- Series temporales de rendimientos simples y log

**Hechos estilizados detectados automáticamente:**
- Colas pesadas (curtosis exc. > 1)
- Asimetría negativa (skewness < 0)
- Agrupamiento de volatilidad (Ljung-Box sobre r² con p < 0.05)
- **Efecto apalancamiento**: correlación entre |r_(t-1)| y r_t (negativa significativa indica el clásico efecto de Black 1976)

Cada bloque incluye `interpretation` textual que explica qué significa el resultado.

#### Por qué log-rendimientos

El proyecto usa **log-rendimientos** como base estadística por tres razones:

1. **Aditividad temporal:** la suma de log-retornos diarios es exactamente el log-retorno acumulado del período. Los retornos simples no son aditivos (hay que multiplicar `(1+r_t)` y restar 1).
2. **Simetría:** una caída del 50% seguida de una subida del 100% se anula con log-retornos (`-0.693 + 0.693 = 0`); con retornos simples queda un sesgo (`-0.5 × 2.0 - 1 = 0`, pero las medias se distorsionan).
3. **Aproximación a la normal:** para retornos pequeños (|r| < 5%) los log-retornos ≈ retornos simples, y se prestan mejor a tests paramétricos como Jarque-Bera.

Esta justificación también aparece en el response del endpoint bajo la clave `log_returns_justification`.

### M3 — Modelos de Volatilidad (EWMA + ARCH/GARCH/EGARCH)

Ajusta cuatro enfoques de volatilidad condicional sobre el mismo activo:

- **EWMA (RiskMetrics)** con λ configurable (default 0.94) vía query `lambda_ewma`. Validado por Pydantic (0 < λ < 1). Devuelve la serie completa, el último valor, la media histórica, la **volatilidad muestral rodante de 30 días** (serie completa para comparación visual) y el promedio rodante.
- **ARCH(1)**, **GARCH(1,1)** y **EGARCH(1,1)** estimados por máxima verosimilitud. Tabla comparativa con Log-Likelihood, AIC, BIC; selección automática del mejor por AIC.

#### Parámetros GARCH(1,1) extraídos en cada llamada
El response incluye `GARCH(1,1).parameters` con:
- `omega` (ω), `alpha` (α), `beta` (β) estimados por MLE
- `persistence` = α + β
- `unconditional_var` = ω/(1−α−β) y `unconditional_vol` = √(ω/(1−α−β))/100
- `mean_reversion` (boolean): true si α+β < 1
- `interpretation` textual

#### Diagnósticos sobre residuos estandarizados
- **Jarque-Bera** (normalidad de residuos): `JB_Stat`, `JB_Pvalue`, `Normal`, `JB_Interpretation` textual
- **ARCH-LM (Engle 1982)** con 5 lags: `lm_stat`, `lm_pvalue`, `passed`, `interpretation`
- Serie de residuos disponible para gráfico

#### Pronóstico de volatilidad
`Forecast_10d`: 10 valores diarios (volatilidad pronosticada por GARCH(1,1)).

#### Tabla comparativa EWMA vs GARCH(1,1) (formato del profesor)
El response incluye `comparison_table.rows` con 6 aspectos: parámetros estimados, varianza incondicional, reversión a la media, costo computacional, captura de asimetría, interpretación.

#### ¿Por qué EWMA y por qué además GARCH?

**Ventajas de EWMA:**
- **Parsimonia:** no requiere estimación de parámetros (con λ fijo) — solo recursión
- **Decay constante:** cada observación pesa proporcionalmente menos por (1−λ)·λ^k
- **Costo computacional mínimo:** una pasada sobre los datos
- **Estándar industria:** RiskMetrics de J.P. Morgan usa λ=0.94 para series diarias

**Limitaciones de EWMA:**
- **No captura asimetría:** una caída grande pesa lo mismo que una subida grande del mismo tamaño absoluto
- **Sin varianza incondicional finita:** el proceso es no-estacionario; no hay σ² al que converger
- **Sin reversión a la media:** la volatilidad no vuelve a un nivel de largo plazo
- **No modela el efecto apalancamiento:** la asimetría sí/no requiere variantes (EGARCH, GJR-GARCH)

**Por qué se necesita ARCH/GARCH además:**
- **Clustering de volatilidad:** GARCH modela explícitamente que σ²_t depende de σ²_(t-1) y de ε²_(t-1)
- **Reversión a la media:** si α+β<1, la varianza reverte al nivel ω/(1−α−β)
- **Asimetría:** EGARCH y GJR-GARCH capturan el efecto apalancamiento (caídas → más volatilidad que subidas equivalentes)
- **Persistencia medible:** α+β cuantifica cuánto persisten los shocks (cerca de 1 = muy persistentes)
- **Diagnósticos formales:** los residuos GARCH se pueden someter a JB y ARCH-LM para validar el ajuste

Los retornos se escalan por 100 antes del fit para mejorar la convergencia numérica del optimizador (el resultado se vuelve a dividir por 100).

### M4 — Riesgo Sistemático (CAPM)

Estima el modelo CAPM por regresión OLS entre los retornos del activo y los del benchmark (S&P 500).

**Endpoints:**
- `GET /capm/{ticker}` — análisis CAPM detallado de un solo activo (con scatter, VaR, etc.)
- `GET /capm` — **tabla resumen consolidada** de los 5 activos del portafolio: Beta, Alpha, R², rendimiento esperado anual, descomposición de varianza, clasificación. Usa Rf real obtenida automáticamente.

**Tasa libre de riesgo automática:** la Rf usada por CAPM se obtiene en este orden:
1. **FRED** (DGS3MO) si `FRED_API_KEY` está configurada — con cache transparente en SQLite
2. **yfinance ^IRX** como fallback si FRED no está disponible
3. **`config.default_rf` del .env** como último recurso

El response devuelve `rf_source` indicando cuál se usó (ej: `"FRED.DGS3MO (2026-05-12)"` o `"yfinance.^IRX"`).

**Cálculos por activo:**
- **Beta** por regresión MCO (`scipy.stats.linregress`) sobre retornos diarios alineados por fecha
- **Alpha de Jensen** = intercepto de la regresión
- **R²** = proporción de varianza del activo explicada por el mercado
- **Beta_StdErr** = error estándar del coeficiente Beta
- **E[R] anual** = Rf + β · (E[Rm] − Rf), formulación CAPM clásica
- **Clasificación:**
  - **Agresivo** si β > 1.05 (amplifica al mercado)
  - **Defensivo** si β < 0.95 (menos sensible)
  - **Neutro** si β ≈ 1 (entre 0.95 y 1.05)

**Descomposición de varianza** (clave para diversificación):
- `var_systematic` = β² · σ²_m (riesgo sistemático no diversificable)
- `var_idiosyncratic` = σ²_total − σ²_systematic (riesgo específico del activo)
- `systematic_share` = porcentaje de varianza explicado por el mercado
- `interpretation` textual

#### Riesgo sistemático vs. no sistemático: por qué importa

El riesgo total de un activo se descompone en:

> **σ²_total = β² · σ²_mercado + σ²_ε**
>
> └─── sistemático ─────┘ └ idiosincrático ┘

- **Riesgo sistemático (β² · σ²_m):** sensibilidad al mercado. Refleja el ciclo macro, política monetaria, eventos sistémicos. **NO se puede diversificar** combinando activos del mismo mercado.
- **Riesgo idiosincrático (σ²_ε):** específico del activo (gestión, sector, eventos individuales). **SÍ se reduce** combinando activos descorrelacionados.

A medida que el portafolio crece con activos descorrelacionados, la varianza idiosincrática tiende a cero (Ley de los Grandes Números), pero la sistemática persiste. Esto es **el límite teórico de la diversificación**: por más activos que agregues, no puedes eliminar el riesgo sistemático del mercado en el que operas.

Esta es exactamente la razón por la que el M6 (Markowitz) busca el portafolio óptimo respetando este límite, y el M11 (Stress Testing) somete al portafolio a shocks sistemáticos para medir su exposición residual.

### M5 — Valor en Riesgo (VaR) y CVaR

Calcula VaR y CVaR bajo tres metodologías sobre el retorno del portafolio:
- **Histórico** (no paramétrico): percentil empírico de los retornos observados — no asume distribución.
- **Paramétrico** (asume Normal): `VaR = μ + σ·z_α`, `CVaR = μ − σ·φ(z_α)/(1−α)`.
- **Monte Carlo**: simulación de ≥10,000 retornos sobre `Normal(μ_emp, σ_emp)` con **semilla configurable** (default 42) para reproducibilidad.

**Endpoint:** `POST /var`

**Request body Pydantic:**
- `tickers: list[str]`, `weights: list[float]` (validados: deben sumar 1)
- `confidence: float ∈ [0.80, 0.99]` (default 0.95) — el response siempre incluye también VaR al 99%
- `n_simulations: int ≥ 10,000` (default 10,000)
- `seed: int` (default 42)
- `lookback_days: int ≥ 250` (default 500) — ventana mínima exigida por Kupiec

**Response Pydantic anidado:**
- `parametric`, `historical`, `montecarlo`: cada uno con `VaR_daily`, `VaR_annual`, `CVaR_daily`, `CVaR_annual`
- `cvar`: bloque dedicado con CVaR por método + interpretación textual de Expected Shortfall
- `var_99`: mismo bloque pero al 99% de confianza
- `comparison_table`: tabla con interpretación de diferencias entre métodos
- `kupiec_test`: bloque con `lookback_days`, `by_method` (Kupiec sobre los 3), `passes_summary`, `methods_passing`, `interpretation`
- `chart_data`: serie del retorno del portafolio + `vertical_lines` para graficar VaR/CVaR como líneas verticales sobre el histograma

#### CVaR (Expected Shortfall)
CVaR responde a la pregunta "si el VaR se excede, ¿cuál es la pérdida esperada?". Es siempre ≥ VaR y captura la **severidad del riesgo de cola** que el VaR deja afuera por construcción.

#### Backtesting de Kupiec (POF)
El test de Proportion of Failures aplica a los **3 métodos** simultáneamente:
- Cuenta excedencias N en una ventana de T días (mínimo 250)
- Estadístico `LR_POF = -2·ln[(1-p)^(T-N)·p^N] + 2·ln[(1-N/T)^(T-N)·(N/T)^N]`
- Distribución H0: chi²(1). Se rechaza H0 si `LR_POF > 3.84` (al 95%)
- **Verdict textual:** `modelo correcto` / `subestima el riesgo` / `sobreestima el riesgo`
- `interpretation`: explicación completa con tasa observada vs esperada

#### Visualización
`chart_data.portfolio_returns` contiene la serie de retornos del portafolio. `chart_data.vertical_lines` mapea cada VaR/CVaR (95% y 99%) al nivel donde el frontend debe dibujar la línea vertical sobre el histograma.

### M6 — Optimización de Portafolio (Markowitz por Monte Carlo + QP)

#### Formulación explícita del problema (M-II-4)

```
minimizar:    wᵀ Σ w                    (varianza del portafolio)
sujeto a:     wᵀ μ  = μ*                (rendimiento objetivo, opcional)
              Σᵢ wᵢ = 1                 (los pesos suman 1)
              wᵢ ≥ 0   ∀ i              (no-negatividad — versión long-only)
```

donde **w** es el vector de pesos, **μ** es el vector de rendimientos esperados anualizados, **Σ** es la matriz de covarianzas anualizada, y **μ\*** el rendimiento objetivo. Es un **problema cuadrático convexo** (función objetivo cuadrática + restricciones lineales).

#### Implementación

Cuatro niveles conviven en el módulo:

1. **Simulación Monte Carlo (10,000 portafolios)** con pesos aleatorios y Σwᵢ = 1, para visualizar el conjunto factible.
2. **Programación cuadrática explícita (SLSQP)** que resuelve numéricamente:
   - **Mínima varianza global**: `min wᵀΣw  s.t. Σwᵢ = 1`
   - **Máximo Sharpe**: `max (wᵀμ − Rƒ)/√(wᵀΣw)  s.t. Σwᵢ = 1`
   En **dos versiones**: long-only (wᵢ ≥ 0) y con short-selling permitido (wᵢ ∈ [-1, 1]).
3. **Frontera eficiente paramétrica**: 30 puntos resolviendo `min wᵀΣw  s.t. wᵀμ = μ*` para cada μ* en el grid `[μ_min, μ_max]`. Devuelve `efficient_frontier_curve_long_only` y `efficient_frontier_curve_with_short` con `target_returns`, `min_volatility`, `weights` por punto y `n_converged`.
4. **Rendimiento objetivo individual** (endpoint `/portfolio/target`): dada una tasa anual deseada, encuentra la composición exacta.

#### Comparación con/sin no-negatividad (M-II-5)

El response `comparison_long_only_vs_short_allowed` incluye:

- **`composition_table`**: filas por activo con pesos en % para min-var y max-Sharpe, en ambas versiones (long-only y short-permitido)
- **`zero_weight_in_long_only`**: lista de activos que quedan en cero (esquinas del conjunto factible)
- **`short_positions_when_allowed`**: lista de activos en posición corta cuando se permite
- **`restriction_cost`**: costo cuantificado de imponer no-negatividad
  - `delta_sharpe_max`, `delta_return_max`, `delta_volatility_max` sobre el max-Sharpe
  - `delta_min_variance` sobre el portafolio de mínima varianza
- **`interpretation`** textual

#### Discusión del costo de la restricción

Cuando se impone wᵢ ≥ 0:
- **Pierde flexibilidad** para combinar posiciones cortas con largas (no puede usar venta en corto para reducir varianza)
- **Gana realismo** para inversionistas minoristas que no pueden vender en corto fácilmente
- La frontera con no-negatividad es **un subconjunto** de la frontera sin restricción → siempre la "domina por debajo": misma volatilidad o más para un retorno objetivo dado
- En este portafolio: ΔSharpe ≈ +0.058 al permitir short, ΔVol_min ≈ 0 → la restricción no es vinculante en mín-varianza pero sí cuesta algo en max-Sharpe

La matriz de correlación entre activos se visualiza como heatmap interactivo (escala RdBu de −1 a +1).

### M7 — Señales y Alertas Técnicas (con persistencia)

Evalúa de forma automática las **cinco reglas técnicas** exigidas por el instructivo
sobre el último registro disponible de cada activo y genera señales accionables con
**explicaciones en lenguaje natural** prudentes (sin recomendaciones absolutas).

**Reglas implementadas (`SignalGenerator.evaluate_all`):**
1. **Cruce del MACD** — línea MACD cruzando línea de señal (con fallback al histograma si las líneas no están disponibles).
2. **RSI extremo** — sobrecompra/sobreventa con umbrales configurables.
3. **Bandas de Bollinger** — precio tocando o cruzando banda superior/inferior (±k·σ).
4. **Cruce de medias móviles** — Golden cross (SMA20 cruza por encima de SMA50) y Death cross (lo opuesto).
5. **Oscilador Estocástico** — %K cruzando %D en zonas extremas.

La lógica vive en una **clase `SignalGenerator`** (`backend/app/services/signals.py`)
con un método por regla (`macd_cross`, `rsi_extreme`, `bollinger_touch`,
`moving_average_cross`, `stochastic_signal`) y un agregador `evaluate_all`.

**Endpoints:**
- `GET /api/v1/signals/{ticker}` y alias `GET /alertas/{ticker}` — devuelve un `SignalReport` (Pydantic) con ticker, timestamp, lista de señales tipadas, thresholds usados y `persisted_count`.
- `GET /alertas` (sin ticker) — **panel consolidado**: itera sobre todos los activos del portafolio y retorna un reporte por activo + total agregado.
- `GET /api/v1/signals/{ticker}/history` y alias `GET /alertas/{ticker}/history` — historial persistido de señales.

**Umbrales configurables vía query parameters** (validados por Pydantic):
- `rsi_overbought` (50–99, default 70)
- `rsi_oversold` (1–50, default 30)
- `bollinger_std` (0 < σ ≤ 5, default 2.0)
- `stoch_overbought` (50–99, default 80)
- `stoch_oversold` (1–50, default 20)
- `persist` (bool, default true)

**Persistencia en `signals_log`** (SQLAlchemy): cada señal disparada se guarda con
`timestamp`, `ticker`, `rule`, `value` y `note`. Se evita duplicar señales del mismo
ticker/regla/día.

**Modelo Pydantic `SignalReport`** (response model del endpoint):
```python
class SignalItem(BaseModel):
    id: str
    rule: Optional[str]
    type: str            # "buy" | "sell"
    value: Optional[float]
    msg: str
    explanation: Optional[str]   # interpretación en lenguaje simple
```

### M9 — Renta Fija (Curva de Rendimiento + Nelson-Siegel + Bono)

Combina dos servicios financieros del mercado de tasas, encapsulados como clases
en `backend/app/services/`:

#### Clase `YieldCurve` (`services/yield_curve.py`)

- **Descarga de la curva del Tesoro EE.UU.** desde 6 series FRED:
  `DGS3MO`, `DGS1`, `DGS2`, `DGS5`, `DGS10`, `DGS30` (cubre todos los vencimientos clave).
- **Cache transparente en SQLite** vía tabla `fred_cache` con TTL 24h. `cache_status` por serie (`hit`/`miss`/`stale_used`) aparece en el response.
- **Fallback declarado**: si `FRED_API_KEY` no está configurada o falla, devuelve curva DEMO marcada como `source: "fallback_demo"` — el endpoint nunca finge que el dato es real de FRED.
- **`fit_nelson_siegel(maturities, yields)`** — ajusta los 4 parámetros (β₀, β₁, β₂, λ) por **mínimos cuadrados no lineales** con `scipy.optimize.least_squares` y bounds en λ ∈ [0.05, 30]. Reporta `rmse`, `fitted_yields`, `converged` (bool) y dos niveles de interpretación:
  - `interpretation`: diagnóstico cualitativo de la forma (normal / plana / invertida).
  - `param_interpretation`: explicación de cada parámetro (β₀ = nivel de largo plazo, β₁ = pendiente corto−largo, β₂ = curvatura, λ = velocidad de decay).
- **`spot_rate(tau)`** — yield estimado a cualquier plazo τ usando los parámetros ajustados (escalar o lista).
- **`interpolated_curve(n=50)`** — curva spot densa entre 0.25 y 30 años para visualización suave en el frontend (clave `interpolated_curve` en el response).

#### Clase `Bond` (`services/bond.py`)

Bono sintético con cupón fijo. Parámetros validados por Pydantic (`BondRequest`):
`face_value > 0`, `coupon_rate ∈ [0,1]`, `maturity_years ∈ (0,100]`, `yield_rate ∈ [0,1]`, `frequency ∈ {1,2,4,12}`.

Métodos analíticos:
- **`price()`** — descuento de flujos sobre el yield actual o un yield alternativo.
- **`macaulay_duration()`** — tiempo promedio ponderado de los flujos.
- **`modified_duration()`** — sensibilidad porcentual del precio: D* = D/(1+y/m).
- **`convexity()`** — curvatura del precio respecto al yield, corrección de segundo orden.
- **`shock_sensitivity()`** — para cada shock en `(-200, -100, -50, +50, +100, +200)` bp, compara **tres aproximaciones**:
  - `duration_only_pct`: ΔP/P ≈ −D*·Δy (lineal con duración)
  - `duration_convex_pct`: ΔP/P ≈ −D*·Δy + ½·C·(Δy)² (segundo orden)
  - `exact_pct`: reprice exacto descontando todos los flujos a (y + Δy)

`summary()` devuelve precio, las tres duraciones, convexidad, tabla completa de shocks e interpretación cualitativa según la duración (corto/intermedio/largo).

### M10 — Opciones Europeas (Black-Scholes + Greeks)

Implementa la fórmula de Black-Scholes para call/put con sus cinco Greeks (Delta, Gamma, Vega, Theta, Rho), verificación numérica de paridad put-call y cálculo de volatilidad implícita por Newton-Raphson. Validado contra valores de referencia (call ATM S=K=100, T=1, r=5%, σ=20% → 10.4506).

### M11 — Stress Testing

Aplica escenarios extremos sobre un portafolio (mercado, tasa, volatilidad y combinado "tormenta perfecta"). Cada escenario propaga el shock por activo según Beta y σ, retornando pérdida estimada absoluta y porcentual, impacto desagregado y un peor caso identificado con interpretación.

### M12 — Predicción ML (Random Forest + Singleton)

Pipeline completo `train → joblib → load → predict` con clasificación direccional (`buy`/`hold`/`sell`) sobre features técnicas (retornos rezagados, RSI, MACD, volatilidad rodante). El predictor implementa el **patrón Singleton** (carga del modelo una sola vez al primer request) y persiste cada predicción en `PredictionLog` para auditoría.

> ⚠️ **El módulo ML es una herramienta analítica académica.** No constituye recomendación financiera ni garantía de rentabilidad. Las predicciones son probabilísticas y no consideran costos de transacción, liquidez ni cambios de régimen.

### M8 — Macro y Benchmark

Combina dos sub-módulos del análisis de portafolio: el **panel macroeconómico** que
contextualiza el entorno de tasas, inflación y tipo de cambio, y la **comparación
portafolio óptimo vs. benchmark** que cuantifica el desempeño relativo.

#### Panel macroeconómico (`GET /api/v1/macro` / alias `GET /macro`)

Response tipado por Pydantic (`MacroIndicators`):

| Campo               | Fuente primaria          | Fallback           |
|---------------------|--------------------------|--------------------|
| `rf_rate`           | FRED.DGS3MO              | yfinance ^IRX      |
| `treasury_10y`      | FRED.DGS10               | yfinance ^TNX      |
| `spx_ytd`           | yfinance ^GSPC           | _(no fallback)_    |
| `inflation_yoy`     | FRED.CPIAUCSL (YoY calc) | _(opcional)_       |
| `usdcop`            | yfinance COP=X           | _(opcional)_       |

Cada métrica reporta su `_source` explícitamente — el endpoint **no finge** que el
dato viene de FRED cuando usa fallback. Cache transparente en SQLite con
**TTL 24h** vía `fred_cache` (controlado por `services/fred_service.py`).
`cache_status` por serie (`hit` / `miss` / `stale_used`) viaja en el response cuando FRED está habilitado.

#### Comparación con benchmark (`GET /api/v1/benchmark` / alias `GET /benchmark`)

Extiende el CAPM del M4 con métricas de desempeño relativo del portafolio Max Sharpe
contra el S&P 500. La función `compute_benchmark` (reutilizada también en
`/api/v1/all` y `data.js`) devuelve:

- **Curvas base 100**: `Port_Cum`, `Bench_Cum` con `Dates` alineadas
- **Series de drawdown**: `Port_DD`, `Bench_DD`
- **Métricas por activo** (Port y Bench): `Ann_Return`, `Ann_Volatility`, `Sharpe`, `Max_Drawdown`
- **Métricas relativas**: `Jensen_Alpha` (anualizado), `Beta`, `Tracking_Error` (anual), `Information_Ratio`, `R_Squared`
- **`interpretation`** textual: si el alpha es positivo/negativo, si el IR es favorable, si el Sharpe del portafolio supera al benchmark; sin recomendaciones absolutas
- **`benchmark_ticker`** y **`rf_rate`**: garantizan trazabilidad y consistencia con el CAPM del M4

#### Justificación del benchmark `^GSPC`

Todos los activos cotizan en USD en NYSE/NASDAQ. Coherencia de divisa (Rf y benchmark
ambos en USD), estándar académico para CAPM (Sharpe 1964, Lintner 1965), disponible
directamente en yfinance. La métrica `usdcop` se incluye para contextualizar el
poder adquisitivo en pesos colombianos pero NO se mezcla en la lógica del benchmark.

#### Significancia estadística del Alpha

El endpoint reporta `Jensen_Alpha` anualizado pero **no calcula la significancia
estadística** (t-stat sobre el intercepto de la regresión). Está documentado como
pendiente — el cálculo es trivial pero se deja para una fase posterior para no
alterar el contrato del endpoint.

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

### Cache transparente de precios (Yahoo Finance → SQLite)

`api/services/price_service.py` implementa la estrategia recomendada por el instructivo:

> *"si el dato existe en BD y la fecha es reciente, leer de BD; si no, llamar a la API externa y persistir el resultado antes de retornarlo"*

Estados posibles del cache (`cache_status`):
- `hit` — datos frescos en BD (TTL: 1 día)
- `miss` — descarga + persiste por primera vez
- `refresh` — fuerza descarga (`?fresh=true`)
- `stale_used` — yfinance falla, se reutiliza cache aunque vencido
- `unavailable` — no hay cache ni respuesta de yfinance

El cliente HTTP de yfinance tiene **3 reintentos con backoff exponencial** (1.5s, 3.0s) ante fallos de red. El cliente FRED implementa el mismo patrón.

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
python -m uvicorn backend.app.main:app --port 8001 --reload

# 5. Abrir frontend/dashboard.html en el navegador
# (apertura directa como archivo local o mediante servidor estático)
```

### Generar snapshot estático (opcional)

```bash
python generate_data.py
```

Descarga datos actuales de Yahoo Finance y escribe `data.js`. El dashboard lo carga automáticamente, lo que permite usar la herramienta sin tener el backend activo.

### Documentación interactiva de la API

Con el backend corriendo: `http://localhost:8001/docs`

### Ejecutar la suite de tests

```bash
pip install pytest "httpx<0.28"        # si no están en requirements
python -m pytest tests/ -v             # 16 tests, ~8s
```

Los tests usan SQLite **en memoria** y override de `Depends(get_db)` —
no tocan la BD real ni dependen de yfinance/FRED.

### Ejecutar con Docker

```bash
# Build de la imagen multi-stage (~200 MB final)
docker build -t risklab-usta .
docker run -p 8001:8001 \
    -e JWT_SECRET=tu_clave_segura \
    -e FRED_API_KEY=tu_key_opcional \
    risklab-usta

# Alternativa con hot-reload para desarrollo
docker compose up --build
```

El frontend (`dashboard/dashboard.html`) se abre directamente en el navegador
y consume la API. Si despliegas en cloud, recuerda actualizar `API_BASE` en
el HTML para apuntar a la URL pública.

### Integración continua

`.github/workflows/ci.yml` ejecuta `pytest` en cada push a `main`/`develop` o PR.
No requiere `FRED_API_KEY` real — los tests usan datos sintéticos y fallback DEMO.

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
| GET | `/api/v1/signals/{ticker}` | Señales técnicas automáticas + persistencia — M7 |
| GET | `/api/v1/signals/{ticker}/history` | Historial de señales persistidas — M7 |
| GET | `/alertas` | Panel consolidado de señales para todos los activos del portafolio — M7 |
| GET | `/alertas/{ticker}` | Alias corto del endpoint de señales por activo — M7 |
| GET | `/alertas/{ticker}/history` | Alias corto del historial persistido — M7 |
| GET | `/api/v1/curva-rendimiento` | Curva FRED + ajuste Nelson-Siegel — M9 |
| POST | `/api/v1/bono/duracion` | Bono sintético: precio, duración, convexidad — M9 |
| POST | `/api/v1/opcion/precio` | Black-Scholes + Greeks + paridad put-call — M10 |
| POST | `/api/v1/stress` | Stress testing del portafolio — M11 |
| POST | `/api/v1/predict` | Predicción ML direccional con logging — M12 |
| GET | `/api/v1/predict/info` | Metadata del modelo ML cargado — M12 |
| GET | `/api/v1/precios/{ticker}` | Precios OHLCV con cache transparente en SQLAlchemy |
| GET | `/api/v1/precios/cache` | Resumen del cache de precios (n_assets, n_prices) |
| GET | `/api/v1/opcion/precio/{ticker}` | Opción sobre un activo del portafolio (S y σ obtenidos automáticamente) |
| GET | `/api/v1/macro` | Contexto macro tipado (Rf, T10Y, S&P YTD, inflación, USDCOP) — M8 |
| GET | `/api/v1/benchmark` | Comparación Portafolio óptimo vs benchmark con Alpha, TE e IR — M8 |
| GET | `/benchmark` | Alias corto de la comparación contra benchmark — M8 |
| GET | `/api/v1/all` | Todos los módulos en una sola llamada |

**Parámetros comunes:**

| Parámetro | Rango | Default | Descripción |
|-----------|-------|---------|-------------|
| `start_date` | desde 2020-01-01 | — | Fecha de inicio (YYYY-MM-DD) |
| `end_date` | hasta hoy | — | Fecha de cierre (YYYY-MM-DD) |
| `confidence` | 0.80 – 0.99 | 0.95 | Nivel de confianza para VaR |
| `n_simulations` | 1,000 – 100,000 | 10,000 | Iteraciones Monte Carlo |
| `target_return` | −0.50 – 2.00 | — | Rendimiento anual objetivo para M6 |
| `lambda_ewma` | 0.0 – 1.0 (excl.) | 0.94 | Factor de decaimiento EWMA en M3 |
| `include_qp` | bool | `true` | Incluye QP determinista en M6 |
| `rsi_overbought` | 50 – 99 | 70 | Umbral de sobrecompra RSI en M7 |
| `rsi_oversold` | 1 – 50 | 30 | Umbral de sobreventa RSI en M7 |
| `persist` | bool | `true` | Guarda señales disparadas en `signals_log` |

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

### Render.com (preparado, despliegue manual)

El archivo `render.yaml` configura automáticamente el runtime Python, el build con `pip install`, el comando de inicio con `uvicorn`, la generación automática de `JWT_SECRET` y un disco persistente de 1 GB para la base de datos SQLite.

**Pasos para desplegar:**
1. Sube el repo a GitHub (push de la rama deseada)
2. Crea cuenta gratuita en https://render.com
3. **New → Web Service** → conecta el repositorio (Render detecta `render.yaml` automáticamente)
4. En el panel UI configura las env vars: `FRED_API_KEY` (opcional), `JWT_SECRET` (Render puede generar una aleatoria)
5. **Limitación free-tier:** el servicio duerme tras 15 min sin tráfico (cold start ~30s). Para una demo en vivo, hacer warmup con `curl` 1 minuto antes.

> Nota: el deploy efectivo requiere acción manual del usuario en Render UI; el repositorio está listo pero no se desplegó automáticamente desde este entorno.

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
