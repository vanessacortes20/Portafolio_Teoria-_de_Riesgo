# Guion de Sustentación — RiskLab USTA

> Documento de apoyo para la defensa oral del Proyecto Integrador de Riesgo Financiero.
> Estructura: bloques cortos por tema. No leer literalmente — usar como mapa.

---

## 1. Apertura (1 min)

> "RiskLab USTA es una plataforma de análisis cuantitativo de riesgo financiero. Está construida con FastAPI en el backend y un dashboard HTML/Plotly en el frontend. Cubre los 12 módulos del nuevo instructivo, persiste resultados en SQLite vía SQLAlchemy ORM y sirve un modelo de Machine Learning con patrón Singleton."

**Activos:** NU · AMZN · SONY · XOM · WPM · Benchmark: ^GSPC.

---

## 1.5. Marco conceptual de riesgo financiero (M-I-1, 1 min)

> "Antes de los indicadores, definamos el lenguaje. **Riesgo financiero** es la probabilidad de pérdida económica por incertidumbre futura. Se descompone en cuatro categorías:
>
> - **Riesgo de mercado** — pérdidas por movimientos adversos en precios. Es el foco central: lo cuantificamos con VaR/CVaR (M5), GARCH (M3), Beta CAPM (M4) y stress (M11).
> - **Riesgo de crédito** — incumplimiento de pago de una contraparte. Aparece indirectamente en M9: la curva de tesoros EE.UU. asume emisor libre de crédito.
> - **Riesgo de liquidez** — incapacidad de cerrar posición a precio razonable. Lo reconocemos como limitación: el VaR no incorpora bid-ask ni profundidad.
> - **Riesgo operativo** — fallos en personas, procesos o sistemas. Se mitiga a nivel de plataforma con JWT, validación Pydantic y persistencia ORM.
>
> Cada uno de los 12 módulos responde cuantitativamente a una o varias de estas categorías."

---

## 2. Arquitectura (2 min)

```
[ Dashboard HTML + Plotly ]
        ↕ fetch (JSON)
[ FastAPI (api/main.py) ]
        │  Pydantic v2 + Depends + JWT
        ├── api/logic.py        ← cálculos puros M1–M8
        ├── api/services/       ← FRED, YieldCurve, Bond, Options, Stress
        ├── api/ml/             ← train.py + predictor.py (Singleton)
        ├── api/db_models.py    ← SQLAlchemy ORM (6 modelos)
        └── api/database.py     ← sqlite3 directo (auth — convive)
```

**Decisión clave:** dos capas de persistencia coexisten (sqlite3 directo para auth, SQLAlchemy ORM para datos nuevos) en la misma BD `data/risklab_users.db`. Esto preserva el flujo de autenticación funcionando sin riesgo de regresión.

---

## 3. SQLAlchemy ORM (1 min)

6 modelos: `Asset`, `Price`, `Portfolio`, `PredictionLog`, `SignalLog`, `FredCache`. Sesión inyectada con `Depends(get_db)`. `Base.metadata.create_all()` se llama en startup (idempotente). Las tablas legacy (`users`, `reset_tokens`) **no se tocaron**.

---

## 4. FRED (1 min)

Servicio en `api/services/fred_service.py` con cache transparente en `FredCache` (TTL 24h). Soporta DGS3MO, DGS10, CPIAUCSL, DGS1/DGS2/DGS5/DGS30. Si `FRED_API_KEY` no está o es placeholder, se activa fallback a yfinance (`/api/v1/macro`) o curva DEMO marcada como `source: fallback_demo` (`/curva-rendimiento`).

---

## 4.5 Rendimientos y log-rendimientos (M2, 1 min)

> "Para el M2 calculamos retornos simples y logarítmicos. Usamos log-rendimientos como base estadística por tres razones: son **aditivos en el tiempo** (la suma de log-rets diarios es el log-ret acumulado), son **simétricos** (una caída del 50% seguida de una subida del 100% se anula), y se aproximan a los simples cuando son pequeños. Esto los hace ideales para tests paramétricos."
>
> "El endpoint `/rendimientos/{ticker}` expone media, desviación, asimetría y curtosis para ambas series; pruebas Jarque-Bera y Shapiro-Wilk con interpretación textual del p-valor; histograma con curva normal superpuesta; Q-Q; boxplot con cuartiles y outliers; y un panel de hechos estilizados con detección automática de **colas pesadas** (curtosis exc. > 1), **agrupamiento de volatilidad** (Ljung-Box sobre r²) y **efecto apalancamiento** (correlación entre |r_(t-1)| y r_t)."
>
> "En la práctica vemos que los 5 activos rechazan normalidad — eso justifica usar VaR histórico y Monte Carlo en M5 y modelos GARCH en M3, en vez de confiar solo en el supuesto normal."

---

## 5. EWMA vs GARCH (M3, 2 min)

**EWMA RiskMetrics:** σ²ₜ = λσ²ₜ₋₁ + (1-λ)r²ₜ₋₁, λ=0.94 default, configurable vía `?lambda_ewma=`.

**Ventajas de EWMA:** parsimonia (0 parámetros estimados), decay exponencial constante, costo computacional mínimo (recursión).

**Limitaciones:** no captura asimetría, no tiene varianza incondicional finita, no modela reversión a la media.

**GARCH(1,1):** σ²ₜ = ω + αε²ₜ₋₁ + βσ²ₜ₋₁ con tres parámetros estimados por máxima verosimilitud. Si α+β<1 la varianza reverte al nivel ω/(1−α−β) (varianza incondicional). Persistencia α+β cuantifica cuánto duran los shocks.

**¿Por qué se necesita GARCH además de EWMA?** Por cuatro razones:
1. Para tener varianza incondicional medible (objetivo a largo plazo).
2. Para que el modelo tenga reversión a la media.
3. Para capturar asimetría (vía EGARCH o GJR-GARCH).
4. Para someter los residuos a diagnósticos formales (JB y ARCH-LM).

El endpoint `/volatilidad/{ticker}` devuelve los parámetros ω, α, β estimados, persistencia, varianza incondicional, los tres modelos (ARCH, GARCH, EGARCH) con AIC/BIC, residuos estandarizados, JB+ARCH-LM con interpretación, pronóstico a 10 días y la **tabla comparativa estructurada** EWMA vs GARCH(1,1).

El **test ARCH-LM** (Engle 1982) sobre residuos GARCH(1,1) con 5 lags es el test que valida si el modelo capturó adecuadamente la heterocedasticidad: p > 0.05 → no quedan efectos ARCH residuales; p ≤ 0.05 → el modelo deja heterocedasticidad sin capturar (necesita más lags).

---

## 6. Markowitz QP (M6, 2 min)

Tres niveles conviven:
1. **Monte Carlo (10k portafolios)** para visualizar el conjunto factible
2. **QP determinista (SLSQP)** que resuelve `min wᵀΣw  s.t. Σwᵢ=1` para mín varianza, y `max (μᵀw − Rf)/√(wᵀΣw)` para máx Sharpe
3. **Comparación con/sin no-negatividad**: long-only (wᵢ≥0) vs short permitido (wᵢ∈[-1,1])

Con el portafolio actual NU es el primero en quedar en 0 en long-only y en posición corta cuando se permite. La ganancia de Sharpe es marginal — el long-only ya está cerca del óptimo no restringido.

---

## 7. Señales persistidas (M7, 1 min)

Cada llamada a `/api/v1/signals/{ticker}` persiste las señales disparadas en `signals_log` (vía `Depends(get_db)`), con dedup por `(ticker, regla, día)`. Umbrales `rsi_overbought`, `rsi_oversold`, `bollinger_std` configurables y validados por Pydantic. Endpoint `/api/v1/signals/{ticker}/history` devuelve el historial.

---

## 8. Renta Fija (M9, 2 min)

**Curva FRED** (DGS3MO, DGS1, DGS2, DGS5, DGS10, DGS30) + **Nelson-Siegel** ajustado por `scipy.optimize.least_squares`:

```
y(τ) = β₀ + β₁ · ((1-e^(-τ/λ))/(τ/λ)) + β₂ · ((1-e^(-τ/λ))/(τ/λ) - e^(-τ/λ))
```

β₀=nivel, β₁=pendiente, β₂=curvatura, λ=decay. Reporta RMSE.

---

## 9. Bono sintético: duración y convexidad (M9, 1 min)

Clase `Bond` con:
- **Duración Macaulay:** D = Σ(t·CFₜ/(1+y)ᵗ) / P
- **Modificada:** D* = D/(1+y/m)
- **Convexidad:** C = Σ(t(t+1/m)·CFₜ/(1+y/m)^(tm+2)) / P / (1+y/m)²

Comparación de tres aproximaciones por shock de tasa: lineal con D, segundo orden con C, reprice exacto. Endpoint POST `/api/v1/bono/duracion`.

---

## 10. Opciones Black-Scholes + Greeks (M10, 2 min)

```
d₁ = [ln(S/K) + (r + σ²/2)T] / (σ√T)
d₂ = d₁ − σ√T
Call = S·N(d₁) − K·e^(-rT)·N(d₂)
Put  = K·e^(-rT)·N(-d₂) − S·N(-d₁)
```

Las **5 Greeks** (Δ, Γ, ν, Θ, ρ) se calculan analíticamente. El método `put_call_parity_check` verifica numéricamente que `C - P = S - K·e^(-rT)` con `abs_diff < 1e-6`. Validado contra valor de referencia: S=K=100, T=1, r=5%, σ=20% → call=10.4506.

---

## 11. Stress Testing (M11, 1 min)

Clase `StressTester` aplica shocks tipados (mercado, tasa, volatilidad, combinado) sobre el portafolio. Cada activo se mueve según su Beta y σ. Endpoint POST `/api/v1/stress` retorna pérdida estimada absoluta y porcentual, impacto desagregado y peor escenario con interpretación. **Pesos validados** por `model_validator` para garantizar Σwᵢ = 1.

---

## 12. Machine Learning + Singleton (M12, 2 min)

Pipeline completo train→joblib→load→predict:
- `api/ml/train.py` entrena `RandomForestClassifier(n_estimators=200, max_depth=8)` con features `ret_lag_1..3, rsi, macd_hist, vol_5, vol_20`
- Target: dirección del retorno del día siguiente (sell/hold/buy)
- `train_test_split(shuffle=False)` para respetar orden temporal
- `api/ml/predictor.py` implementa **Singleton** vía `__new__`: el modelo se carga UNA sola vez al primer request
- Cada predicción se persiste en `PredictionLog` con `model_version`, `ticker`, `input_features`, `prediction`, `timestamp`

**⚠️ Disclaimer obligatorio:** modelo académico de demostración. NO es recomendación financiera.

---

## 13. Tests pytest (1 min)

`tests/conftest.py` configura TestClient con BD SQLite en memoria + override de `Depends(get_db)`. **16 tests** cubren:
- RSI sobre series alcista/bajista conocidas
- VaR paramétrico vs valor analítico (norm.ppf)
- Paridad put-call (con strike igual y distinto)
- `/docs` y `/openapi.json` responden
- Validaciones 422: T=0 en opciones, frequency inválida en bono, NaN en features ML, pesos≠1 en stress
- Esquemas válidos en /predict, /stress, /bono/duracion, /curva-rendimiento

Sin dependencias de yfinance ni FRED en vivo. CI corre `pytest` en cada push.

---

## 14. Docker (1 min)

**Dockerfile multi-stage:** builder con compiladores + runtime sin ellos. Imagen final ~200 MB (vs 600 MB single-stage). `docker-compose.yml` para desarrollo local con hot-reload. `.dockerignore` excluye `.env`, BD, backups, tests y docs.

---

## 15. CI/Deploy (1 min)

`.github/workflows/ci.yml` ejecuta pytest en push/PR. **Render.yaml** preparado pero el deploy requiere acción manual: subir el repo a GitHub, conectar a Render, configurar `JWT_SECRET` y `FRED_API_KEY` en el panel UI. El servicio free-tier duerme tras 15 min de inactividad — hacer warmup antes de la demo.

---

## 16. Limitaciones reconocidas (1 min)

- ML accuracy ~40% sobre 3 clases — modelo de demostración, no apto para producción
- Yahoo Finance puede limitar requests; FRED requiere API key gratuita (con fallback DEMO)
- Solo 2/5 activos están en S&P 500 (AMZN, XOM); el resto cotiza en NYSE pero no integra el índice — defendido como "correlación funcional con riesgo sistémico USD"
- WPM dispara warning de convergencia en EGARCH (no bloquea, conocido)
- VaR paramétrico asume normalidad — el M2 demuestra que se rechaza, por eso M5 tiene 3 métodos en paralelo

---

## 17. Cierre (1 min)

> "El proyecto cubre los 12 módulos exigidos por el nuevo instructivo respetando el principio de **conservar + extender**: no se eliminó ningún endpoint, login, dashboard ni capa anterior. Las cinco fases del plan se ejecutaron en rama dedicada `feature/instructivo-iii` con commits atómicos en español. El sistema está listo para desplegarse en Render con un push."

---

## Apéndice: comandos rápidos para la demo

```bash
# Arrancar backend
python -m uvicorn api.main:app --port 8001

# Login
curl -X POST http://localhost:8001/auth/login \
    -d "username=admin&password=Admin2025%21" \
    -H "Content-Type: application/x-www-form-urlencoded"

# Probar endpoints clave
curl http://localhost:8001/api/v1/curva-rendimiento
curl -X POST http://localhost:8001/api/v1/opcion/precio \
    -H "Content-Type: application/json" \
    -d '{"S":100,"K":100,"T":1,"r":0.05,"sigma":0.20,"option_type":"call"}'

# Tests
python -m pytest tests/ -v

# Docker
docker compose up --build
```

---

*Última actualización: cierre Fase 5 — listo para sustentación.*
