# RiskLab USTA — Plataforma de Análisis Cuantitativo de Riesgo Financiero

**Proyecto Integrador — Riesgo Financiero**  
Universidad Santo Tomás (USTA) · 2024–2026

---

## 1. Descripción General

RiskLab USTA es una plataforma web de análisis cuantitativo de riesgo financiero que integra nueve módulos (M1–M9) desarrollados en Python/FastAPI (backend) y HTML/Plotly.js (frontend). La plataforma descarga datos en tiempo real desde Yahoo Finance y permite analizar activos individuales y portafolios multi-activo bajo los marcos teóricos de Markowitz, CAPM, modelos ARCH/GARCH y métricas de riesgo VaR/CVaR.

---

## 2. Arquitectura del Sistema

```
proyecto_2/
├── api/
│   ├── main.py          # FastAPI app — endpoints REST + autenticación JWT
│   ├── logic.py         # Funciones de cómputo estadístico (M1–M7)
│   ├── data.py          # Descarga de datos históricos (yfinance)
│   └── database.py      # SQLite — usuarios, tokens de reset, sesiones
├── dashboard.html        # Frontend completo (HTML + CSS + JS + Plotly.js)
├── generate_data.py      # Generador de data.js (datos estáticos de respaldo)
├── data.js               # Snapshot estático generado por generate_data.py
├── requirements.txt      # Dependencias Python
├── Dockerfile            # Imagen Docker de producción
├── Procfile              # Comando de inicio para plataformas PaaS (Render/Heroku)
├── render.yaml           # Configuración de despliegue en Render.com
└── .env.example          # Variables de entorno requeridas
```

### Patrón de datos híbrido

El dashboard carga por defecto el snapshot estático `data.js` (generado por `generate_data.py`) para garantizar disponibilidad sin backend. Al presionar **"Recalcular (API)"**, el frontend consume los endpoints REST y actualiza todos los módulos con datos en tiempo real desde Yahoo Finance.

---

## 3. Módulos Implementados

| Módulo | Nombre | Técnicas principales |
|--------|--------|----------------------|
| M1 | Análisis Técnico | SMA, EMA, RSI, MACD, Bollinger Bands, Estocástico |
| M2 | Distribución de Rendimientos | Retornos simples y logarítmicos, Jarque-Bera, Shapiro-Wilk, Q-Q plot, hechos estilizados |
| M3 | Modelos de Volatilidad | ARCH(1), GARCH(1,1), EGARCH(1,1) — AIC/BIC, pronóstico 10 días |
| M4 | Riesgo Sistemático (CAPM) | Beta, Alpha, R², retorno esperado diario y anualizado, scatter vs. benchmark |
| M5 | VaR y CVaR | Histórico, Paramétrico Normal, Monte Carlo; backtesting con Test de Kupiec (POF) |
| M6 | Optimización de Portafolio | Frontera eficiente Markowitz (10,000 simulaciones), optimización por rendimiento objetivo (SLSQP) |
| M7 | Señales y Alertas | Reglas sobre RSI, MACD, Bandas de Bollinger con explicaciones en lenguaje natural |
| M8 | Portafolio vs. Benchmark | Alpha de Jensen, Beta, Tracking Error, Information Ratio, Max Drawdown, retorno acumulado vs. S&P 500; contexto macro (Rf, T10Y, S&P YTD) |
| M9 | Oportunidades de Mejora | Tres mejoras priorizadas con impacto y esfuerzo estimados |

---

## 4. Instalación y Ejecución Local

### Requisitos previos

- Python 3.10+
- pip

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
# Editar .env: cambiar JWT_SECRET por un valor aleatorio seguro

# 4. Iniciar el backend
uvicorn api.main:app --port 8001 --reload

# 5. Abrir dashboard.html en el navegador
# (abrir directamente el archivo o servir con un servidor estático)
```

### Generar datos estáticos (opcional)

```bash
python generate_data.py
```

Este comando descarga datos de Yahoo Finance y genera `data.js` con el snapshot más reciente. El dashboard lo carga automáticamente al abrir `dashboard.html`.

---

## 5. Autenticación

La plataforma implementa autenticación JWT completa:

- **Registro**: `POST /auth/register` — nombre, apellido, teléfono, cédula, email, usuario, contraseña (mínimo 8 caracteres, confirmación requerida).
- **Login**: `POST /auth/login` — usuario o correo + contraseña → `access_token` JWT.
- **Perfil**: `GET /auth/me` — devuelve datos del usuario autenticado.
- **Cambio de contraseña**: `POST /auth/change-password`.
- **Restablecimiento**: `POST /auth/reset-password` + `POST /auth/reset-password/confirm` con token de 1 hora.
- **Admin**: `GET /auth/users` — lista todos los usuarios (solo rol `admin`).

Tokens firmados con HS256, TTL configurable vía `JWT_TTL_MINUTES` (default: 60 min). Contraseñas hasheadas con BCrypt.

**Usuarios demo** (creados automáticamente al iniciar):

| Usuario | Contraseña | Rol |
|---------|-----------|-----|
| `admin` | `admin123` | admin |
| `demo` | `demo1234` | user |

---

## 6. Endpoints REST principales

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | `/` | Estado de la API |
| GET | `/api/v1/technical/{ticker}` | Indicadores técnicos M1/M7 |
| GET | `/api/v1/returns/{ticker}` | Estadísticas de retornos M2 |
| GET | `/api/v1/volatility/{ticker}` | Modelos ARCH/GARCH M3 |
| GET | `/api/v1/risk/{ticker}` | CAPM, VaR, CVaR M4/M5 |
| GET | `/api/v1/risk/{ticker}/backtest` | Backtesting VaR — Test de Kupiec M5 |
| GET | `/api/v1/portfolio/optimize` | Frontera eficiente Markowitz M6 |
| GET | `/api/v1/portfolio/target` | Optimización por rendimiento objetivo M6 |
| GET | `/api/v1/signals/{ticker}` | Señales técnicas automáticas M7 |
| GET | `/api/v1/macro` | Indicadores macro (Rf, T10Y, S&P YTD) M8 |
| GET | `/api/v1/all` | Todos los módulos en una sola llamada |

**Parámetros comunes:**
- `start_date` / `end_date` (YYYY-MM-DD): rango histórico (mínimo: 2020-01-01)
- `confidence` (0.80–0.99): nivel de confianza VaR (default: 0.95)
- `n_simulations` (1,000–100,000): iteraciones Monte Carlo (default: 10,000)

Documentación interactiva: `http://localhost:8001/docs`

---

## 7. Variables de Entorno

| Variable | Descripción | Default |
|----------|-------------|---------|
| `JWT_SECRET` | Clave secreta para firmar tokens JWT | Generada aleatoriamente |
| `JWT_TTL_MINUTES` | Tiempo de vida del token en minutos | `60` |
| `PORTFOLIO_TICKERS` | Tickers del portafolio separados por coma | `NU,AMZN,SONY,XOM,WPM` |
| `BENCHMARK_TICKER` | Ticker del benchmark | `^GSPC` |
| `RISK_FREE_RATE` | Tasa libre de riesgo anual (default) | `0.04` |

---

## 8. Fundamentos Matemáticos

### VaR y CVaR (M5)

- **Histórico**: percentil empírico de la distribución de retornos.
- **Paramétrico**: `VaR = μ + σ · z_α`, `CVaR = μ − σ · φ(z_α)/(1−α)`, donde `z_α = Φ⁻¹(1−α)`.
- **Monte Carlo**: simulación de `N` retornos normales `~ N(μ, σ²)`, percentil del resultado.

### Test de Kupiec — POF (M5)

Estadístico de razón de verosimilitud:

```
LR = −2 [ N·ln(p/p̂) + (T−N)·ln((1−p)/(1−p̂)) ] ~ χ²(1) bajo H₀
```

donde `T` = observaciones totales, `N` = excepciones, `p` = tasa esperada (1−α), `p̂ = N/T`. Se rechaza el modelo si `p-value < 0.05`.

### CAPM (M4)

```
E[Rᵢ] = Rƒ + βᵢ · (E[Rₘ] − Rƒ)
βᵢ = Cov(Rᵢ, Rₘ) / Var(Rₘ)
```

`Rƒ` diaria: `(1 + Rƒ_anual)^(1/252) − 1`. Beta estimada por regresión OLS.

### Sharpe Ratio (M6)

```
SR = (Rₚ − Rƒ) / σₚ
```

Todos los Sharpe Ratio usan la tasa libre de riesgo anualizada (fuente: `^IRX` de Yahoo Finance).

### Optimización de Markowitz (M6)

10,000 portafolios aleatorios con pesos `wᵢ ≥ 0`, `Σwᵢ = 1`. Frontera eficiente trazada como nube de Volatilidad vs. Retorno coloreada por Sharpe.

Optimización por rendimiento objetivo con SLSQP:

```
min  σₚ = √(wᵀΣw)
s.t. Σwᵢ = 1,   wᵀμ = r_objetivo,   wᵢ ≥ 0
```

### Modelos ARCH/GARCH (M3)

- **ARCH(p)**: `σₜ² = ω + Σαᵢεₜ₋ᵢ²`
- **GARCH(1,1)**: `σₜ² = ω + α·εₜ₋₁² + β·σₜ₋₁²`
- **EGARCH(1,1)**: `ln σₜ² = ω + α·(|zₜ₋₁| − E|zₜ₋₁|) + γ·zₜ₋₁ + β·ln σₜ₋₁²`

Comparados por AIC y BIC. Pronóstico de volatilidad a 10 días con el GARCH(1,1).

### Alpha de Jensen y métricas benchmark (M8)

```
α_Jensen = Rₚ_anual − [Rƒ + β·(Rₘ_anual − Rƒ)]
TE = std(Rₚ − Rₘ) · √252
IR = (Rₚ_anual − Rₘ_anual) / TE
```

---

## 9. Despliegue en la Nube

### Docker

```bash
docker build -t risklab-usta .
docker run -p 8001:8001 -e JWT_SECRET=tu_clave_segura risklab-usta
```

### Render.com

El archivo `render.yaml` configura automáticamente:
- Runtime Python
- Comando de build: `pip install -r requirements.txt`
- Comando de inicio: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
- `JWT_SECRET` generado automáticamente por Render
- Disco persistente de 1 GB para la base de datos SQLite

### Heroku / Railway

Usar el `Procfile` incluido. Configurar `JWT_SECRET` como variable de entorno en la plataforma.

---

## 10. Dependencias Principales

| Paquete | Uso |
|---------|-----|
| `fastapi` | Framework REST API |
| `uvicorn` | Servidor ASGI |
| `yfinance` | Descarga de datos históricos de Yahoo Finance |
| `pandas` / `numpy` | Manipulación de datos y cálculo matricial |
| `scipy` | Tests estadísticos, optimización SLSQP |
| `arch` | Modelos ARCH/GARCH/EGARCH |
| `statsmodels` | Test de Ljung-Box para clustering de volatilidad |
| `bcrypt` | Hashing seguro de contraseñas |
| `python-jose[cryptography]` | Generación y verificación de tokens JWT |
| `pydantic[email]` | Validación de datos y esquemas |
| `plotly` (CDN) | Visualizaciones interactivas en el frontend |

---

## 11. Estructura de Datos (data.js / /api/v1/all)

```json
{
  "generated_at": "2025-04-20 10:00:00",
  "tickers": ["NU", "AMZN", "SONY", "XOM", "WPM"],
  "rf_rate": 0.043,
  "rf_source": "^IRX Yahoo Finance",
  "technical": { "<ticker>": [ { "Date": "...", "Close": ..., "SMA_20": ..., ... } ] },
  "returns":   { "<ticker>": { "stats": {...}, "normality": {...}, "stylized_facts": {...} } },
  "volatility":{ "<ticker>": { "ARCH(1)": {...}, "GARCH(1,1)": {...}, "EGARCH(1,1)": {...}, "Forecast_10d": [...] } },
  "risk":      { "<ticker>": { "capm": {...}, "var": {...}, "var_99": {...} } },
  "portfolio": { "Max_Sharpe": {...}, "Min_Volatility": {...}, "Correlation": {...}, "Frontier": {...} },
  "benchmark": { "Port": {...}, "Bench": {...}, "Jensen_Alpha": ..., "Beta": ..., ... }
}
```

---

*Desarrollado con Python 3.11 · FastAPI · Plotly.js · Yahoo Finance API*  
*Universidad Santo Tomás — Proyecto Integrador Riesgo Financiero*
